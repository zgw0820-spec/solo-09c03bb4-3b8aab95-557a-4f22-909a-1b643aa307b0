"""Read-path optimization: maintained counts, snapshot reads, big history."""

from __future__ import annotations

import sqlite3
import threading
import time

from app.alarm.service import COUNT_GROUPS, GROUP_OF_STATE, AlarmService


def ground_truth(db_path: str) -> tuple[dict[str, int], int]:
    """Independent full-table aggregation: the definition of correct."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT state, COUNT(*) FROM alarm_rounds GROUP BY state"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM alarm_rounds").fetchone()[0]
    finally:
        conn.close()
    counts = {group: 0 for group in COUNT_GROUPS}
    for state, n in rows:
        counts[GROUP_OF_STATE[state]] += n
    return counts, total


def assert_counts_match(service: AlarmService) -> dict:
    state = service.get_state()
    expected, total = ground_truth(service.db_path)
    assert state["counts_ok"] is True
    assert state["counts"] == expected
    assert state["total_rounds"] == total
    return state


def seed_rounds(db_path: str, states: list[str]) -> None:
    """Bulk-insert terminal rounds with the given states."""
    conn = sqlite3.connect(db_path)
    try:
        base = 1_700_000_000.0
        conn.executemany(
            """
            INSERT INTO alarm_rounds
                (notification_no, triggered_at, recovered_at, state,
                 attempts, needs_verify, updated_at)
            VALUES (?, ?, ?, ?, 1, 0, ?)
            """,
            [
                (f"ALM-{i:06d}", base + i, base + i + 5, state, base + i + 5)
                for i, state in enumerate(states, 1)
            ],
        )
        conn.commit()
    finally:
        conn.close()


def test_counts_track_interleaved_state_changes(alarm, notifier, clock):
    """Trigger/recover/deliver/suppress interleaved: counts must always equal
    a ground-truth GROUP BY over the very same committed rows."""
    for round_no in range(6):
        alarm.trigger()
        assert_counts_match(alarm)
        if round_no % 3 == 2:
            alarm.set_maintenance(clock() + 3600)
        alarm.tick()
        assert_counts_match(alarm)
        alarm.recover()
        state = assert_counts_match(alarm)
        assert state["active_round"] is None
        if round_no % 3 == 2:
            alarm.end_maintenance()
            assert_counts_match(alarm)
    # Six rounds, all delivered before recovery -> all sent.
    final = alarm.get_state()
    assert final["counts"]["sent"] == 6
    assert final["total_rounds"] == 6
    assert sum(final["counts"].values()) == 6
    assert [r["state"] for r in final["rounds"]] == ["sent"] * 6


def test_counts_cover_suppressed_and_skipped_flow(alarm, notifier, clock):
    alarm.set_maintenance(clock() + 60)
    alarm.trigger()
    state = assert_counts_match(alarm)
    assert state["counts"]["suppressed"] == 1

    alarm.recover()  # recovered inside maintenance -> skipped
    state = assert_counts_match(alarm)
    assert state["counts"]["skipped"] == 1
    assert state["counts"]["suppressed"] == 0

    clock.advance(61)  # window expires; nothing left to backfill
    alarm.tick()
    state = assert_counts_match(alarm)
    assert state["counts"]["skipped"] == 1
    assert notifier[0].recent(limit=10) == []


def test_old_database_is_upgraded_and_counts_rebuilt(tmp_path, notifier, clock):
    """A v1 database has no counts tables/triggers; opening it must upgrade
    the schema and serve correct statistics derived from the rounds table."""
    _store, url = notifier
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE maintenance_windows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at REAL NOT NULL,
            ends_at REAL NOT NULL,
            closed_at REAL,
            close_reason TEXT,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE alarm_rounds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            notification_no TEXT UNIQUE,
            triggered_at REAL NOT NULL,
            recovered_at REAL,
            maintenance_id INTEGER REFERENCES maintenance_windows(id),
            state TEXT NOT NULL,
            scheduled_at REAL,
            accepted_at REAL,
            attempts INTEGER NOT NULL DEFAULT 0,
            last_attempt_at REAL,
            next_attempt_at REAL,
            delivery_started_at REAL,
            needs_verify INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            note TEXT,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX idx_alarm_rounds_state ON alarm_rounds(state)")
    conn.commit()
    conn.close()
    seed_rounds(str(db), ["sent"] * 40 + ["skipped"] * 7 + ["cancelled"] * 3)

    service = AlarmService(db, url, clock=clock, start_worker=False, retry_base=0.01)
    try:
        state = assert_counts_match(service)
        assert state["counts"] == {
            "suppressed": 0,
            "pending": 0,
            "sent": 40,
            "skipped": 7,
            "cancelled": 3,
        }
        assert state["total_rounds"] == 50
        # New writes on the upgraded database keep the counters exact.
        service.trigger()
        service.tick()
        state = assert_counts_match(service)
        assert state["counts"]["sent"] == 41
        assert state["total_rounds"] == 51
    finally:
        service.close()


def test_corrupted_counts_are_rebuilt_not_displayed(alarm, notifier):
    """Simulate a failed/partial counter maintenance: the next read must
    detect the mismatch and repair it instead of showing wrong numbers."""
    alarm.trigger()
    alarm.tick()
    state = alarm.get_state()
    assert state["counts"]["sent"] == 1

    conn = sqlite3.connect(alarm.db_path)
    conn.execute("UPDATE alarm_state_counts SET n = 9999 WHERE state = 'sent'")
    conn.commit()
    conn.close()

    state = alarm.get_state()
    assert state["counts_ok"] is True  # repaired before serving
    assert state["counts"]["sent"] == 1
    assert state["total_rounds"] == 1


def test_corrupted_total_is_rebuilt_not_displayed(alarm, notifier):
    alarm.trigger()
    alarm.tick()
    conn = sqlite3.connect(alarm.db_path)
    conn.execute("UPDATE alarm_meta SET value = '123456' WHERE key = 'total_rounds'")
    conn.commit()
    conn.close()

    state = alarm.get_state()
    assert state["counts_ok"] is True
    assert state["total_rounds"] == 1
    assert state["counts"]["sent"] == 1


def test_restart_with_broken_counts_recovers_truth(alarm_factory, notifier, clock):
    """Counter maintenance failure + service restart: startup rebuild must
    converge to the rounds table, never to the stale counters."""
    service = alarm_factory()
    service.trigger()
    service.tick()
    service.recover()
    service.trigger()
    service.recover()  # pending, never attempted -> cancelled
    service.close()

    # Wreck the summary tables as if maintenance had failed mid-way.
    conn = sqlite3.connect(service.db_path)
    conn.execute("DELETE FROM alarm_state_counts")
    conn.execute("UPDATE alarm_meta SET value = '0' WHERE key = 'total_rounds'")
    conn.commit()
    conn.close()

    restarted = alarm_factory()
    state = restarted.get_state()
    assert state["counts_ok"] is True
    assert state["counts"]["sent"] == 1
    assert state["counts"]["cancelled"] == 1
    assert state["total_rounds"] == 2
    assert_counts_match(restarted)


def test_dropped_counts_table_self_heals(alarm, notifier):
    """If the summary table itself is lost, the read path recreates and
    repopulates it rather than failing or inventing numbers."""
    alarm.trigger()
    alarm.tick()
    conn = sqlite3.connect(alarm.db_path)
    conn.execute("DROP TABLE alarm_state_counts")
    conn.commit()
    conn.close()

    state = alarm.get_state()
    assert state["counts_ok"] is True
    assert state["counts"]["sent"] == 1
    assert state["total_rounds"] == 1


def test_counts_unavailable_is_reported_not_faked(alarm, notifier, monkeypatch):
    """If the summary cannot be read or rebuilt, the API must say so instead
    of serving plausible-looking wrong numbers; the list still works."""
    alarm.trigger()
    alarm.tick()
    assert alarm.get_state()["counts_ok"] is True

    def broken_select(*args, **kwargs):
        raise sqlite3.OperationalError("no such table: alarm_state_counts")

    monkeypatch.setattr(alarm, "_read_counts", broken_select)
    monkeypatch.setattr(alarm, "_rebuild_counts_locked", lambda: None)
    state = alarm.get_state()
    assert state["counts_ok"] is False
    assert state["counts"] is None
    assert state["total_rounds"] is None
    assert len(state["rounds"]) == 1  # list unaffected
    assert state["rounds"][0]["state"] == "sent"


def test_large_history_interleaved_states(alarm, notifier, clock):
    """200k stored rounds: the page still gets the newest 100 plus exact
    full-history counts, and the live flow (suppress -> expire -> deliver ->
    recover) keeps list and counters consistent with committed data."""
    store, _ = notifier
    seed_rounds(
        alarm.db_path,
        ["sent"] * 150_000 + ["skipped"] * 30_000 + ["cancelled"] * 20_000,
    )

    state = assert_counts_match(alarm)
    assert state["total_rounds"] == 200_000
    assert state["counts"] == {
        "suppressed": 0,
        "pending": 0,
        "sent": 150_000,
        "skipped": 30_000,
        "cancelled": 20_000,
    }
    assert len(state["rounds"]) == 100
    ids = [r["id"] for r in state["rounds"]]
    assert ids == sorted(ids, reverse=True)
    assert ids[0] == 200_000

    # Live flow on top of the big history: suppress, expire, deliver, recover.
    alarm.set_maintenance(clock() + 30)
    created = alarm.trigger()
    assert created["state"] == "suppressed"
    state = assert_counts_match(alarm)
    assert state["counts"]["suppressed"] == 1
    assert state["rounds"][0]["id"] == created["id"]
    assert state["rounds"][0]["state"] == "suppressed"

    clock.advance(31)
    state = alarm.get_state()  # read path performs the expiry transition
    assert state["rounds"][0]["state"] == "pending"
    assert state["counts"]["pending"] == 1
    assert state["counts"]["suppressed"] == 0

    alarm.tick()
    state = assert_counts_match(alarm)
    assert state["rounds"][0]["state"] == "sent"
    assert state["counts"]["sent"] == 150_001
    assert state["counts"]["pending"] == 0
    delivered = store.recent(limit=10)
    assert [r["notification_no"] for r in delivered] == [created["notification_no"]]

    alarm.recover()
    state = assert_counts_match(alarm)
    assert state["total_rounds"] == 200_001
    assert state["active_round"] is None
    assert len(state["rounds"]) == 100

    # Polling must stay cheap: no linear scan of the 200k-row history.
    # (Full-scan implementation measured ~35ms avg here; maintained counters
    # measure ~3ms. Threshold sits between the two with wide headroom.)
    alarm.get_state()  # warm any caches
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        alarm.get_state()
        samples.append(time.perf_counter() - start)
    avg = sum(samples) / len(samples)
    assert avg < 0.025, f"get_state too slow with 200k rounds: avg={avg*1000:.1f}ms"


def test_polling_does_not_wait_for_delivery_lock(alarm, notifier):
    """While a writer holds the processing lock, a poll must still complete
    from its own snapshot instead of queueing behind it."""
    alarm.trigger()
    alarm.tick()
    alarm.recover()

    acquired = threading.Event()
    release = threading.Event()

    def hold_lock():
        with alarm._lock:
            acquired.set()
            release.wait(5)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert acquired.wait(5)
    try:
        start = time.perf_counter()
        state = alarm.get_state()
        elapsed = time.perf_counter() - start
    finally:
        release.set()
        holder.join(timeout=5)
    assert elapsed < 1.0
    assert state["counts_ok"] is True
    assert state["counts"]["sent"] == 1
    assert len(state["rounds"]) == 1
