"""Tests for anomaly_scheduler: run times, timezone-aware days, and storing
findings without duplicates. No real database is used."""

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

import anomaly_scheduler
from activity_data import split_by_day
from anomaly_scheduler import (HERD_SEPARATION, LOW_FEEDING, next_run_time, parse_run_hours,
                               run_once, store_anomalies)

ROME = ZoneInfo("Europe/Rome")
HOURS = (0, 6, 12, 18)


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# ── schedule ────────────────────────────────────────────────────────────────

def test_parse_run_hours():
    assert parse_run_hours("18, 0,6,12") == (0, 6, 12, 18)
    with pytest.raises(ValueError):
        parse_run_hours("0,24")


def test_next_run_is_the_next_slot_in_farm_time():
    # 2026-11-19 09:30 UTC = 10:30 in Rome (winter, UTC+1) -> 12:00 Rome = 11:00 UTC
    assert next_run_time(utc(2026, 11, 19, 9, 30), HOURS, ROME) == utc(2026, 11, 19, 11, 0)


def test_run_exactly_at_a_slot_schedules_the_following_one():
    assert next_run_time(utc(2026, 11, 19, 5, 0), HOURS, ROME) == utc(2026, 11, 19, 11, 0)


def test_after_the_last_slot_runs_at_midnight_next_day():
    # 20:00 Rome -> 00:00 Rome next day = 23:00 UTC same day
    assert next_run_time(utc(2026, 11, 19, 19, 0), HOURS, ROME) == utc(2026, 11, 19, 23, 0)


def test_summer_time_slots_shift_in_utc():
    # 2026-07-01 03:00 UTC = 05:00 Rome (summer, UTC+2) -> 06:00 Rome = 04:00 UTC
    assert next_run_time(utc(2026, 7, 1, 3, 0), HOURS, ROME) == utc(2026, 7, 1, 4, 0)


def test_day_clocks_go_back():
    # 2026-10-25: 03:00 summer -> 02:00 winter. 04:00 UTC = 05:00 Rome -> 06:00 = 05:00 UTC
    assert next_run_time(utc(2026, 10, 25, 4, 0), HOURS, ROME) == utc(2026, 10, 25, 5, 0)


def test_day_clocks_go_forward():
    # 2026-03-29: 02:00 winter -> 03:00 summer. 03:30 UTC = 05:30 Rome -> 06:00 = 04:00 UTC
    assert next_run_time(utc(2026, 3, 29, 3, 30), HOURS, ROME) == utc(2026, 3, 29, 4, 0)


def test_farm_days_follow_summer_and_winter_time():
    # 2026-10-25 in Rome lasts 25 hours: 22:00 UTC on the 24th to 23:00 UTC on the 25th
    chunks = split_by_day(utc(2026, 10, 24, 22, 0), utc(2026, 10, 25, 23, 0), ROME)
    assert chunks == [(date(2026, 10, 25), 25 * 3600.0)]


# ── storing findings ────────────────────────────────────────────────────────

class FakeDB:
    """In-memory stand-in for Anomalies_set / Anomalies_detection."""

    def __init__(self, lock_free=True):
        self.types = {}          # name -> id
        self.detections = []     # (device_id, anomaly_id, time)
        self.lock_free = lock_free
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class FakeCursor:
    def __init__(self, db):
        self.db = db
        self.result = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        if sql.startswith("SELECT pg_try_advisory_lock"):
            self.result = [(self.db.lock_free,)]
        elif sql.startswith("SELECT pg_advisory_unlock"):
            self.result = [(True,)]
        elif sql == "SELECT Anomaly_id, Name FROM Anomalies_set":
            self.result = [(i, n) for n, i in self.db.types.items()]
        elif sql.startswith("INSERT INTO Anomalies_set"):
            new_id = len(self.db.types) + 1
            self.db.types[params[0]] = new_id
            self.result = [(new_id,)]
        elif sql == "SELECT Device_id, Anomaly_id, Time FROM Anomalies_detection":
            # Postgres returns timestamptz in the session timezone; simulate UTC.
            self.result = [(d, a, t.astimezone(timezone.utc)) for d, a, t in self.db.detections]
        else:
            raise AssertionError(f"unexpected SQL: {sql}")

    def executemany(self, sql, rows):
        assert sql.startswith("INSERT INTO Anomalies_detection")
        self.db.detections.extend(rows)

    def fetchone(self):
        return self.result[0]

    def fetchall(self):
        return list(self.result)


FOUND = {
    (5, LOW_FEEDING, datetime(2026, 11, 14, tzinfo=ROME)),
    (5, HERD_SEPARATION, utc(2026, 10, 20, 16, 6, 22, 162281)),
}


def test_findings_are_stored_with_their_types():
    db = FakeDB()
    assert store_anomalies(db, FOUND) == 2
    assert set(db.types) == {LOW_FEEDING, HERD_SEPARATION}
    assert len(db.detections) == 2


def test_running_again_does_not_duplicate():
    db = FakeDB()
    store_anomalies(db, FOUND)
    assert store_anomalies(db, FOUND) == 0
    assert len(db.detections) == 2


def test_only_new_findings_are_added():
    db = FakeDB()
    store_anomalies(db, FOUND)
    more = FOUND | {(6, LOW_FEEDING, datetime(2026, 11, 15, tzinfo=ROME))}
    assert store_anomalies(db, more) == 1
    assert len(db.detections) == 3 and len(db.types) == 2


def test_same_moment_in_another_timezone_is_not_a_new_finding():
    db = FakeDB()
    store_anomalies(db, FOUND)
    # 00:00 Rome on 14 Nov == 23:00 UTC on 13 Nov
    assert store_anomalies(db, {(5, LOW_FEEDING, utc(2026, 11, 13, 23, 0))}) == 0


def test_run_is_skipped_when_another_instance_holds_the_lock(monkeypatch):
    db = FakeDB(lock_free=False)
    monkeypatch.setattr(anomaly_scheduler, "get_connection", lambda: db)
    assert run_once(ROME) is None
    assert db.detections == [] and db.closed
