"""
anomaly_scheduler.py
─────────────────────────────────────────────────────────────────────────────
Runs the three anomaly detectors on a fixed daily schedule inside the
graize-app service and stores what they find in the database.

Schedule. A daemon thread (start_scheduler, called from app.py's startup
hook) runs the detectors once at startup — to catch up after downtime — and
then at each hour in run_hours (default 00, 06, 12, 18) in the farm's
timezone (default Europe/Rome, which follows summer/winter time). Errors are
logged and never stop the thread or affect ingest; the next run just tries
again. Each run re-evaluates all data, which takes seconds at current
volume.

What gets stored. Anomaly types live in Anomalies_set (created on first
use); each finding is one Anomalies_detection row (device, type, time):
    low_feeding      feeding check,           time = 00:00 of that farm day
    high_resting     daily resting check,     time = 00:00 of that farm day
    long_lying_bout  longest lying bout check, time = when the bout started
    herd_separation  separation check,        time = when the bout started
A row is only inserted if the same (device, type, time) isn't stored yet,
so repeated runs don't create duplicates. Anomalies are never deleted: a
finding that a later run no longer reports (e.g. a day that gained data)
stays in the table.

One run at a time. If graize-app runs in more than one place (e.g. Docker
on the VM and the portal), each copy has its own scheduler; a PostgreSQL
advisory lock makes a run skip when another copy is already running one.

The detector scripts (detect_*_anomalies.py) still work on their own from
the command line; this module reuses their evaluate() functions with their
default thresholds.
"""

import logging
import threading
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import detect_feeding_anomalies as feeding
import detect_inactivity_anomalies as inactivity
import detect_separation_anomalies as separation
from activity_data import fetch_activities_by_device, get_connection, local_midnight

logger = logging.getLogger("graize-app.anomalies")

DEFAULT_TIMEZONE = "Europe/Rome"
DEFAULT_RUN_HOURS = (0, 6, 12, 18)
ADVISORY_LOCK_KEY = 0x67726169   # arbitrary app-wide key ("grai") for pg_try_advisory_lock

LOW_FEEDING = "low_feeding"
HIGH_RESTING = "high_resting"
LONG_LYING_BOUT = "long_lying_bout"
HERD_SEPARATION = "herd_separation"


def parse_run_hours(text):
    """"0,6,12,18" -> (0, 6, 12, 18); rejects hours outside 0-23."""
    hours = tuple(sorted({int(h) for h in text.split(",") if h.strip()}))
    if not hours or any(h < 0 or h > 23 for h in hours):
        raise ValueError(f"run hours must be a comma-separated list of 0-23, got {text!r}")
    return hours


def next_run_time(now, run_hours, tz):
    """First run time strictly after `now` (aware), at one of run_hours
    o'clock in tz. Returned as an aware datetime in tz."""
    local_now = now.astimezone(tz)
    for day_offset in range(2):
        day = local_now.date() + timedelta(days=day_offset)
        for hour in sorted(run_hours):
            candidate = datetime.combine(day, time(hour), tzinfo=tz)
            if candidate > local_now:
                return candidate
    raise ValueError("run_hours is empty")


def find_anomalies(activities_by_device, fixes_by_device, tz):
    """Runs all detectors with their default thresholds and returns the set
    of findings {(device_id, anomaly_type, aware_time), ...}."""
    found = set()
    for row in feeding.evaluate(activities_by_device, tz):
        if row["status"] == "anomaly":
            found.add((row["device_id"], LOW_FEEDING, local_midnight(row["date"], tz)))
    for row in inactivity.evaluate(activities_by_device, tz):
        if row["rest_flag"]:
            found.add((row["device_id"], HIGH_RESTING, local_midnight(row["date"], tz)))
        if row["bout_flag"]:
            found.add((row["device_id"], LONG_LYING_BOUT, row["bout_start"]))
    _, _, _, bouts_by_device = separation.evaluate(fixes_by_device, activities_by_device, tz)
    for device_id, bouts in bouts_by_device.items():
        for bout in bouts:
            found.add((device_id, HERD_SEPARATION, bout["start"]))
    return found


def store_anomalies(conn, found):
    """Inserts the findings not stored yet; creates missing anomaly types.
    Returns the number of rows inserted. Doesn't commit."""
    with conn.cursor() as cur:
        cur.execute("SELECT Anomaly_id, Name FROM Anomalies_set")
        type_ids = {name: anomaly_id for anomaly_id, name in cur.fetchall()}
        for name in sorted({anomaly_type for _, anomaly_type, _ in found} - type_ids.keys()):
            cur.execute("INSERT INTO Anomalies_set (Name) VALUES (%s) RETURNING Anomaly_id", (name,))
            type_ids[name] = cur.fetchone()[0]

        cur.execute("SELECT Device_id, Anomaly_id, Time FROM Anomalies_detection")
        existing = {(d, a, t.astimezone(timezone.utc)) for d, a, t in cur.fetchall()}
        new_rows = sorted({(d, type_ids[name], t.astimezone(timezone.utc))
                           for d, name, t in found} - existing)
        if new_rows:
            cur.executemany(
                "INSERT INTO Anomalies_detection (Device_id, Anomaly_id, Time) VALUES (%s, %s, %s)",
                new_rows)
    return len(new_rows)


def run_once(tz):
    """One full detection run. Returns the number of new anomalies stored,
    or None if another instance held the lock. Raises on DB/detector errors."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
            if not cur.fetchone()[0]:
                logger.info("Anomaly run skipped: another instance is already running one")
                return None
        try:
            activities = fetch_activities_by_device(conn)
            fixes = separation.fetch_valid_fixes_by_device(conn)
            found = find_anomalies(activities, fixes, tz)
            inserted = store_anomalies(conn, found)
            conn.commit()
            logger.info("Anomaly run done: %d finding(s), %d new stored", len(found), inserted)
            return inserted
        finally:
            conn.rollback()   # no-op after commit; clears a failed transaction before unlocking
            with conn.cursor() as cur:
                cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
    finally:
        conn.close()


def _safe_run(tz):
    try:
        run_once(tz)
    except Exception:
        logger.exception("Anomaly run failed; will retry at the next scheduled time")


def _loop(stop_event, tz, run_hours, run_at_start):
    if run_at_start:
        _safe_run(tz)
    while not stop_event.is_set():
        next_run = next_run_time(datetime.now(timezone.utc), run_hours, tz)
        logger.info("Next anomaly run at %s", next_run.isoformat())
        wait_s = (next_run - datetime.now(timezone.utc)).total_seconds()
        if stop_event.wait(max(wait_s, 0)):
            break
        _safe_run(tz)


def start_scheduler(timezone_name=DEFAULT_TIMEZONE, run_hours=DEFAULT_RUN_HOURS,
                    run_at_start=True):
    """Starts the scheduler daemon thread. Returns (thread, stop_event);
    set stop_event to stop it."""
    tz = ZoneInfo(timezone_name)
    stop_event = threading.Event()
    thread = threading.Thread(target=_loop, args=(stop_event, tz, tuple(run_hours), run_at_start),
                              name="anomaly-scheduler", daemon=True)
    thread.start()
    logger.info("Anomaly scheduler started: runs at %s:00 %s%s",
                ", ".join(f"{h:02d}" for h in sorted(run_hours)), timezone_name,
                ", plus once now" if run_at_start else "")
    return thread, stop_event
