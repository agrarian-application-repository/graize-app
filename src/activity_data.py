"""
activity_data.py
─────────────────────────────────────────────────────────────────────────────
Shared helpers for the anomaly detectors (detect_feeding_anomalies.py,
detect_inactivity_anomalies.py, detect_separation_anomalies.py): DB access,
reporting-gap handling, per-day behaviour totals, the idle-collar check and
the command-line flags the behaviour detectors share.

Why gap handling lives here
───────────────────────────
Inertial_activities rows are built by splitting each reading's slice
[Time_i, Time_i+1) in half between its two predictions (see
backfill_inertial_activities.py). Readings normally arrive every few
minutes, so rows are a few minutes long. When a collar goes quiet, though,
the slice before the next reading spans the whole silence and the last
prediction gets stretched over it: rows of days or even weeks exist in
the data, labelled with whatever the collar said last. That time was never
observed.

mark_gaps() turns every such row (longer than max_row_s) — and any hole
between consecutive rows — into an explicit gap entry with label None, so
downstream code can skip unobserved time instead of counting it as
behaviour.

Why an idle-collar check
────────────────────────
Collars have spent weeks lying unused (e.g. device 3 until ~2026-10-16:
~24h "resting", no foraging, stationary GPS). A live cow ruminates for
hours every day — on 88 normal collar-days the minimum was ~2h — while an
idle collar shows almost none. Days below --min-rumination-hours are
reported "no_animal" and not judged by the behaviour detectors, and their
GPS fixes are dropped by the separation detector (idle_collar_days).
"""

from collections import defaultdict
from datetime import datetime, time, timedelta, timezone

import psycopg2

from app import settings

# A row longer than this means the readings around it were > 2x apart,
# i.e. the collar stopped reporting. Normal rows are ~2-5 min.
DEFAULT_MAX_ROW_MINUTES = 15.0
DEFAULT_MIN_COVERAGE_HOURS = 18.0    # hours actually observed before a day is judged
DEFAULT_MIN_RUMINATION_HOURS = 1.0   # below this (per 24h) the collar is not on a live animal

RUMINATING_LABEL = "ruminating"


def get_connection():
    return psycopg2.connect(**{
        "host": settings.db_host,
        "port": settings.db_port,
        "dbname": settings.db_name,
        "user": settings.db_user,
        "password": settings.db_password,
    })


def fetch_device_euis(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT Device_id, EUI FROM Devices_set")
        return dict(cur.fetchall())


def fetch_activities_by_device(conn) -> dict:
    """Returns {device_id: [(time_start, time_end, label), ...]}, time-ordered."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ia.Device_id, ia.Time_start, ia.Time_end, s.Name
            FROM Inertial_activities ia
            JOIN Inertial_activities_set s ON s.Inertial_activity_id = ia.Inertial_activity_id
            ORDER BY ia.Device_id, ia.Time_start
            """
        )
        rows = cur.fetchall()

    by_device = defaultdict(list)
    for device_id, time_start, time_end, label in rows:
        by_device[device_id].append((time_start, time_end, label))
    return by_device


def mark_gaps(activities, max_row_s):
    """activities: [(time_start, time_end, label), ...] time-ordered for one device.

    Returns the same timeline with unobserved time made explicit: any row
    longer than max_row_s, and any hole longer than max_row_s between one
    row's end and the next row's start, becomes (start, end, None).
    Zero/negative-length rows are dropped."""
    out = []
    prev_end = None
    for time_start, time_end, label in activities:
        if time_end <= time_start:
            continue
        if prev_end is not None and (time_start - prev_end).total_seconds() > max_row_s:
            out.append((prev_end, time_start, None))
        if (time_end - time_start).total_seconds() > max_row_s:
            out.append((time_start, time_end, None))
        else:
            out.append((time_start, time_end, label))
        prev_end = time_end if prev_end is None else max(prev_end, time_end)
    return out


def as_tz(tz):
    """Accepts a tzinfo (e.g. ZoneInfo("Europe/Rome"), which follows summer/winter
    time) or a fixed offset in hours from UTC, and returns a tzinfo."""
    if isinstance(tz, (int, float)):
        return timezone(timedelta(hours=tz))
    return tz


def local_date(ts, tz):
    """Calendar date of an aware timestamp in tz (tzinfo or hours from UTC)."""
    return ts.astimezone(as_tz(tz)).date()


def local_midnight(date, tz):
    """Aware datetime of 00:00 on `date` in tz (tzinfo or hours from UTC)."""
    return datetime.combine(date, time.min, tzinfo=as_tz(tz))


def split_by_day(time_start, time_end, tz=0.0):
    """Split [time_start, time_end) (aware timestamps) into per-local-calendar-day
    (date, seconds) chunks, where "local" is tz — a tzinfo or a fixed offset in
    hours from UTC. With a real timezone, summer/winter time days are 23/25 h."""
    tz = as_tz(tz)
    chunks = []
    cursor = time_start
    while cursor < time_end:
        day = local_date(cursor, tz)
        chunk_end = min(time_end, local_midnight(day + timedelta(days=1), tz))
        chunks.append((day, (chunk_end - cursor).total_seconds()))
        cursor = chunk_end
    return chunks


def compute_daily_label_times(activities, tz=0.0) -> dict:
    """activities: output of mark_gaps() for one device.
    Returns {local_date: {"observed_s": float, <label>: float, ...}} with the
    observed (gap-free) time and the seconds carrying each behaviour label."""
    daily = defaultdict(lambda: defaultdict(float))
    for time_start, time_end, row_label in activities:
        if row_label is None:
            continue
        for date, seconds in split_by_day(time_start, time_end, tz):
            daily[date]["observed_s"] += seconds
            daily[date][row_label] += seconds
    return daily


def per_24h(totals, label):
    """Seconds of `label` in a day's totals, normalised to hours per 24h observed."""
    return totals.get(label, 0.0) / totals["observed_s"] * 24


def day_precheck(totals, min_coverage_h, min_rumination_h):
    """Shared guards for a collar-day before any behaviour check.
    Returns "inconclusive" (too little observed), "no_animal" (no rumination,
    so the collar isn't on a live cow), or None when the day can be judged."""
    if totals.get("observed_s", 0.0) < min_coverage_h * 3600:
        return "inconclusive"
    if per_24h(totals, RUMINATING_LABEL) < min_rumination_h:
        return "no_animal"
    return None


def idle_collar_days(activities_by_device, max_row_s, tz, min_rumination_h,
                     min_observed_h=1.0) -> set:
    """Returns {(device_id, local_date), ...} for collar-days with at least
    min_observed_h of observed behaviour but less than min_rumination_h of
    rumination per 24h — a collar not on a live animal. Days without any
    behaviour data aren't included (unknown, not idle)."""
    idle = set()
    for device_id, activities in activities_by_device.items():
        daily = compute_daily_label_times(mark_gaps(activities, max_row_s), tz)
        for date, totals in daily.items():
            if (totals["observed_s"] >= min_observed_h * 3600
                    and per_24h(totals, RUMINATING_LABEL) < min_rumination_h):
                idle.add((device_id, date))
    return idle


def add_common_args(parser):
    """CLI flags shared by the behaviour detectors."""
    parser.add_argument("--min-coverage-hours", type=float, default=DEFAULT_MIN_COVERAGE_HOURS,
                        help=f"Need at least this many hours actually observed that day to judge "
                             f"it (default: {DEFAULT_MIN_COVERAGE_HOURS}).")
    parser.add_argument("--min-rumination-hours", type=float, default=DEFAULT_MIN_RUMINATION_HOURS,
                        help=f"Days ruminating less than this (per 24h) are treated as an idle "
                             f"collar, not judged (default: {DEFAULT_MIN_RUMINATION_HOURS}).")
    parser.add_argument("--max-row-minutes", type=float, default=DEFAULT_MAX_ROW_MINUTES,
                        help=f"Activity rows longer than this are treated as reporting gaps "
                             f"(default: {DEFAULT_MAX_ROW_MINUTES}).")
    parser.add_argument("--utc-offset-hours", type=float, default=0.0,
                        help="Hours to shift UTC timestamps by before computing calendar-day "
                             "boundaries, e.g. 2 for UTC+2 (default: 0).")
    parser.add_argument("--exclude-devices", default="",
                        help="Comma-separated Device_ids to skip (default: none; idle collars "
                             "are caught by the rumination check).")


def parse_excluded(arg) -> set:
    return {int(x) for x in arg.split(",") if x.strip()}


def rounded(value, digits=2):
    return None if value is None else round(value, digits)
