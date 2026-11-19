"""
detect_separation_anomalies.py
─────────────────────────────────────────────────────────────────────────────
Fixed-threshold anomaly detector for an animal separating from the herd:
flags stretches of at least --min-duration-minutes where a collar is more
than --distance-threshold-m (default 100 m) from every other collar while
the other collars stay together.

1. Idle collars out. Collar-days without rumination (activity_data.
   idle_collar_days — the collar isn't on a live animal) are dropped
   entirely, both as the checked animal and as herd members.

2. Herd snapshot. Collars report asynchronously, so for each valid GPS fix
   at time T, every other collar's nearest valid fix within
   --time-tolerance-minutes of T is pulled. Two numbers come out:
       distance_m  — distance to the NEAREST other collar. Large means the
                     animal is far from every other animal.
       cohesion_m  — how close the other collars are to each other: the
                     median, over the others, of each one's distance to its
                     nearest neighbour among the others (needs >= 2 others).
   Nearest-neighbour distance rather than distance to a herd centroid: with
   only two or three collars the centroid is just a midpoint, and an animal
   standing apart drags it far enough to make its herd-mates look separated.

3. Who left whom. With only one other collar there's no telling which of
   the two animals wandered off, so a fix needs at least --min-herd-size
   other collars (default 2), otherwise it's "no_herd". A fix is then:
       "ok"          distance_m <= threshold
       "separated"   distance_m >  threshold and cohesion_m <= threshold
                     (far from everyone, while the others are together)
       "herd_split"  distance_m >  threshold and cohesion_m >  threshold
                     (the others are scattered too — not one animal leaving)
   Two animals leaving side by side aren't flagged: each has the other as a
   close neighbour.

4. Bouts. Consecutive "separated" fixes are merged into a bout; inconclusive
   fixes in between don't break it (missing information, not evidence the
   animal rejoined) unless the time since the last separated fix exceeds
   --time-tolerance-minutes. Only bouts lasting at least
   --min-duration-minutes are reported, so single noisy GPS fixes aren't
   flagged on their own.

Why a fixed 100 m: collars have been fitted, removed and re-fitted, so a
rolling baseline doesn't describe one stable herd. From all fixes with >= 2
other collars (as of 2026-11-19, idle days removed): cows together sit at a
median of 11 m (middle half up to 22 m); a separate cluster sits at
200-440 m (an idle collar on 23-24 Jul, and pasture vs. stall groups in
Nov). Any threshold from ~75 m to ~110 m gives the same bouts: the only
separation among cows in normal periods is device 5 on 2026-10-20 (194 m,
22 min); below ~75 m routine 60-100 m grazing drift starts being flagged.

Known limitation: animals kept apart on purpose (e.g. one cow on pasture
while the others are in the stall) are reported as separated — the
detector doesn't know where each animal is supposed to be.

Only coordinates passing app.py's is_valid_coordinate() are used (filters
the INT32_MAX no-fix sentinel).

Usage:
    python detect_separation_anomalies.py --utc-offset-hours 2
    python detect_separation_anomalies.py --utc-offset-hours 2 --distance-threshold-m 75
    python detect_separation_anomalies.py --utc-offset-hours 2 --output separations.csv
"""

import argparse
import csv
import os
import sys
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import timedelta
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activity_data import (DEFAULT_MAX_ROW_MINUTES, DEFAULT_MIN_RUMINATION_HOURS,
                           fetch_activities_by_device, fetch_device_euis, get_connection,
                           idle_collar_days, local_date, parse_excluded)
from app import is_valid_coordinate, LAT_SCALED_BOUND, LON_SCALED_BOUND
from classify_movement import haversine, LAT_LON_SCALE

DEFAULT_DISTANCE_THRESHOLD_M = 100.0
DEFAULT_TIME_TOLERANCE_MINUTES = 15.0
DEFAULT_MIN_HERD_SIZE = 2
DEFAULT_MIN_DURATION_MINUTES = 15.0

STATUSES = ("separated", "ok", "herd_split", "no_herd")


def fetch_valid_fixes_by_device(conn) -> dict:
    """Returns {device_id: [(time, lat_deg, lon_deg), ...]}, time-ordered, with
    the INT32_MAX no-fix sentinel and other out-of-range coordinates excluded."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT Device_id, Time, Latitude, Longitude
            FROM Location_history
            WHERE Latitude > %s AND Latitude < %s
              AND Longitude > %s AND Longitude < %s
            ORDER BY Device_id, Time
            """,
            (-LAT_SCALED_BOUND, LAT_SCALED_BOUND, -LON_SCALED_BOUND, LON_SCALED_BOUND)
        )
        rows = cur.fetchall()

    by_device = defaultdict(list)
    for device_id, time, lat, lon in rows:
        if is_valid_coordinate(lat, lon):
            by_device[device_id].append((time, lat / LAT_LON_SCALE, lon / LAT_LON_SCALE))
    return by_device


def drop_idle_fixes(by_device, idle_days, tz):
    """Removes fixes that fall on an idle collar-day {(device_id, local_date)}
    (tz: tzinfo or hours from UTC, the same one the idle days were built with)."""
    return {d: [f for f in fixes if (d, local_date(f[0], tz)) not in idle_days]
            for d, fixes in by_device.items()}


def nearest_fix(times, fixes, target_time, tolerance):
    """times/fixes: parallel lists for one device, time-ordered.
    Returns the (time, lat, lon) in fixes closest to target_time within
    tolerance (a timedelta), or None if none qualifies."""
    idx = bisect_left(times, target_time)
    best = None
    best_diff = None
    for i in (idx - 1, idx):
        if 0 <= i < len(times):
            diff = abs(times[i] - target_time)
            if diff <= tolerance and (best_diff is None or diff < best_diff):
                best, best_diff = i, diff
    return fixes[best] if best is not None else None


def herd_cohesion(herd):
    """herd: [(lat, lon), ...] of the OTHER animals, len >= 2. Median over
    them of each one's distance to its nearest neighbour within the herd."""
    return median(min(haversine(a[0], a[1], b[0], b[1]) for j, b in enumerate(herd) if j != i)
                  for i, a in enumerate(herd))


def measure_readings(by_device, tolerance):
    """Returns {device_id: [{"time", "herd_size", "distance_m", "cohesion_m"}, ...]},
    time-ordered. distance_m is None with no other collar aligned within
    tolerance; cohesion_m is None with fewer than two."""
    device_ids = sorted(d for d in by_device if by_device[d])
    times_by_device = {d: [t for t, _, _ in by_device[d]] for d in device_ids}

    results = {}
    for device_id in device_ids:
        readings = []
        for time, lat, lon in by_device[device_id]:
            herd = []
            for other_id in device_ids:
                if other_id == device_id:
                    continue
                fix = nearest_fix(times_by_device[other_id], by_device[other_id], time, tolerance)
                if fix is not None:
                    herd.append((fix[1], fix[2]))
            readings.append({
                "time": time,
                "herd_size": len(herd),
                "distance_m": min((haversine(lat, lon, p[0], p[1]) for p in herd), default=None),
                "cohesion_m": herd_cohesion(herd) if len(herd) >= 2 else None,
            })
        results[device_id] = readings
    return results


def judge_readings(readings, threshold_m, min_herd_size):
    """Adds "status" to each reading of one device, in place."""
    for r in readings:
        if r["herd_size"] < min_herd_size:
            r["status"] = "no_herd"
        elif r["distance_m"] <= threshold_m:
            r["status"] = "ok"
        elif r["cohesion_m"] <= threshold_m:
            r["status"] = "separated"
        else:
            r["status"] = "herd_split"


def build_separation_bouts(readings, max_gap):
    """readings: judged readings of one device, time-ordered.
    Merges "separated" readings into bouts. "ok" ends a bout; inconclusive
    readings don't, unless more than max_gap has passed since the bout's
    last separated reading.
    Returns [{"start", "end", "max_distance_m", "min_herd_size"}, ...]."""
    bouts = []
    current = None
    for r in readings:
        if current is not None and r["time"] - current["end"] > max_gap:
            bouts.append(current)
            current = None
        if r["status"] == "separated":
            if current is None:
                current = {"start": r["time"], "end": r["time"],
                           "max_distance_m": r["distance_m"], "min_herd_size": r["herd_size"]}
            else:
                current["end"] = r["time"]
                current["max_distance_m"] = max(current["max_distance_m"], r["distance_m"])
                current["min_herd_size"] = min(current["min_herd_size"], r["herd_size"])
        elif r["status"] == "ok" and current is not None:
            bouts.append(current)
            current = None
    if current is not None:
        bouts.append(current)
    return bouts


def detect(by_device, tolerance, min_herd_size, threshold_m, min_duration_minutes):
    """Runs the whole pipeline on {device_id: [(time, lat, lon), ...]} (idle
    fixes already dropped). Returns (readings_by_device, {device_id: [bout, ...]}),
    keeping only bouts lasting at least min_duration_minutes."""
    readings_by_device = measure_readings(by_device, tolerance)
    bouts_by_device = {}
    for device_id, readings in readings_by_device.items():
        judge_readings(readings, threshold_m, min_herd_size)
        bouts_by_device[device_id] = [
            b for b in build_separation_bouts(readings, tolerance)
            if (b["end"] - b["start"]).total_seconds() / 60 >= min_duration_minutes]
    return readings_by_device, bouts_by_device


def evaluate(fixes_by_device, activities_by_device, tz,
             threshold_m=DEFAULT_DISTANCE_THRESHOLD_M,
             tolerance=timedelta(minutes=DEFAULT_TIME_TOLERANCE_MINUTES),
             min_herd_size=DEFAULT_MIN_HERD_SIZE,
             min_duration_minutes=DEFAULT_MIN_DURATION_MINUTES,
             min_rumination_h=DEFAULT_MIN_RUMINATION_HOURS, excluded=()):
    """Full pipeline from DB data: drops excluded devices and idle collar-days,
    then runs detect(). fixes_by_device: output of fetch_valid_fixes_by_device;
    activities_by_device: output of fetch_activities_by_device (for the
    idle-collar check); tz: farm timezone (tzinfo or hours from UTC).
    Returns (fixes, kept_fixes, readings_by_device, bouts_by_device)."""
    fixes = {d: f for d, f in fixes_by_device.items() if d not in excluded}
    idle_days = idle_collar_days(activities_by_device, DEFAULT_MAX_ROW_MINUTES * 60, tz,
                                 min_rumination_h)
    kept = drop_idle_fixes(fixes, idle_days, tz)
    readings_by_device, bouts_by_device = detect(kept, tolerance, min_herd_size, threshold_m,
                                                 min_duration_minutes)
    return fixes, kept, readings_by_device, bouts_by_device


def main():
    parser = argparse.ArgumentParser(
        description="Flag an animal staying far from every other animal while the rest of "
                     "the herd stays together.")
    parser.add_argument("--distance-threshold-m", type=float, default=DEFAULT_DISTANCE_THRESHOLD_M,
                        help=f"Separated when farther than this from the nearest other collar "
                             f"(default: {DEFAULT_DISTANCE_THRESHOLD_M}).")
    parser.add_argument("--time-tolerance-minutes", type=float, default=DEFAULT_TIME_TOLERANCE_MINUTES,
                        help=f"Max time gap when aligning another device's fix to the checked "
                             f"reading, and max gap inside a bout (default: {DEFAULT_TIME_TOLERANCE_MINUTES}).")
    parser.add_argument("--min-herd-size", type=int, default=DEFAULT_MIN_HERD_SIZE,
                        help=f"Need at least this many OTHER collars aligned in time; below 2 it's "
                             f"impossible to tell which animal left (default: {DEFAULT_MIN_HERD_SIZE}).")
    parser.add_argument("--min-duration-minutes", type=float, default=DEFAULT_MIN_DURATION_MINUTES,
                        help=f"Only flag a separation bout lasting at least this long "
                             f"(default: {DEFAULT_MIN_DURATION_MINUTES}).")
    parser.add_argument("--min-rumination-hours", type=float, default=DEFAULT_MIN_RUMINATION_HOURS,
                        help=f"Collar-days ruminating less than this (per 24h) are idle collars "
                             f"and their fixes are dropped (default: {DEFAULT_MIN_RUMINATION_HOURS}).")
    parser.add_argument("--utc-offset-hours", type=float, default=0.0,
                        help="Hours to shift UTC timestamps by when deciding which local day a "
                             "fix belongs to for the idle-collar check (default: 0).")
    parser.add_argument("--exclude-devices", default="",
                        help="Comma-separated Device_ids to drop entirely (default: none).")
    parser.add_argument("--output", help="Optional CSV path for the flagged-bout report.")
    args = parser.parse_args()

    tolerance = timedelta(minutes=args.time_tolerance_minutes)
    excluded = parse_excluded(args.exclude_devices)

    conn = get_connection()
    euis = fetch_device_euis(conn)
    all_fixes = fetch_valid_fixes_by_device(conn)
    activities = fetch_activities_by_device(conn)
    conn.close()

    fixes, by_device, readings_by_device, bouts_by_device = evaluate(
        all_fixes, activities, args.utc_offset_hours, args.distance_threshold_m, tolerance,
        args.min_herd_size, args.min_duration_minutes, args.min_rumination_hours, excluded)

    report_rows = []
    for device_id in sorted(bouts_by_device):
        for bout in bouts_by_device[device_id]:
            report_rows.append({
                "device_id": device_id,
                "eui": euis.get(device_id, ""),
                "start": bout["start"].isoformat(),
                "end": bout["end"].isoformat(),
                "duration_minutes": round((bout["end"] - bout["start"]).total_seconds() / 60, 1),
                "max_distance_m": round(bout["max_distance_m"], 1),
                "min_herd_size": bout["min_herd_size"],
            })

    print("Fixes per device:\n")
    print(f"{'device':<8}{'fixes':<8}{'idle_dropped':<14}" + "".join(f"{s:<14}" for s in STATUSES))
    for device_id in sorted(fixes):
        counts = Counter(r["status"] for r in readings_by_device.get(device_id, []))
        dropped = len(fixes[device_id]) - len(by_device[device_id])
        print(f"{device_id:<8}{len(fixes[device_id]):<8}{dropped:<14}"
              + "".join(f"{counts.get(s, 0):<14}" for s in STATUSES))

    print(f"\n\nSeparation bouts >= {args.min_duration_minutes}min "
          f"(> {args.distance_threshold_m}m from the nearest other collar, "
          f">= {args.min_herd_size} other collars, others together):\n")
    print(f"{'device':<8}{'eui':<20}{'start':<34}{'end':<34}{'dur_min':<9}{'max_m':<9}")
    for row in report_rows:
        print(f"{row['device_id']:<8}{row['eui']:<20}{row['start']:<34}{row['end']:<34}"
              f"{row['duration_minutes']:<9}{row['max_distance_m']:<9}")
    print(f"\n{len(report_rows)} separation bout(s) flagged.")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["device_id", "eui", "start", "end",
                                                     "duration_minutes", "max_distance_m",
                                                     "min_herd_size"])
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
