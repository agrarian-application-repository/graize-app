"""
detect_inactivity_anomalies.py
─────────────────────────────────────────────────────────────────────────────
Fixed-threshold anomaly detector for "prolonged inactivity or resting",
from Inertial_activities (label laying_resting). Two checks per collar-day:

  1. Daily resting — laying_resting time normalised to a full day
     (resting_s / observed_s * 24) above --rest-threshold-hours
     (default 13 h/day).

  2. Longest lying bout — the longest unbroken laying_resting stretch that
     starts that day, above --bout-threshold-hours (default 3.5 h). One
     unusually long lie is a stronger health signal (lameness, illness, a
     downer animal) than a high total. Predictions flip every few minutes
     and one stray non-resting reading would otherwise split a long rest
     in two, so non-resting interruptions totalling <= --bridge-minutes are
     absorbed into the bout (bout length = wall-clock time from first to
     last resting reading). A reporting gap always ends a bout.

A day is "anomaly" if either check fires; the report shows both.

Why fixed thresholds and not a per-animal rolling baseline: collars have
been fitted, removed and re-fitted, often on different animals and with
weeks of silence in between, so a collar's recent history is not one
animal's normal. Thresholds were chosen from the distribution of all
collar-days (as of 2026-11-19), 88 normal live-animal days vs 14 stall days
(14-18 Nov):
  - daily resting: normal 4.5-15.1 h (median 10.4, middle half 8.8-12.0),
    stall 11.8-21.4 h. 13 h flags 4 normal days (device 2 on 5-7 Aug, just
    before its collar went quiet; device 1 on 19 Oct) and 12/14 stall days.
  - longest bout: normal 0.55-5.25 h (median 1.5, middle half 1.1-2.0),
    stall 2.3-38.6 h. 3.5 h flags 5 normal days (devices 1/6 on 17-22 Oct,
    right after fitting; device 1 on 4 Aug) and 13/14 stall days.

Shared guards (activity_data): unobserved time is removed first
(mark_gaps); days with too little observed time are "inconclusive"; days
without rumination are "no_animal" (idle collar) — day_precheck.

Time_start/Time_end are stored in UTC; --utc-offset-hours shifts day
boundaries to the farm's local day (e.g. 2 for UTC+2).

Usage:
    python detect_inactivity_anomalies.py --utc-offset-hours 2
    python detect_inactivity_anomalies.py --utc-offset-hours 2 --rest-threshold-hours 15
    python detect_inactivity_anomalies.py --utc-offset-hours 2 --output inactivity.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activity_data import (DEFAULT_MAX_ROW_MINUTES, DEFAULT_MIN_COVERAGE_HOURS,
                           DEFAULT_MIN_RUMINATION_HOURS, add_common_args,
                           compute_daily_label_times, day_precheck, fetch_activities_by_device,
                           fetch_device_euis, get_connection, local_date, mark_gaps,
                           parse_excluded, per_24h, rounded)

RESTING_LABEL = "laying_resting"

DEFAULT_REST_THRESHOLD_HOURS = 13.0   # flag days resting more than this (per 24h observed)
DEFAULT_BOUT_THRESHOLD_HOURS = 3.5    # flag days whose longest lying bout exceeds this
DEFAULT_BRIDGE_MINUTES = 6.0          # ~one reading: absorb a single stray non-resting prediction

FIELDNAMES = ["device_id", "eui", "date", "observed_hours", "resting_hours_24h",
              "longest_bout_hours", "bout_start", "bout_end", "rest_flag", "bout_flag", "status"]


def find_resting_bouts(observed, bridge_s):
    """observed: output of mark_gaps() for one device (gaps have label None).
    Returns [{"start", "end"}, ...]: stretches of laying_resting where any
    non-resting interruption totals <= bridge_s. Reporting gaps end a bout."""
    bouts = []
    current = None
    interruption_s = 0.0
    for time_start, time_end, label in observed:
        if label == RESTING_LABEL:
            if current is None:
                current = {"start": time_start, "end": time_end}
            else:
                current["end"] = time_end
            interruption_s = 0.0
        elif current is not None:
            interruption_s += (time_end - time_start).total_seconds()
            if label is None or interruption_s > bridge_s:
                bouts.append(current)
                current = None
                interruption_s = 0.0
    if current is not None:
        bouts.append(current)
    return bouts


def longest_bout_by_day(bouts, tz):
    """Returns {local_date: bout} with the longest bout starting on each day
    (tz: tzinfo or hours from UTC)."""
    longest = {}
    for bout in bouts:
        date = local_date(bout["start"], tz)
        if date not in longest or (bout["end"] - bout["start"]) > (
                longest[date]["end"] - longest[date]["start"]):
            longest[date] = bout
    return longest


def judge_day(totals, longest_bout_h, rest_threshold_h, bout_threshold_h,
              min_coverage_h, min_rumination_h):
    """totals: {"observed_s", <label>: seconds, ...} for one collar-day;
    longest_bout_h: longest bout starting that day (0 if none).
    Returns (status, resting_24h, rest_flag, bout_flag)."""
    status = day_precheck(totals, min_coverage_h, min_rumination_h)
    if status == "inconclusive":
        return status, None, False, False
    resting_24h = per_24h(totals, RESTING_LABEL)
    if status == "no_animal":
        return status, resting_24h, False, False
    rest_flag = resting_24h > rest_threshold_h
    bout_flag = longest_bout_h > bout_threshold_h
    return ("anomaly" if rest_flag or bout_flag else "ok"), resting_24h, rest_flag, bout_flag


def evaluate(activities_by_device, tz, rest_threshold_h=DEFAULT_REST_THRESHOLD_HOURS,
             bout_threshold_h=DEFAULT_BOUT_THRESHOLD_HOURS, bridge_minutes=DEFAULT_BRIDGE_MINUTES,
             min_coverage_h=DEFAULT_MIN_COVERAGE_HOURS,
             min_rumination_h=DEFAULT_MIN_RUMINATION_HOURS,
             max_row_minutes=DEFAULT_MAX_ROW_MINUTES, euis=None, excluded=()):
    """Judges every collar-day. activities_by_device: output of
    fetch_activities_by_device; tz: farm timezone (tzinfo or hours from UTC).
    Returns one report row per collar-day; "date" is a datetime.date and
    "bout_start"/"bout_end" are aware datetimes (or None)."""
    euis = euis or {}
    rows = []
    for device_id in sorted(activities_by_device):
        if device_id in excluded:
            continue
        observed = mark_gaps(activities_by_device[device_id], max_row_minutes * 60)
        daily = compute_daily_label_times(observed, tz)
        longest = longest_bout_by_day(find_resting_bouts(observed, bridge_minutes * 60), tz)
        for date in sorted(daily):
            bout = longest.get(date)
            bout_h = (bout["end"] - bout["start"]).total_seconds() / 3600 if bout else 0.0
            status, resting_24h, rest_flag, bout_flag = judge_day(
                daily[date], bout_h, rest_threshold_h, bout_threshold_h,
                min_coverage_h, min_rumination_h)
            rows.append({
                "device_id": device_id,
                "eui": euis.get(device_id, ""),
                "date": date,
                "observed_hours": round(daily[date]["observed_s"] / 3600, 2),
                "resting_hours_24h": rounded(resting_24h),
                "longest_bout_hours": round(bout_h, 2),
                "bout_start": bout["start"] if bout else None,
                "bout_end": bout["end"] if bout else None,
                "rest_flag": rest_flag,
                "bout_flag": bout_flag,
                "status": status,
            })
    return rows


def fmt(value):
    return "-" if value is None else value


def main():
    parser = argparse.ArgumentParser(
        description="Flag collar-days with unusually much resting or an unusually long lying bout.")
    parser.add_argument("--rest-threshold-hours", type=float, default=DEFAULT_REST_THRESHOLD_HOURS,
                        help=f"Flag a day whose resting, normalised to 24h, exceeds this "
                             f"(default: {DEFAULT_REST_THRESHOLD_HOURS}).")
    parser.add_argument("--bout-threshold-hours", type=float, default=DEFAULT_BOUT_THRESHOLD_HOURS,
                        help=f"Flag a day whose longest lying bout exceeds this "
                             f"(default: {DEFAULT_BOUT_THRESHOLD_HOURS}).")
    parser.add_argument("--bridge-minutes", type=float, default=DEFAULT_BRIDGE_MINUTES,
                        help=f"Non-resting interruptions totalling up to this many minutes don't "
                             f"break a lying bout (default: {DEFAULT_BRIDGE_MINUTES}).")
    add_common_args(parser)
    parser.add_argument("--output", help="Optional CSV path to write the full per-day report to.")
    args = parser.parse_args()

    excluded = parse_excluded(args.exclude_devices)

    conn = get_connection()
    euis = fetch_device_euis(conn)
    by_device = fetch_activities_by_device(conn)
    conn.close()

    report_rows = evaluate(by_device, args.utc_offset_hours, args.rest_threshold_hours,
                           args.bout_threshold_hours, args.bridge_minutes,
                           args.min_coverage_hours, args.min_rumination_hours,
                           args.max_row_minutes, euis, excluded)
    for r in report_rows:
        r["date"] = r["date"].isoformat()
        for key in ("bout_start", "bout_end"):
            r[key] = r[key].isoformat() if r[key] else None

    print(f"{'device':<8}{'eui':<20}{'date':<12}{'obs_h':<8}{'rest24h':<9}{'bout_h':<8}"
          f"{'flags':<12}{'status':<14}")
    for r in report_rows:
        flags = "+".join(f for f, on in (("rest", r["rest_flag"]), ("bout", r["bout_flag"])) if on)
        print(f"{r['device_id']:<8}{r['eui']:<20}{r['date']:<12}{r['observed_hours']:<8}"
              f"{fmt(r['resting_hours_24h']):<9}{r['longest_bout_hours']:<8}{flags or '-':<12}"
              f"{r['status']:<14}")

    counts = {s: sum(1 for r in report_rows if r["status"] == s)
              for s in ("anomaly", "ok", "no_animal", "inconclusive")}
    n_rest = sum(r["rest_flag"] for r in report_rows)
    n_bout = sum(r["bout_flag"] for r in report_rows)
    print(f"\n{counts['anomaly']} anomalous day(s) out of {counts['anomaly'] + counts['ok']} judged "
          f"({n_rest} with resting > {args.rest_threshold_hours}h/24h, {n_bout} with a bout "
          f"> {args.bout_threshold_hours}h); {counts['no_animal']} idle-collar day(s), "
          f"{counts['inconclusive']} with < {args.min_coverage_hours}h observed.")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
