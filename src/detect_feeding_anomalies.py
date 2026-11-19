"""
detect_feeding_anomalies.py
─────────────────────────────────────────────────────────────────────────────
Fixed-threshold anomaly detector: flags collar-days where foraging falls
below --threshold-hours (default 6 h/day).

Why a fixed threshold and not a per-animal rolling baseline: collars have
been fitted, removed and re-fitted, often on different animals and with
weeks of silence in between, so "this collar's recent history" is not a
reliable picture of one animal's normal. The threshold was chosen from the
distribution of all collar-days instead (as of 2026-11-19): 88 normal
live-animal days spanned 4.7-16.8 h (median 8.5 h, middle half 7.3-9.8 h),
while stall days (14-18 Nov, where the behaviour model doesn't recognise
bunk feeding) sat at 0-1.3 h. 6 h catches every stall day and flags ~5%
of normal days (the lowest, 4.7-5.7 h); 6.5 h flagged ~15%, mostly
light eaters (devices 5/6, ~7 h normal) dipping just below it.

Per device, per local calendar day:

  1. Unobserved time is removed first (activity_data.mark_gaps).
  2. Days with too little observed time are "inconclusive"; days without
     rumination are "no_animal" (idle collar) — activity_data.day_precheck.
  3. Foraging is normalised to a full day (foraging_s / observed_s * 24);
     below --threshold-hours → "anomaly", otherwise "ok".

Time_start/Time_end are stored in UTC; --utc-offset-hours shifts day
boundaries to the farm's local day (e.g. 2 for UTC+2).

Usage:
    python detect_feeding_anomalies.py --utc-offset-hours 2
    python detect_feeding_anomalies.py --utc-offset-hours 2 --threshold-hours 5
    python detect_feeding_anomalies.py --utc-offset-hours 2 --output feeding.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from activity_data import (RUMINATING_LABEL, add_common_args, compute_daily_label_times,
                           day_precheck, fetch_activities_by_device, fetch_device_euis,
                           get_connection, mark_gaps, parse_excluded, per_24h, rounded)

FORAGING_LABEL = "foraging"

DEFAULT_THRESHOLD_HOURS = 6.0        # flag days with less foraging than this (per 24h observed)

FIELDNAMES = ["device_id", "eui", "date", "observed_hours", "foraging_hours",
              "foraging_hours_24h", "ruminating_hours_24h", "status"]


def judge_day(totals, threshold_h, min_coverage_h, min_rumination_h):
    """totals: {"observed_s", <label>: seconds, ...} for one collar-day.
    Returns (status, foraging_24h, ruminating_24h); the 24h values are None
    when the day isn't observed enough to normalise."""
    status = day_precheck(totals, min_coverage_h, min_rumination_h)
    if status == "inconclusive":
        return status, None, None
    foraging_24h = per_24h(totals, FORAGING_LABEL)
    ruminating_24h = per_24h(totals, RUMINATING_LABEL)
    if status is None:
        status = "anomaly" if foraging_24h < threshold_h else "ok"
    return status, foraging_24h, ruminating_24h


def fmt(value):
    return "-" if value is None else value


def main():
    parser = argparse.ArgumentParser(
        description="Flag collar-days with less foraging than a fixed threshold.")
    parser.add_argument("--threshold-hours", type=float, default=DEFAULT_THRESHOLD_HOURS,
                        help=f"Flag a day whose foraging, normalised to 24h, is below this "
                             f"(default: {DEFAULT_THRESHOLD_HOURS}).")
    add_common_args(parser)
    parser.add_argument("--output", help="Optional CSV path to write the full per-day report to.")
    args = parser.parse_args()

    excluded = parse_excluded(args.exclude_devices)

    conn = get_connection()
    euis = fetch_device_euis(conn)
    by_device = fetch_activities_by_device(conn)
    conn.close()

    report_rows = []
    for device_id in sorted(by_device):
        if device_id in excluded:
            continue
        observed = mark_gaps(by_device[device_id], args.max_row_minutes * 60)
        daily = compute_daily_label_times(observed, args.utc_offset_hours)
        for date in sorted(daily):
            totals = daily[date]
            status, foraging_24h, ruminating_24h = judge_day(
                totals, args.threshold_hours, args.min_coverage_hours, args.min_rumination_hours)
            report_rows.append({
                "device_id": device_id,
                "eui": euis.get(device_id, ""),
                "date": date.isoformat(),
                "observed_hours": round(totals["observed_s"] / 3600, 2),
                "foraging_hours": round(totals.get(FORAGING_LABEL, 0.0) / 3600, 2),
                "foraging_hours_24h": rounded(foraging_24h),
                "ruminating_hours_24h": rounded(ruminating_24h),
                "status": status,
            })

    print(f"{'device':<8}{'eui':<20}{'date':<12}{'obs_h':<8}{'forage24h':<11}{'rumin24h':<10}{'status':<14}")
    for r in report_rows:
        print(f"{r['device_id']:<8}{r['eui']:<20}{r['date']:<12}{r['observed_hours']:<8}"
              f"{fmt(r['foraging_hours_24h']):<11}{fmt(r['ruminating_hours_24h']):<10}{r['status']:<14}")

    counts = {s: sum(1 for r in report_rows if r["status"] == s)
              for s in ("anomaly", "ok", "no_animal", "inconclusive")}
    print(f"\n{counts['anomaly']} anomalous day(s) out of {counts['anomaly'] + counts['ok']} judged "
          f"(foraging < {args.threshold_hours}h/24h); {counts['no_animal']} idle-collar day(s) "
          f"(rumination < {args.min_rumination_hours}h), {counts['inconclusive']} with "
          f"< {args.min_coverage_hours}h observed.")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(report_rows)
        print(f"Full report written to {args.output}")


if __name__ == "__main__":
    main()
