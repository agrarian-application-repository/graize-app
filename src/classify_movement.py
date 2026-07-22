"""
classify_movement.py
─────────────────────────────────────────────────────────────────────────────
Threshold-based movement classification for a single hour-long GPS window.

Reads N (lat, lon) fixes sampled at a fixed low frequency (default: every
5 minutes, 12 fixes = 1 hour) and outputs one of three movement classes:

    static    — animal barely moved; potential welfare concern
    grazing   — normal slow drift within a bounded area
    displaced — animal reached an unusually large distance from its start;
                potential escape, fence breach, or disturbance

Features computed:
    path_length              — total distance covered (sum of haversine steps)
    max_displacement_from_start — furthest point reached from the first fix
    motion_fraction          — fraction of steps exceeding STEP_THRESHOLD_M

Usage:
    python classify_movement.py fixes.csv
    python classify_movement.py fixes.csv --lat-col latitude --lon-col longitude
    python classify_movement.py fixes.csv --static-threshold 150 --displaced-threshold 400
"""

import argparse
import math
import os
import sys

import pandas as pd

# ── Tunable parameters ───────────────────────────────────────────────────────

# Scale factor applied to raw lat/lon values before computing distances.
# Set to 1.0 if the CSV already contains decimal degrees.
# Set to 1e7 if values are stored as int32 (e.g. 443980368 → 44.398°).
LAT_LON_SCALE = 1e7

# Minimum step distance (metres) to count as actual movement vs. GPS jitter.
STEP_THRESHOLD_M = 30.0

# Total path length (metres) below which the animal is classified as static.
STATIC_PATH_THRESHOLD_M = 100.0

# Max displacement from the starting fix (metres) above which the animal is
# classified as displaced.
DISPLACED_MAX_DISP_THRESHOLD_M = 300.0

# ── Haversine distance ───────────────────────────────────────────────────────

_EARTH_RADIUS_M = 6_371_000.0


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


# ── Feature extraction ───────────────────────────────────────────────────────

def extract_features(lats: list, lons: list) -> dict:
    n = len(lats)
    if n < 2:
        raise ValueError(f"Need at least 2 fixes, got {n}.")

    steps = [haversine(lats[i], lons[i], lats[i + 1], lons[i + 1]) for i in range(n - 1)]
    displacements_from_start = [haversine(lats[0], lons[0], lats[i], lons[i]) for i in range(1, n)]

    path_length = sum(steps)
    max_displacement = max(displacements_from_start)
    motion_fraction = sum(1 for s in steps if s > STEP_THRESHOLD_M) / len(steps)

    return {
        "path_length_m":              round(path_length, 1),
        "max_displacement_from_start_m": round(max_displacement, 1),
        "motion_fraction":            round(motion_fraction, 3),
        "n_fixes":                    n,
    }


# ── Classification ───────────────────────────────────────────────────────────

def classify(features: dict,
             static_threshold: float = STATIC_PATH_THRESHOLD_M,
             displaced_threshold: float = DISPLACED_MAX_DISP_THRESHOLD_M) -> str:
    if features["path_length_m"] < static_threshold:
        return "static"
    if features["max_displacement_from_start_m"] > displaced_threshold:
        return "displaced"
    return "grazing"


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Classify a sequence of GPS fixes into static / grazing / displaced.")
    parser.add_argument("fixes_csv",
                        help="CSV file with lat/lon fixes, one row per fix, ordered in time.")
    parser.add_argument("--lat-col", default="lat",
                        help="Name of the latitude column (default: lat).")
    parser.add_argument("--lon-col", default="lon",
                        help="Name of the longitude column (default: lon).")
    parser.add_argument("--static-threshold", type=float, default=STATIC_PATH_THRESHOLD_M,
                        help=f"Path length (m) below which → static (default: {STATIC_PATH_THRESHOLD_M}).")
    parser.add_argument("--displaced-threshold", type=float, default=DISPLACED_MAX_DISP_THRESHOLD_M,
                        help=f"Max displacement (m) above which → displaced (default: {DISPLACED_MAX_DISP_THRESHOLD_M}).")
    args = parser.parse_args()

    if not os.path.exists(args.fixes_csv):
        sys.exit(f"Error: file not found: {args.fixes_csv}")

    df = pd.read_csv(args.fixes_csv)

    for col in (args.lat_col, args.lon_col):
        if col not in df.columns:
            sys.exit(f"Error: column '{col}' not found. Available: {list(df.columns)}")

    lats = (df[args.lat_col] / LAT_LON_SCALE).tolist()
    lons = (df[args.lon_col] / LAT_LON_SCALE).tolist()

    try:
        features = extract_features(lats, lons)
    except ValueError as e:
        sys.exit(f"Error: {e}")

    label = classify(features, args.static_threshold, args.displaced_threshold)

    print(f"\nResult:  {label.upper()}\n")
    print(f"  Fixes:                       {features['n_fixes']}")
    print(f"  Path length:                 {features['path_length_m']:.1f} m")
    print(f"  Max displacement from start: {features['max_displacement_from_start_m']:.1f} m")
    print(f"  Motion fraction:             {features['motion_fraction']:.1%}")
    print(f"\nThresholds used:")
    print(f"  static    if path_length < {args.static_threshold:.0f} m")
    print(f"  displaced if max_displacement > {args.displaced_threshold:.0f} m")
    print(f"  grazing   otherwise\n")


if __name__ == "__main__":
    main()
