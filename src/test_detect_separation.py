"""
Regression tests for detect_separation_anomalies, modelled on the real
2026-07-23..29 data: devices 1 and 2 grazing together (~13m apart) while
device 3's collar sat stationary ~340m away. Device 3 must come out as
separated and devices 1/2 must not.
"""

import math
import random
from datetime import date, datetime, timedelta, timezone

from detect_separation_anomalies import DEFAULT_DISTANCE_THRESHOLD_M, detect, drop_idle_fixes

BASE_LAT, BASE_LON = 44.398, 8.951
M_PER_DEG_LAT = 111_320.0
CADENCE = timedelta(minutes=5.5)
TOLERANCE = timedelta(minutes=15)


def offset(lat, lon, north_m, east_m):
    return (lat + north_m / M_PER_DEG_LAT,
            lon + east_m / (M_PER_DEG_LAT * math.cos(math.radians(lat))))


def simulate(days, separated_collar=True, seed=0):
    """Two cows wandering together, plus (optionally) a stationary collar 340m north."""
    rng = random.Random(seed)
    start = datetime(2026, 7, 23, tzinfo=timezone.utc)
    by_device = {1: [], 2: [], 3: []}
    herd_lat, herd_lon = BASE_LAT, BASE_LON
    t = start
    while t < start + timedelta(days=days):
        herd_lat, herd_lon = offset(herd_lat, herd_lon, rng.gauss(0, 3), rng.gauss(0, 3))
        for dev, skew in ((1, 0), (2, 40)):
            ts = t + timedelta(seconds=skew)   # asynchronous reporting
            by_device[dev].append((ts, *offset(herd_lat, herd_lon, rng.gauss(0, 8), rng.gauss(0, 8))))
        if separated_collar:
            by_device[3].append((t + timedelta(seconds=20),
                                 *offset(BASE_LAT, BASE_LON, 340 + rng.gauss(0, 5), rng.gauss(0, 5))))
        t += CADENCE
    return {d: fixes for d, fixes in by_device.items() if fixes}


def run(by_device, min_herd_size=2):
    return detect(by_device, TOLERANCE, min_herd_size=min_herd_size,
                  threshold_m=DEFAULT_DISTANCE_THRESHOLD_M, min_duration_minutes=15.0)


def test_stationary_collar_away_from_herd_is_separated():
    readings, bouts = run(simulate(days=2))
    assert all(r["status"] == "separated" for r in readings[3])
    assert bouts[3], "the stationary collar should produce a separation bout"
    assert all(b["max_distance_m"] > 300 for b in bouts[3])


def test_herd_mates_are_not_flagged_because_of_the_absent_one():
    readings, bouts = run(simulate(days=2))
    assert bouts[1] == [] and bouts[2] == []
    for dev in (1, 2):
        assert not any(r["status"] == "separated" for r in readings[dev])


def test_two_collars_alone_are_never_judged():
    readings, bouts = run(simulate(days=2, separated_collar=False))
    for dev in (1, 2):
        assert {r["status"] for r in readings[dev]} == {"no_herd"}
        assert bouts[dev] == []


def test_idle_collar_days_are_dropped_and_no_longer_flagged():
    fixes = simulate(days=2)
    idle = {(3, date(2026, 7, 23)), (3, date(2026, 7, 24))}
    filtered = drop_idle_fixes(fixes, idle, utc_offset_hours=0)
    assert filtered[3] == [] and len(filtered[1]) == len(fixes[1])
    _, bouts = run(filtered)
    assert all(b == [] for b in bouts.values())
