"""Tests for the fixed-threshold inactivity detector."""

from datetime import datetime, timedelta

from activity_data import mark_gaps
from detect_inactivity_anomalies import (DEFAULT_BOUT_THRESHOLD_HOURS,
                                         DEFAULT_REST_THRESHOLD_HOURS, find_resting_bouts,
                                         judge_day)

H = 3600


def day(observed_h, resting_h, ruminating_h):
    rest_of_day = observed_h - resting_h - ruminating_h
    return {"observed_s": observed_h * H, "laying_resting": resting_h * H,
            "ruminating": ruminating_h * H, "foraging": rest_of_day * H}


def judge(totals, bout_h):
    return judge_day(totals, bout_h, DEFAULT_REST_THRESHOLD_HOURS, DEFAULT_BOUT_THRESHOLD_HOURS,
                     min_coverage_h=18, min_rumination_h=1.0)


def test_normal_pasture_day_is_ok():
    assert judge(day(24, 10.4, 4.4), 1.5)[0] == "ok"


def test_stall_day_with_high_resting_and_long_bout_is_anomaly():
    # e.g. device 1 on 2026-11-16: 19.1h resting, 16.1h bout
    status, _, rest_flag, bout_flag = judge(day(24, 19.1, 4.3), 16.1)
    assert status == "anomaly" and rest_flag and bout_flag


def test_either_check_alone_is_enough():
    # e.g. device 5 on 2026-11-16: 16.4h resting but 4.3h bout; and a long bout alone
    assert judge(day(24, 16.4, 7.0), 2.0)[2:] == (True, False)
    assert judge(day(24, 10.0, 5.0), 4.0)[0] == "anomaly"


def test_resting_threshold_is_on_the_24h_normalised_value():
    # 11h resting in 20h observed = 13.2h/24h -> above 13
    status, resting_24h, rest_flag, _ = judge(day(20, 11.0, 4.0), 1.0)
    assert rest_flag and round(resting_24h, 1) == 13.2


def test_idle_collar_is_not_judged():
    # e.g. device 3 in July: ~24h "resting", no rumination, endless bout
    assert judge(day(24, 24.0, 0.0), 110.0)[0] == "no_animal"


def test_single_stray_reading_does_not_split_a_bout_but_a_gap_does():
    t = datetime(2026, 1, 1)

    def seq(a, b, label):
        return [(t + timedelta(minutes=x), t + timedelta(minutes=min(x + 5, b)), label)
                for x in range(a, b, 5)]

    rows = (seq(0, 60, "laying_resting") + seq(60, 65, "ruminating")       # 5 min: bridged
            + seq(65, 120, "laying_resting") + seq(120, 130, "foraging")   # 10 min: breaks
            + seq(130, 200, "laying_resting") + seq(240, 250, "laying_resting"))  # 40 min hole
    bouts = find_resting_bouts(mark_gaps(rows, 15 * 60), 6 * 60)
    spans = [((b["start"] - t).seconds // 60, (b["end"] - t).seconds // 60) for b in bouts]
    assert spans == [(0, 120), (130, 200), (240, 250)]
