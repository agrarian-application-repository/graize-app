"""Tests for the fixed-threshold feeding detector (detect_feeding_anomalies.judge_day)."""

from detect_feeding_anomalies import DEFAULT_THRESHOLD_HOURS, judge_day

H = 3600


def day(observed_h, foraging_h, ruminating_h):
    return {"observed_s": observed_h * H, "foraging": foraging_h * H,
            "ruminating": ruminating_h * H,
            "laying_resting": (observed_h - foraging_h - ruminating_h) * H}


def judge(totals):
    return judge_day(totals, threshold_h=DEFAULT_THRESHOLD_HOURS, min_coverage_h=18,
                     min_rumination_h=1.0)


def test_normal_pasture_day_is_ok():
    assert judge(day(24, 8.5, 4.4))[0] == "ok"


def test_stall_day_with_almost_no_foraging_is_anomaly():
    # e.g. device 5 on 2026-11-15: 0.05h foraging, 6h ruminating
    assert judge(day(24, 0.05, 6.0))[0] == "anomaly"


def test_threshold_is_on_the_24h_normalised_value():
    # 4.5h foraging in 20h observed = 5.4h/24h -> below 6
    status, foraging_24h, _ = judge(day(20, 4.5, 4.0))
    assert status == "anomaly" and round(foraging_24h, 2) == 5.4


def test_light_eater_just_above_threshold_is_ok():
    # e.g. device 6 on 2026-10-20: 6.1h foraging
    assert judge(day(24, 6.1, 7.3))[0] == "ok"


def test_idle_collar_without_rumination_is_not_judged():
    # e.g. device 3 in July: ~24h "resting", no rumination, no foraging
    assert judge(day(24, 0.0, 0.0))[0] == "no_animal"


def test_short_observation_is_inconclusive():
    assert judge(day(10, 0.0, 3.0))[0] == "inconclusive"
