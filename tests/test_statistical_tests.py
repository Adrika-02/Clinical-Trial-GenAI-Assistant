import numpy as np

from src.stats.statistical_tests import (
    welch_t_test, cohens_d, chi_square_test, kruskal_wallis_test, confidence_interval,
)


def test_welch_t_test_detects_known_difference():
    rng = np.random.default_rng(0)
    group_a = rng.normal(60, 10, 200)
    group_b = rng.normal(50, 10, 200)
    result = welch_t_test(group_a, group_b)
    assert result.p_value < 0.05
    assert result.significant is True
    assert result.extra["cohens_d"] > 0.5


def test_welch_t_test_no_difference():
    rng = np.random.default_rng(1)
    group_a = rng.normal(50, 10, 200)
    group_b = rng.normal(50, 10, 200)
    result = welch_t_test(group_a, group_b)
    assert result.significant is False


def test_cohens_d_zero_for_identical_distributions():
    rng = np.random.default_rng(2)
    data = rng.normal(0, 1, 500)
    result = cohens_d(data, data)
    assert abs(result.statistic) < 1e-9
    assert result.extra["interpretation"] == "negligible"


def test_chi_square_detects_association():
    rng = np.random.default_rng(3)
    arm = np.array(["A"] * 100 + ["B"] * 100)
    outcome = np.array([True] * 70 + [False] * 30 + [True] * 30 + [False] * 70)
    result = chi_square_test(arm, outcome)
    assert result.p_value < 0.001
    assert result.significant is True


def test_kruskal_wallis_detects_difference():
    rng = np.random.default_rng(4)
    g1 = rng.normal(0, 1, 100)
    g2 = rng.normal(2, 1, 100)
    g3 = rng.normal(4, 1, 100)
    result = kruskal_wallis_test(g1, g2, g3, group_names=["low", "mid", "high"])
    assert result.p_value < 0.001


def test_confidence_interval_contains_true_mean():
    rng = np.random.default_rng(5)
    data = rng.normal(100, 15, 1000)
    ci = confidence_interval(data)
    assert ci["lower"] < 100 < ci["upper"]
