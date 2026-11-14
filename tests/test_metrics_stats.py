import math

import pytest

from src.metrics.metrics import (
    accuracy, benefit_breakdown, escalation_rate, mean_field, summarize,
)
from src.metrics.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff


def oc(esc, correct, small_ok, strong_ok, cost):
    return {"escalated": esc, "correct": correct, "small_answer_correct": small_ok,
            "strong_correct": strong_ok, "cost_seconds": cost,
            "overhead_seconds": 0.0, "latency_seconds": cost}


def test_benefit_breakdown_counts():
    outcomes = [
        oc(True, True, False, True, 11),    # fixed
        oc(True, True, True, True, 11),     # wasted
        oc(True, False, False, False, 11),  # both wrong
        oc(True, False, True, False, 11),   # harmed
        oc(False, True, True, True, 1),     # not escalated: ignored
    ]
    assert benefit_breakdown(outcomes) == {
        "escalated": 4, "fixed": 1, "wasted": 1, "both_wrong": 1, "harmed": 1}


def test_summarize_and_empty_input():
    outcomes = [oc(False, True, True, True, 1), oc(False, True, True, True, 1),
                oc(True, False, False, False, 11), oc(True, True, False, True, 11)]
    s = summarize(outcomes)
    assert s["n"] == 4
    assert s["accuracy"] == pytest.approx(0.75)
    assert s["cost_per_q"] == pytest.approx(6.0)
    assert s["escalation_rate"] == pytest.approx(0.5)
    assert s["n_fixed"] == 1 and s["n_both_wrong"] == 1
    assert math.isnan(accuracy([])) and math.isnan(escalation_rate([]))


def test_mean_field_skips_missing_values():
    rows = [{"x": 1.0}, {"x": None}, {"x": 3.0}, {}]
    assert mean_field(rows, "x") == pytest.approx(2.0)


def test_bootstrap_ci_contains_mean_and_is_reproducible():
    vals = [1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
    a = bootstrap_ci(vals, seed=3)
    b = bootstrap_ci(vals, seed=3)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]
    assert a["mean"] == pytest.approx(0.7)


def test_bootstrap_ci_degenerate():
    r = bootstrap_ci([1.0] * 20)
    assert r["ci_low"] == 1.0 and r["ci_high"] == 1.0


def test_paired_diff_zero_and_positive():
    same = paired_bootstrap_diff([1, 0, 1, 1], [1, 0, 1, 1])
    assert same["mean_diff"] == 0.0 and same["p_value"] == 1.0
    better = paired_bootstrap_diff([1.0] * 20, [0.0] * 20)
    assert better["mean_diff"] == 1.0
    assert better["ci_low"] == 1.0 and better["p_value"] == 0.0


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 0], [1])


def test_mcnemar_one_sided_counts():
    a = [True] * 5 + [False] * 2
    b = [False] * 5 + [False] * 2
    r = mcnemar(a, b)
    assert r["a_right_b_wrong"] == 5 and r["a_wrong_b_right"] == 0
    assert r["p_value"] == pytest.approx(0.0625)


def test_mcnemar_balanced_and_no_discordant():
    a = [True, True, True, False, False, False]
    b = [False, False, False, True, True, True]
    assert mcnemar(a, b)["p_value"] == pytest.approx(1.0)
    assert mcnemar([True, False], [True, False])["p_value"] == 1.0