import math

import pytest

from src.routing.strategies import (
    agreement_score, difficulty_escalation, difficulty_score, escalate_if,
    fit_difficulty, logprob_escalation, logprob_signal, majority,
    majority_answer, oracle, random_escalation, random_uniform,
    self_consistency, small_only, strong_only,
)

SHORT = "Tom has 2 apples."
MEDIUM = "Anna buys 3 pens for 5 dollars each. She pays 20 dollars."
LONG = ("A shop sells 12 cakes on Monday. It sells 15 on Tuesday. On Wednesday "
        "it sells twice as many as Monday. How many cakes did it sell in total "
        "across the three days?")


def make_rec(qid="q1", ref=10, small_ok=True, strong_ok=True, small_secs=1.0,
             strong_secs=10.0, samples=None, logprobs=None, question=SHORT):
    rec = {
        "id": qid, "question": question, "reference_answer": ref,
        "small": {"greedy": {"correct": small_ok, "compute_seconds": small_secs,
                             "extracted": None, "logprobs": logprobs}},
        "strong": {"greedy": {"correct": strong_ok, "compute_seconds": strong_secs}},
    }
    if samples is not None:
        rec["small"]["samples"] = [{"extracted": v, "compute_seconds": 0.5}
                                   for v in samples]
    return rec


@pytest.mark.parametrize("values, answer, agr", [
    ([16, 16, 16], 16.0, 1.0),
    ([42, 38, 42], 42.0, 2 / 3),
    ([1, 2, 3], 1.0, 1 / 3),          # 3-way tie: earliest answer wins
    ([None, None, 5], None, 2 / 3),   # missing answers count as their own value
])
def test_majority_and_agreement(values, answer, agr):
    samples = [{"extracted": v} for v in values]
    ans, a = majority(samples)
    assert ans == answer
    assert a == pytest.approx(agr)
    assert majority_answer(samples) == answer
    assert agreement_score(samples) == pytest.approx(agr)


def test_escalate_if_directions():
    assert escalate_if(0.5, 0.8, "below") is True
    assert escalate_if(0.9, 0.8, "below") is False
    assert escalate_if(0.5, 0.8, "above") is False
    assert escalate_if(0.9, 0.8, "above") is True
    assert escalate_if(None, 0.8, "below") is True      # missing signal escalates
    with pytest.raises(ValueError):
        escalate_if(0.5, 0.8, "sideways")


def test_threshold_between_values_gotcha():
    # 2/3 = 0.6667 is BELOW 0.67, so that threshold escalates the 2/3 case.
    assert escalate_if(2 / 3, 0.67, "below") is True
    assert escalate_if(2 / 3, 0.5, "below") is False
    assert escalate_if(2 / 3, 0.8, "below") is True


def test_logprob_signal_modes():
    lps = [math.log(0.9), math.log(0.5)]
    assert logprob_signal(lps, "mean") == pytest.approx(0.7)
    assert logprob_signal(lps, "min") == pytest.approx(0.5)
    assert logprob_signal(None) is None
    assert logprob_signal([]) is None
    with pytest.raises(ValueError):
        logprob_signal(lps, "median")


def test_logprob_escalation_costs_and_missing():
    high = make_rec(logprobs=[math.log(0.99)] * 5)
    low = make_rec(logprobs=[math.log(0.3)] * 5, small_ok=False)
    none = make_rec(logprobs=None)
    o_high = logprob_escalation(high, 0.8)
    o_low = logprob_escalation(low, 0.8)
    assert o_high["escalated"] is False
    assert o_high["cost_seconds"] == pytest.approx(1.0)
    assert o_low["escalated"] is True
    assert o_low["cost_seconds"] == pytest.approx(11.0)   # small 1 + strong 10
    assert o_low["correct"] is True                       # strong fixes it
    assert o_low["overhead_seconds"] == 0.0               # logprobs come free
    assert logprob_escalation(none, 0.8)["escalated"] is True


def test_random_extremes():
    ids = [f"q{i}" for i in range(50)]
    assert not any(random_escalation(make_rec(qid=i), 0.0, 7)["escalated"] for i in ids)
    assert all(random_escalation(make_rec(qid=i), 1.0, 7)["escalated"] for i in ids)


def test_random_nested_and_rate():
    ids = [f"q{i}" for i in range(200)]
    low = {i for i in ids if random_uniform(i, 7) < 0.3}
    high = {i for i in ids if random_uniform(i, 7) < 0.6}
    assert low <= high
    assert 0.2 < len(low) / 200 < 0.4


def test_random_pre_generation_cost():
    rec = make_rec()
    pre = random_escalation(rec, 1.0, 0, pre_generation=True)
    strict = random_escalation(rec, 1.0, 0, pre_generation=False)
    never = random_escalation(rec, 0.0, 0)
    assert pre["cost_seconds"] == pytest.approx(10.0)       # strong only
    assert strict["cost_seconds"] == pytest.approx(11.0)    # small 1 + strong 10
    assert never["cost_seconds"] == pytest.approx(1.0)


def test_difficulty_scores_order():
    fit = fit_difficulty([SHORT, MEDIUM, LONG])
    assert (difficulty_score(LONG, fit) > difficulty_score(MEDIUM, fit)
            > difficulty_score(SHORT, fit))


def test_difficulty_escalation():
    fit = fit_difficulty([SHORT, MEDIUM, LONG])
    hard = difficulty_escalation(make_rec(question=LONG), 0.0, fit)
    easy = difficulty_escalation(make_rec(question=SHORT), 0.0, fit)
    assert hard["escalated"] is True
    assert hard["cost_seconds"] == pytest.approx(10.0)      # skips the small call
    assert easy["escalated"] is False
    assert easy["cost_seconds"] == pytest.approx(1.0)
    strict = difficulty_escalation(make_rec(question=LONG), 0.0, fit, pre_generation=False)
    assert strict["cost_seconds"] == pytest.approx(11.0)


def test_self_consistency_confident_no_escalation():
    rec = make_rec(ref=16, samples=[16, 16, 16])
    o = self_consistency(rec, 0.8)
    assert o["escalated"] is False
    assert o["correct"] is True
    assert o["cost_seconds"] == pytest.approx(1.5)          # 3 x 0.5
    assert o["overhead_seconds"] == pytest.approx(1.5)


def test_self_consistency_escalation_and_overhead():
    rec = make_rec(ref=42, samples=[42, 38, 42], strong_ok=True)
    o = self_consistency(rec, 0.8)
    assert o["escalated"] is True
    assert o["small_answer_correct"] is True                # majority 42 was right
    assert o["cost_seconds"] == pytest.approx(11.5)         # 1.5 samples + 10 strong
    assert o["overhead_seconds"] == pytest.approx(1.5)
    assert o["strong_seconds"] == pytest.approx(10.0)


def test_self_consistency_confidently_wrong():
    rec = make_rec(ref=42, samples=[38, 38, 42], strong_ok=True)
    o = self_consistency(rec, 0.5)                          # 2/3 >= 0.5: keep small
    assert o["escalated"] is False
    assert o["correct"] is False                            # wrong majority returned


def test_oracle_escalates_only_when_it_helps():
    helps = oracle(make_rec(small_ok=False, strong_ok=True))
    hopeless = oracle(make_rec(small_ok=False, strong_ok=False))
    fine = oracle(make_rec(small_ok=True, strong_ok=False))
    assert helps["escalated"] is True and helps["correct"] is True
    assert hopeless["escalated"] is False and hopeless["correct"] is False
    assert fine["escalated"] is False and fine["correct"] is True


def test_small_only_and_strong_only():
    rec = make_rec(small_ok=False, strong_ok=True)
    a, b = small_only(rec), strong_only(rec)
    assert a["correct"] is False and a["cost_seconds"] == pytest.approx(1.0)
    assert b["correct"] is True and b["cost_seconds"] == pytest.approx(10.0)
    assert a["escalated"] is False and b["escalated"] is True