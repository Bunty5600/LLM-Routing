import copy
import json
import math
import random
from pathlib import Path

import pytest

from src.config import load_config
from src.data.gsm8k import save_items
from src.experiments import tiered_replay as tr
from src.experiments.collect_outputs import output_path
from src.metrics.extra import (
    annotate_calls, calls_for, extra_summary, mean_model_calls, routing_accuracy,
)
from src.routing import strategies as S
from src.routing import tiered as T


# ------------------------------------------------------------------ helpers
def question(n):
    return "x " * n + "apples"


def make_rec(qid="q1", n_words=3, ref=10, small_ok=True, strong_ok=True,
             small_secs=1.0, strong_secs=10.0, samples=None, med_samples=None):
    rec = {"id": qid, "question": question(n_words), "reference_answer": ref,
           "small": {"greedy": {"correct": small_ok, "compute_seconds": small_secs,
                                "extracted": None, "logprobs": None}},
           "strong": {"greedy": {"correct": strong_ok, "compute_seconds": strong_secs}}}
    if samples is not None:
        rec["small"]["samples"] = [{"extracted": v, "compute_seconds": 0.5} for v in samples]
    if med_samples is not None:
        rec["medium"] = {"samples": [{"extracted": v, "compute_seconds": 1.0}
                                     for v in med_samples]}
    return rec


def calibration_recs():
    """Six questions, increasing difficulty; bucket accuracies (3 buckets):
    bucket 1: small 1.0, strong 1.0 | bucket 2: small 0.5, strong 1.0 |
    bucket 3: small 0.0, strong 0.5."""
    small = [True, True, True, False, False, False]
    strong = [True, True, True, True, True, False]
    return [make_rec(f"c{i}", n_words=3 * (i + 1), small_ok=small[i], strong_ok=strong[i])
            for i in range(6)]


def fit_for(recs):
    return S.fit_difficulty([r["question"] for r in recs])


# ------------------------------------------------------- capability profile
def test_bucket_edges_and_bucket_of():
    assert T.bucket_edges([1, 2, 3, 4, 5, 6], 3) == pytest.approx([2.5, 4.5])
    assert T.bucket_edges([1], 3) == []
    edges = [2.5, 4.5]
    assert [T.bucket_of(s, edges) for s in (1, 2, 2.5, 3, 4.5, 6)] == [0, 0, 0, 1, 1, 2]


def test_build_profile_and_decisions():
    cal = calibration_recs()
    fit = fit_for(cal)
    prof = T.build_profile(cal, fit, 3)
    accs = [(b["small_acc"], b["strong_acc"]) for b in prof["buckets"]]
    assert accs == [(1.0, 1.0), (0.5, 1.0), (0.0, 0.5)]

    q = lambda n: question(n)
    # q = 0.85: easy stays small; the two harder buckets go straight to strong
    assert T.profile_decision(q(3), prof, 0.85, fit) is False
    assert T.profile_decision(q(9), prof, 0.85, fit) is True
    assert T.profile_decision(q(18), prof, 0.85, fit) is True
    # q = 0.4: bucket 2 (small acc 0.5) is good enough; bucket 3 (0.0) is not
    assert T.profile_decision(q(9), prof, 0.4, fit) is False
    assert T.profile_decision(q(18), prof, 0.4, fit) is True
    # q = 0 never routes directly
    assert not any(T.profile_decision(q(n), prof, 0.0, fit) for n in (3, 9, 18))


def test_profile_never_escalates_when_strong_is_not_better():
    fit = fit_for(calibration_recs())
    prof = {"edges": [], "buckets": [{"n": 5, "small_acc": 0.6, "strong_acc": 0.5}]}
    assert T.profile_decision(question(5), prof, 0.95, fit) is False


def test_empty_bucket_falls_back_to_overall():
    cal = [make_rec(f"c{i}", n_words=5, small_ok=(i % 2 == 0)) for i in range(4)]
    prof = T.build_profile(cal, fit_for(cal), 3)        # identical scores -> empty buckets
    assert len(prof["buckets"]) == 3
    assert prof["buckets"][-1]["small_acc"] == pytest.approx(prof["overall_small_acc"]) \
        or prof["buckets"][-1]["n"] > 0
    with pytest.raises(ValueError):
        T.build_profile([], fit_for(cal), 3)


def test_capability_router_costs_and_label_independence():
    cal = calibration_recs()
    fit = fit_for(cal)
    prof = T.build_profile(cal, fit, 3)
    hard = make_rec(n_words=18, small_ok=False, strong_ok=True)
    pri = T.capability_router(hard, prof, 0.85, fit, pre_generation=True)
    strict = T.capability_router(hard, prof, 0.85, fit, pre_generation=False)
    easy = T.capability_router(make_rec(n_words=3), prof, 0.85, fit)
    assert pri["escalated"] and pri["cost_seconds"] == pytest.approx(10.0)
    assert strict["cost_seconds"] == pytest.approx(11.0)
    assert not easy["escalated"] and easy["cost_seconds"] == pytest.approx(1.0)
    # the decision never looks at correctness labels
    same = T.capability_router(make_rec(n_words=18, small_ok=True, strong_ok=False),
                               prof, 0.85, fit)
    assert same["escalated"] == pri["escalated"]


# ----------------------------------------------------------------- two-stage
def low_profile():
    return {"edges": [], "buckets": [{"n": 5, "small_acc": 0.2, "strong_acc": 1.0}]}


def high_profile():
    return {"edges": [], "buckets": [{"n": 5, "small_acc": 0.99, "strong_acc": 1.0}]}


def test_two_stage_direct_route():
    fit = fit_for(calibration_recs())
    rec = make_rec(ref=42, samples=[42, 42, 42], small_ok=True, strong_ok=True)
    pri = T.two_stage(rec, low_profile(), 0.85, fit, 0.8, pre_generation=True)
    strict = T.two_stage(rec, low_profile(), 0.85, fit, 0.8, pre_generation=False)
    assert pri["route"] == "direct" and pri["escalated"] is True
    assert pri["cost_seconds"] == pytest.approx(10.0)       # no sampling at all
    assert pri["overhead_seconds"] == 0.0 and pri["model_calls"] == 1
    assert strict["cost_seconds"] == pytest.approx(11.0) and strict["model_calls"] == 2
    assert pri["small_answer_correct"] is True               # majority 42 was right


def test_two_stage_second_stage_matches_e1():
    fit = fit_for(calibration_recs())
    unanimous = make_rec(ref=16, samples=[16, 16, 16])
    split = make_rec(ref=42, samples=[42, 38, 42])
    a = T.two_stage(unanimous, high_profile(), 0.85, fit, 0.8)
    b = T.two_stage(split, high_profile(), 0.85, fit, 0.8)
    assert a["route"] == "stage2" and not a["escalated"]
    assert a["cost_seconds"] == pytest.approx(1.5) and a["model_calls"] == 3
    assert b["escalated"] and b["cost_seconds"] == pytest.approx(11.5) and b["model_calls"] == 4
    # required quality 0 reduces two-stage to plain E1
    e1 = S.self_consistency(split, 0.8)
    zero = T.two_stage(split, low_profile(), 0.0, fit, 0.8)
    assert zero["cost_seconds"] == e1["cost_seconds"] and zero["correct"] == e1["correct"]


# ---------------------------------------------------------------- three-tier
def test_three_tier_paths():
    small_ok = make_rec(ref=16, samples=[16, 16, 16], med_samples=[1, 2, 3])
    t = T.three_tier(small_ok, 0.8, 0.8)
    assert t["tier_final"] == "small" and not t["escalated"]
    assert t["cost_seconds"] == pytest.approx(1.5) and t["model_calls"] == 3

    medium_ok = make_rec(ref=16, samples=[16, 17, 18], med_samples=[16, 16, 16])
    t = T.three_tier(medium_ok, 0.8, 0.8)
    assert t["tier_final"] == "medium" and not t["escalated"]
    assert t["correct"] is True and t["small_answer_correct"] is True
    assert t["cost_seconds"] == pytest.approx(4.5) and t["model_calls"] == 6

    to_strong = make_rec(ref=16, samples=[16, 17, 18], med_samples=[1, 2, 3], strong_ok=True)
    t = T.three_tier(to_strong, 0.8, 0.8)
    assert t["tier_final"] == "strong" and t["escalated"] is True
    assert t["cost_seconds"] == pytest.approx(14.5) and t["model_calls"] == 7
    assert t["overhead_seconds"] == pytest.approx(4.5)
    assert t["correct"] is True and t["small_answer_correct"] is False


def test_three_tier_threshold_gotcha():
    # 2/3 agreement escalates at threshold 0.833 but not at 0.5
    rec = make_rec(ref=16, samples=[16, 16, 17], med_samples=[16, 16, 16])
    assert T.three_tier(rec, 0.5, 0.5)["tier_final"] == "small"
    assert T.three_tier(rec, 0.833, 0.833)["tier_final"] == "medium"


# -------------------------------------------------------------- extra metrics
def oc(esc, small_ok, strong_ok, calls=None):
    d = {"escalated": esc, "small_answer_correct": small_ok, "strong_correct": strong_ok}
    if calls is not None:
        d["model_calls"] = calls
    return d


def test_routing_accuracy_dry_run():
    outs = [oc(True, False, True), oc(True, True, True),
            oc(False, False, False), oc(False, True, True)]
    assert routing_accuracy(outs) == pytest.approx(0.75)
    assert math.isnan(routing_accuracy([]))


def test_mean_calls_and_calls_for():
    assert mean_model_calls([oc(False, True, True, 3), oc(True, True, True, 4)]) == pytest.approx(3.5)
    assert math.isnan(mean_model_calls([oc(False, True, True)]))
    esc, keep = oc(True, False, True), oc(False, True, True)
    assert calls_for("E1", esc, 3) == 4 and calls_for("E1", keep, 3) == 3
    assert calls_for("E2", esc, 3) == 2 and calls_for("A", keep, 3) == 1
    assert calls_for("D", esc, 3, True) == 1 and calls_for("D", esc, 3, False) == 2
    assert calls_for("Ref", esc, 3) == 2
    with pytest.raises(ValueError):
        calls_for("nope", esc, 3)
    annotate_calls([esc, keep], "E1", 3)
    assert esc["model_calls"] == 4
    assert set(extra_summary([esc, keep])) == {"mean_model_calls", "routing_accuracy"}


# -------------------------------------------------------------- end to end
def fake_call(ref, correct, secs, with_logprobs):
    d = {"extracted": float(ref) if correct else float(ref) + 1.0, "correct": correct,
         "compute_seconds": secs, "generated_tokens": 100, "truncated": False}
    if with_logprobs:
        d["logprobs"] = [math.log(0.9 if correct else 0.5)] * 5
    return d


def fake_samples(rng, ref, correct):
    out = []
    for i in range(3):
        ok = rng.random() < (0.85 if correct else 0.25)
        out.append({"extracted": float(ref) if ok else float(ref) + rng.randint(1, 2),
                    "compute_seconds": 0.5, "correct": ok, "sample_index": i + 1})
    return out


def build(cfg, n=24):
    rng = random.Random(0)
    m, med = cfg["models"], cfg["tiered"]["medium_model"]
    for split in ("calibration", "evaluation"):
        items = [{"id": f"{split}-{i}", "question": "x " * rng.randint(3, 40) + f"{i} apples",
                  "reference_answer": i} for i in range(n)]
        save_items(items, Path(cfg["data"]["dir"]) / f"{split}.json")
        for model, role in ((m["small_a"], "small"), (m["small_b"], "small"),
                            (med, "small"), (m["strong"], "strong")):
            path = output_path(split, model, cfg)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for it in items:
                    ref = it["reference_answer"]
                    row = {"id": it["id"], "split": split, "model": model,
                           "reference_answer": ref}
                    if role == "small":
                        ok = rng.random() < 0.6
                        row["greedy"] = fake_call(ref, ok, 1.0, True)
                        row["samples"] = fake_samples(rng, ref, ok)
                    else:
                        row["greedy"] = fake_call(ref, rng.random() < 0.95, 10.0, False)
                    f.write(json.dumps(row) + "\n")


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["results"] = str(tmp_path / "results")
    c["data"]["dir"] = str(tmp_path / "data")
    c["replay"]["random_seeds"] = 10
    c["replay"]["bootstrap_resamples"] = 200
    return c


def test_end_to_end_primary_strict_and_isolation(cfg):
    build(cfg)
    out = tr.run_tiered_replay(cfg, out_subdir="tiered", verbose=False)
    d = Path(cfg["paths"]["results"]) / "tiered"
    for name in ("profile", "summary_two_tier", "curves_two_tier", "cost_matched_two_tier",
                 "paired_two_tier", "summary_three_tier", "paired_three_tier",
                 "per_question_pair1", "per_question_pair2"):
        assert (d / f"{name}.csv").exists()
    assert (d / "run_meta.json").exists()

    assert len(out["profile"]) == 6                        # 3 buckets x 2 pairs
    assert len(out["table"]) == 20 and len(out["paired"]) == 10
    assert len(out["cost_matched"]) == 10
    assert len(out["three_tier"]) == 11 and len(out["paired_three"]) == 6
    assert {r["pair"] for r in out["three_tier"]} == {"pair1"}     # pair 1 only
    assert not (Path(cfg["paths"]["results"]) / "replay").exists()  # old results untouched

    for r in out["table"]:                                  # calls are always reported
        assert r["mean_model_calls"] >= 1.0

    strict = tr.run_tiered_replay(cfg, pre_generation=False,
                                  out_subdir="tiered_strict", verbose=False)
    key = lambda r: (r["pair"], r["group"], r["strategy"], r["setting"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    for r in out["table"]:
        if r["strategy"] in ("P profile-router", "D matched-rate", "C matched-rate",
                             "TS two-stage"):
            assert strict_cost[key(r)] >= r["cost_per_q"] - 1e-9