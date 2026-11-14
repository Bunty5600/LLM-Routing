import copy
import json
import math
import random
from pathlib import Path

import pytest

from src.config import load_config
from src.data.gsm8k import save_items
from src.experiments import routellm_replay as rr
from src.experiments.collect_outputs import output_path
from src.routing.routellm_strategy import load_scores, router_escalation


def make_rec(qid="q1", small_ok=True, strong_ok=True, small_secs=1.0, strong_secs=10.0):
    return {"id": qid, "question": "x", "reference_answer": 10,
            "small": {"greedy": {"correct": small_ok, "compute_seconds": small_secs,
                                 "extracted": None, "logprobs": None}},
            "strong": {"greedy": {"correct": strong_ok, "compute_seconds": strong_secs}}}


SCORES = {"q1": {"score": 0.8, "seconds": 0.05}}


def test_auc_basic_ties_and_missing():
    assert rr.auc([0.9, 0.6, 0.5, 0.7], [True, True, False, False]) == pytest.approx(0.75)
    assert rr.auc([0.5, 0.5], [True, False]) == pytest.approx(0.5)
    assert rr.auc([0.9, 0.1], [True, True]) is None          # no negatives
    assert rr.auc([None, 0.9, 0.1], [True, True, False]) == pytest.approx(1.0)


def test_load_scores_validates(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"meta": {"router": "bert"}, "scores": SCORES}))
    scores, meta = load_scores(path, ["q1"])
    assert scores["q1"]["score"] == 0.8 and meta["router"] == "bert"
    with pytest.raises(ValueError):
        load_scores(path, ["q1", "q2"])                      # q2 has no score
    with pytest.raises(FileNotFoundError):
        load_scores(tmp_path / "missing.json")


def test_router_costs_primary_and_strict():
    rec = make_rec(small_ok=False, strong_ok=True)
    pri = router_escalation(rec, 0.5, SCORES, pre_generation=True)
    strict = router_escalation(rec, 0.5, SCORES, pre_generation=False)
    keep = router_escalation(rec, 0.9, SCORES)
    assert pri["escalated"] and pri["correct"] is True        # strong fixes it
    assert pri["cost_seconds"] == pytest.approx(10.05)        # router + strong
    assert strict["cost_seconds"] == pytest.approx(11.05)     # router + small + strong
    assert pri["overhead_seconds"] == pytest.approx(0.05)
    assert pri["latency_seconds"] == pytest.approx(pri["cost_seconds"])
    assert not keep["escalated"] and keep["correct"] is False
    assert keep["cost_seconds"] == pytest.approx(1.05)        # router + small


def test_decision_ignores_labels():
    a = make_rec(small_ok=True, strong_ok=True)
    b = make_rec(small_ok=False, strong_ok=False)
    for t in (0.5, 0.9):
        assert (router_escalation(a, t, SCORES)["escalated"]
                == router_escalation(b, t, SCORES)["escalated"])


def test_missing_score_raises():
    with pytest.raises(KeyError):
        router_escalation(make_rec(qid="unknown"), 0.5, SCORES)


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
    m = cfg["models"]
    scores = {}
    for split in ("calibration", "evaluation"):
        items = [{"id": f"{split}-{i}",
                  "question": "x " * rng.randint(3, 40) + f"{i} apples",
                  "reference_answer": i} for i in range(n)]
        save_items(items, Path(cfg["data"]["dir"]) / f"{split}.json")
        for it in items:
            scores[it["id"]] = {"split": split, "score": rng.random(), "seconds": 0.05}
        for model, role in ((m["small_a"], "small"), (m["small_b"], "small"),
                            (m["strong"], "strong")):
            if role == "strong" and split == "calibration":
                continue
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
    sp = Path(cfg["routellm"]["scores_file"])
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps({"meta": {"router": "bert", "checkpoint": "fake"},
                              "scores": scores}), encoding="utf-8")


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["results"] = str(tmp_path / "results")
    c["data"]["dir"] = str(tmp_path / "data")
    c["routellm"]["scores_file"] = str(tmp_path / "results" / "routellm_scores.json")
    c["replay"]["random_seeds"] = 10
    c["replay"]["bootstrap_resamples"] = 200
    return c


def test_end_to_end_and_existing_results_untouched(cfg):
    build(cfg)
    out = rr.run_routellm_replay(cfg, out_subdir="routellm", verbose=False)
    d = Path(cfg["paths"]["results"]) / "routellm"
    for name in ("summary_routellm", "curves_routellm", "paired_routellm",
                 "cost_matched_routellm", "auc_routellm", "thresholds_routellm",
                 "per_question_pair1", "per_question_pair2"):
        assert (d / f"{name}.csv").exists()
    assert (d / "run_meta.json").exists()

    r_rows = [r for r in out["table"] if r["strategy"] == "R router"]
    assert len(r_rows) == 4                                   # 2 pairs x 2 operating points
    assert all(r["overhead_per_q"] == pytest.approx(0.05) for r in r_rows)

    strict = rr.run_routellm_replay(cfg, pre_generation=False,
                                    out_subdir="routellm_strict", verbose=False)
    key = lambda r: (r["pair"], r["group"], r["strategy"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    for r in r_rows:
        assert strict_cost[key(r)] >= r["cost_per_q"] - 1e-9

    assert not (Path(cfg["paths"]["results"]) / "replay").exists()  # old outputs not touched