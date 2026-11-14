import copy
import json
import math
import random
from pathlib import Path

import pytest

from src.config import load_config
from src.data.gsm8k import save_items
from src.experiments import replay as rp
from src.experiments.collect_outputs import output_path


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["results"] = str(tmp_path / "results")
    c["data"]["dir"] = str(tmp_path / "data")
    c["replay"]["random_seeds"] = 10
    c["replay"]["bootstrap_resamples"] = 200
    return c


def fake_call(ref, correct, secs, with_logprobs):
    d = {"extracted": float(ref) if correct else float(ref) + 1.0,
         "correct": correct, "compute_seconds": secs,
         "generated_tokens": 100, "truncated": False}
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


def build_cache(cfg, seed=0, n=24):
    rng = random.Random(seed)
    m = cfg["models"]
    for split in ("calibration", "evaluation"):
        items = [{"id": f"{split}-{i}",
                  "question": "x " * rng.randint(3, 40) + f"{i} apples",
                  "reference_answer": i} for i in range(n)]
        save_items(items, Path(cfg["data"]["dir"]) / f"{split}.json")
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
                        ok = rng.random() < 0.95
                        row["greedy"] = fake_call(ref, ok, 10.0, False)
                    f.write(json.dumps(row) + "\n")


def test_candidate_thresholds():
    assert rp.candidate_thresholds([1.0, 0.5, None, 1.0]) == pytest.approx([-0.5, 0.75, 2.0])
    assert rp.candidate_thresholds([None, None]) == []


def test_choose_thresholds_budget_and_recall():
    scores = [1.0, 1.0, 2 / 3, 2 / 3, 1 / 3]
    wrong = [False, False, True, False, True]
    ops = rp.choose_thresholds(scores, wrong, "below", budget=0.3, recall_target=0.7)
    assert ops["budget"]["threshold"] == pytest.approx(0.5)
    assert ops["budget"]["cal_escalation_rate"] == pytest.approx(0.2)
    assert ops["budget"]["cal_error_recall"] == pytest.approx(0.5)
    assert ops["recall"]["threshold"] == pytest.approx(5 / 6)
    assert ops["recall"]["cal_escalation_rate"] == pytest.approx(0.6)
    assert ops["recall"]["cal_error_recall"] == pytest.approx(1.0)
    assert rp.choose_thresholds([None], [True], "below", 0.3, 0.7) is None


def test_matched_threshold_above():
    scores = [3, 1, 2, 5, 4]
    t = rp.matched_threshold_above(scores, 0.4)
    assert t == pytest.approx(3.5)
    assert sum(s > t for s in scores) == 2
    assert not any(s > rp.matched_threshold_above(scores, 0.0) for s in scores)
    assert all(s > rp.matched_threshold_above(scores, 1.0) for s in scores)


def test_random_mean_summary_extremes():
    def rec(i, ok):
        return {"id": f"q{i}", "question": "q",
                "small": {"greedy": {"correct": ok, "compute_seconds": 1.0}},
                "strong": {"greedy": {"correct": True, "compute_seconds": 10.0}}}
    recs = [rec(0, True), rec(1, False), rec(2, False), rec(3, True)]
    none = rp.random_mean_summary(recs, 0.0, [1, 2, 3], True)
    full = rp.random_mean_summary(recs, 1.0, [1, 2, 3], True)
    full_strict = rp.random_mean_summary(recs, 1.0, [1, 2, 3], False)
    assert none["accuracy"] == pytest.approx(0.5)
    assert none["cost_per_q"] == pytest.approx(1.0)
    assert full["accuracy"] == pytest.approx(1.0)
    assert full["cost_per_q"] == pytest.approx(10.0)
    assert full_strict["cost_per_q"] == pytest.approx(11.0)


def test_load_records_joins_and_validates(cfg):
    build_cache(cfg)
    m = cfg["models"]
    recs = rp.load_records(cfg, "evaluation", m["small_a"], m["strong"])
    assert len(recs) == 24
    assert recs[0]["question"].endswith("0 apples")
    assert "samples" in recs[0]["small"] and "strong" in recs[0]

    with pytest.raises(ValueError):                 # strong rows have no samples
        rp.load_records(cfg, "evaluation", m["strong"])

    path = output_path("evaluation", m["strong"], cfg)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):                 # one question missing
        rp.load_records(cfg, "evaluation", m["small_a"], m["strong"])


def test_end_to_end_and_strict_sensitivity(cfg):
    build_cache(cfg)
    out = rp.run_replay(cfg, out_subdir="replay", verbose=False)
    assert len(out["table"]) == 2 * 19
    assert len(out["paired"]) == 2 * 16
    assert len(out["transfer"]) == 18
    for name in ("summary", "curves", "paired", "transfer", "posthoc",
                 "per_question_pair1", "per_question_pair2"):
        assert (Path(cfg["paths"]["results"]) / "replay" / f"{name}.csv").exists()

    for pair in ("pair1", "pair2"):
        base = {r["strategy"]: r for r in out["table"]
                if r["pair"] == pair and r["group"] == "baselines"}
        a, b, ref = base["A small-only"], base["B strong-only"], base["Ref oracle"]
        assert a["cost_per_q"] == pytest.approx(1.0)
        assert b["cost_per_q"] == pytest.approx(10.0)
        assert ref["accuracy"] >= max(a["accuracy"], b["accuracy"]) - 1e-9

    strict = rp.run_replay(cfg, pre_generation=False,
                           out_subdir="replay_strict", verbose=False)
    key = lambda r: (r["pair"], r["group"], r["strategy"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    for r in out["table"]:
        if r["strategy"] in ("C matched-rate", "D matched-rate"):
            assert strict_cost[key(r)] >= r["cost_per_q"] - 1e-9