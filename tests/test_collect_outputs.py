import copy
import json

import pytest

from src.config import load_config
from src.experiments import collect_outputs as co

ITEMS = [
    {"id": "t-1", "question": "Q one", "reference_answer": 10},
    {"id": "t-2", "question": "Q two", "reference_answer": 11},
]


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["results"] = str(tmp_path / "results")
    return c


@pytest.fixture
def fake_out():
    """What the fake model returns. Tests change it to simulate truncation."""
    return {"text": "so it is. Final answer: 10", "tokens": 10}


@pytest.fixture
def calls(monkeypatch, fake_out):
    log = []

    def fake_generate(model, prompt, temperature=0.0, seed=0, sample_index=0,
                      logprobs=False, use_cache=True, cfg=None):
        log.append((temperature, seed, sample_index))
        return {
            "text": fake_out["text"], "prompt_tokens": 50,
            "generated_tokens": fake_out["tokens"], "prompt_eval_seconds": 0.1,
            "eval_seconds": 0.9, "compute_seconds": 1.0, "load_seconds": 0.0,
            "logprobs": [-0.1, -0.2] if logprobs else None,
            "timestamp": "t", "from_cache": False,
        }

    monkeypatch.setattr(co, "generate", fake_generate)
    monkeypatch.setattr(co, "warmup", lambda model, cfg=None: 2.5)
    return log


def test_small_model_collects_greedy_and_samples(cfg, calls):
    path = co.collect_model("m:1b", ITEMS, "pilot", cfg,
                            samples=True, logprobs=True, verbose=False)
    rows = co.load_rows(path)
    assert len(rows) == 2
    assert len(calls) == 2 * (1 + 3)          # per question: 1 greedy + 3 samples
    assert [r["greedy"]["correct"] for r in rows] == [True, False]  # ref 10, 11
    assert [s["sample_index"] for s in rows[0]["samples"]] == [1, 2, 3]
    assert [s["seed"] for s in rows[0]["samples"]] == [43, 44, 45]
    assert rows[0]["greedy"]["logprobs"] == [-0.1, -0.2]
    assert rows[0]["greedy"]["truncated"] is False


def test_strong_model_greedy_only(cfg, calls):
    path = co.collect_model("m:7b", ITEMS, "pilot", cfg,
                            samples=False, logprobs=False, verbose=False)
    rows = co.load_rows(path)
    assert len(calls) == 2                     # one greedy call per question
    assert "samples" not in rows[0]
    assert "logprobs" not in rows[0]["greedy"]


def test_resume_skips_done_and_survives_partial_line(cfg, calls):
    path = co.collect_model("m:1b", ITEMS[:1], "pilot", cfg, samples=True,
                            logprobs=True, verbose=False)
    assert len(calls) == 4
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"id": "broken", "gree')     # simulated crash mid-write
    co.collect_model("m:1b", ITEMS, "pilot", cfg, samples=True,
                     logprobs=True, verbose=False)
    assert len(calls) == 8                     # only t-2 was run (4 more calls)
    ids = [r["id"] for r in co.load_rows(path)]
    assert ids == ["t-1", "t-2"]               # partial line ignored, nothing lost


def test_warmup_load_time_logged_separately(cfg, calls):
    path = co.collect_model("m:1b", ITEMS[:1], "pilot", cfg, verbose=False)
    meta = path.with_suffix(".meta.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(meta)["warmup_load_seconds"] == 2.5


def test_truncated_without_marker_counts_as_no_answer(cfg, calls, fake_out):
    # The last number in this text is 10, which would match the reference.
    # Because the answer was cut off with no 'Final answer', it must NOT count.
    fake_out["text"] = "12 - 2 = 10 and then we continue with 7 * 3 = 10"
    fake_out["tokens"] = 512                   # equals num_predict in the config
    path = co.collect_model("m:1b", ITEMS[:1], "pilot", cfg, verbose=False)
    g = co.load_rows(path)[0]["greedy"]
    assert g["truncated"] is True
    assert g["extracted"] is None
    assert g["correct"] is False


def test_truncated_with_marker_keeps_the_answer(cfg, calls, fake_out):
    fake_out["text"] = "so it is. Final answer: 10\nBut let me double check..."
    fake_out["tokens"] = 512
    path = co.collect_model("m:1b", ITEMS[:1], "pilot", cfg, verbose=False)
    g = co.load_rows(path)[0]["greedy"]
    assert g["truncated"] is True
    assert g["extracted"] == 10.0
    assert g["correct"] is True