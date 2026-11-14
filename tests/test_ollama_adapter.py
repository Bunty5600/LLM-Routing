import copy

# pyright: reportMissingImports=false
import pytest

from src.config import load_config
from src.models import ollama_adapter as oa


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data
        self.text = text

    def json(self):
        if self._data is None:
            raise ValueError("malformed json")
        return self._data


GOOD = {
    "response": "12 - 2 = 10. Final answer: 10",
    "prompt_eval_count": 61,
    "prompt_eval_duration": 100_000_000,   # 0.1 s
    "eval_count": 20,
    "eval_duration": 1_000_000_000,        # 1.0 s
    "load_duration": 4_000_000_000,        # 4.0 s
    "total_duration": 5_200_000_000,
    "logprobs": [
        {"token": "12", "logprob": -0.1, "bytes": [49, 50]},
        {"token": " -", "logprob": -0.5, "bytes": [32, 45]},
    ],
}


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(load_config())
    c["paths"]["cache"] = str(tmp_path / "cache")  # isolated cache per test
    c["ollama"]["max_retries"] = 3
    return c


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(oa.time, "sleep", lambda s: None)  # no real waiting


def test_cache_key_changes_with_every_input():
    base = dict(model="m", prompt="p", temperature=0.7, seed=1, sample_index=0,
                logprobs=False, num_ctx=4096, num_predict=512)
    key = oa.make_cache_key(**base)
    assert key == oa.make_cache_key(**base)  # same input, same key
    for field, value in [("model", "m2"), ("prompt", "p2"), ("temperature", 0.0),
                         ("seed", 2), ("sample_index", 1), ("logprobs", True)]:
        changed = dict(base)
        changed[field] = value
        assert oa.make_cache_key(**changed) != key


def test_second_call_uses_cache(cfg, monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(200, GOOD)

    monkeypatch.setattr(oa.requests, "post", fake_post)
    a = oa.generate("m", "p", cfg=cfg)
    b = oa.generate("m", "p", cfg=cfg)

    assert len(calls) == 1                       # server hit only once
    assert a["from_cache"] is False and b["from_cache"] is True
    assert a["text"] == b["text"]
    assert a["compute_seconds"] == pytest.approx(1.1)   # 0.1 + 1.0, load excluded
    assert b["compute_seconds"] == pytest.approx(1.1)   # timing replayed from cache
    assert a["load_seconds"] == pytest.approx(4.0)


def test_retry_then_success(cfg, monkeypatch):
    queue = [FakeResponse(500, text="boom"),
             FakeResponse(200, None),            # malformed JSON
             FakeResponse(200, GOOD)]
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        return queue.pop(0)

    monkeypatch.setattr(oa.requests, "post", fake_post)
    rec = oa.generate("m", "p", cfg=cfg)
    assert len(calls) == 3
    assert rec["text"].endswith("10")


def test_all_attempts_fail_raises(cfg, monkeypatch):
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append(1)
        return FakeResponse(500, text="boom")

    monkeypatch.setattr(oa.requests, "post", fake_post)
    with pytest.raises(RuntimeError):
        oa.generate("m", "p", cfg=cfg)
    assert len(calls) == 3


def test_logprobs_requested_and_parsed(cfg, monkeypatch):
    bodies = []

    def fake_post(url, json=None, timeout=None):
        bodies.append(json)
        return FakeResponse(200, GOOD)

    monkeypatch.setattr(oa.requests, "post", fake_post)

    rec = oa.generate("m", "p", logprobs=True, cfg=cfg)
    assert bodies[0]["logprobs"] is True
    assert rec["logprobs"] == [-0.1, -0.5]

    rec2 = oa.generate("m", "p2", logprobs=False, cfg=cfg)
    assert "logprobs" not in bodies[1]
    assert rec2["logprobs"] is None