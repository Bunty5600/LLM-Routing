"""Ollama HTTP adapter with a disk cache.

Every call returns one record with text, token counts, timings and
(optionally) per-token log-probabilities. Records are cached on disk, so a
rerun is free AND replays the originally recorded timings.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from functools import lru_cache

import requests  # pyright: ignore[reportMissingImports, reportMissingModuleSource]

from src.config import load_config, resolve

NS = 1e9


@lru_cache(maxsize=1)
def _default_cfg():
    return load_config()


def _ns_to_s(value):
    """Ollama reports durations in nanoseconds; missing -> 0.0 seconds."""
    return (value or 0) / NS


def make_cache_key(model, prompt, temperature, seed, sample_index,
                   logprobs, num_ctx, num_predict):
    """Stable hash of everything that can change the output.

    Dry run: model 'm', prompt 'p', temperature 0.7, seed 42, sample_index 1
    -> one hash. Change sample_index to 2 -> a different hash, so the three
    samples of one question get three separate cache files.
    """
    payload = {
        "model": model, "prompt": prompt, "temperature": temperature,
        "seed": seed, "sample_index": sample_index, "logprobs": logprobs,
        "num_ctx": num_ctx, "num_predict": num_predict,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_cache(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None  # corrupt cache file: ignore it and recompute


def _write_cache(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: no half-written cache files


def _build_record(data, model, temperature, seed, sample_index, wall, want_logprobs):
    text = data["response"]  # KeyError if missing -> treated as a failed attempt
    prompt_s = _ns_to_s(data.get("prompt_eval_duration"))
    eval_s = _ns_to_s(data.get("eval_duration"))

    lps = None
    if want_logprobs:
        raw = data.get("logprobs")
        if raw:
            lps = [e["logprob"] for e in raw
                   if isinstance(e, dict) and "logprob" in e]

    return {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "sample_index": sample_index,
        "text": text,
        "prompt_tokens": data.get("prompt_eval_count"),
        "generated_tokens": data.get("eval_count"),
        "prompt_eval_seconds": prompt_s,
        "eval_seconds": eval_s,
        # Primary cost: prompt processing + generation, EXCLUDING model load.
        "compute_seconds": prompt_s + eval_s,
        "load_seconds": _ns_to_s(data.get("load_duration")),
        "total_seconds": _ns_to_s(data.get("total_duration")),
        "wall_seconds": wall,  # client-side stopwatch, a sanity check
        "logprobs": lps,       # None when not requested or not returned
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def generate(model, prompt, temperature=0.0, seed=0, sample_index=0,
             logprobs=False, use_cache=True, cfg=None):
    """Call a local Ollama model once (or return the cached record).

    Returns a dict (see _build_record) plus 'from_cache' (True/False).

    Dry run with real numbers from your check: prompt_eval 0.099 s +
    eval 1.157 s -> compute_seconds = 1.256. The 4.39 s load is stored in
    load_seconds and is NOT part of compute_seconds.
    """
    cfg = cfg or _default_cfg()
    gen, oll = cfg["generation"], cfg["ollama"]
    num_ctx, num_predict = gen["num_ctx"], gen["num_predict"]

    key = make_cache_key(model, prompt, temperature, seed, sample_index,
                         logprobs, num_ctx, num_predict)
    cache_path = resolve(cfg["paths"]["cache"]) / f"{key}.json"

    if use_cache:
        cached = _read_cache(cache_path)
        if cached is not None:
            cached["from_cache"] = True
            return cached

    body = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": oll["keep_alive"],
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    }
    if logprobs:
        body["logprobs"] = True

    url = f"{oll['base_url']}/api/generate"
    max_retries = oll["max_retries"]
    last_err = None
    record = None

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.perf_counter()
            resp = requests.post(url, json=body, timeout=oll["timeout_seconds"])
            wall = time.perf_counter() - t0
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            data = resp.json()  # malformed JSON raises ValueError
            record = _build_record(data, model, temperature, seed,
                                   sample_index, wall, logprobs)
            break
        except (requests.RequestException, ValueError, RuntimeError, KeyError) as e:
            last_err = e
            time.sleep(2 ** (attempt - 1))  # waits 1 s, then 2 s, then 4 s ...

    if record is None:
        raise RuntimeError(
            f"Ollama call failed after {max_retries} attempts "
            f"(model={model}): {last_err}"
        )

    if use_cache:
        _write_cache(cache_path, record)

    out = dict(record)
    out["from_cache"] = False
    return out


def warmup(model, cfg=None):
    """Load the model with a tiny throwaway call (never cached).

    Call this once per model BEFORE timed collection, so the slow first-call
    effect you saw earlier (54 s load, 41 s prompt) does not pollute results.
    Returns the load time in seconds, to be logged separately.
    """
    rec = generate(model, "Reply with the word OK.", temperature=0.0,
                   seed=0, use_cache=False, cfg=cfg)
    return rec["load_seconds"]