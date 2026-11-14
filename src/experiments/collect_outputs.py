"""Collect model outputs once, cache them, and write one JSONL row per question.

Replay later reads these files, so everything a strategy could need is stored:
the greedy answer, k sampled answers (small models), token counts, timings
and per-token logprobs.

Scoring rule for truncated answers: if the model hit the token limit AND never
wrote a 'Final answer' marker, the answer counts as MISSING (extracted = None),
because the 'last number in the text' fallback would pick up a meaningless
number from the middle of a calculation.

Dry run (illustrative numbers): small model, k = 3, question 'train-5238',
reference 12.
  greedy call   -> text '... Final answer: 12', compute 4.6 s, 178 tokens
  sample 1..3   -> answers 12, 12, 10 (seeds 43, 44, 45; temperature 0.7)
  one JSONL row -> {id, greedy: {...}, samples: [{...}, {...}, {...}]}
Truncation dry run: 512 tokens generated (limit 512), text ends '... = 0.0083'
  and has no 'Final answer' -> truncated = True, extracted = None, correct = False.
The file is appended to after every question, so a crash loses at most the
question in progress, and rerunning the same command resumes where it stopped.
"""
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import resolve
from src.evaluation.answer_extract import extract_final_number, is_correct
from src.models.ollama_adapter import generate, warmup

_STORED = ("text", "prompt_tokens", "generated_tokens", "prompt_eval_seconds",
           "eval_seconds", "compute_seconds", "load_seconds", "timestamp")

_MARKER = re.compile(r"final\s+answer", re.IGNORECASE)


def safe_name(model):
    """'qwen2.5:3b' -> 'qwen2.5_3b' (safe for a Windows file name)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def output_path(split_name, model, cfg):
    return resolve(cfg["paths"]["results"]) / f"{split_name}__{safe_name(model)}.jsonl"


def load_rows(path):
    """Read a JSONL file. Skips blank lines and a half-written last line."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # crash left a partial line: ignore it
    return rows


def _ensure_newline_end(path):
    """If a crash left a partial line, end it so the next row starts cleanly."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return
    with open(p, "rb") as f:
        f.seek(-1, 2)
        last = f.read(1)
    if last != b"\n":
        with open(p, "ab") as f:
            f.write(b"\n")


def _call_fields(rec, ref, with_logprobs, num_predict):
    out = {k: rec.get(k) for k in _STORED}
    truncated = (rec.get("generated_tokens") or 0) >= num_predict
    pred = extract_final_number(rec["text"])
    if truncated and not _MARKER.search(rec["text"]):
        pred = None  # cut off before answering: do not trust a mid-text number
    out["truncated"] = truncated
    out["extracted"] = pred
    out["correct"] = is_correct(pred, ref)
    if with_logprobs:
        out["logprobs"] = rec.get("logprobs")
    return out


def collect_model(model, items, split_name, cfg, samples=False,
                  logprobs=False, verbose=True):
    """Run one model over all items. Returns the output file path.

    samples=True  -> also k sampled answers per question (small models)
    logprobs=True -> store per-token logprobs of the greedy answer
    """
    path = output_path(split_name, model, cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    done = {r["id"] for r in load_rows(path)}
    todo = [it for it in items if it["id"] not in done]
    if not todo:
        if verbose:
            print(f"[{model}] {split_name}: all {len(items)} questions already collected")
        return path

    gen = cfg["generation"]
    seed, k, num_predict = cfg["seed"], gen["k_samples"], gen["num_predict"]

    # Load the model with a throwaway call so load time never lands in a
    # per-question measurement. The load time is logged separately.
    load_s = warmup(model, cfg)
    with open(path.with_suffix(".meta.jsonl"), "a", encoding="utf-8") as mf:
        mf.write(json.dumps({
            "model": model, "split": split_name, "warmup_load_seconds": load_s,
            "questions_to_run": len(todo), "samples": samples,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }) + "\n")
    if verbose:
        print(f"[{model}] {split_name}: {len(todo)} to run "
              f"({len(done)} already done), warm-up load {load_s:.1f} s")

    _ensure_newline_end(path)
    t_start = time.perf_counter()

    with open(path, "a", encoding="utf-8") as out:
        for n, item in enumerate(todo, start=1):
            ref = item["reference_answer"]
            prompt = gen["prompt_template"].format(question=item["question"])

            g = generate(model, prompt, temperature=gen["greedy_temperature"],
                         seed=seed, sample_index=0, logprobs=logprobs, cfg=cfg)
            row = {
                "id": item["id"], "split": split_name, "model": model,
                "reference_answer": ref,
                "greedy": _call_fields(g, ref, logprobs, num_predict),
            }

            if samples:
                sample_rows = []
                for i in range(1, k + 1):
                    s = generate(model, prompt, temperature=gen["sample_temperature"],
                                 seed=seed + i, sample_index=i, logprobs=False, cfg=cfg)
                    sf = _call_fields(s, ref, False, num_predict)
                    sf["sample_index"] = i
                    sf["seed"] = seed + i
                    sample_rows.append(sf)
                row["samples"] = sample_rows

            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            if verbose:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / n * (len(todo) - n)
                mark = "ok " if row["greedy"]["correct"] else "WRONG"
                print(f"  {n:>3}/{len(todo)} {item['id']:<12} greedy {mark} "
                      f"{row['greedy']['compute_seconds']:.1f}s "
                      f"| elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")
    return path