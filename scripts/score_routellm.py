"""Score every calibration and evaluation question with RouteLLM's BERT router.

Run with the routellm environment (NOT your main Python):

    ..\\venv-routellm\\Scripts\\python.exe scripts\\score_routellm.py

Output: results/routellm_scores.json (path set in configs/experiment.yaml).
The file is NEW; an existing file is never overwritten unless --force is given.

Input to the router: the raw question text only (no prompt template, no answer,
no ground truth). Timing: wall-clock seconds for each router call after one
unrecorded warm-up call; router load time is stored separately in 'meta'.

A placeholder (NOT a real key) is set for this process only, because importing
RouteLLM creates an OpenAI client at import time. The BERT router itself does
not call OpenAI.

Dry run (illustrative numbers): question 'train-5238' -> score 0.569 in 0.089 s
-> records['train-5238'] = {'split': 'calibration', 'score': 0.569, 'seconds': 0.089}.
Replay later escalates a question when its score is above a threshold that is
calibrated on the calibration split only.
"""
import os

os.environ["OPENAI_API_KEY"] = "placeholder-not-a-real-key"

import argparse  # noqa: E402
import importlib.metadata as md  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("calibration", "evaluation")
WARMUP_TEXT = "A warm-up question: what is 2 + 2?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing scores file")
    args = ap.parse_args()

    with open(ROOT / "configs" / "experiment.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rcfg = cfg["routellm"]
    out_path = ROOT / rcfg["scores_file"]
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists. Nothing was changed.")
        print("Use --force only if you really want to overwrite it.")
        return

    data = {}
    for split in SPLITS:
        with open(ROOT / cfg["data"]["dir"] / f"{split}.json", "r", encoding="utf-8") as f:
            data[split] = json.load(f)
    print({s: len(v) for s, v in data.items()}, "questions to score")

    print(f"Loading router '{rcfg['router']}' from {rcfg['checkpoint']} ...")
    t0 = time.perf_counter()
    try:
        from routellm.routers.routers import BERTRouter
        router = BERTRouter(rcfg["checkpoint"])
    except Exception as e:
        print(f"\nFAILED to load the router: {type(e).__name__}: {e}")
        sys.exit(1)
    load_seconds = time.perf_counter() - t0
    print(f"Router loaded in {load_seconds:.1f} s")

    router.calculate_strong_win_rate(WARMUP_TEXT)  # warm-up, not recorded

    records = {}
    for split in SPLITS:
        for item in data[split]:
            if item["id"] in records:
                raise ValueError(f"duplicate question id: {item['id']}")
            t = time.perf_counter()
            score = float(router.calculate_strong_win_rate(item["question"]))
            seconds = time.perf_counter() - t
            if not math.isfinite(score):
                raise ValueError(f"non-finite score for {item['id']}")
            records[item["id"]] = {"split": split, "score": score, "seconds": seconds}

    versions = {}
    for pkg in ("routellm", "torch", "transformers", "huggingface-hub"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            versions[pkg] = "not installed"

    meta = {
        "router": rcfg["router"],
        "checkpoint": rcfg["checkpoint"],
        "input": "raw question text only",
        "device": "cpu",
        "router_load_seconds": load_seconds,
        "n_scored": len(records),
        "versions": versions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "scores": records}, f, indent=1)
    os.replace(tmp, out_path)  # atomic: no half-written file

    print(f"\n{'split':<13}{'n':>5}{'min':>8}{'median':>8}{'max':>8}{'mean s/q':>10}")
    for split in SPLITS:
        recs = [r for r in records.values() if r["split"] == split]
        sc = [r["score"] for r in recs]
        print(f"{split:<13}{len(recs):>5}{min(sc):>8.3f}{statistics.median(sc):>8.3f}"
              f"{max(sc):>8.3f}{statistics.mean(r['seconds'] for r in recs):>10.3f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()