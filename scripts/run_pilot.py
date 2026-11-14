"""Pilot: every candidate model on the pilot questions, then a summary table.

Usage:  python scripts\\run_pilot.py
        python scripts\\run_pilot.py --models qwen2.5:3b qwen2.5:7b
        python scripts\\run_pilot.py --with-samples      (adds k sampled answers, slower)
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.data.gsm8k import load_items
from src.experiments.collect_outputs import collect_model, load_rows


def installed_models(cfg):
    r = requests.get(f"{cfg['ollama']['base_url']}/api/tags", timeout=10)
    return {m["name"] for m in r.json().get("models", [])}


def summarize(rows, num_predict):
    n = len(rows)
    g = [r["greedy"] for r in rows]
    return {
        "n": n,
        "accuracy": sum(x["correct"] for x in g) / n,
        "no_number": sum(x["extracted"] is None for x in g),
        "truncated": sum((x["generated_tokens"] or 0) >= num_predict for x in g),
        "mean_seconds": sum(x["compute_seconds"] for x in g) / n,
        "mean_tokens": sum((x["generated_tokens"] or 0) for x in g) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--with-samples", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    models = args.models or cfg["models"]["pilot_candidates"]
    items = load_items(resolve("data/pilot.json"))
    num_predict = cfg["generation"]["num_predict"]

    have = installed_models(cfg)
    missing = [m for m in models if m not in have]
    for m in missing:
        print(f"SKIPPING {m}: not installed. Run: ollama pull {m}")
    models = [m for m in models if m in have]

    summaries = {}
    for model in models:  # one model at a time, so load times never mix
        path = collect_model(model, items, "pilot", cfg,
                             samples=args.with_samples, logprobs=True)
        rows = load_rows(path)
        if rows:
            summaries[model] = summarize(rows, num_predict)

    if not summaries:
        print("No results to summarise.")
        return

    print("\n=== PILOT SUMMARY (greedy answers) ===")
    print(f"{'model':<16}{'n':>4}{'acc':>8}{'no-num':>8}{'trunc':>7}"
          f"{'sec/q':>8}{'tok/q':>8}")
    for m, s in summaries.items():
        print(f"{m:<16}{s['n']:>4}{s['accuracy']*100:>7.0f}%{s['no_number']:>8}"
              f"{s['truncated']:>7}{s['mean_seconds']:>8.1f}{s['mean_tokens']:>8.0f}")

    strong = cfg["models"]["strong"]
    if strong in summaries:
        print(f"\n=== PAIR CHECK vs strong = {strong} ===")
        print("Rule: strong beats small by >= 15 points AND small is wrong 25-60%.")
        print("With n = 25 each question is worth 4 points, so treat this as rough.")
        for m, s in summaries.items():
            if m == strong:
                continue
            gap = (summaries[strong]["accuracy"] - s["accuracy"]) * 100
            wrong = (1 - s["accuracy"]) * 100
            ok = gap >= 15 and 25 <= wrong <= 60
            print(f"{m:<16} gap {gap:>5.0f} pts | small wrong {wrong:>4.0f}% | "
                  f"{'MEETS rule' if ok else 'does not meet rule'}")


if __name__ == "__main__":
    main()