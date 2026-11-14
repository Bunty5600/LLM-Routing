"""Main collection for one model on one split.
Small models:  greedy answer (+ logprobs) and k sampled answers.
Strong model:  greedy answer only.

Usage:  python scripts\\run_collect.py --model qwen2.5:1.5b --split calibration --role small
        python scripts\\run_collect.py --model qwen2.5:1.5b --split evaluation  --role small
        python scripts\\run_collect.py --model qwen2.5:7b   --split evaluation  --role strong
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve
from src.data.gsm8k import load_items
from src.experiments.collect_outputs import collect_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", required=True, choices=["pilot", "calibration", "evaluation"])
    ap.add_argument("--role", required=True, choices=["small", "strong"])
    args = ap.parse_args()

    cfg = load_config()
    items = load_items(resolve(f"data/{args.split}.json"))
    small = args.role == "small"
    path = collect_model(args.model, items, args.split, cfg,
                         samples=small, logprobs=small)
    print(f"\nDone. Output file: {path}")


if __name__ == "__main__":
    main()