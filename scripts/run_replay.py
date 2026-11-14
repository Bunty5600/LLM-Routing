"""Run the replay twice: primary (baselines skip the small call when they
escalate) and strict (every question pays the small call first).

Usage:  python scripts\\run_replay.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments.replay import run_replay


def main():
    cfg = load_config()
    try:
        primary = run_replay(cfg, out_subdir="replay")
        strict = run_replay(cfg, pre_generation=False,
                            out_subdir="replay_strict", verbose=False)
    except ValueError as e:
        print(f"\nCannot run replay: {e}")
        sys.exit(1)

    key = lambda r: (r["pair"], r["group"], r["strategy"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    print("\n=== SENSITIVITY: baselines that decide before generating ===")
    print("cost s/q with 'skip the small call' (primary) vs 'pay for it' (strict)")
    print(f"{'pair':<7}{'group':<14}{'strategy':<17}{'primary':>9}{'strict':>9}")
    for r in primary["table"]:
        if r["strategy"] in ("C matched-rate", "D matched-rate"):
            print(f"{r['pair']:<7}{r['group']:<14}{r['strategy']:<17}"
                  f"{r['cost_per_q']:>9.2f}{strict_cost[key(r)]:>9.2f}")
    print("\nStrict results: results\\replay_strict\\")


if __name__ == "__main__":
    main()