"""Run the synopsis-style replay: primary cost convention, then the strict one.

Use your MAIN Python:

    python scripts\\run_tiered_replay.py

Needs the cached outputs, including qwen2.5:3b (calibration + evaluation) and
qwen2.5:7b on the calibration split. Writes only to results\\tiered\\ and
results\\tiered_strict\\.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments.tiered_replay import run_tiered_replay


def main():
    cfg = load_config()
    try:
        primary = run_tiered_replay(cfg, out_subdir="tiered")
        strict = run_tiered_replay(cfg, pre_generation=False,
                                   out_subdir="tiered_strict", verbose=False)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nCannot run the tiered replay: {type(e).__name__}: {e}")
        sys.exit(1)

    key = lambda r: (r["pair"], r["group"], r["strategy"], r["setting"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    watch = ("P profile-router", "D matched-rate", "C matched-rate", "TS two-stage")
    print("\n=== SENSITIVITY: cost s/q, small call skipped (primary) vs paid (strict) ===")
    print(f"{'pair':<7}{'strategy':<34}{'primary':>9}{'strict':>9}")
    for r in primary["table"]:
        if r["strategy"] in watch:
            label = f"{r['strategy']} {r['setting']}"
            print(f"{r['pair']:<7}{label:<34}{r['cost_per_q']:>9.2f}"
                  f"{strict_cost[key(r)]:>9.2f}")
    print("\nStrict results: results\\tiered_strict\\")


if __name__ == "__main__":
    main()