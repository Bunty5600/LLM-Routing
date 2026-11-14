"""Run the RouteLLM replay: primary cost convention, then the strict one.

Use your MAIN Python (not the routellm environment):

    python scripts\\run_routellm_replay.py

Needs results\\routellm_scores.json (made by scripts\\score_routellm.py).
Writes only to results\\routellm\\ and results\\routellm_strict\\.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments.routellm_replay import run_routellm_replay


def main():
    cfg = load_config()
    try:
        primary = run_routellm_replay(cfg, out_subdir="routellm")
        strict = run_routellm_replay(cfg, pre_generation=False,
                                     out_subdir="routellm_strict", verbose=False)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nCannot run the RouteLLM replay: {e}")
        sys.exit(1)

    key = lambda r: (r["pair"], r["group"], r["strategy"])
    strict_cost = {key(r): r["cost_per_q"] for r in strict["table"]}
    print("\n=== SENSITIVITY: cost s/q, small call skipped (primary) vs paid (strict) ===")
    print(f"{'pair':<7}{'group':<11}{'strategy':<17}{'primary':>9}{'strict':>9}")
    for r in primary["table"]:
        if r["strategy"] in ("R router", "R matched-rate", "D matched-rate",
                             "C matched-rate"):
            print(f"{r['pair']:<7}{r['group']:<11}{r['strategy']:<17}"
                  f"{r['cost_per_q']:>9.2f}{strict_cost[key(r)]:>9.2f}")
    print("\nStrict results: results\\routellm_strict\\")


if __name__ == "__main__":
    main()