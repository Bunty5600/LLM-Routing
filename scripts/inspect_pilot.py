"""Inspect pilot outputs: truncated answers, wrong answers, escalation
headroom against the strong model, and (if present) sample agreement.

Usage:  python scripts\\inspect_pilot.py
        python scripts\\inspect_pilot.py --models qwen2.5:1.5b llama3.2:1b

Dry run for agreement (illustrative): samples 42, 38, 42 -> counts {42: 2, 38: 1}
-> top count 2 -> agreement 2/3 = 0.67. Samples 16, 16, 16 -> 3/3 = 1.00.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments.collect_outputs import load_rows, output_path


def agreement(samples):
    """Share of samples that match the most common answer (None counts as its own value)."""
    keys = [None if s["extracted"] is None else round(s["extracted"], 6)
            for s in samples]
    top = Counter(keys).most_common(1)[0][1]
    return top / len(keys)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--models", nargs="*", default=cfg["models"]["pilot_candidates"])
    args = ap.parse_args()

    num_predict = cfg["generation"]["num_predict"]
    strong = cfg["models"]["strong"]
    data = {}
    for m in args.models:
        rows = load_rows(output_path(args.split, m, cfg))
        if rows:
            data[m] = {r["id"]: r for r in rows}

    # 1. Truncated and wrong answers per model
    print("=== TRUNCATED ANSWERS (hit the token limit) ===")
    any_trunc = False
    for m, rows in data.items():
        for qid, r in rows.items():
            g = r["greedy"]
            if (g["generated_tokens"] or 0) >= num_predict:
                any_trunc = True
                tail = g["text"][-90:].replace("\n", " ")
                print(f"{m:<14} {qid:<12} correct={g['correct']} "
                      f"extracted={g['extracted']} tokens={g['generated_tokens']}")
                print(f"{'':<14} ends with: ...{tail}")
    if not any_trunc:
        print("none")

    print("\n=== WRONG QUESTION IDS PER MODEL ===")
    for m, rows in data.items():
        wrong = [qid for qid, r in rows.items() if not r["greedy"]["correct"]]
        print(f"{m:<14} {len(wrong)} wrong: {', '.join(wrong)}")

    # 2. Escalation headroom against the strong model
    if strong in data:
        print(f"\n=== ESCALATION HEADROOM vs {strong} (greedy, same questions) ===")
        print(f"{'small model':<14}{'both right':>11}{'small wrong,':>14}"
              f"{'small right,':>14}{'both wrong':>11}")
        print(f"{'':<14}{'':>11}{'strong right':>14}{'strong wrong':>14}{'':>11}")
        for m, rows in data.items():
            if m == strong:
                continue
            c = Counter()
            for qid, r in rows.items():
                if qid not in data[strong]:
                    continue
                s_ok = r["greedy"]["correct"]
                b_ok = data[strong][qid]["greedy"]["correct"]
                c[(s_ok, b_ok)] += 1
            print(f"{m:<14}{c[(True, True)]:>11}{c[(False, True)]:>14}"
                  f"{c[(True, False)]:>14}{c[(False, False)]:>11}")

    # 3. Sample agreement, for models that were run with samples
    for m, rows in data.items():
        with_s = [r for r in rows.values() if r.get("samples")]
        if not with_s:
            continue
        print(f"\n=== SAMPLE AGREEMENT for {m} (n = {len(with_s)}) ===")
        buckets = {}
        for r in with_s:
            a = round(agreement(r["samples"]), 2)
            buckets.setdefault(a, []).append(r["greedy"]["correct"])
        print(f"{'agreement':>10}{'questions':>11}{'greedy accuracy':>17}")
        for a in sorted(buckets, reverse=True):
            oks = buckets[a]
            print(f"{a:>10.2f}{len(oks):>11}{sum(oks) / len(oks) * 100:>16.0f}%")
        t = [s["generated_tokens"] or 0 for r in with_s for s in r["samples"]]
        print(f"mean tokens per sample: {sum(t) / len(t):.0f}")


if __name__ == "__main__":
    main()