"""Print paired comparisons from a paired*.csv file.

'a - b' is accuracy of strategy a minus strategy b on the SAME questions.
'excl 0' = yes means the 95% paired-bootstrap interval is entirely on one
side of zero. McNemar counts: questions where only a was right / only b.

Usage:  python scripts\\show_paired.py
        python scripts\\show_paired.py --subdir routellm --file paired_routellm.csv
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="replay")
    ap.add_argument("--file", default="paired.csv")
    args = ap.parse_args()
    cfg = load_config()
    path = resolve(cfg["paths"]["results"]) / args.subdir / args.file
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"{'pair':<7}{'group':<11}{'a':<5}{'b':<16}{'acc a-b (pts)':>14}"
          f"{'95% CI':>18}{'excl 0':>8}{'McNemar p':>11}{'a only/b only':>15}"
          f"{'cost a-b (s)':>14}{'cost CI':>18}")
    for r in rows:
        d, lo, hi = (float(r[k]) * 100 for k in
                     ("acc_diff_a_minus_b", "acc_diff_ci_low", "acc_diff_ci_high"))
        c, clo, chi = (float(r[k]) for k in
                       ("cost_diff_a_minus_b", "cost_diff_ci_low", "cost_diff_ci_high"))
        excl = "yes" if (lo > 0 or hi < 0) else "no"
        print(f"{r['pair']:<7}{r['group']:<11}{r['a']:<5}{r['b']:<16}{d:>+14.1f}"
              f"{f'[{lo:+.1f}, {hi:+.1f}]':>18}{excl:>8}{float(r['mcnemar_p']):>11.3f}"
              f"{r['a_right_b_wrong'] + '/' + r['a_wrong_b_right']:>15}"
              f"{c:>+14.2f}{f'[{clo:+.2f}, {chi:+.2f}]':>18}")
    print("\nWith many comparisons, a few 'yes' rows are expected by chance alone.")


if __name__ == "__main__":
    main()