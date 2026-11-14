"""Sweep of the capability-profile (P) and two-stage (TS) routers over the quality
target, paired tests at two targets declared BEFORE their results were seen, and
paired tests for the 3B medium tier (pair 1 only).

Use your MAIN Python:   python scripts\\show_tiered_sweep.py
Offline: no models, no Ollama. Writes and changes no file.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.experiments import replay as RP
from src.metrics.metrics import escalation_rate, summarize
from src.routing import strategies as S
from src.routing import tiered as T

DECLARED_Q = (0.7, 0.8)      # declared before these results were seen


def pline(r):
    d, lo, hi = (float(r[k]) * 100 for k in
                 ("acc_diff_a_minus_b", "acc_diff_ci_low", "acc_diff_ci_high"))
    excl = "yes" if (lo > 0 or hi < 0) else "no"
    return (f"  {r['a']:<17}vs {r['b']:<22}{d:>+6.1f} pts [{lo:+.1f}, {hi:+.1f}] "
            f"excl0={excl:<3} McNemar p={float(r['mcnemar_p']):.3f} "
            f"({r['a_right_b_wrong']}/{r['a_wrong_b_right']})  "
            f"cost {float(r['cost_diff_a_minus_b']):+.2f} s "
            f"[{float(r['cost_diff_ci_low']):+.2f}, {float(r['cost_diff_ci_high']):+.2f}]")


def srow(label, q, outs):
    s = summarize(outs)
    print(f"{label:<13}{q:>6.2f}{s['escalation_rate'] * 100:>8.0f}%{s['accuracy'] * 100:>8.1f}%"
          f"{s['cost_per_q']:>10.2f}{s['overhead_per_q']:>10.2f}")


def main():
    cfg = load_config()
    rp, tc = cfg["replay"], cfg["tiered"]
    pre = rp["pre_generation"]
    ctx = {"pre_generation": pre, "mode": rp["logprob_mode"]}
    res = rp["bootstrap_resamples"]
    models = {"pair1": cfg["models"]["small_a"], "pair2": cfg["models"]["small_b"],
              "strong": cfg["models"]["strong"]}
    grid, qgrid = tc["agreement_thresholds"], tc["quality_grid"]

    for pair in ("pair1", "pair2"):
        P = RP.prepare_pair(cfg, models[pair], models["strong"])
        cal = RP.load_records(cfg, "calibration", models[pair], models["strong"])
        fit, recs = P["fit"], P["eval"]
        c = dict(ctx, fit=fit, random_rates=rp["random_rates"])
        profile = T.build_profile(cal, fit, tc["n_buckets"])
        A, B = RP.apply("A", recs, None, c), RP.apply("B", recs, None, c)
        E1 = {t: RP.apply("E1", recs, t, c) for t in grid}
        d_eval = [S.difficulty_score(r["question"], fit) for r in recs]

        print(f"\n=== {pair.upper()}: {models[pair]} + {models['strong']} | SWEEP OVER REQUIRED QUALITY ===")
        print(f"{'strategy':<13}{'q':>6}{'strong%':>9}{'acc':>9}{'cost s/q':>10}{'overhead':>10}")
        srow("A small-only", 0.0, A)
        for q in qgrid:
            srow("P", q, [T.capability_router(r, profile, q, fit, pre) for r in recs])
            for t in grid:
                srow(f"TS t={t}", q, [T.two_stage(r, profile, q, fit, t, pre) for r in recs])
        srow("B strong-only", 1.0, B)

        print(f"\n--- {pair}: paired tests at declared q = {DECLARED_Q} ---")
        for q in DECLARED_Q:
            po = [T.capability_router(r, profile, q, fit, pre) for r in recs]
            rate = escalation_rate(po)
            dm = RP.apply("D", recs, RP.matched_threshold_above(d_eval, rate), c)
            print(f"q = {q}: P sends {rate * 100:.0f}% straight to the strong model")
            for b_name, b in (("D matched-rate", dm), ("A small-only", A), ("B strong-only", B)):
                print(pline(RP.paired_row(pair, "P", "P", b_name, po, b, res)))
            for t in grid:
                tso = [T.two_stage(r, profile, q, fit, t, pre) for r in recs]
                print(pline(RP.paired_row(pair, "TS", f"TS t={t}", f"E1 t={t}", tso, E1[t], res)))
                print(pline(RP.paired_row(pair, "TS", f"TS t={t}", "B strong-only", tso, B, res)))

        if pair == "pair1":
            med_rows = {r["id"]: r for r in
                        RP.load_records(cfg, "evaluation", tc["medium_model"])}
            recs3, med_recs = [], []
            for r in recs:
                m = med_rows[r["id"]]["small"]
                base = {"id": r["id"], "question": r["question"],
                        "reference_answer": r["reference_answer"], "strong": r["strong"]}
                recs3.append(dict(base, small=r["small"], medium=m))
                med_recs.append(dict(base, small=m))
            M = [S.small_only(r) for r in med_recs]
            E1m = [S.self_consistency(r, 0.833) for r in med_recs]
            TT = [T.three_tier(r, 0.833, 0.833) for r in recs3]
            print("\n--- pair1: 3B medium-only vs the alternatives (declared before seeing results) ---")
            for b_name, b in (("B strong-only", B), ("A small-only", A),
                              ("E1 1.5B->7B t=.833", E1[0.833]),
                              ("E1 3B->7B t=.833", E1m), ("3T t=.833/.833", TT)):
                print(pline(RP.paired_row(pair, "M", "M 3B-only", b_name, M, b, res)))
    print("\nWith many comparisons, a few 'yes' rows are expected by chance alone.")


if __name__ == "__main__":
    main()