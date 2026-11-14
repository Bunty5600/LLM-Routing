"""Replay of the synopsis-style strategies. No model is run here.

Reads the cached JSONL outputs (small, medium and strong models) and writes
results to a NEW folder (results/tiered, results/tiered_strict). Existing
results are not touched.

What is compared:
  P  = capability-profile router (static; difficulty buckets measured on the
       calibration split; operating point required quality = 0.85, the number
       from the synopsis, fixed before any result was seen; a sweep is also run)
  TS = two-stage router (P first, then E1 on the small model)
  3T = three-tier cascade (small E1 -> medium E1 -> strong), pair 1 only
The test that isolates the value of stage 1 is TS vs E1 at the SAME E1 threshold.
Matched-rate comparisons ignore overhead; the cost-matched table does not.
"""
import json

from src.config import resolve
from src.experiments import replay as RP
from src.experiments.plots import curve_points, interp_accuracy
from src.metrics.extra import annotate_calls, extra_summary
from src.metrics.metrics import escalation_rate, summarize
from src.routing import strategies as S
from src.routing import tiered as T


def _ts_name(t):
    return f"TS_t{t}"


def _row(pair, group, strategy, setting, threshold, outcomes, res, **extra):
    row = RP.make_row(pair, group, strategy, setting, threshold, outcomes, res)
    row.update(extra_summary(outcomes))
    row.update(extra)
    return row


def _share(outcomes, key, value):
    return sum(1 for o in outcomes if o.get(key) == value) / len(outcomes)


# ------------------------------------------------------------------ two-tier
def _analyze_pair(out, pair, cfg, models, ctx, seeds, res, k):
    rp, tc = cfg["replay"], cfg["tiered"]
    pre = ctx["pre_generation"]
    grid, qgrid, qstar = tc["agreement_thresholds"], tc["quality_grid"], tc["required_quality"]

    P = RP.prepare_pair(cfg, models[pair], models["strong"])
    cal_full = RP.load_records(cfg, "calibration", models[pair], models["strong"])
    fit, recs = P["fit"], P["eval"]
    c = dict(ctx, fit=fit, random_rates=rp["random_rates"])

    profile = T.build_profile(cal_full, fit, tc["n_buckets"])
    edges = profile["edges"]
    for b, bk in enumerate(profile["buckets"]):
        out["profile"].append({
            "pair": pair, "bucket": b + 1, "n_calibration": bk["n"],
            "small_acc": bk["small_acc"], "strong_acc": bk["strong_acc"],
            "score_above": edges[b - 1] if b > 0 else None,
            "score_up_to": edges[b] if b < len(edges) else None})

    A = annotate_calls(RP.apply("A", recs, None, c), "A", k, pre)
    B = annotate_calls(RP.apply("B", recs, None, c), "B", k, pre)
    R = annotate_calls(RP.apply("Ref", recs, None, c), "Ref", k, pre)
    E1 = {t: annotate_calls(RP.apply("E1", recs, t, c), "E1", k, pre) for t in grid}
    TS = {t: [T.two_stage(r, profile, qstar, fit, t, pre) for r in recs] for t in grid}
    Pq = annotate_calls([T.capability_router(r, profile, qstar, fit, pre) for r in recs],
                        "P", k, pre)

    rate = escalation_rate(Pq)                       # matched baselines use the rate only
    d_eval = [S.difficulty_score(r["question"], fit) for r in recs]
    dm_t = RP.matched_threshold_above(d_eval, rate)
    Dm = annotate_calls(RP.apply("D", recs, dm_t, c), "D", k, pre)
    rm = RP.random_mean_summary(recs, rate, seeds, pre)
    crow = RP.random_row(pair, "P", rate, rm, len(seeds))
    crow["mean_model_calls"] = 1.0 if pre else 1.0 + rate
    crow["routing_accuracy"] = None

    tb = out["table"]
    tb += [_row(pair, "baselines", "A small-only", "-", None, A, res),
           _row(pair, "baselines", "B strong-only", "-", None, B, res),
           _row(pair, "baselines", "Ref oracle", "uses labels", None, R, res)]
    for t in grid:
        tb.append(_row(pair, "E1", "E1", f"t={t}", t, E1[t], res))
    tb += [_row(pair, "P", "P profile-router", f"q={qstar}", qstar, Pq, res),
           _row(pair, "P", "D matched-rate", f"threshold {dm_t:.3f}", dm_t, Dm, res),
           crow]
    for t in grid:
        tb.append(_row(pair, "TS", "TS two-stage", f"q={qstar}, t={t}", t, TS[t], res,
                       share_direct=_share(TS[t], "route", "direct")))

    pr = out["paired"]
    for t in grid:
        pr.append(RP.paired_row(pair, f"TS/E1 {t}", "TS", "E1", TS[t], E1[t], res))
    pr += [RP.paired_row(pair, "P/D", "P", "D matched", Pq, Dm, res),
           RP.paired_row(pair, "P/ends", "P", "A small-only", Pq, A, res),
           RP.paired_row(pair, "P/ends", "P", "B strong-only", Pq, B, res)]

    # Curves: the existing ones plus the capability router and two-stage sweeps.
    curves = RP.curves_for_pair(pair, {"eval": recs, "cal": P["cal"]}, c, seeds, k)
    for q in qgrid:
        po = [T.capability_router(r, profile, q, fit, pre) for r in recs]
        row = {"pair": pair, "strategy": "P", "param": q}
        row.update(summarize(po))
        curves.append(row)
        for t in grid:
            tso = [T.two_stage(r, profile, q, fit, t, pre) for r in recs]
            row = {"pair": pair, "strategy": _ts_name(t), "param": q}
            row.update(summarize(tso))
            curves.append(row)
    out["curves"] += curves

    # Cost-matched: every curve's accuracy at the SAME cost as each signal.
    comps = ["C", "D", "E1", "P"] + [_ts_name(t) for t in grid]
    signals = ([("E1", f"t={t}", E1[t]) for t in grid]
               + [("P", f"q={qstar}", Pq)]
               + [(_ts_name(t), f"q={qstar}", TS[t]) for t in grid])
    for name, point, outs in signals:
        s = summarize(outs)
        row = {"pair": pair, "signal": name, "point": point,
               "cost_per_q": s["cost_per_q"], "accuracy": s["accuracy"]}
        for comp in comps:
            row[f"acc_{comp}_at_same_cost"] = interp_accuracy(
                curve_points(curves, pair, comp), s["cost_per_q"])
        out["cost_matched"].append(row)

    for r in recs:
        sc = S.difficulty_score(r["question"], fit)
        out["per_question"][pair].append({
            "pair": pair, "id": r["id"], "difficulty_score": sc,
            "bucket": T.bucket_of(sc, edges) + 1,
            "direct_route": T.profile_decision(r["question"], profile, qstar, fit),
            "e1_agreement": S.agreement_score(r["small"]["samples"]),
            "small_correct": r["small"]["greedy"]["correct"],
            "strong_correct": r["strong"]["greedy"]["correct"]})
    return P


# ---------------------------------------------------------------- three-tier
def _analyze_three_tier(out, pair, P, cfg, ctx, res, k):
    tc = cfg["tiered"]
    pre = ctx["pre_generation"]
    grid = tc["agreement_thresholds"]
    c = dict(ctx, fit=P["fit"])

    med_rows = {r["id"]: r for r in RP.load_records(cfg, "evaluation", tc["medium_model"])}
    recs3, med_recs = [], []
    for r in P["eval"]:
        m = med_rows[r["id"]]["small"]
        base = {"id": r["id"], "question": r["question"],
                "reference_answer": r["reference_answer"], "strong": r["strong"]}
        recs3.append(dict(base, small=r["small"], medium=m))
        med_recs.append(dict(base, small=m))

    A = annotate_calls(RP.apply("A", P["eval"], None, c), "A", k, pre)
    M = annotate_calls([S.small_only(r) for r in med_recs], "A", k, pre)
    B = annotate_calls(RP.apply("B", P["eval"], None, c), "B", k, pre)
    E1s = {t: annotate_calls([S.self_consistency(r, t) for r in P["eval"]], "E1", k, pre)
           for t in grid}
    E1m = {t: annotate_calls([S.self_consistency(r, t) for r in med_recs], "E1", k, pre)
           for t in grid}
    TT = {(t1, t2): [T.three_tier(r, t1, t2) for r in recs3] for t1 in grid for t2 in grid}

    tb = out["three_tier"]
    tb += [_row(pair, "three-tier", "A small-only", "1.5B", None, A, res),
           _row(pair, "three-tier", "M medium-only", "3B", None, M, res),
           _row(pair, "three-tier", "B strong-only", "7B", None, B, res)]
    for t in grid:
        tb.append(_row(pair, "three-tier", "E1 small->strong", f"t={t}", t, E1s[t], res))
    for t in grid:
        tb.append(_row(pair, "three-tier", "E1 medium->strong", f"t={t}", t, E1m[t], res))
    for (t1, t2), outs in TT.items():
        tb.append(_row(pair, "three-tier", "3T three-tier", f"t1={t1}, t2={t2}", t1, outs, res,
                       share_small=_share(outs, "tier_final", "small"),
                       share_medium=_share(outs, "tier_final", "medium"),
                       share_strong=_share(outs, "tier_final", "strong")))

    for t in grid:
        grp = f"3T {t}"
        out["paired_three"] += [
            RP.paired_row(pair, grp, "3T", "E1 small", TT[(t, t)], E1s[t], res),
            RP.paired_row(pair, grp, "3T", "E1 med", TT[(t, t)], E1m[t], res),
            RP.paired_row(pair, grp, "3T", "B strong", TT[(t, t)], B, res)]


# ---------------------------------------------------------------- main entry
def run_tiered_replay(cfg, pre_generation=None, out_subdir="tiered", verbose=True):
    rp, tc = cfg["replay"], cfg["tiered"]
    pre = rp["pre_generation"] if pre_generation is None else pre_generation
    ctx = {"pre_generation": pre, "mode": rp["logprob_mode"]}
    seeds = list(range(cfg["seed"], cfg["seed"] + rp["random_seeds"]))
    res = rp["bootstrap_resamples"]
    k = cfg["generation"]["k_samples"]
    models = {"pair1": cfg["models"]["small_a"], "pair2": cfg["models"]["small_b"],
              "strong": cfg["models"]["strong"]}

    out = {"profile": [], "table": [], "curves": [], "paired": [], "cost_matched": [],
           "three_tier": [], "paired_three": [], "per_question": {"pair1": [], "pair2": []},
           "models": models, "pre_generation": pre, "tiered_cfg": tc}
    for pair in ("pair1", "pair2"):
        P = _analyze_pair(out, pair, cfg, models, ctx, seeds, res, k)
        if pair in tc["three_tier_pairs"]:
            _analyze_three_tier(out, pair, P, cfg, ctx, res, k)

    d = resolve(cfg["paths"]["results"]) / out_subdir
    RP.write_csv(d / "profile.csv", out["profile"])
    RP.write_csv(d / "summary_two_tier.csv", out["table"])
    RP.write_csv(d / "curves_two_tier.csv", out["curves"])
    RP.write_csv(d / "cost_matched_two_tier.csv", out["cost_matched"])
    RP.write_csv(d / "paired_two_tier.csv", out["paired"])
    RP.write_csv(d / "summary_three_tier.csv", out["three_tier"])
    RP.write_csv(d / "paired_three_tier.csv", out["paired_three"])
    for pair, rows in out["per_question"].items():
        RP.write_csv(d / f"per_question_{pair}.csv", rows)
    with open(d / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"models": models, "pre_generation": pre, "tiered": tc}, f, indent=1)

    if verbose:
        print_report(out)
        print(f"\nCSV files written to: {d}")
    return out


# ---------------------------------------------------------------- printing
def _pct(x):
    return f"{x * 100:5.1f}%" if x is not None else f"{'-':>6}"


def _num(x, spec):
    return format(x, spec) if x is not None else f"{'-':>{len(format(0, spec))}}"


def _ci(r):
    if r.get("acc_ci_low") is not None:
        return f"[{r['acc_ci_low'] * 100:3.0f}-{r['acc_ci_high'] * 100:3.0f}]"
    return f"(sd {r.get('accuracy_sd_over_seeds', 0) * 100:.1f})"


def _label(r):
    st = r.get("setting")
    return r["strategy"] if st in (None, "-", "") else f"{r['strategy']} {st}"


def print_report(out):
    m, tc = out["models"], out["tiered_cfg"]
    print("\n=== SYNOPSIS-STYLE STRATEGIES (offline replay) ===")
    print("cost rule: " + ("pre-generation routers skip the small call when they escalate "
                           "(primary)" if out["pre_generation"] else
                           "STRICT: every question pays the small call first"))
    print(f"required quality {tc['required_quality']} (predefined) | buckets {tc['n_buckets']} "
          f"| E1 thresholds {tc['agreement_thresholds']}")

    print("\n=== CAPABILITY PROFILE (calibration split; small buckets are noisy) ===")
    print(f"{'pair':<7}{'bucket':>7}{'n':>5}{'small acc':>11}{'strong acc':>12}  difficulty range")
    for r in out["profile"]:
        lo = "-inf" if r["score_above"] is None else f"{r['score_above']:.2f}"
        hi = "+inf" if r["score_up_to"] is None else f"{r['score_up_to']:.2f}"
        print(f"{r['pair']:<7}{r['bucket']:>7}{r['n_calibration']:>5}"
              f"{_pct(r['small_acc']):>11}{_pct(r['strong_acc']):>12}  ({lo}, {hi}]")

    for pair in ("pair1", "pair2"):
        rows = [r for r in out["table"] if r["pair"] == pair]
        print(f"\n--- {pair.upper()}: small = {m[pair]}, strong = {m['strong']} (n = {rows[0]['n']}) ---")
        print(f"{'strategy':<34}{'strong%':>8}{'acc':>8}{'95% CI':>11}{'cost s/q':>10}"
              f"{'overhead':>10}{'calls':>7}{'route acc':>10}{'direct':>8}")
        for r in rows:
            print(f"{_label(r):<34}{_pct(r['escalation_rate']):>8}{_pct(r['accuracy']):>8}"
                  f"{_ci(r):>11}{r['cost_per_q']:>10.2f}{r['overhead_per_q']:>10.2f}"
                  f"{_num(r.get('mean_model_calls'), '7.2f')}"
                  f"{_pct(r.get('routing_accuracy')):>10}{_pct(r.get('share_direct')):>8}")

    comps = ["C", "D", "E1", "P"] + [_ts_name(t) for t in tc["agreement_thresholds"]]
    print("\n=== COST-MATCHED: accuracy of each curve at the SAME cost as the signal ===")
    print("(linear interpolation; descriptive, no intervals)")
    print(f"{'pair':<7}{'signal':<11}{'point':<9}{'cost s':>8}{'its acc':>9}"
          + "".join(f"{c_:>10}" for c_ in comps))
    for r in out["cost_matched"]:
        cells = "".join(f"{_pct(r[f'acc_{c_}_at_same_cost']):>10}" for c_ in comps)
        print(f"{r['pair']:<7}{r['signal']:<11}{r['point']:<9}{r['cost_per_q']:>8.2f}"
              f"{_pct(r['accuracy']):>9}{cells}")

    if out["three_tier"]:
        print("\n=== THREE-TIER CASCADE (pair 1: 1.5B -> 3B -> 7B) ===")
        print(f"{'strategy':<34}{'strong%':>8}{'sm/med/str':>16}{'acc':>8}{'95% CI':>11}"
              f"{'cost s/q':>10}{'overhead':>10}{'calls':>7}")
        for r in out["three_tier"]:
            if r.get("share_small") is not None:
                split = (f"{r['share_small'] * 100:.0f}/{r['share_medium'] * 100:.0f}/"
                         f"{r['share_strong'] * 100:.0f}")
            else:
                split = "-"
            print(f"{_label(r):<34}{_pct(r['escalation_rate']):>8}{split:>16}"
                  f"{_pct(r['accuracy']):>8}{_ci(r):>11}{r['cost_per_q']:>10.2f}"
                  f"{r['overhead_per_q']:>10.2f}{_num(r.get('mean_model_calls'), '7.2f')}")