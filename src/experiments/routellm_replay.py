"""Replay of the RouteLLM BERT-router baseline (pre-generation routing).

Reads the scores written by scripts/score_routellm.py and the cached model
outputs. No model and no router is run here.

What is compared (state this in the paper):
  * R = RouteLLM's pretrained BERT router score, thresholded with OUR
    calibration split. The router was trained for a different model pair and
    task mix, so this is NOT RouteLLM's own operating point.
  * R decides from the question text before any model runs (like D). E1/E2
    decide after the small model has answered. Their costs are not identical.
  * Router seconds are charged on EVERY question. When R escalates, the small
    call is skipped (primary run) or still paid (strict run).

Dry run for AUC (illustrative): wrong questions score [0.9, 0.6], right
questions score [0.5, 0.7] -> pairs where the wrong one scores higher: 0.9>0.5,
0.9>0.7, 0.6>0.5 = 3 of 4 -> AUC = 0.75. AUC 0.5 = no information.
"""
import json

from src.config import resolve
from src.data.gsm8k import load_items
from src.evaluation.answer_extract import is_correct
from src.experiments import replay as RP
from src.experiments.plots import curve_points, interp_accuracy
from src.metrics.metrics import escalation_rate, summarize
from src.routing import strategies as S
from src.routing.routellm_strategy import load_scores, router_escalation


def auc(scores, positive):
    """P(random positive scores higher than random negative); ties count 0.5."""
    pairs = [(s, p) for s, p in zip(scores, positive) if s is not None]
    pos = [s for s, p in pairs if p]
    neg = [s for s, p in pairs if not p]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def calibrate_router(cfg, small_model, scores):
    """Thresholds from the CALIBRATION split only (small-model correctness)."""
    rp = cfg["replay"]
    cal = RP.load_records(cfg, "calibration", small_model, require_samples=False)
    s = [scores[r["id"]]["score"] for r in cal]
    wrong = [not r["small"]["greedy"]["correct"] for r in cal]
    ops = RP.choose_thresholds(s, wrong, "above", rp["escalation_budget"],
                               rp["error_recall_target"])
    return ops, s


def _analyze_pair(out, pair, cfg, models, scores, ctx, seeds, res, k):
    rp = cfg["replay"]
    pre = ctx["pre_generation"]
    P = RP.prepare_pair(cfg, models[pair], models["strong"])
    ops_r, cal_scores = calibrate_router(cfg, models[pair], scores)
    if ops_r is None:
        raise ValueError(f"{pair}: no router scores on the calibration split")
    c = dict(ctx, fit=P["fit"], random_rates=rp["random_rates"])
    recs = P["eval"]
    r_eval = [scores[r["id"]]["score"] for r in recs]
    d_eval = [S.difficulty_score(r["question"], P["fit"]) for r in recs]

    def route(t):
        return [router_escalation(r, t, scores, pre) for r in recs]

    A = RP.apply("A", recs, None, c)
    B = RP.apply("B", recs, None, c)
    out["table"] += [RP.make_row(pair, "baselines", "A small-only", "-", None, A, res),
                     RP.make_row(pair, "baselines", "B strong-only", "-", None, B, res)]

    calibrated = []                       # (signal, operating point, outcomes)
    for op in RP.OPS:
        t = ops_r[op]["threshold"]
        ro = route(t)
        rate = escalation_rate(ro)
        out["thresholds"].append({
            "pair": pair, "point": op, "threshold": t,
            "cal_escalation_rate": ops_r[op]["cal_escalation_rate"],
            "cal_error_recall": ops_r[op]["cal_error_recall"],
            "eval_escalation_rate": rate})

        # Group R/op: R against D and random at R's own escalation rate.
        dm_t = RP.matched_threshold_above(d_eval, rate)
        dm = RP.apply("D", recs, dm_t, c)
        rm = RP.random_mean_summary(recs, rate, seeds, pre)
        e1_t = P["ops"]["E1"][op]["threshold"]
        e1 = RP.apply("E1", recs, e1_t, c)
        d_t = P["ops"]["D"][op]["threshold"]
        d_own = RP.apply("D", recs, d_t, c)
        g = f"R/{op}"
        out["table"] += [
            RP.make_row(pair, g, "R router", f"calibrated:{op}", t, ro, res),
            RP.make_row(pair, g, "D matched-rate", f"threshold {dm_t:.3f}", dm_t, dm, res),
            RP.random_row(pair, g, rate, rm, len(seeds)),
            RP.make_row(pair, g, "E1 (own point)", f"calibrated:{op}", e1_t, e1, res),
            RP.make_row(pair, g, "D (own point)", f"calibrated:{op}", d_t, d_own, res)]
        out["paired"] += [RP.paired_row(pair, g, "R", "D matched-rate", ro, dm, res),
                          RP.paired_row(pair, g, "R", "A small-only", ro, A, res),
                          RP.paired_row(pair, g, "R", "B strong-only", ro, B, res)]

        # Group E1/op: R and D both set to E1's escalation rate.
        e1_rate = escalation_rate(e1)
        rm_t = RP.matched_threshold_above(r_eval, e1_rate)
        r_m = route(rm_t)
        dm2_t = RP.matched_threshold_above(d_eval, e1_rate)
        dm2 = RP.apply("D", recs, dm2_t, c)
        g2 = f"E1/{op}"
        out["table"] += [
            RP.make_row(pair, g2, "E1", f"calibrated:{op}", e1_t, e1, res),
            RP.make_row(pair, g2, "R matched-rate", f"threshold {rm_t:.3f}", rm_t, r_m, res),
            RP.make_row(pair, g2, "D matched-rate", f"threshold {dm2_t:.3f}", dm2_t, dm2, res)]
        out["paired"] += [RP.paired_row(pair, g2, "R", "E1", r_m, e1, res),
                          RP.paired_row(pair, g2, "R", "D matched-rate", r_m, dm2, res)]
        calibrated += [("R", op, ro), ("E1", op, e1), ("D", op, d_own)]

    # Curve of R over candidate thresholds, and the cost-matched table.
    r_curve = []
    for t in RP.candidate_thresholds(cal_scores):
        row = {"pair": pair, "strategy": "R", "param": t}
        row.update(summarize(route(t)))
        r_curve.append(row)
    out["curves"] += r_curve
    all_curves = RP.curves_for_pair(pair, P, c, seeds, k) + r_curve
    for sig, op, outs in calibrated:
        s = summarize(outs)
        row = {"pair": pair, "signal": sig, "operating_point": op,
               "cost_per_q": s["cost_per_q"], "accuracy": s["accuracy"]}
        for strat in ("C", "D", "E1", "E2", "R"):
            row[f"acc_{strat}_at_same_cost"] = interp_accuracy(
                curve_points(all_curves, pair, strat), s["cost_per_q"])
        out["cost_matched"].append(row)

    # Informativeness (offline, uses labels, no threshold involved).
    wrong = [not r["small"]["greedy"]["correct"] for r in recs]
    e1_wrong = [not is_correct(S.majority_answer(r["small"]["samples"]),
                               r["reference_answer"]) for r in recs]
    e2_prob = [S.logprob_signal(r["small"]["greedy"].get("logprobs"), ctx["mode"])
               for r in recs]
    e1_dis = [1.0 - S.agreement_score(r["small"]["samples"]) for r in recs]
    for name, sc, target, lab in (
            ("R router score", r_eval, "greedy answer wrong", wrong),
            ("D difficulty score", d_eval, "greedy answer wrong", wrong),
            ("E1 disagreement", e1_dis, "greedy answer wrong", wrong),
            ("E1 disagreement", e1_dis, "majority answer wrong", e1_wrong),
            ("E2 low confidence", [None if p is None else 1.0 - p for p in e2_prob],
             "greedy answer wrong", wrong)):
        out["auc"].append({"pair": pair, "signal": name, "target": target,
                           "auc": auc(sc, lab), "n_positive": sum(lab), "n": len(lab)})

    for i, r in enumerate(recs):
        row = {"pair": pair, "id": r["id"], "router_score": r_eval[i],
               "router_seconds": scores[r["id"]]["seconds"],
               "small_correct": r["small"]["greedy"]["correct"],
               "strong_correct": r["strong"]["greedy"]["correct"]}
        for op, outs in ((o, x) for s_, o, x in calibrated if s_ == "R"):
            row[f"escalate_R_{op}"] = outs[i]["escalated"]
        out["per_question"][pair].append(row)


def run_routellm_replay(cfg, pre_generation=None, out_subdir="routellm", verbose=True):
    rp, rl = cfg["replay"], cfg["routellm"]
    pre = rp["pre_generation"] if pre_generation is None else pre_generation
    ctx = {"pre_generation": pre, "mode": rp["logprob_mode"]}
    seeds = list(range(cfg["seed"], cfg["seed"] + rp["random_seeds"]))
    res = rp["bootstrap_resamples"]
    k = cfg["generation"]["k_samples"]
    models = {"pair1": cfg["models"]["small_a"], "pair2": cfg["models"]["small_b"],
              "strong": cfg["models"]["strong"]}

    ids = []
    for split in ("calibration", "evaluation"):
        ids += [it["id"] for it in
                load_items(resolve(f"{cfg['data']['dir']}/{split}.json"))]
    scores, meta = load_scores(resolve(rl["scores_file"]), ids)

    out = {"table": [], "curves": [], "paired": [], "cost_matched": [], "auc": [],
           "thresholds": [], "per_question": {"pair1": [], "pair2": []},
           "models": models, "pre_generation": pre, "router_meta": meta}
    for pair in ("pair1", "pair2"):
        _analyze_pair(out, pair, cfg, models, scores, ctx, seeds, res, k)

    out_dir = resolve(cfg["paths"]["results"]) / out_subdir
    RP.write_csv(out_dir / "summary_routellm.csv", out["table"])
    RP.write_csv(out_dir / "curves_routellm.csv", out["curves"])
    RP.write_csv(out_dir / "paired_routellm.csv", out["paired"])
    RP.write_csv(out_dir / "cost_matched_routellm.csv", out["cost_matched"])
    RP.write_csv(out_dir / "auc_routellm.csv", out["auc"])
    RP.write_csv(out_dir / "thresholds_routellm.csv", out["thresholds"])
    for pair, rows in out["per_question"].items():
        RP.write_csv(out_dir / f"per_question_{pair}.csv", rows)
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"router_meta": meta, "pre_generation": pre, "models": models,
                   "thresholds": out["thresholds"]}, f, indent=1)

    if verbose:
        print_report(out)
        print(f"\nCSV files written to: {out_dir}")
    return out


# ---------------------------------------------------------------- printing
def _pct(x):
    return f"{x * 100:5.1f}%"


def _ci(r):
    if r.get("acc_ci_low") is not None:
        return f"[{r['acc_ci_low'] * 100:3.0f}-{r['acc_ci_high'] * 100:3.0f}]"
    return f"(sd {r.get('accuracy_sd_over_seeds', 0) * 100:.1f})"


def print_report(out):
    meta, m = out["router_meta"], out["models"]
    print("\n=== ROUTELLM BASELINE (BERT router, scores from the separate environment) ===")
    print(f"checkpoint {meta.get('checkpoint')} | input: {meta.get('input')} | "
          f"router load {meta.get('router_load_seconds', 0):.1f} s")
    print("cost rule: " + ("D/R/random skip the small call when they escalate (primary)"
                           if out["pre_generation"] else
                           "STRICT: every question pays the small call first"))
    for pair in ("pair1", "pair2"):
        n = next(r["n"] for r in out["table"] if r["pair"] == pair)
        print(f"\n--- {pair.upper()}: small = {m[pair]}, strong = {m['strong']} (n = {n}) ---")
        print(f"{'group':<11}{'strategy':<17}{'esc':>7}{'acc':>8}{'95% CI':>11}"
              f"{'cost s/q':>10}{'overhead':>10}")
        for r in out["table"]:
            if r["pair"] == pair:
                print(f"{r['group']:<11}{r['strategy']:<17}{_pct(r['escalation_rate']):>7}"
                      f"{_pct(r['accuracy']):>8}{_ci(r):>11}{r['cost_per_q']:>10.2f}"
                      f"{r['overhead_per_q']:>10.2f}")

    print("\n=== ROUTER THRESHOLDS: calibration vs evaluation ===")
    print(f"{'pair':<7}{'point':<9}{'thr':>7}{'cal esc':>9}{'cal recall':>12}{'eval esc':>10}")
    for r in out["thresholds"]:
        print(f"{r['pair']:<7}{r['point']:<9}{r['threshold']:>7.3f}"
              f"{_pct(r['cal_escalation_rate']):>9}{_pct(r['cal_error_recall']):>12}"
              f"{_pct(r['eval_escalation_rate']):>10}")

    print("\n=== COST-MATCHED: accuracy of each curve at the SAME cost as the signal ===")
    print("(linear interpolation along each curve; descriptive, no intervals)")
    print(f"{'pair':<7}{'signal':<8}{'point':<9}{'cost s':>8}{'its acc':>9}"
          f"{'C':>8}{'D':>8}{'E1':>8}{'E2':>8}{'R':>8}")
    f = lambda x: f"{x * 100:7.1f}%" if x is not None else f"{'n/a':>8}"
    for r in out["cost_matched"]:
        print(f"{r['pair']:<7}{r['signal']:<8}{r['operating_point']:<9}"
              f"{r['cost_per_q']:>8.2f}{r['accuracy'] * 100:>8.1f}%"
              f"{f(r['acc_C_at_same_cost'])}{f(r['acc_D_at_same_cost'])}"
              f"{f(r['acc_E1_at_same_cost'])}{f(r['acc_E2_at_same_cost'])}"
              f"{f(r['acc_R_at_same_cost'])}")

    print("\n=== INFORMATIVENESS: AUC for spotting wrong small-model answers (evaluation) ===")
    print("0.5 = no information, 1.0 = perfect. Uses labels offline only; no intervals.")
    print(f"{'pair':<7}{'signal':<20}{'target':<24}{'AUC':>7}{'wrong/n':>10}")
    for r in out["auc"]:
        a = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
        print(f"{r['pair']:<7}{r['signal']:<20}{r['target']:<24}{a:>7}"
              f"{str(r['n_positive']) + '/' + str(r['n']):>10}")