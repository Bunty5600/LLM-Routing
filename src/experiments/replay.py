"""Replay: simulate every strategy offline from the cached JSONL files.

No model is called here. Every strategy reads recorded answers and recorded
seconds and sums the cost of exactly the calls it would have made.

Calibration rules (labels from the CALIBRATION split only):
  * The strong model was not run on calibration, so thresholds are chosen from
    the small model's own correctness.
  * Operating point "budget": among thresholds that escalate <= budget of the
    calibration questions, pick the one catching the most small-model errors.
  * Operating point "recall": the lowest-escalation threshold that catches at
    least `error_recall_target` of the small model's errors.
Matched baselines use only the escalation RATE of a signal result, never labels.

Dry run for choosing a threshold (illustrative): E1 agreement scores
  [1.0, 1.0, 0.667, 0.667, 0.333], small wrong on questions 3 and 5.
  Candidate thresholds are midpoints between scores. Threshold 0.5 escalates
  one question (rate 0.2) and catches 1 of 2 errors (recall 0.5); threshold
  0.833 escalates three (rate 0.6) and catches both (recall 1.0).
  Budget 0.3 -> threshold 0.5. Recall target 0.7 -> threshold 0.833.
"""
import csv
from pathlib import Path
from statistics import mean, pstdev

from src.config import resolve
from src.data.gsm8k import load_items
from src.evaluation.answer_extract import is_correct
from src.experiments.collect_outputs import load_rows, output_path
from src.metrics.metrics import escalation_rate, summarize
from src.metrics.stats import bootstrap_ci, mcnemar, paired_bootstrap_diff
from src.routing import strategies as S

SIGNALS = ("E1", "E2", "D")
OPS = ("budget", "recall")


# ------------------------------------------------------------------ loading
def load_records(cfg, split, small_model, strong_model=None, require_samples=True):
    """Join questions with cached rows. Raises ValueError if anything is missing."""
    items = load_items(resolve(f"{cfg['data']['dir']}/{split}.json"))
    small_rows = {r["id"]: r for r in load_rows(output_path(split, small_model, cfg))}
    strong_rows = {}
    if strong_model:
        strong_rows = {r["id"]: r for r in load_rows(output_path(split, strong_model, cfg))}

    missing, recs = [], []
    for it in items:
        sm = small_rows.get(it["id"])
        st = strong_rows.get(it["id"]) if strong_model else None
        if sm is None or (strong_model and st is None):
            missing.append(it["id"])
            continue
        if require_samples and not sm.get("samples"):
            raise ValueError(f"{small_model} / {split}: rows have no sampled answers. "
                             "Collect it with --role small.")
        rec = {"id": it["id"], "question": it["question"],
               "reference_answer": it["reference_answer"],
               "small": {"greedy": sm["greedy"], "samples": sm.get("samples")}}
        if strong_model:
            rec["strong"] = {"greedy": st["greedy"]}
        recs.append(rec)

    if missing:
        raise ValueError(
            f"{split}: {len(missing)} of {len(items)} questions are missing for "
            f"{small_model}" + (f" / {strong_model}" if strong_model else "") +
            ". Finish collection first (the run_collect commands are resume-safe).")
    return recs


# -------------------------------------------------------------- calibration
def calibration_data(cal_recs, mode):
    fit = S.fit_difficulty([r["question"] for r in cal_recs])
    greedy_wrong = [not r["small"]["greedy"]["correct"] for r in cal_recs]
    e1_wrong = [not is_correct(S.majority_answer(r["small"]["samples"]),
                               r["reference_answer"]) for r in cal_recs]
    return {
        "fit": fit,
        "E1": {"direction": "below", "wrong": e1_wrong,
               "scores": [S.agreement_score(r["small"]["samples"]) for r in cal_recs]},
        "E2": {"direction": "below", "wrong": greedy_wrong,
               "scores": [S.logprob_signal(r["small"]["greedy"].get("logprobs"), mode)
                          for r in cal_recs]},
        "D": {"direction": "above", "wrong": greedy_wrong,
              "scores": [S.difficulty_score(r["question"], fit) for r in cal_recs]},
    }


def candidate_thresholds(scores):
    """Midpoints between observed scores, plus one 'never' and one 'always' value."""
    vals = sorted({s for s in scores if s is not None})
    if not vals:
        return []
    mids = [(a + b) / 2.0 for a, b in zip(vals, vals[1:])]
    return [vals[0] - 1.0] + mids + [vals[-1] + 1.0]


def rate_and_recall(scores, wrong, threshold, direction):
    esc = [S.escalate_if(s, threshold, direction) for s in scores]
    rate = sum(esc) / len(esc)
    n_wrong = sum(wrong)
    caught = sum(1 for e, w in zip(esc, wrong) if e and w)
    return rate, (caught / n_wrong if n_wrong else 1.0)


def choose_thresholds(scores, wrong, direction, budget, recall_target):
    """Return {"budget": {...}, "recall": {...}} or None if there are no scores."""
    cands = candidate_thresholds(scores)
    if not cands:
        return None
    stats = []
    for t in cands:
        rate, recall = rate_and_recall(scores, wrong, t, direction)
        stats.append((t, rate, recall))

    within = [x for x in stats if x[1] <= budget + 1e-9]
    pick_b = (max(within, key=lambda x: (x[2], -x[1])) if within
              else min(stats, key=lambda x: x[1]))
    enough = [x for x in stats if x[2] >= recall_target - 1e-9]
    pick_r = (min(enough, key=lambda x: (x[1], -x[2])) if enough
              else max(stats, key=lambda x: (x[2], -x[1])))

    def pack(x):
        return {"threshold": x[0], "cal_escalation_rate": x[1], "cal_error_recall": x[2]}
    return {"budget": pack(pick_b), "recall": pack(pick_r)}


def matched_threshold_above(scores, target_rate):
    """Threshold t so that 'score > t' escalates about target_rate of questions.

    Uses scores only, no labels. Dry run: scores [3, 1, 2, 5, 4], target 0.4
    -> m = 2 questions -> sorted high-to-low [5, 4, 3, 2, 1] -> t = (4 + 3) / 2
    = 3.5 -> scores above 3.5 are 5 and 4, so exactly 2 are escalated.
    """
    n = len(scores)
    order = sorted(scores, reverse=True)
    m = round(target_rate * n)
    if m <= 0:
        return order[0]
    if m >= n:
        return order[-1] - 1.0
    return (order[m - 1] + order[m]) / 2.0


# --------------------------------------------------------------- simulation
def apply(strategy, recs, threshold, ctx):
    pre = ctx["pre_generation"]
    if strategy == "A":
        return [S.small_only(r) for r in recs]
    if strategy == "B":
        return [S.strong_only(r) for r in recs]
    if strategy == "Ref":
        return [S.oracle(r) for r in recs]
    if strategy == "E1":
        return [S.self_consistency(r, threshold) for r in recs]
    if strategy == "E2":
        return [S.logprob_escalation(r, threshold, ctx["mode"]) for r in recs]
    if strategy == "D":
        return [S.difficulty_escalation(r, threshold, ctx["fit"], pre) for r in recs]
    raise ValueError(f"unknown strategy: {strategy}")


def random_mean_summary(recs, p, seeds, pre_generation):
    """Random escalation at rate p, averaged over many seeds."""
    sums = [summarize([S.random_escalation(r, p, sd, pre_generation) for r in recs])
            for sd in seeds]
    out = {"n": len(recs)}
    for key in ("accuracy", "cost_per_q", "overhead_per_q", "latency_per_q",
                "escalation_rate"):
        out[key] = mean(s[key] for s in sums)
    out["accuracy_sd_over_seeds"] = pstdev([s["accuracy"] for s in sums])
    return out


def make_row(pair, group, strategy, setting, threshold, outcomes, resamples, seed=0):
    row = summarize(outcomes)
    ci = bootstrap_ci([1.0 if o["correct"] else 0.0 for o in outcomes],
                      n_resamples=resamples, seed=seed)
    row.update({"pair": pair, "group": group, "strategy": strategy,
                "setting": setting, "threshold": threshold,
                "acc_ci_low": ci["ci_low"], "acc_ci_high": ci["ci_high"]})
    return row


def random_row(pair, group, rate, rm, n_seeds):
    row = dict(rm)
    row.update({"pair": pair, "group": group, "strategy": "C matched-rate",
                "setting": f"p={rate:.3f}, mean of {n_seeds} seeds", "threshold": rate})
    return row


def paired_row(pair, group, a_name, b_name, a, b, resamples, seed=0):
    a_ok = [bool(o["correct"]) for o in a]
    b_ok = [bool(o["correct"]) for o in b]
    acc = paired_bootstrap_diff([float(x) for x in a_ok], [float(x) for x in b_ok],
                                n_resamples=resamples, seed=seed)
    cost = paired_bootstrap_diff([o["cost_seconds"] for o in a],
                                 [o["cost_seconds"] for o in b],
                                 n_resamples=resamples, seed=seed)
    mc = mcnemar(a_ok, b_ok)
    return {"pair": pair, "group": group, "a": a_name, "b": b_name,
            "acc_diff_a_minus_b": acc["mean_diff"], "acc_diff_ci_low": acc["ci_low"],
            "acc_diff_ci_high": acc["ci_high"], "acc_diff_p_boot": acc["p_value"],
            "mcnemar_p": mc["p_value"], "a_right_b_wrong": mc["a_right_b_wrong"],
            "a_wrong_b_right": mc["a_wrong_b_right"],
            "cost_diff_a_minus_b": cost["mean_diff"],
            "cost_diff_ci_low": cost["ci_low"], "cost_diff_ci_high": cost["ci_high"]}


# ------------------------------------------------------------- pair analysis
def prepare_pair(cfg, small_model, strong_model):
    rp = cfg["replay"]
    cal_recs = load_records(cfg, "calibration", small_model)
    eval_recs = load_records(cfg, "evaluation", small_model, strong_model)
    cal = calibration_data(cal_recs, rp["logprob_mode"])
    ops = {}
    for sig in SIGNALS:
        c = cal[sig]
        ops[sig] = choose_thresholds(c["scores"], c["wrong"], c["direction"],
                                     rp["escalation_budget"], rp["error_recall_target"])
    return {"small": small_model, "cal": cal, "fit": cal["fit"],
            "eval": eval_recs, "ops": ops}


def per_question_rows(pair, P, ctx):
    rows = []
    mode = ctx["mode"]
    for r in P["eval"]:
        agr = S.agreement_score(r["small"]["samples"])
        e2 = S.logprob_signal(r["small"]["greedy"].get("logprobs"), mode)
        d = S.difficulty_score(r["question"], P["fit"])
        row = {"pair": pair, "id": r["id"],
               "small_correct": r["small"]["greedy"]["correct"],
               "strong_correct": r["strong"]["greedy"]["correct"],
               "e1_majority_correct": is_correct(S.majority_answer(r["small"]["samples"]),
                                                 r["reference_answer"]),
               "e1_agreement": agr, "e2_score": e2, "difficulty_score": d}
        for sig, score, direction in (("E1", agr, "below"), ("E2", e2, "below"),
                                      ("D", d, "above")):
            if P["ops"][sig] is None:
                continue
            for op in OPS:
                t = P["ops"][sig][op]["threshold"]
                row[f"escalate_{sig}_{op}"] = S.escalate_if(score, t, direction)
        rows.append(row)
    return rows


def curves_for_pair(pair, P, c, seeds, k):
    recs, pre = P["eval"], c["pre_generation"]
    rows = []

    def add(strategy, param, summary):
        row = {"pair": pair, "strategy": strategy, "param": param}
        row.update(summary)
        rows.append(row)

    for strategy in ("A", "B", "Ref"):
        add(strategy, "", summarize(apply(strategy, recs, None, c)))
    for p in c["random_rates"]:
        add("C", p, random_mean_summary(recs, p, seeds, pre))
    for j in range(1, k + 2):                      # k = 3 -> 0.167, 0.5, 0.833, 1.167
        t = (j - 0.5) / k
        add("E1", t, summarize(apply("E1", recs, t, c)))
    for sig in ("E2", "D"):
        for t in candidate_thresholds(P["cal"][sig]["scores"]):
            add(sig, t, summarize(apply(sig, recs, t, c)))
    return rows


def posthoc_rows(pair, curves):
    """Cheapest curve point within 2 points of strong-only (read off, not tuned)."""
    b = next(r for r in curves if r["strategy"] == "B")
    target = b["accuracy"] - 0.02
    rows = []
    for strat in ("E1", "E2", "D", "C"):
        cand = [r for r in curves if r["strategy"] == strat
                and r["accuracy"] >= target - 1e-12]
        row = {"pair": pair, "strategy": strat,
               "strong_only_accuracy": b["accuracy"], "strong_only_cost": b["cost_per_q"],
               "note": "post-hoc: read off the evaluation curve, not tuned on it"}
        if cand:
            best = min(cand, key=lambda r: (r["cost_per_q"], r["escalation_rate"]))
            row.update({"param": best["param"],
                        "escalation_rate": best["escalation_rate"],
                        "accuracy": best["accuracy"], "cost_per_q": best["cost_per_q"],
                        "cost_vs_strong_only": best["cost_per_q"] / b["cost_per_q"]})
        else:
            row["result"] = "never within 2 points of strong-only"
        rows.append(row)
    return rows


def analyze_pair(pair, P, ctx, rp, seeds, resamples, k):
    c = dict(ctx, fit=P["fit"], random_rates=rp["random_rates"])
    recs, pre = P["eval"], ctx["pre_generation"]
    n_seeds = len(seeds)
    table, paired = [], []

    A = apply("A", recs, None, c)
    B = apply("B", recs, None, c)
    R = apply("Ref", recs, None, c)
    table += [make_row(pair, "baselines", "A small-only", "-", None, A, resamples),
              make_row(pair, "baselines", "B strong-only", "-", None, B, resamples),
              make_row(pair, "baselines", "Ref oracle", "uses labels", None, R, resamples)]

    d_eval = [S.difficulty_score(r["question"], P["fit"]) for r in recs]

    for sig in ("E1", "E2"):
        if P["ops"][sig] is None:
            continue
        for op in OPS:
            t = P["ops"][sig][op]["threshold"]
            group = f"{sig}/{op}"
            so = apply(sig, recs, t, c)
            rate = escalation_rate(so)
            dm_t = matched_threshold_above(d_eval, rate)
            dm = apply("D", recs, dm_t, c)
            rm = random_mean_summary(recs, rate, seeds, pre)
            table.append(make_row(pair, group, sig, f"calibrated:{op}", t, so, resamples))
            table.append(make_row(pair, group, "D matched-rate",
                                  f"threshold {dm_t:.3f}", dm_t, dm, resamples))
            table.append(random_row(pair, group, rate, rm, n_seeds))
            paired += [paired_row(pair, group, sig, "D matched-rate", so, dm, resamples),
                       paired_row(pair, group, sig, "A small-only", so, A, resamples),
                       paired_row(pair, group, sig, "B strong-only", so, B, resamples)]

    for op in OPS:                                  # difficulty baseline, own calibration
        t = P["ops"]["D"][op]["threshold"]
        group = f"D/{op}"
        do = apply("D", recs, t, c)
        rate = escalation_rate(do)
        rm = random_mean_summary(recs, rate, seeds, pre)
        table.append(make_row(pair, group, "D", f"calibrated:{op}", t, do, resamples))
        table.append(random_row(pair, group, rate, rm, n_seeds))
        paired += [paired_row(pair, group, "D", "A small-only", do, A, resamples),
                   paired_row(pair, group, "D", "B strong-only", do, B, resamples)]

    curves = curves_for_pair(pair, P, c, seeds, k)
    return (table, curves, paired, posthoc_rows(pair, curves),
            per_question_rows(pair, P, c))


def transfer_rows(P1, P2, ctx, seeds, resamples):
    """Pair-1 thresholds unchanged on pair 2, versus recalibrated on pair 2."""
    rows = []
    pre = ctx["pre_generation"]
    for sig in SIGNALS:
        if P1["ops"][sig] is None or P2["ops"][sig] is None:
            continue
        for op in OPS:
            t1 = P1["ops"][sig][op]["threshold"]
            t2 = P2["ops"][sig][op]["threshold"]
            for pair, setting, P, t in (("pair1", "pair1 own", P1, t1),
                                        ("pair2", "pair2 unchanged threshold", P2, t1),
                                        ("pair2", "pair2 recalibrated", P2, t2)):
                c = dict(ctx, fit=P["fit"])
                outs = apply(sig, P["eval"], t, c)
                row = make_row(pair, f"{sig}/{op}", sig, setting, t, outs, resamples)
                rate = escalation_rate(outs)
                rm = random_mean_summary(P["eval"], rate, seeds, pre)
                row["matched_random_accuracy"] = rm["accuracy"]
                row["acc_minus_random"] = row["accuracy"] - rm["accuracy"]
                if sig != "D":
                    d_eval = [S.difficulty_score(r["question"], P["fit"]) for r in P["eval"]]
                    dm = apply("D", P["eval"], matched_threshold_above(d_eval, rate), c)
                    d_acc = sum(o["correct"] for o in dm) / len(dm)
                    row["matched_difficulty_accuracy"] = d_acc
                    row["acc_minus_difficulty"] = row["accuracy"] - d_acc
                rows.append(row)
    return rows


# ---------------------------------------------------------------- output
def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for r in rows:
        for key in r:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _pct(x):
    return f"{x * 100:5.1f}%"


def _ci(r):
    if "acc_ci_low" in r:
        return f"[{r['acc_ci_low'] * 100:3.0f}-{r['acc_ci_high'] * 100:3.0f}]"
    return f"(sd {r.get('accuracy_sd_over_seeds', 0) * 100:.1f})"


def print_report(out):
    m = out["models"]
    for pair in ("pair1", "pair2"):
        print(f"\n=== {pair.upper()}: small = {m[pair]}, strong = {m['strong']} "
              f"(evaluation set, n = {next(r['n'] for r in out['table'] if r['pair'] == pair)}) ===")
        print(f"{'group':<14}{'strategy':<17}{'esc':>7}{'acc':>8}{'95% CI':>11}"
              f"{'cost s/q':>10}{'overhead':>10}")
        for r in out["table"]:
            if r["pair"] != pair:
                continue
            print(f"{r['group']:<14}{r['strategy']:<17}{_pct(r['escalation_rate']):>7}"
                  f"{_pct(r['accuracy']):>8}{_ci(r):>11}{r['cost_per_q']:>10.2f}"
                  f"{r['overhead_per_q']:>10.2f}")

    print("\n=== TRANSFER: pair-1 threshold unchanged vs recalibrated ===")
    print(f"{'group':<12}{'setting':<27}{'thr':>7}{'esc':>7}{'acc':>8}"
          f"{'vs random':>11}{'vs diff.':>10}")
    for r in out["transfer"]:
        vd = r.get("acc_minus_difficulty", "")
        vd = f"{vd * 100:+9.1f}" if vd != "" else f"{'-':>9}"
        print(f"{r['group']:<12}{r['setting']:<27}{r['threshold']:>7.3f}"
              f"{_pct(r['escalation_rate']):>7}{_pct(r['accuracy']):>8}"
              f"{r['acc_minus_random'] * 100:>+10.1f} {vd}")
    print("(vs random / vs diff. = accuracy points above the random / difficulty "
          "baseline at the same escalation rate)")

    print("\n=== POST-HOC: cheapest point within 2 points of strong-only ===")
    for r in out["posthoc"]:
        if "result" in r:
            print(f"{r['pair']} {r['strategy']:<3} {r['result']}")
        else:
            print(f"{r['pair']} {r['strategy']:<3} param {r['param']:.3f}  esc "
                  f"{_pct(r['escalation_rate'])}  acc {_pct(r['accuracy'])}  "
                  f"cost {r['cost_per_q']:.2f}s = {r['cost_vs_strong_only'] * 100:.0f}% "
                  f"of strong-only")


def run_replay(cfg, pre_generation=None, out_subdir="replay", verbose=True):
    rp = cfg["replay"]
    pre = rp["pre_generation"] if pre_generation is None else pre_generation
    ctx = {"pre_generation": pre, "mode": rp["logprob_mode"]}
    seeds = list(range(cfg["seed"], cfg["seed"] + rp["random_seeds"]))
    resamples = rp["bootstrap_resamples"]
    k = cfg["generation"]["k_samples"]
    models = {"pair1": cfg["models"]["small_a"], "pair2": cfg["models"]["small_b"],
              "strong": cfg["models"]["strong"]}

    P = {n: prepare_pair(cfg, models[n], models["strong"]) for n in ("pair1", "pair2")}

    out = {"table": [], "curves": [], "paired": [], "posthoc": [],
           "per_question": {}, "models": models, "pre_generation": pre}
    for n in ("pair1", "pair2"):
        t, c, pr, ph, pq = analyze_pair(n, P[n], ctx, rp, seeds, resamples, k)
        out["table"] += t
        out["curves"] += c
        out["paired"] += pr
        out["posthoc"] += ph
        out["per_question"][n] = pq
    out["transfer"] = transfer_rows(P["pair1"], P["pair2"], ctx, seeds, resamples)

    out_dir = resolve(cfg["paths"]["results"]) / out_subdir
    write_csv(out_dir / "summary.csv", out["table"])
    write_csv(out_dir / "curves.csv", out["curves"])
    write_csv(out_dir / "paired.csv", out["paired"])
    write_csv(out_dir / "transfer.csv", out["transfer"])
    write_csv(out_dir / "posthoc.csv", out["posthoc"])
    for n, rows in out["per_question"].items():
        write_csv(out_dir / f"per_question_{n}.csv", rows)

    if verbose:
        print_report(out)
        print(f"\nCSV files written to: {out_dir}")
    return out