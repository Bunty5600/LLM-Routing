"""Figures and the cost-matched comparison, built from results/replay/*.csv.

matplotlib is imported only when a figure is drawn (see _plt), so the replay
code can import the helpers in this module without loading matplotlib.

Dry run for the cost-matched table (illustrative numbers): a curve has points
(cost s, accuracy) = (2.5, 0.71), (14.5, 0.89), (25.5, 0.95). A signal sits at
cost 8.5 s. 8.5 is halfway between 2.5 and 14.5, so the interpolated accuracy
of that curve at 8.5 s is halfway between 0.71 and 0.89 = 0.80. A cost outside
the curve's range gives None (shown as n/a), never an extrapolation.
"""
import csv

import numpy as np

from src.config import resolve
from src.experiments.replay import write_csv

COLORS = {"C": "#888888", "D": "#d95f02", "E1": "#1b9e77", "E2": "#7570b3",
          "A": "#000000", "B": "#e7298a", "Ref": "#66a61e"}
NAMES = {"C": "Random (matched rate)", "D": "Difficulty baseline",
         "E1": "Self-consistency (E1)", "E2": "Logprob (E2)", "A": "Small-only",
         "B": "Strong-only", "Ref": "Oracle (ceiling)"}
BARS = [("Small-only", "baselines", "A small-only"),
        ("Strong-only", "baselines", "B strong-only"),
        ("E1 budget", "E1/budget", "E1"), ("E1 recall", "E1/recall", "E1"),
        ("E2 budget", "E2/budget", "E2"), ("E2 recall", "E2/recall", "E2"),
        ("D budget", "D/budget", "D"), ("D recall", "D/recall", "D")]
TRANSFER_GROUPS = ["E1/budget", "E1/recall", "E2/budget", "E2/recall",
                   "D/budget", "D/recall"]
TRANSFER_SETTINGS = [("pair1 own", "#1b9e77"),
                     ("pair2 unchanged threshold", "#d95f02"),
                     ("pair2 recalibrated", "#7570b3")]


# ------------------------------------------------------------------ helpers
def _plt():
    """Import matplotlib only when needed (it may be blocked on some machines)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if v == "":
                r[k] = None
                continue
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def curve_points(curves, pair, strategy, sort_field="cost_per_q"):
    pts = [r for r in curves if r["pair"] == pair and r["strategy"] == strategy]
    return sorted(pts, key=lambda r: r[sort_field])


def interp_accuracy(points, cost):
    """Accuracy of a curve (points sorted by cost) at `cost`, or None if outside."""
    xs = [p["cost_per_q"] for p in points]
    ys = [p["accuracy"] for p in points]
    if not xs or cost < min(xs) - 1e-9 or cost > max(xs) + 1e-9:
        return None
    return float(np.interp(cost, xs, ys))


def cost_matched_rows(curves, summary):
    rows = []
    for r in summary:
        if (r["strategy"] not in ("E1", "E2", "D")
                or not str(r.get("setting") or "").startswith("calibrated")):
            continue
        row = {"pair": r["pair"], "signal": r["strategy"],
               "operating_point": str(r["setting"]).split(":")[1],
               "cost_per_q": r["cost_per_q"], "accuracy": r["accuracy"]}
        for s in ("C", "D", "E1", "E2"):
            row[f"acc_{s}_at_same_cost"] = interp_accuracy(
                curve_points(curves, r["pair"], s), r["cost_per_q"])
        rows.append(row)
    return rows


def print_cost_matched(rows):
    print("\n=== COST-MATCHED: accuracy of each curve at the SAME cost as the signal ===")
    print("(linear interpolation along each curve; E1's curve has only 4 points)")
    print(f"{'pair':<7}{'signal':<8}{'point':<9}{'cost s':>8}{'its acc':>9}"
          f"{'C':>8}{'D':>8}{'E1':>8}{'E2':>8}")
    f = lambda x: f"{x * 100:7.1f}%" if x is not None else f"{'n/a':>8}"
    for r in rows:
        print(f"{r['pair']:<7}{r['signal']:<8}{r['operating_point']:<9}"
              f"{r['cost_per_q']:>8.2f}{r['accuracy'] * 100:>8.1f}%"
              f"{f(r['acc_C_at_same_cost'])}{f(r['acc_D_at_same_cost'])}"
              f"{f(r['acc_E1_at_same_cost'])}{f(r['acc_E2_at_same_cost'])}")


def _style():
    _plt().rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                            "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    _plt().close(fig)


def _title(pair, models):
    small = models["pair1"] if pair == "pair1" else models["pair2"]
    n = "1" if pair == "pair1" else "2"
    return f"Pair {n}: {small} + {models['strong']}"


# ------------------------------------------------------------------ figures
def fig_curves(curves, summary, models, path, x_field):
    """Accuracy vs cost (x_field cost_per_q) or vs escalation rate."""
    plt = _plt()
    _style()
    scale = 100.0 if x_field == "escalation_rate" else 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, pair in zip(axes, ("pair1", "pair2")):
        for s in ("C", "D", "E1", "E2"):
            pts = curve_points(curves, pair, s, x_field)
            ax.plot([p[x_field] * scale for p in pts],
                    [p["accuracy"] * 100 for p in pts], "-o", ms=3, lw=1.2,
                    color=COLORS[s], label=NAMES[s])
        for s, mk in (("A", "s"), ("B", "s"), ("Ref", "D")):
            p = next(r for r in curves if r["pair"] == pair and r["strategy"] == s)
            ax.scatter([p[x_field] * scale], [p["accuracy"] * 100], marker=mk,
                       s=45, color=COLORS[s], zorder=5, label=NAMES[s])
        for r in summary:                       # calibrated points with 95% CI
            if (r["pair"] == pair and r["strategy"] in ("E1", "E2", "D")
                    and str(r.get("setting") or "").startswith("calibrated")):
                lo = (r["accuracy"] - r["acc_ci_low"]) * 100
                hi = (r["acc_ci_high"] - r["accuracy"]) * 100
                ax.errorbar(r[x_field] * scale, r["accuracy"] * 100,
                            yerr=[[lo], [hi]], fmt="none", capsize=3,
                            elinewidth=1.2, ecolor=COLORS[r["strategy"]], zorder=4)
        ax.set_title(_title(pair, models))
        ax.set_xlabel("Mean compute per question (s)" if x_field == "cost_per_q"
                      else "Escalation rate (%)")
    axes[0].set_ylabel("Accuracy on evaluation set (%)")
    axes[1].legend(loc="lower right", fontsize=7.5)
    _save(fig, path)


def fig_latency(summary, models, path):
    plt = _plt()
    _style()
    lookup = {(r["pair"], r["group"], r["strategy"]): r for r in summary}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, pair in zip(axes, ("pair1", "pair2")):
        for i, (label, group, strat) in enumerate(BARS):
            r = lookup.get((pair, group, strat))
            if r is None:
                continue
            base = r["latency_per_q"] - r["overhead_per_q"]
            ax.bar(i, base, color="#4c78a8")
            ax.bar(i, r["overhead_per_q"], bottom=base, color="#f58518",
                   hatch="//", edgecolor="white")
        ax.set_xticks(range(len(BARS)))
        ax.set_xticklabels([b[0] for b in BARS], rotation=35, ha="right")
        ax.set_title(_title(pair, models))
    axes[0].set_ylabel("Mean latency per question (s)")
    axes[0].bar(0, 0, color="#4c78a8", label="model calls")
    axes[0].bar(0, 0, color="#f58518", hatch="//", label="signal overhead (E1 samples)")
    axes[0].legend(fontsize=8)
    _save(fig, path)


def fig_transfer(transfer, path):
    plt = _plt()
    _style()
    fig, ax = plt.subplots(figsize=(11, 4.4))
    width = 0.26
    for j, (setting, color) in enumerate(TRANSFER_SETTINGS):
        for i, group in enumerate(TRANSFER_GROUPS):
            r = next((x for x in transfer
                      if x["group"] == group and x["setting"] == setting), None)
            if r is None:
                continue
            xpos = i + (j - 1) * width
            ax.bar(xpos, r["accuracy"] * 100, width, color=color,
                   label=setting if i == 0 else None)
            ax.errorbar(xpos, r["accuracy"] * 100,
                        yerr=[[(r["accuracy"] - r["acc_ci_low"]) * 100],
                              [(r["acc_ci_high"] - r["accuracy"]) * 100]],
                        fmt="none", ecolor="black", capsize=2, elinewidth=1)
            ax.plot([xpos], [r["matched_random_accuracy"] * 100], marker="_",
                    color="black", ms=14, mew=2,
                    label="random at same rate" if (i == 0 and j == 0) else None)
            ax.text(xpos, r["acc_ci_high"] * 100 + 1, f"{r['escalation_rate'] * 100:.0f}%",
                    ha="center", fontsize=7)
    ax.set_xticks(range(len(TRANSFER_GROUPS)))
    ax.set_xticklabels(TRANSFER_GROUPS)
    ax.set_ylabel("Accuracy on evaluation set (%)")
    ax.set_title("Threshold transfer (numbers above bars = escalation rate)")
    ax.set_ylim(40, 105)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def make_all(cfg, subdir="replay"):
    in_dir = resolve(cfg["paths"]["results"]) / subdir
    out_dir = resolve(cfg["paths"]["figures"]) / ("" if subdir == "replay" else subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = read_csv(in_dir / "curves.csv")
    summary = read_csv(in_dir / "summary.csv")
    transfer = read_csv(in_dir / "transfer.csv")
    models = {"pair1": cfg["models"]["small_a"], "pair2": cfg["models"]["small_b"],
              "strong": cfg["models"]["strong"]}

    fig_curves(curves, summary, models, out_dir / "fig1_quality_vs_cost.png", "cost_per_q")
    fig_curves(curves, summary, models, out_dir / "fig2_accuracy_vs_escalation.png",
               "escalation_rate")
    fig_latency(summary, models, out_dir / "fig3_latency.png")
    fig_transfer(transfer, out_dir / "fig4_transfer.png")

    rows = cost_matched_rows(curves, summary)
    write_csv(in_dir / "cost_matched.csv", rows)
    print_cost_matched(rows)
    print(f"\nFigures (png + pdf) written to: {out_dir}")
    return rows