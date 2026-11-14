import pytest

from src.experiments import plots as pl

MODELS = {"pair1": "small-a", "pair2": "small-b", "strong": "big"}


def pt(pair, strat, acc, cost, esc, overhead=0.0):
    return {"pair": pair, "strategy": strat, "param": 0.5, "accuracy": acc,
            "cost_per_q": cost, "escalation_rate": esc, "overhead_per_q": overhead,
            "latency_per_q": cost}


def fake_curves():
    rows = []
    for pair in ("pair1", "pair2"):
        rows += [pt(pair, "A", 0.70, 2.5, 0.0), pt(pair, "B", 0.95, 25.0, 1.0),
                 pt(pair, "Ref", 0.96, 10.0, 0.25)]
        for s in ("C", "D", "E1", "E2"):
            rows += [pt(pair, s, 0.70, 2.5, 0.0), pt(pair, s, 0.82, 14.0, 0.5),
                     pt(pair, s, 0.95, 25.0, 1.0)]
    return rows


def srow(pair, group, strat, setting, acc=0.85, cost=12.0, esc=0.4, ovh=0.0):
    return {"pair": pair, "group": group, "strategy": strat, "setting": setting,
            "accuracy": acc, "acc_ci_low": acc - 0.07, "acc_ci_high": acc + 0.07,
            "cost_per_q": cost, "escalation_rate": esc, "overhead_per_q": ovh,
            "latency_per_q": cost}


def fake_summary():
    rows = []
    for pair in ("pair1", "pair2"):
        rows += [srow(pair, "baselines", "A small-only", "-", 0.7, 2.5, 0.0),
                 srow(pair, "baselines", "B strong-only", "-", 0.95, 25.0, 1.0)]
        for label, group, strat in pl.BARS[2:]:
            op = group.split("/")[1]
            rows.append(srow(pair, group, strat, f"calibrated:{op}", ovh=1.0))
    return rows


def fake_transfer():
    rows = []
    for group in pl.TRANSFER_GROUPS:
        for setting, _ in pl.TRANSFER_SETTINGS:
            r = srow("pair1", group, group.split("/")[0], setting)
            r["matched_random_accuracy"] = 0.78
            rows.append(r)
    return rows


def test_interp_accuracy():
    pts = [{"cost_per_q": 2.5, "accuracy": 0.71}, {"cost_per_q": 14.5, "accuracy": 0.89},
           {"cost_per_q": 25.5, "accuracy": 0.95}]
    assert pl.interp_accuracy(pts, 8.5) == pytest.approx(0.80)
    assert pl.interp_accuracy(pts, 2.5) == pytest.approx(0.71)
    assert pl.interp_accuracy(pts, 1.0) is None       # below the curve: no extrapolation
    assert pl.interp_accuracy(pts, 30.0) is None
    assert pl.interp_accuracy([], 5.0) is None


def test_cost_matched_rows():
    rows = pl.cost_matched_rows(fake_curves(), fake_summary())
    assert len(rows) == 2 * 6                          # 2 pairs x (E1,E2,D) x (budget,recall)
    r = rows[0]
    assert r["cost_per_q"] == pytest.approx(12.0)
    # curve between (2.5, .70) and (14.0, .82): at 12.0 -> 0.70 + 0.12 * 9.5/11.5
    assert r["acc_C_at_same_cost"] == pytest.approx(0.70 + 0.12 * 9.5 / 11.5)


def test_figures_are_written(tmp_path):
    pl.fig_curves(fake_curves(), fake_summary(), MODELS, tmp_path / "a.png", "cost_per_q")
    pl.fig_curves(fake_curves(), fake_summary(), MODELS, tmp_path / "b.png", "escalation_rate")
    pl.fig_latency(fake_summary(), MODELS, tmp_path / "c.png")
    pl.fig_transfer(fake_transfer(), tmp_path / "d.png")
    for name in ("a", "b", "c", "d"):
        assert (tmp_path / f"{name}.png").stat().st_size > 0
        assert (tmp_path / f"{name}.pdf").stat().st_size > 0