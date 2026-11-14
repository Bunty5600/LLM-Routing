"""Metrics over a list of outcome dicts (see strategies.py)."""
import math


def _mean(values):
    return sum(values) / len(values) if values else math.nan


def accuracy(outcomes):
    return _mean([1.0 if o["correct"] else 0.0 for o in outcomes])


def mean_field(outcomes, field):
    """Mean of a numeric field. Missing (None) values are skipped, not zeroed."""
    vals = [float(o[field]) for o in outcomes if o.get(field) is not None]
    return _mean(vals)


def escalation_rate(outcomes):
    return _mean([1.0 if o["escalated"] else 0.0 for o in outcomes])


def benefit_breakdown(outcomes):
    """Among escalated questions, what did escalation actually do?

    fixed      = small wrong, strong right   (escalation helped)
    both_wrong = small wrong, strong wrong   (escalation could not help)
    wasted     = small right, strong right   (unnecessary spend)
    harmed     = small right, strong wrong   (escalation made it worse)
    """
    out = {"escalated": 0, "fixed": 0, "both_wrong": 0, "wasted": 0, "harmed": 0}
    for o in outcomes:
        if not o["escalated"]:
            continue
        out["escalated"] += 1
        s, t = o["small_answer_correct"], o["strong_correct"]
        if t and not s:
            out["fixed"] += 1
        elif not t and not s:
            out["both_wrong"] += 1
        elif t and s:
            out["wasted"] += 1
        else:
            out["harmed"] += 1
    return out


def summarize(outcomes):
    s = {
        "n": len(outcomes),
        "accuracy": accuracy(outcomes),
        "cost_per_q": mean_field(outcomes, "cost_seconds"),
        "overhead_per_q": mean_field(outcomes, "overhead_seconds"),
        "latency_per_q": mean_field(outcomes, "latency_seconds"),
        "escalation_rate": escalation_rate(outcomes),
    }
    s.update({f"n_{k}": v for k, v in benefit_breakdown(outcomes).items()})
    return s