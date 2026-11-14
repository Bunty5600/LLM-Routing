"""Extra metrics: model calls per question and routing accuracy.

Routing accuracy = how often the escalate / do-not-escalate decision equals the
oracle's decision. The oracle escalates exactly when the small answer is wrong
AND the strong answer is right (escalating when the strong model is also wrong
buys nothing). Uses labels offline only.

Dry run: four questions, (small right?, strong right?, escalated?):
  (no, yes, yes) -> ideal yes, did yes -> hit
  (yes, yes, yes) -> ideal no,  did yes -> miss (wasted)
  (no, no, no)   -> ideal no,  did no  -> hit
  (yes, yes, no) -> ideal no,  did no  -> hit
routing accuracy = 3/4 = 0.75.
"""
import math


def mean_model_calls(outcomes):
    vals = [o["model_calls"] for o in outcomes if o.get("model_calls") is not None]
    return sum(vals) / len(vals) if vals else math.nan


def routing_accuracy(outcomes):
    if not outcomes:
        return math.nan
    hits = 0
    for o in outcomes:
        ideal = (not o["small_answer_correct"]) and bool(o["strong_correct"])
        hits += 1 if bool(o["escalated"]) == ideal else 0
    return hits / len(outcomes)


def extra_summary(outcomes):
    return {"mean_model_calls": mean_model_calls(outcomes),
            "routing_accuracy": routing_accuracy(outcomes)}


def calls_for(strategy, outcome, k, pre_generation=True):
    """Model calls one question costs, for the strategies in strategies.py.

    k = number of sampled answers (E1). 'esc' is 1 when the strong model ran.
    Pre-generation routers (D, P, C) make ONE call in the primary convention;
    in the strict convention an escalated question also pays the small call.
    """
    esc = 1 if outcome["escalated"] else 0
    if strategy in ("A", "B"):
        return 1
    if strategy == "Ref":
        return 1 + esc
    if strategy == "E1":
        return k + esc
    if strategy == "E2":
        return 1 + esc
    if strategy in ("D", "P", "C", "R"):
        return 1 if pre_generation else 1 + esc
    raise ValueError(f"unknown strategy: {strategy}")


def annotate_calls(outcomes, strategy, k, pre_generation=True):
    for o in outcomes:
        o["model_calls"] = calls_for(strategy, o, k, pre_generation)
    return outcomes