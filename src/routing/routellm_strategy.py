"""RouteLLM BERT-router score as an escalation strategy, evaluated by replay.

The router is NOT run here. Its scores were computed once, in the separate
routellm environment, by scripts/score_routellm.py and saved to JSON. This
module only reads those scores, so it needs no torch and no OpenAI client.

The decision uses the router score alone: ground truth is never an input.

Dry run (illustrative numbers): router score 0.8, threshold 0.5 -> escalate.
Small greedy call 1.0 s, strong call 10.0 s, router 0.05 s.
  primary (pre_generation=True):  cost = 0.05 + 0 + 10.0   = 10.05 s
  strict  (pre_generation=False): cost = 0.05 + 1.0 + 10.0 = 11.05 s
Score 0.3 -> no escalation -> cost = 0.05 + 1.0 = 1.05 s in both conventions.
The router's 0.05 s is reported as overhead and is paid on every question.
"""
import json
from pathlib import Path

from src.routing import strategies as S


def load_scores(path, required_ids=None):
    """Return (scores_by_id, meta). Raises with a clear message if unusable."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"RouteLLM scores file not found: {p}. Run scripts\\score_routellm.py "
            "with the routellm environment first.")
    with open(p, "r", encoding="utf-8") as f:
        blob = json.load(f)
    scores = blob["scores"]
    if required_ids is not None:
        missing = [i for i in required_ids if i not in scores]
        if missing:
            raise ValueError(
                f"{len(missing)} question ids have no RouteLLM score "
                f"(first: {missing[0]}). Re-run scripts\\score_routellm.py.")
    return scores, blob.get("meta", {})


def router_escalation(rec, threshold, scores, pre_generation=True):
    """Escalate when the router score is above the threshold.

    The library's own rule is 'score >= threshold'. Our thresholds are
    midpoints between observed scores, so '>' and '>=' give the same decisions;
    '>' keeps this consistent with the difficulty baseline.
    """
    entry = scores[rec["id"]]
    esc = S.escalate_if(entry["score"], threshold, "above")
    g = rec["small"]["greedy"]
    small_seconds = 0.0 if (esc and pre_generation) else g["compute_seconds"]
    out = S._route(rec, esc, small_seconds, g["correct"])
    router_s = float(entry["seconds"])
    out["overhead_seconds"] = router_s          # the router is the signal's own cost
    out["cost_seconds"] += router_s
    out["latency_seconds"] += router_s
    out["router_seconds"] = router_s
    return out