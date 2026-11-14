"""Escalation strategies as pure functions over cached records. No model calls.

A record is one question:
  {"id", "question", "reference_answer",
   "small":  {"greedy": {...}, "samples": [{...}, ...]},
   "strong": {"greedy": {...}}}
where the inner dicts are rows written by collect_outputs.py.

Every strategy returns an 'outcome' dict:
  escalated, correct, small_answer_correct, strong_correct,
  small_seconds, strong_seconds, overhead_seconds, cost_seconds, latency_seconds

Cost conventions:
  cost_seconds     = all small-model seconds + strong-model seconds the strategy
                     would have spent (recorded compute seconds, load excluded).
  overhead_seconds = seconds spent only to compute the escalation signal
                     (E1: the k sampled calls; all other strategies: 0).
  latency_seconds  = same sum, because calls run one after another on one machine.
"""
import hashlib
import math
import re
from collections import Counter
from statistics import mean, pstdev

from src.evaluation.answer_extract import is_correct


# ---------------------------------------------------------------- helpers
def _small(rec):
    return rec["small"]["greedy"]


def _strong(rec):
    return rec["strong"]["greedy"]


def _key(x):
    return None if x is None else round(float(x), 6)


def _outcome(final_correct, small_answer_correct, strong_correct, escalated,
             small_seconds, strong_seconds, overhead_seconds):
    cost = small_seconds + strong_seconds
    return {
        "escalated": bool(escalated),
        "correct": bool(final_correct),
        "small_answer_correct": bool(small_answer_correct),
        "strong_correct": bool(strong_correct),
        "small_seconds": small_seconds,
        "strong_seconds": strong_seconds,
        "overhead_seconds": overhead_seconds,
        "cost_seconds": cost,
        "latency_seconds": cost,
    }


def _route(rec, escalated, small_seconds, small_answer_correct, overhead_seconds=0.0):
    """Shared rule: if escalated, the strong answer is final and its time is paid."""
    t = _strong(rec)
    final = t["correct"] if escalated else small_answer_correct
    strong_seconds = t["compute_seconds"] if escalated else 0.0
    return _outcome(final, small_answer_correct, t["correct"], escalated,
                    small_seconds, strong_seconds, overhead_seconds)


# ---------------------------------------------------------------- signals
def majority(samples):
    """Return (majority_answer, agreement) for a list of sampled answers.

    Dry run: answers 42, 38, 42 -> counts {42: 2, 38: 1} -> (42.0, 2/3).
    Answers 16, 16, 16 -> (16.0, 1.0). Answers 1, 2, 3 (a 3-way tie) -> the
    earliest answer wins -> (1.0, 1/3). A missing answer (None) counts as its
    own value, so [None, None, 5] -> (None, 2/3).
    """
    keys = [_key(s["extracted"]) for s in samples]
    value, count = Counter(keys).most_common(1)[0]
    return value, count / len(keys)


def agreement_score(samples):
    return majority(samples)[1]


def majority_answer(samples):
    return majority(samples)[0]


def logprob_signal(logprobs, mode="mean"):
    """Confidence from per-token log-probabilities of the small model's answer.

    'mean' = average token probability, 'min' = lowest token probability.
    Dry run: logprobs [ln 0.9, ln 0.5] -> probabilities [0.9, 0.5]
    -> mean 0.7, min 0.5. No logprobs -> None.
    """
    if not logprobs:
        return None
    probs = [math.exp(lp) for lp in logprobs]
    if mode == "mean":
        return sum(probs) / len(probs)
    if mode == "min":
        return min(probs)
    raise ValueError(f"unknown mode: {mode}")


def difficulty_features(question):
    """Cheap input-only features: (word count, number count, sentence count)."""
    words = len(re.findall(r"\w+", question))
    numbers = len(re.findall(r"\d+(?:[.,]\d+)*", question))
    sentences = max(1, len([s for s in re.split(r"[.!?]+", question) if s.strip()]))
    return (words, numbers, sentences)


def fit_difficulty(questions):
    """Learn feature means/stds. Call this on the CALIBRATION questions only."""
    cols = list(zip(*[difficulty_features(q) for q in questions]))
    return {"means": [mean(c) for c in cols],
            "stds": [pstdev(c) or 1.0 for c in cols]}


def difficulty_score(question, fit):
    """Average z-score of the three features. Higher = looks harder."""
    f = difficulty_features(question)
    z = [(x - m) / s for x, m, s in zip(f, fit["means"], fit["stds"])]
    return sum(z) / len(z)


def random_uniform(qid, seed):
    """A fixed pseudo-random number in [0, 1) for each (seed, question id).

    Because it depends only on the id, the set escalated at p = 0.3 is always
    a subset of the set escalated at p = 0.6, so a sweep over p is consistent.
    """
    h = hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def escalate_if(score, threshold, direction):
    """direction 'below': escalate when score < threshold; 'above': score > threshold.

    A missing score (None) escalates: if the signal is unavailable we cannot
    vouch for the small answer.
    """
    if score is None:
        return True
    if direction == "below":
        return score < threshold
    if direction == "above":
        return score > threshold
    raise ValueError(f"unknown direction: {direction}")


# ---------------------------------------------------------------- strategies
def small_only(rec):
    """A: the small model's single greedy answer."""
    g = _small(rec)
    return _outcome(g["correct"], g["correct"], _strong(rec)["correct"],
                    False, g["compute_seconds"], 0.0, 0.0)


def strong_only(rec):
    """B: the strong model's single greedy answer (counted as 'escalated')."""
    t = _strong(rec)
    return _outcome(t["correct"], _small(rec)["correct"], t["correct"],
                    True, 0.0, t["compute_seconds"], 0.0)


def random_escalation(rec, p, seed, pre_generation=True):
    """C: escalate with probability p, independent of the question.

    pre_generation=True : the decision is made before any generation, so an
        escalated question skips the small call (small_seconds = 0).
    pre_generation=False: every question first pays the small call.
    """
    g = _small(rec)
    esc = random_uniform(rec["id"], seed) < p
    small_seconds = 0.0 if (esc and pre_generation) else g["compute_seconds"]
    return _route(rec, esc, small_seconds, g["correct"])


def difficulty_escalation(rec, threshold, fit, pre_generation=True):
    """D: escalate when the input-only difficulty score is above the threshold."""
    g = _small(rec)
    score = difficulty_score(rec["question"], fit)
    esc = escalate_if(score, threshold, "above")
    small_seconds = 0.0 if (esc and pre_generation) else g["compute_seconds"]
    return _route(rec, esc, small_seconds, g["correct"])


def self_consistency(rec, threshold):
    """E1: sample the small model k times; escalate if agreement < threshold.

    Not escalated -> the majority answer is returned.
    Dry run (illustrative): samples 42, 38, 42, each 0.5 s, strong 10 s.
      agreement 2/3, threshold 0.8 -> escalate.
      cost = 3 x 0.5 + 10 = 11.5 s, overhead = 1.5 s.
    """
    samples = rec["small"]["samples"]
    answer, agr = majority(samples)
    esc = escalate_if(agr, threshold, "below")
    small_ok = is_correct(answer, rec["reference_answer"])
    secs = sum(s["compute_seconds"] for s in samples)
    return _route(rec, esc, secs, small_ok, overhead_seconds=secs)


def logprob_escalation(rec, threshold, mode="mean"):
    """E2: escalate when the small model's token confidence is below threshold.

    The logprobs come with the greedy call, so the signal itself costs nothing.
    """
    g = _small(rec)
    score = logprob_signal(g.get("logprobs"), mode)
    esc = escalate_if(score, threshold, "below")
    return _route(rec, esc, g["compute_seconds"], g["correct"])


def oracle(rec):
    """Ref: escalate exactly when small is wrong and strong is right.

    Uses the ground truth, so it is a ceiling for comparison, never deployable.
    """
    g, t = _small(rec), _strong(rec)
    esc = (not g["correct"]) and t["correct"]
    return _route(rec, esc, g["compute_seconds"], g["correct"])