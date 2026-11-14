"""Capability-profile router, two-stage router and three-tier cascade.

Pure functions over cached records (no model calls). A record is one question:
  {"id", "question", "reference_answer",
   "small":  {"greedy": {...}, "samples": [{...}, ...]},
   "medium": {"greedy": {...}, "samples": [{...}, ...]},   (three-tier only)
   "strong": {"greedy": {...}}}

Ground truth enters ONLY in build_profile (offline, calibration split). Every
live decision uses the question text and, for the E1 checks, the sampled answers.

Dry run for the capability profile (illustrative numbers). Calibration difficulty
scores sorted: [-1.0, -0.5, 0.0, 0.4, 0.9, 1.5], 3 buckets -> edges
[-0.25, 0.65] (midpoints between the 2nd/3rd and 4th/5th scores).
  bucket 0 (easiest): small right 90%, strong right 100%
  bucket 1:           small right 70%, strong right 100%
  bucket 2 (hardest): small right 40%, strong right 90%
Required quality 0.85: bucket 0 stays with the small model (0.90 >= 0.85);
buckets 1 and 2 go straight to the strong model.

Dry run for three_tier (illustrative; samples cost 0.5 s small, 1.0 s medium,
strong 10 s): small samples 16,16,16 -> unanimous -> stop. cost 1.5 s, 3 calls.
Small samples 16,17,18 (disagree) -> medium samples 16,16,16 -> stop at the
medium tier: cost 1.5 + 3.0 = 4.5 s, 6 calls. Medium samples also disagree ->
strong answers: cost 4.5 + 10 = 14.5 s, 7 calls.
"""
from src.evaluation.answer_extract import is_correct
from src.routing import strategies as S


# ------------------------------------------------------- capability profile
def bucket_edges(scores, n_buckets):
    """Cut points between buckets: midpoints at the 1/B, 2/B, ... positions.

    Dry run: scores [1,2,3,4,5,6], 3 buckets -> positions 2 and 4 ->
    edges [(2+3)/2, (4+5)/2] = [2.5, 4.5].
    """
    s = sorted(scores)
    n = len(s)
    if n < 2 or n_buckets < 2:
        return []
    edges = []
    for b in range(1, n_buckets):
        i = min(max(round(n * b / n_buckets), 1), n - 1)
        edges.append((s[i - 1] + s[i]) / 2.0)
    return edges


def bucket_of(score, edges):
    """0-based bucket index: the number of edges the score lies above."""
    return sum(1 for e in edges if score > e)


def build_profile(cal_recs, fit, n_buckets):
    """Measure small and strong accuracy per difficulty bucket on CALIBRATION.

    An empty bucket falls back to the overall accuracies.
    """
    if not cal_recs:
        raise ValueError("no calibration records for the capability profile")
    scores = [S.difficulty_score(r["question"], fit) for r in cal_recs]
    edges = bucket_edges(scores, n_buckets)
    n_b = len(edges) + 1
    counts, small_ok, strong_ok = [0] * n_b, [0] * n_b, [0] * n_b
    for r, sc in zip(cal_recs, scores):
        b = bucket_of(sc, edges)
        counts[b] += 1
        small_ok[b] += 1 if r["small"]["greedy"]["correct"] else 0
        strong_ok[b] += 1 if r["strong"]["greedy"]["correct"] else 0
    n = len(cal_recs)
    overall_small, overall_strong = sum(small_ok) / n, sum(strong_ok) / n
    buckets = []
    for b in range(n_b):
        if counts[b]:
            buckets.append({"n": counts[b], "small_acc": small_ok[b] / counts[b],
                            "strong_acc": strong_ok[b] / counts[b]})
        else:
            buckets.append({"n": 0, "small_acc": overall_small,
                            "strong_acc": overall_strong})
    return {"edges": edges, "buckets": buckets,
            "overall_small_acc": overall_small, "overall_strong_acc": overall_strong}


def profile_decision(question, profile, required_quality, fit):
    """True = send this question straight to the strong model.

    Rule: the bucket's measured small-model accuracy is below the required
    quality AND the strong model is measurably better in that bucket.
    """
    score = S.difficulty_score(question, fit)
    bk = profile["buckets"][bucket_of(score, profile["edges"])]
    return bool(bk["small_acc"] < required_quality
                and bk["strong_acc"] > bk["small_acc"])


def capability_router(rec, profile, required_quality, fit, pre_generation=True):
    """Static capability-profile router (decides before any model runs)."""
    esc = profile_decision(rec["question"], profile, required_quality, fit)
    g = rec["small"]["greedy"]
    small_seconds = 0.0 if (esc and pre_generation) else g["compute_seconds"]
    return S._route(rec, esc, small_seconds, g["correct"])


# ------------------------------------------------------------- two-stage
def two_stage(rec, profile, required_quality, fit, t_agree, pre_generation=True):
    """Stage 1: capability profile may send the question straight to the strong
    model. Stage 2: otherwise sample the small model and escalate when agreement
    is below t_agree (exactly E1).

    required_quality = 0 never routes directly, so two_stage reduces to E1.
    """
    samples = rec["small"]["samples"]
    if profile_decision(rec["question"], profile, required_quality, fit):
        t = rec["strong"]["greedy"]
        small_ok = is_correct(S.majority_answer(samples), rec["reference_answer"])
        small_secs = 0.0 if pre_generation else rec["small"]["greedy"]["compute_seconds"]
        out = S._outcome(t["correct"], small_ok, t["correct"], True,
                         small_secs, t["compute_seconds"], 0.0)
        out["model_calls"] = 1 if pre_generation else 2
        out["route"] = "direct"
        return out
    out = S.self_consistency(rec, t_agree)
    out["model_calls"] = len(samples) + (1 if out["escalated"] else 0)
    out["route"] = "stage2"
    return out


# ------------------------------------------------------------- three-tier
def three_tier(rec, t_small, t_medium):
    """Small E1 -> medium E1 -> strong. 'escalated' means the strong model answered."""
    ref = rec["reference_answer"]
    strong = rec["strong"]["greedy"]

    s_samples = rec["small"]["samples"]
    s_ans, s_agr = S.majority(s_samples)
    s_secs = sum(x["compute_seconds"] for x in s_samples)
    s_ok = is_correct(s_ans, ref)
    if not S.escalate_if(s_agr, t_small, "below"):
        out = S._outcome(s_ok, s_ok, strong["correct"], False, s_secs, 0.0, s_secs)
        out["tier_final"] = "small"
        out["model_calls"] = len(s_samples)
        return out

    m_samples = rec["medium"]["samples"]
    m_ans, m_agr = S.majority(m_samples)
    m_secs = sum(x["compute_seconds"] for x in m_samples)
    m_ok = is_correct(m_ans, ref)
    non_strong = s_secs + m_secs
    calls = len(s_samples) + len(m_samples)
    if not S.escalate_if(m_agr, t_medium, "below"):
        out = S._outcome(m_ok, m_ok, strong["correct"], False, non_strong, 0.0, non_strong)
        out["tier_final"] = "medium"
    else:
        out = S._outcome(strong["correct"], m_ok, strong["correct"], True,
                         non_strong, strong["compute_seconds"], non_strong)
        out["tier_final"] = "strong"
        calls += 1
    out["model_calls"] = calls
    return out