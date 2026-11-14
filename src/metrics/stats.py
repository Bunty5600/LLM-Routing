"""Bootstrap confidence intervals, paired comparisons and McNemar's test.

All randomness uses fixed seeds, so results are reproducible.
"""
import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(values, n_resamples=10000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for the mean, resampling questions.

    Dry run: 5 questions, values [1, 1, 0, 1, 0] -> mean 0.6. Each resample
    draws 5 questions with replacement (for example 1, 1, 1, 0, 1 -> 0.8).
    After 10,000 resamples the 2.5th and 97.5th percentiles of those means
    form the interval.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(arr.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n": int(arr.size)}


def paired_bootstrap_diff(a, b, n_resamples=10000, alpha=0.05, seed=0):
    """Bootstrap the mean of (a - b) on the SAME questions.

    Questions are resampled together, so the pairing is kept. The p-value is
    two-sided: twice the smaller share of resamples on either side of zero.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("a and b must be non-empty and the same length")
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min(float(np.mean(means <= 0)), float(np.mean(means >= 0))))
    return {"mean_diff": float(d.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": p, "n": int(d.size)}


def mcnemar(a_correct, b_correct):
    """Exact McNemar test on paired right/wrong results.

    Only questions where the two disagree matter.
    Dry run: A right and B wrong on 5 questions, the reverse on 0 -> n = 5,
    p = 2 x 0.5^5 = 0.0625. With 3 and 3, p = 1.0. With no disagreements,
    p = 1.0.
    """
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("a and b must be the same length")
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
    return {"a_right_b_wrong": only_a, "a_wrong_b_right": only_b, "p_value": p}