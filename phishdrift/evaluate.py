"""Metrics, threshold selection, and clustered uncertainty.

Design commitments, all of which exist to keep the benchmark-versus-live
comparison honest:

**Accuracy is not reported as a headline.** At the operational base rate a
constant "never phishing" forecast scores well on accuracy while having zero
skill. TSS (Youden's J: recall minus false-alarm rate) is zero for any constant
forecast by construction, so the zero line *is* the no-skill baseline. Brier
score is reported alongside because it is proper -- it cannot be improved by
misreporting confidence.

**Thresholds are selected on a validation split and then frozen.** Selecting a
threshold on the data you are about to score is the single most common way an
evaluation becomes optimistic, and it is a bug this project's sibling (Helios)
had and fixed. ``select_threshold`` takes validation arrays only; the caller is
responsible for never passing it test or live data.

**Uncertainty is clustered by registrable domain.** URLs from one domain are
not independent samples: a phishing kit deployed across 200 paths on one host
is closer to one observation than to 200. Resampling rows i.i.d. would make
confidence intervals several times too narrow. We resample whole domains.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


# --------------------------------------------------------------------------
# Point metrics
# --------------------------------------------------------------------------

def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y).astype(bool)
    pred = np.asarray(pred).astype(bool)
    return (
        int((pred & y).sum()),       # tp
        int((pred & ~y).sum()),      # fp
        int((~pred & y).sum()),      # fn
        int((~pred & ~y).sum()),     # tn
    )


def tss_from_confusion(tp: int, fp: int, fn: int, tn: int) -> float:
    """True Skill Statistic = recall - false alarm rate. Zero for any constant forecast."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    return recall - far


def brier(y: np.ndarray, prob: np.ndarray) -> float:
    return float(np.mean((np.asarray(prob) - np.asarray(y)) ** 2))


def brier_skill(y: np.ndarray, prob: np.ndarray) -> float:
    """Brier skill score against the climatological (constant base rate) forecast."""
    y = np.asarray(y, dtype=float)
    base = float(y.mean())
    ref = float(np.mean((base - y) ** 2))
    return 1.0 - brier(y, prob) / ref if ref > 0 else float("nan")


@dataclass
class Scores:
    n: int
    base_rate: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float
    false_alarm_rate: float
    precision: float
    f1: float
    accuracy: float
    tss: float
    brier: float
    brier_skill: float

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}


def score_at(y: np.ndarray, prob: np.ndarray, threshold: float) -> Scores:
    """Full metric set at a given decision threshold."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    pred = prob >= threshold
    tp, fp, fn, tn = confusion(y, pred)

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return Scores(
        n=len(y),
        base_rate=float(np.mean(y)),
        threshold=float(threshold),
        tp=tp, fp=fp, fn=fn, tn=tn,
        recall=recall,
        false_alarm_rate=far,
        precision=precision,
        f1=f1,
        accuracy=(tp + tn) / len(y) if len(y) else 0.0,
        tss=recall - far,
        brier=brier(y, prob),
        brier_skill=brier_skill(y, prob),
    )


# --------------------------------------------------------------------------
# Threshold selection
# --------------------------------------------------------------------------

def _candidate_thresholds(prob: np.ndarray, max_candidates: int = 512) -> np.ndarray:
    """Deduplicated quantile grid over observed scores.

    Sweeping every unique probability is O(n) thresholds and needlessly slow on
    200k rows; a quantile grid lands candidates where the scores actually are,
    which a uniform 0..1 grid does not when probabilities pile up near 0.
    """
    prob = np.asarray(prob, dtype=float)
    qs = np.linspace(0.0, 1.0, max_candidates)
    grid = np.unique(np.quantile(prob, qs))
    return np.unique(np.concatenate([[0.0], grid, [1.0]]))


def select_threshold(y_val: np.ndarray, prob_val: np.ndarray,
                     objective: str = "tss") -> float:
    """Choose a decision threshold on VALIDATION data only.

    ``objective`` is the whole experiment. The field's default is ``"f1"``,
    which balances precision against recall and is therefore dragged toward the
    base rate of the data it was tuned on. ``"tss"`` maximises Youden's J, which
    is invariant to base rate. The pre-registered hypothesis is that the f1
    point collapses when the operational base rate differs from the benchmark's
    and the tss point survives.

    Never call this with test or live data.
    """
    if objective not in ("f1", "tss"):
        raise ValueError(f"objective must be 'f1' or 'tss', got {objective!r}")

    y_val = np.asarray(y_val)
    best_thr, best_val = 0.5, -np.inf
    for thr in _candidate_thresholds(prob_val):
        s = score_at(y_val, prob_val, thr)
        value = s.f1 if objective == "f1" else s.tss
        if value > best_val:
            best_thr, best_val = float(thr), value
    return best_thr


def peak_tss(y: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Threshold-free ceiling: the best TSS achievable in hindsight, and its threshold.

    Reported only as an upper bound. It is *not* a deployable number -- it picks
    the threshold on the very data being scored -- and every table that shows it
    also shows the frozen-threshold score beside it.
    """
    best_thr, best = 0.5, -np.inf
    for thr in _candidate_thresholds(prob):
        s = score_at(y, prob, thr)
        if s.tss > best:
            best_thr, best = float(thr), s.tss
    return best, best_thr


# --------------------------------------------------------------------------
# Clustered uncertainty
# --------------------------------------------------------------------------

def cluster_bootstrap_ci(y: np.ndarray, prob: np.ndarray, threshold: float,
                         groups: np.ndarray | None = None,
                         n_resamples: int = 1000,
                         statistic: str = "tss",
                         seed: int = 20260920) -> tuple[float, float]:
    """Percentile 95% CI for a metric, resampling whole groups with replacement.

    ``groups`` should be registrable domains. Passing ``None`` falls back to
    i.i.d. row resampling, which is only appropriate when rows are genuinely
    independent (e.g. one aggregated forecast per day).
    """
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=float)
    rng = np.random.default_rng(seed)

    if groups is None:
        index_pool = [np.arange(len(y))]
        keys = np.zeros(1, dtype=int)
    else:
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        sorted_groups = groups[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_groups[1:] != sorted_groups[:-1]])
        index_pool = np.split(order, boundaries[1:])
        keys = np.arange(len(index_pool))

    def stat(idx: np.ndarray) -> float:
        s = score_at(y[idx], prob[idx], threshold)
        return s.tss if statistic == "tss" else getattr(s, statistic)

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = rng.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([index_pool[k] for k in picked])
        draws[i] = stat(idx)

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def paired_difference_ci(y_a, prob_a, thr_a, y_b, prob_b, thr_b,
                         groups_a=None, groups_b=None,
                         n_resamples: int = 1000, seed: int = 20260920
                         ) -> tuple[float, float, float]:
    """CI for the gap between two evaluations, resampled independently.

    Returns ``(point_difference, lo, hi)`` for ``A - B``. The two cells are
    resampled independently because they are different corpora -- there is no
    row-level pairing to preserve, unlike two models scored on one test set.
    A CI excluding zero is the claim that the gap is real.
    """
    rng = np.random.default_rng(seed)

    def pools(groups, n):
        if groups is None:
            return [np.arange(n)], np.zeros(1, dtype=int)
        groups = np.asarray(groups)
        order = np.argsort(groups, kind="stable")
        sg = groups[order]
        bounds = np.flatnonzero(np.r_[True, sg[1:] != sg[:-1]])
        pool = np.split(order, bounds[1:])
        return pool, np.arange(len(pool))

    pool_a, keys_a = pools(groups_a, len(y_a))
    pool_b, keys_b = pools(groups_b, len(y_b))

    y_a, prob_a = np.asarray(y_a), np.asarray(prob_a, dtype=float)
    y_b, prob_b = np.asarray(y_b), np.asarray(prob_b, dtype=float)

    point = score_at(y_a, prob_a, thr_a).tss - score_at(y_b, prob_b, thr_b).tss

    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        ia = np.concatenate([pool_a[k] for k in rng.choice(keys_a, len(keys_a), replace=True)])
        ib = np.concatenate([pool_b[k] for k in rng.choice(keys_b, len(keys_b), replace=True)])
        draws[i] = score_at(y_a[ia], prob_a[ia], thr_a).tss - score_at(y_b[ib], prob_b[ib], thr_b).tss

    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(point), float(lo), float(hi)
