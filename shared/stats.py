"""AUROC and clustered bootstrap CIs.

Both kill tables quote intervals of the same shape: design P6 asks for "AUROC per
image on the 20%, 3 seeds, mean and bootstrap CI over images", and P15 for
"item-level bootstrap CIs (2,000 resamples, clustered by image)".  Clustering by
image is not a detail: a K2 image contributes several conditions and a K1 image
contributes both of its control edits, so resampling rows would understate the
interval.

Added to ``shared/`` rather than ``idea91/`` because IDEA-11's tables quote the
same intervals; recorded in ``notes/DEVIATIONS.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_RESAMPLES = 2000  # P15
BOOTSTRAP_SEED = 0


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.hi - self.lo) / 2.0

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with ties handled by mid-ranks.

    ``labels`` is 0/1; returns nan if either class is empty (an undefined AUROC
    must not silently read as 0.5).
    """
    labels = np.asarray(labels).astype(bool).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if labels.size != scores.size:
        raise ValueError(f"{labels.size} labels vs {scores.size} scores")
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cluster_index(clusters: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    uniq, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    groups = [np.flatnonzero(inverse == k) for k in range(uniq.size)]
    return uniq, groups


def clustered_bootstrap(
    statistic,
    clusters: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    """Percentile CI for ``statistic(row_indices)``, resampling whole clusters.

    Returns ``(lo, hi, draws)``; ``draws`` lets a caller report a distribution.
    """
    _, groups = _cluster_index(clusters)
    if not groups:
        return float("nan"), float("nan"), np.array([])
    rng = np.random.default_rng(seed)
    n = len(groups)
    draws = np.empty(n_resamples, dtype=float)
    for b in range(n_resamples):
        picked = rng.integers(0, n, size=n)
        idx = np.concatenate([groups[p] for p in picked])
        draws[b] = statistic(idx)
    good = draws[np.isfinite(draws)]
    if good.size == 0:
        return float("nan"), float("nan"), draws
    lo, hi = np.percentile(good, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), draws


def auroc_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    clusters: np.ndarray | None = None,
    **kwargs,
) -> Interval:
    """AUROC with a bootstrap CI clustered by ``clusters`` (default: by row)."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=float)
    clusters = np.arange(labels.size) if clusters is None else np.asarray(clusters)
    point = auroc(labels, scores)
    lo, hi, _ = clustered_bootstrap(
        lambda idx: auroc(labels[idx], scores[idx]), clusters, **kwargs
    )
    return Interval(point=point, lo=lo, hi=hi, n=int(labels.size))


def rate_ci(
    successes: np.ndarray,
    clusters: np.ndarray | None = None,
    **kwargs,
) -> Interval:
    """A proportion with a bootstrap CI clustered by image (P15).

    ``successes`` is a boolean array over the items in the denominator.
    """
    successes = np.asarray(successes).astype(bool)
    if successes.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    clusters = np.arange(successes.size) if clusters is None else np.asarray(clusters)
    point = float(successes.mean())
    lo, hi, _ = clustered_bootstrap(lambda idx: float(successes[idx].mean()), clusters, **kwargs)
    return Interval(point=point, lo=lo, hi=hi, n=int(successes.size))


def paired_difference_ci(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray | None = None,
    **kwargs,
) -> Interval:
    """Mean of ``a - b`` with a clustered bootstrap CI.

    Acceptance check 3's calibration line: "the paired difference in same-box
    between human-clean and verifier-clean items is reported with its bootstrap
    CI ... and does not gate".
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError(f"paired arrays differ in length: {a.size} vs {b.size}")
    d = a - b
    clusters = np.arange(d.size) if clusters is None else np.asarray(clusters)
    lo, hi, _ = clustered_bootstrap(lambda idx: float(d[idx].mean()), clusters, **kwargs)
    return Interval(point=float(d.mean()) if d.size else float("nan"), lo=lo, hi=hi, n=int(d.size))


def cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Two raters, categorical labels (acceptance check 3: human versus V1)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.size != b.size:
        raise ValueError("kappa needs paired labels")
    if a.size == 0:
        return float("nan")
    cats = np.unique(np.concatenate([a, b]))
    obs = float((a == b).mean())
    exp = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return float("nan") if exp == 1.0 else float((obs - exp) / (1.0 - exp))


def fleiss_kappa(counts: np.ndarray) -> float:
    """Many raters, categorical labels (P11: human-human agreement).

    ``counts`` is items x categories, each row summing to the number of raters.
    """
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    n_raters = counts.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        raise ValueError("Fleiss kappa needs the same number of raters on every item")
    n = float(n_raters[0])
    if n < 2 or n_items == 0:
        return float("nan")
    p_i = (np.square(counts).sum(axis=1) - n) / (n * (n - 1.0))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n)
    p_e = float(np.square(p_j).sum())
    return float("nan") if p_e == 1.0 else float((p_bar - p_e) / (1.0 - p_e))
