"""Statistics with no GPU or heavy imports, so they are easy to unit test.

Prompts are not independent: one behavior appears once as a direct request and again inside every
jailbreak template. So confidence intervals resample whole behaviors (a cluster bootstrap), not
single prompts.

Where it fits: analyze.py and robustness.py (the last step of generate -> judge -> analyze) call these small
helpers to turn piles of yes/no judgments into rates, error bars and p-values. Nothing here touches a GPU or the
disk, so the unit tests can run it in a second.

A quick cheat sheet of the ideas used below:
  - Bootstrap: to see how much a number would wobble, re-draw the data at random (with replacement) thousands of
    times and recompute it each time. The spread of those answers is the error bar.
  - Cluster bootstrap: the same, but you re-draw whole groups (here: whole behaviors) instead of single prompts.
    Like regrading exam papers by drawing whole students, not loose pages, because a student's pages go together.
  - McNemar test: for before/after yes/no data on the same items, only the items that flipped carry information.
  - Holm correction: when you run many tests, some will look "significant" by luck. Holm raises the bar to account
    for that, without being as harsh as plain Bonferroni.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def cluster_arrays(df, cond_mask, cluster_col: str, flag_col: str, clusters: list):
    """Per-cluster (successes, counts) for the rows selected by cond_mask, aligned to `clusters`."""
    sub = df[cond_mask]
    # reindex(clusters) puts every condition on the same list of behaviors; a behavior with no rows becomes 0/0.
    g =sub.groupby(cluster_col)[flag_col].agg(["sum", "count"]).reindex(clusters).fillna(0)
    return g["sum"].to_numpy(dtype=float), g["count"].to_numpy(dtype=float)


def boot_indices(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """Cluster indices to resample. Reuse the same array for every condition so comparisons stay paired."""
    # A fixed seed makes the "random" draws repeatable, so re-running gives the same error bars.
    # Each row is one pretend re-run of the study: n_clusters behaviors picked at random, repeats allowed.
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def ratio(s: np.ndarray, n: np.ndarray, idx: np.ndarray | None = None):
    """Point estimate (idx=None) or the bootstrap distribution of sum(s)/sum(n)."""
    # Ratio of totals (not an average of per-behavior rates), so a behavior with more prompts counts for more.
    if idx is None:
        return float(s.sum() / n.sum()) if n.sum() else float("nan")
    # s[idx] builds every pretend re-run at once. A re-run with no prompts at all gives NaN, not a crash.
    num, den = s[idx].sum(axis=1), n[idx].sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Confidence interval from a bootstrap: the middle 95% of the re-drawn answers (2.5th to 97.5th percentile)."""
    # nanpercentile quietly skips the NaN re-runs produced by ratio() above.
    return (float(np.nanpercentile(samples, 100 * alpha / 2)), float(np.nanpercentile(samples, 100 * (1 - alpha / 2))))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes (same prompts under two conditions).

    n01 = prompts where only `b` succeeded, n10 = only `a` (the code's names for the two kinds of flip).
    Two-sided exact binomial test on the discordant pairs.
    """
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    # Prompts that gave the same answer both times tell us nothing about a change, so only the "flippers" count.
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    # If nothing changed, flips toward each side should be a fair coin toss. The binomial test asks how
    # surprising the observed split is under that coin. No flips at all means no evidence, so p = 1.
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
    return {"n10": n10, "n01": n01, "n_discordant": n, "p": p, "n_pairs": int(len(a))}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    m = len(pvals)
    # Walk from the smallest p-value up. The smallest gets the strictest penalty (times m), the next a bit less,
    # and so on.
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        # "running" keeps adjusted values from ever going down as raw p-values go up (that would be odd),
        # and min(1.0, ...) keeps them valid probabilities.
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_mde(n_pairs: int, p_discordant: float, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Smallest difference (as a fraction) McNemar can detect with ~80% power at alpha 0.05.

    Assumes independent prompts. With clustered prompts the real figure is larger, so treat it as a lower bound.
    """
    if n_pairs <= 0:
        return float("nan")
    # 1.96 and 0.84 are the standard normal-curve cutoffs for 5% false alarms and 80% power. Think of it as
    # "how big must a change be before we would reliably notice it, given how few prompts flipped?"
    return (z_alpha + z_power) * math.sqrt(p_discordant / n_pairs)


def cohens_kappa(x: np.ndarray, y: np.ndarray) -> float:
    """Agreement between two yes/no labelers (say, judge vs human), corrected for agreement by pure luck.

    1 is perfect agreement, 0 is no better than chance. Used when checking the judge against hand labels.
    """
    x, y = np.asarray(x).astype(int), np.asarray(y).astype(int)
    po = float((x == y).mean())
    # pe = how often they would agree by chance, given how often each one says "yes".
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    """Plain (non-clustered) bootstrap CI for an average. Fine for items that really are independent."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return percentile_ci(values[idx].mean(axis=1), alpha)


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points where lower x and lower y are both better."""
    keep = []
    for i, (xi, yi) in enumerate(points):
        # A point is "dominated" if some other point is at least as good on both axes and strictly better on one.
        # Like comparing phones: if another one is cheaper AND better, nobody has a reason to pick this one.
        dominated = any(
            (xj <= xi and yj <= yi) and (xj < xi or yj < yi) for j, (xj, yj) in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])
