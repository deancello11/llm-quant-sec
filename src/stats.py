"""Statistics with no GPU or heavy imports, so they are easy to unit test.

Prompts are not independent: one behavior appears once as a direct request and again inside every
jailbreak template. So confidence intervals resample whole behaviors (a cluster bootstrap), not
single prompts.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binomtest


def cluster_arrays(df, cond_mask, cluster_col: str, flag_col: str, clusters: list):
    """Per-cluster (successes, counts) for the rows selected by cond_mask, aligned to `clusters`."""
    sub = df[cond_mask]
    g = sub.groupby(cluster_col)[flag_col].agg(["sum", "count"]).reindex(clusters).fillna(0)
    return g["sum"].to_numpy(dtype=float), g["count"].to_numpy(dtype=float)


def boot_indices(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """Cluster indices to resample. Reuse the same array for every condition so comparisons stay paired."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_clusters, size=(n_boot, n_clusters))


def ratio(s: np.ndarray, n: np.ndarray, idx: np.ndarray | None = None):
    """Point estimate (idx=None) or the bootstrap distribution of sum(s)/sum(n)."""
    if idx is None:
        return float(s.sum() / n.sum()) if n.sum() else float("nan")
    num, den = s[idx].sum(axis=1), n[idx].sum(axis=1)
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    return (float(np.nanpercentile(samples, 100 * alpha / 2)), float(np.nanpercentile(samples, 100 * (1 - alpha / 2))))


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact McNemar test on paired binary outcomes (same prompts under two conditions).

    b01 = prompts where only `b` succeeded, b10 = only `a`. Two-sided exact binomial test on the discordant pairs.
    """
    a, b = np.asarray(a).astype(bool), np.asarray(b).astype(bool)
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n = n10 + n01
    p = 1.0 if n == 0 else float(binomtest(n10, n, 0.5, alternative="two-sided").pvalue)
    return {"n10": n10, "n01": n01, "n_discordant": n, "p": p, "n_pairs": int(len(a))}


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the original order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def mcnemar_mde(n_pairs: int, p_discordant: float, z_alpha: float = 1.96, z_power: float = 0.84) -> float:
    """Smallest difference (as a fraction) McNemar can detect with ~80% power at alpha 0.05.

    Assumes independent prompts. With clustered prompts the real figure is larger, so treat it as a lower bound.
    """
    if n_pairs <= 0:
        return float("nan")
    return (z_alpha + z_power) * math.sqrt(p_discordant / n_pairs)


def cohens_kappa(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x).astype(int), np.asarray(y).astype(int)
    po = float((x == y).mean())
    pe = float(x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean()))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 10000, seed: int = 0, alpha: float = 0.05):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    return percentile_ci(values[idx].mean(axis=1), alpha)


def pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points where lower x and lower y are both better."""
    keep = []
    for i, (xi, yi) in enumerate(points):
        dominated = any(
            (xj <= xi and yj <= yi) and (xj < xi or yj < yi) for j, (xj, yj) in enumerate(points) if j != i
        )
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: points[i][0])
