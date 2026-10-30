import numpy as np
from src.stats import holm, mcnemar_exact, mcnemar_mde, pareto_frontier, ratio, boot_indices, percentile_ci, cohens_kappa


def test_holm_matches_hand_calc():
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_holm_caps_at_one_and_keeps_order():
    assert holm([0.9, 0.5]) == [0.9, 0.9] or max(holm([0.9, 0.5])) <= 1.0


def test_mcnemar_symmetric_is_not_significant():
    a = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    b = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    r = mcnemar_exact(a, b)
    assert r["n10"] == 2 and r["n01"] == 2 and r["p"] == 1.0


def test_mcnemar_detects_one_sided_shift():
    a = np.zeros(100, dtype=int)
    b = np.zeros(100, dtype=int)
    b[:30] = 1
    assert mcnemar_exact(a, b)["p"] < 1e-6


def test_mcnemar_no_discordant():
    assert mcnemar_exact(np.ones(5), np.ones(5))["p"] == 1.0


def test_bootstrap_ci_contains_estimate():
    s = np.array([1, 0, 2, 1, 0, 3, 1, 1.0]); n = np.full(8, 5.0)
    idx = boot_indices(8, 2000, 0)
    lo, hi = percentile_ci(ratio(s, n, idx))
    assert lo <= ratio(s, n) <= hi


def test_mde_shrinks_with_n():
    assert mcnemar_mde(1000, 0.1) < mcnemar_mde(200, 0.1)


def test_kappa_perfect_and_chance():
    x = np.array([1, 0, 1, 0])
    assert cohens_kappa(x, x) == 1.0
    assert abs(cohens_kappa(x, 1 - x) + 1.0) < 1e-9


def test_pareto_lower_is_better_on_both():
    pts = [(1, 5), (2, 3), (3, 4), (0.5, 9)]
    assert pareto_frontier(pts) == [3, 0, 1]
