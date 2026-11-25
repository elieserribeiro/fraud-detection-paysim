import numpy as np
import pytest

from fraud.costs import best_threshold, cost_curve, financial_at_threshold, legit_weight_for
from fraud.ensemble import (TailRank, best_weights, first_half_mask, fit_tail_ranks,
                            stack_matrix, stack_oof, weight_grid, weights_oof)


def test_cost_curve_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.05).astype(int)
    score = rng.random(n) + 0.8 * y
    amount = rng.lognormal(8, 1, n)
    cost, catch = 300.0, 0.9
    thr = best_threshold(y, amount, score, cost, catch)
    brute = max(
        financial_at_threshold(y, amount, score, t, cost, catch)["beneficio_liquido"]
        for t in np.unique(score)
    )
    got = financial_at_threshold(y, amount, score, thr, cost, catch)["beneficio_liquido"]
    assert got == pytest.approx(brute)
    assert got > 0


def test_best_threshold_flags_nothing_when_cost_too_high():
    y = np.array([1, 0, 0, 0])
    thr = best_threshold(y, np.array([10.0, 1, 1, 1]), np.array([0.9, 0.1, 0.2, 0.3]), 1e6, 1.0)
    assert thr == float("inf")


def test_financial_numbers():
    y = np.array([1, 1, 0, 0])
    amount = np.array([100.0, 300.0, 50.0, 10.0])
    score = np.array([0.9, 0.2, 0.8, 0.1])
    r = financial_at_threshold(y, amount, score, 0.5, investigation_cost=10, catch_rate=0.5, legit_weight=2.0)
    assert (r["fraudes_pegas"], r["falsos_positivos"], r["alertas"]) == (1, 1, 2)
    assert r["valor_poupado"] == pytest.approx(50.0)
    assert r["custo_investigacao"] == pytest.approx(10 * (1 + 2.0))
    assert r["beneficio_liquido"] == pytest.approx(50 - 30)


def test_legit_weight_scales_to_real_rate():
    y = np.array([1] * 34 + [0] * 9966)  # 0,34%
    w = legit_weight_for(y)
    assert 2 < w < 3.5


def test_tail_rank_is_monotonic_and_keeps_tail_resolution():
    rng = np.random.default_rng(1)
    val = rng.random(100000)
    tr = TailRank().fit(val)
    t = tr.transform(np.array([0.1, 0.5, 0.99, 0.9999, 5.0]))
    assert np.all(np.diff(t) > 0) or np.all(np.diff(t) >= 0)
    assert t[3] > t[2] + 1  # a cauda superior é esticada


def test_stack_and_weights_use_signal_and_no_leak_between_folds():
    rng = np.random.default_rng(2)
    n = 6000
    y = (rng.random(n) < 0.03).astype(int)
    good = rng.random(n) + 1.5 * y
    noise = rng.random(n)
    scores = {"good": good, "noise": noise}
    ranks = fit_tail_ranks(scores)
    Q = stack_matrix(scores, ranks, ["good", "noise"])
    fold = first_half_mask(np.arange(n))
    s = stack_oof(Q, y, fold)
    w = weights_oof(Q, y, fold)
    from sklearn.metrics import average_precision_score as ap
    assert ap(y, s) > ap(y, noise) + 0.1
    assert ap(y, w) > ap(y, noise) + 0.1
    assert best_weights(Q, y)[0] >= 0.75  # peso concentrado no modelo bom
    assert sum(1 for _ in weight_grid(3, 4)) == 15
