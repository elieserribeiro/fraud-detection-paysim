import numpy as np
import pytest

from fraud.costs import best_threshold, financial_at_threshold
from fraud.thresholds import decision_levels, evaluate_levels, optimize_levels

KW = dict(inv_cost=300.0, auth_cost=10.0, inv_catch=0.95, auth_catch=0.7)


def _data(seed=0, n=250):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.08).astype(int)
    score = np.round(rng.random(n) + 0.9 * y, 2)        # com empates de propósito
    amount = rng.lognormal(7, 1, n)
    return y, amount, score


def test_decision_levels_mapping():
    lv = decision_levels(np.array([0.1, 0.5, 0.9, 0.99]), 0.5, 0.9)
    assert lv.tolist() == [0, 1, 2, 2]


def test_optimizer_matches_brute_force_over_all_cut_pairs():
    y, amount, score = _data()
    w = 2.5
    opt = optimize_levels(y, amount, score, legit_weight=w, **KW)
    cuts = np.r_[np.unique(score), np.inf]
    best = max(
        evaluate_levels(y, amount, score, a, b, legit_weight=w, **KW)["beneficio_liquido"]
        for a in cuts for b in cuts if a <= b
    )
    got = evaluate_levels(y, amount, score, opt["t_auth"], opt["t_inv"], legit_weight=w, **KW)
    assert got["beneficio_liquido"] == pytest.approx(best)
    assert opt["beneficio_liquido"] == pytest.approx(best)
    assert opt["t_auth"] <= opt["t_inv"]


def test_levels_cost_adds_up():
    y, amount, score = _data(1)
    r = evaluate_levels(y, amount, score, 0.8, 1.4, legit_weight=1.7, **KW)
    assert sum(n["custo_operacional"] for n in r["niveis"]) == pytest.approx(r["custo_operacional"])
    assert sum(n["transacoes"] for n in r["niveis"]) == len(y)
    assert sum(n["pct_transacoes_reais"] for n in r["niveis"]) == pytest.approx(100.0)
    assert r["beneficio_liquido"] == pytest.approx(r["valor_poupado"] - r["custo_operacional"])


def test_approve_everything_has_zero_benefit():
    y, amount, score = _data(2)
    r = evaluate_levels(y, amount, score, np.inf, np.inf, legit_weight=1.0, **KW)
    assert r["beneficio_liquido"] == pytest.approx(0.0)


def test_without_auth_level_matches_two_level_rule():
    y, amount, score = _data(3)
    w = 3.0
    opt = optimize_levels(y, amount, score, inv_cost=300.0, auth_cost=1e9, inv_catch=0.95, auth_catch=0.0,
                          legit_weight=w)
    thr = best_threshold(y, amount, score, 300.0, 0.95, w)
    ref = financial_at_threshold(y, amount, score, thr, 300.0, 0.95, w)["beneficio_liquido"]
    assert opt["beneficio_liquido"] == pytest.approx(ref)
    assert opt["beneficio_dois_niveis"] == pytest.approx(ref)


def test_three_levels_never_worse_than_two():
    for seed in range(5):
        y, amount, score = _data(seed + 10)
        opt = optimize_levels(y, amount, score, legit_weight=2.0, **KW)
        assert opt["beneficio_liquido"] >= opt["beneficio_dois_niveis"] - 1e-9


def test_weights_equivalent_to_legit_weight_and_replication():
    y, amount, score = _data(4)
    w = 3
    weights = np.where(y == 1, 1.0, float(w))
    a = optimize_levels(y, amount, score, legit_weight=w, **KW)
    b = optimize_levels(y, amount, score, weights=weights, **KW)
    assert a["beneficio_liquido"] == pytest.approx(b["beneficio_liquido"])
    ea = evaluate_levels(y, amount, score, a["t_auth"], a["t_inv"], legit_weight=w, **KW)
    eb = evaluate_levels(y, amount, score, a["t_auth"], a["t_inv"], weights=weights, **KW)
    assert ea["beneficio_liquido"] == pytest.approx(eb["beneficio_liquido"])
    # replicar as legítimas w vezes dá o mesmo resultado que o peso inteiro
    rep = np.r_[np.arange(len(y)), np.repeat(np.flatnonzero(y == 0), w - 1)]
    er = evaluate_levels(y[rep], amount[rep], score[rep], a["t_auth"], a["t_inv"], **KW)
    assert er["beneficio_liquido"] == pytest.approx(ea["beneficio_liquido"])
