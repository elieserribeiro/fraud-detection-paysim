import numpy as np
import pandas as pd
import pytest

from fraud.metrics import evaluate, metrics_at_threshold, recall_at_precision
from fraud.split import temporal_split_by_rows


def test_metrics_perfect_and_random():
    y = np.array([0] * 90 + [1] * 10)
    perfect = y.astype(float)
    r = evaluate(y, perfect)
    assert r["pr_auc"] == pytest.approx(1.0)
    assert r["bestF1_recall"] == pytest.approx(1.0)
    assert r["bestF1_fpr"] == 0.0

    rng = np.random.default_rng(0)
    r2 = evaluate(y, rng.random(100))
    assert r2["pr_auc"] < 0.5


def test_metrics_at_threshold_counts_and_value_recall():
    y = np.array([1, 1, 0, 0, 1])
    score = np.array([0.9, 0.4, 0.8, 0.1, 0.7])
    amount = np.array([100.0, 300.0, 50.0, 10.0, 100.0])
    m = metrics_at_threshold(y, score, 0.5, amount)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["value_recall"] == pytest.approx(200 / 500)


def test_recall_at_precision_returns_zero_when_impossible():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.1, 0.9, 0.2, 0.8])  # pior que aleatório
    assert recall_at_precision(y, score, 0.99) == 0.0


def test_split_by_rows_balances_sizes_and_keeps_order():
    rng = np.random.default_rng(1)
    # volume concentrado nos primeiros steps, como no PaySim
    steps = np.concatenate([rng.integers(1, 100, 9000), rng.integers(100, 744, 1000)])
    df = pd.DataFrame({"step": steps, "isFraud": 0})
    tr, va, te = temporal_split_by_rows(df, 0.6, 0.8)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr["step"].max() < va["step"].min() <= va["step"].max() < te["step"].min()
    assert 0.5 < len(tr) / len(df) < 0.75


def test_scenarios_drop_expected_columns():
    from fraud.features import SCENARIOS, build_features

    df = pd.DataFrame({
        "step": [1, 2], "type": ["TRANSFER", "CASH_OUT"], "amount": [10.0, 20.0],
        "oldbalanceOrg": [10.0, 50.0], "newbalanceOrig": [0.0, 30.0],
        "oldbalanceDest": [0.0, 5.0], "newbalanceDest": [10.0, 25.0],
    })
    cols = {k: set(build_features(df, v).columns) for k, v in SCENARIOS.items()}
    assert "errorBalanceOrig" in cols["A"] and "errorBalanceOrig" not in cols["B"]
    assert "amount_equals_old_org" in cols["B"] and "amount_equals_old_org" not in cols["C"]
    assert {"hour", "day"} <= cols["C"] and not ({"hour", "day"} & cols["D"])
    assert cols["D"] < cols["C"] < cols["B"] < cols["A"]
    # aceita letra ou nome
    assert list(build_features(df, "B").columns) == list(build_features(df, "sem_saldos_pos").columns)


def test_make_split_uses_config_mode():
    from fraud.split import make_split

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"step": np.concatenate([rng.integers(1, 50, 900), rng.integers(50, 300, 100)])})
    cfg = {"split": {"mode": "por_linhas", "train_end": 0.6, "val_end": 0.8}}
    a = make_split(df, cfg)
    cfg["split"]["mode"] = "por_steps"
    b = make_split(df, cfg)
    assert len(a[0]) != len(b[0])
    with pytest.raises(ValueError):
        make_split(df, {"split": {"mode": "x", "train_end": 0.6, "val_end": 0.8}})


def test_anomaly_prep_and_fill():
    from fraud.anomaly import AnomalyPrep, fill_out_of_scope, in_scope

    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame({
        "step": rng.integers(1, 100, n),
        "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n),
        "amount": rng.lognormal(9, 1, n), "oldbalanceOrg": rng.lognormal(10, 1, n),
        "newbalanceOrig": 0.0, "oldbalanceDest": rng.lognormal(10, 1, n), "newbalanceDest": 1.0,
    })
    mask = in_scope(df)
    assert mask.sum() < n
    prep = AnomalyPrep("sem_esvaziamento").fit(df[mask])
    X = prep.transform(df[mask])
    assert X.shape == (mask.sum(), len(prep.cols)) and np.isfinite(X).all()
    assert abs(X.mean()) < 0.5  # padronizado
    assert np.abs(X).max() <= 5.0  # corte em ±5
    full = fill_out_of_scope(np.ones(mask.sum()), mask)
    assert (full[mask] == 1).all() and (full[~mask] < 1).all()


def test_anomaly_prep_clips_rare_binary_feature():
    from fraud.anomaly import AnomalyPrep

    n = 2000
    rng = np.random.default_rng(4)
    legit = pd.DataFrame({
        "step": rng.integers(1, 50, n), "type": "TRANSFER",
        "amount": rng.lognormal(9, 1, n), "oldbalanceOrg": rng.lognormal(10, 1, n),
        "newbalanceOrig": 0.0, "oldbalanceDest": rng.lognormal(10, 1, n), "newbalanceDest": 1.0,
    })
    legit.loc[:1, "amount"] = legit.loc[:1, "oldbalanceOrg"]  # 2 casos "valor = saldo" em 2000
    fraud = legit.head(5).copy()
    fraud["amount"] = fraud["oldbalanceOrg"]
    prep = AnomalyPrep("sem_saldos_pos").fit(legit)
    assert "amount_equals_old_org" in prep.cols
    idx = prep.cols.index("amount_equals_old_org")
    assert prep.scaler.scale_[idx] < 0.05  # desvio minúsculo, como no PaySim
    assert prep.transform(fraud)[:, idx].max() == 5.0  # sem o corte seria > 20
