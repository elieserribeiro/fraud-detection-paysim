import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from fraud.features import build_features
from fraud.scoring import (contributions, level_names, load_model, logit, score_transactions, top_reasons)


def _toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    typ = rng.choice(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN"], n)
    old = rng.lognormal(10, 1.5, n)
    amt = rng.lognormal(8, 1, n)
    fraud = np.isin(typ, ["CASH_OUT", "TRANSFER"]) & (rng.random(n) < 0.05)
    amt[fraud] = old[fraud]
    return pd.DataFrame({"step": rng.integers(1, 700, n), "type": typ, "amount": amt, "oldbalanceOrg": old,
                         "oldbalanceDest": rng.lognormal(9, 1, n)}), fraud.astype(int)


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    df, y = _toy()
    m = xgb.XGBClassifier(n_estimators=40, max_depth=4, tree_method="hist").fit(build_features(df, "sem_saldos_pos"), y)
    p = tmp_path_factory.mktemp("m") / "m.json"
    m.save_model(str(p))
    return load_model(p), df, y


def test_json_roundtrip_scores_and_ranks_fraud_higher(model):
    m, df, y = model
    s = score_transactions(m, df)
    assert s.shape == (len(df),) and 0 <= s.min() and s.max() <= 1
    assert s[y == 1].mean() > s[y == 0].mean() + 0.3


def test_contributions_sum_to_margin(model):
    m, df, _ = model
    c = contributions(m, df.head(50))
    s = score_transactions(m, df.head(50))
    p = 1 / (1 + np.exp(-c.sum(axis=1).values))
    np.testing.assert_allclose(p, s, rtol=1e-4, atol=1e-5)


def test_top_reasons_labels_and_order(model):
    m, df, y = model
    row = df[y == 1].head(1)
    c = contributions(m, row).iloc[0]
    x = build_features(row, "sem_saldos_pos").iloc[0]
    r = top_reasons(c, x, k=3)
    assert len(r) == 3 and r["contribuicao"].abs().is_monotonic_decreasing
    assert "Valor igual a TODO o saldo de origem" in set(top_reasons(c, x, k=14)["atributo"])


def test_missing_column_error(model):
    m, df, _ = model
    with pytest.raises(ValueError):
        score_transactions(m, df.drop(columns=["amount"]))


def test_level_names_and_logit():
    assert level_names(np.array([0.1, 0.6, 0.99]), 0.5, 0.9) == ["aprovar", "autenticar", "investigar"]
    assert logit(0.5) == pytest.approx(0.0) and logit(0.0) == -12.0
