import numpy as np
import pandas as pd
import pytest

from fraud.data import dev_sample
from fraud.features import VARIANTS, build_features, feature_names
from fraud.split import temporal_split


@pytest.fixture
def fake_paysim():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame(
        {
            "step": rng.integers(1, 744, n),
            "type": rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"], n),
            "amount": rng.lognormal(8, 1.5, n),
            "oldbalanceOrg": rng.lognormal(9, 2, n),
            "oldbalanceDest": rng.lognormal(9, 2, n),
            "isFraud": (rng.random(n) < 0.01).astype(int),
        }
    )
    df["newbalanceOrig"] = (df.oldbalanceOrg - df.amount).clip(lower=0)
    df["newbalanceDest"] = df.oldbalanceDest + df.amount
    return df.sort_values("step").reset_index(drop=True)


def test_temporal_split_has_no_overlap_and_is_ordered(fake_paysim):
    tr, va, te = temporal_split(fake_paysim, 0.6, 0.8)
    assert len(tr) + len(va) + len(te) == len(fake_paysim)
    assert tr["step"].max() < va["step"].min()
    assert va["step"].max() < te["step"].min()


def test_temporal_split_rejects_bad_fractions(fake_paysim):
    with pytest.raises(ValueError):
        temporal_split(fake_paysim, 0.8, 0.6)


def test_variants_drop_post_transaction_balances(fake_paysim):
    com = build_features(fake_paysim, "com_saldos")
    sem = build_features(fake_paysim, "sem_saldos_pos")
    for col in ("newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest"):
        assert col in com.columns
        assert col not in sem.columns
    assert set(sem.columns) < set(com.columns)


@pytest.mark.parametrize("variant", VARIANTS)
def test_features_have_no_nan_and_match_names(fake_paysim, variant):
    X = build_features(fake_paysim, variant)
    assert not X.isna().any().any()
    assert list(X.columns) == feature_names(variant)
    assert len(X) == len(fake_paysim)


def test_dev_sample_keeps_all_frauds(fake_paysim):
    s = dev_sample(fake_paysim, 0.1)
    assert s["isFraud"].sum() == fake_paysim["isFraud"].sum()
    assert len(s) < len(fake_paysim)
