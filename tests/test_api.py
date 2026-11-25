import json

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from fastapi.testclient import TestClient

from fraud.features import build_features


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("app_data")
    rng = np.random.default_rng(0)
    n = 2000
    typ = rng.choice(["CASH_OUT", "TRANSFER", "PAYMENT"], n)
    old = rng.lognormal(10, 1.5, n)
    amt = rng.lognormal(8, 1, n)
    y = (np.isin(typ, ["CASH_OUT", "TRANSFER"]) & (rng.random(n) < 0.05)).astype(int)
    amt[y == 1] = old[y == 1]
    df = pd.DataFrame({"step": rng.integers(1, 700, n), "type": typ, "amount": amt, "oldbalanceOrg": old,
                       "oldbalanceDest": rng.lognormal(9, 1, n)})
    m = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(build_features(df, "sem_saldos_pos"), y)
    m.save_model(str(d / "xgboost_B.json"))
    (d / "thresholds.json").write_text(json.dumps({"thresholds": {"B": {"x": {
        "t_auth": 0.3, "t_inv": 0.8, "padrao": True}}}}), encoding="utf-8")
    import os
    os.environ["FRAUD_APP_DATA"] = str(d)
    from api import main
    main.bundle.cache_clear()
    yield TestClient(main.app)
    main.bundle.cache_clear()


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["cortes"] == {"autenticar": 0.3, "investigar": 0.8}


def test_score_fraud_pattern_is_investigated(client):
    r = client.post("/score", json={"step": 200, "type": "TRANSFER", "amount": 50000.0,
                                    "oldbalanceOrg": 50000.0, "oldbalanceDest": 0.0}).json()
    assert r["nivel"] == "investigar" and r["score"] > 0.8 and len(r["motivos"]) == 5


def test_batch_and_validation(client):
    itens = [{"step": 10, "type": "PAYMENT", "amount": 100.0, "oldbalanceOrg": 5000.0, "oldbalanceDest": 0.0}] * 3
    r = client.post("/score/lote", json=itens)
    assert r.status_code == 200 and len(r.json()) == 3
    bad = client.post("/score", json={"step": 1, "type": "PIX", "amount": 1.0, "oldbalanceOrg": 0.0, "oldbalanceDest": 0.0})
    assert bad.status_code == 422
