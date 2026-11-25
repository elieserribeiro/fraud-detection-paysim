"""Pontuação de transações: score de risco, nível de decisão e explicação (valores SHAP).

Usado pelo painel Streamlit e pela API. Carrega o modelo em JSON nativo do XGBoost (portável entre
versões) e usa o cenário B (``sem_saldos_pos``): só informações disponíveis ANTES de aprovar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import build_features
from .thresholds import LEVELS, decision_levels

VARIANT = "sem_saldos_pos"
REQUIRED_COLUMNS = ["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest"]

# Maiores valores do PaySim completo (6,36 mi de transações). Acima disso o modelo não tem referência.
PAYSIM_MAX = {"amount": 92_445_516.64, "oldbalanceOrg": 59_585_040.37, "oldbalanceDest": 356_015_889.35}
RULE_LABEL = "regra de precaução: valores fora da faixa conhecida pelo modelo"

FEATURE_LABELS = {
    "type_CASH_IN": "Tipo: depósito (CASH_IN)",
    "type_CASH_OUT": "Tipo: saque (CASH_OUT)",
    "type_DEBIT": "Tipo: débito",
    "type_PAYMENT": "Tipo: pagamento",
    "type_TRANSFER": "Tipo: transferência",
    "hour": "Hora do dia",
    "day": "Dia da simulação",
    "amount_log": "Valor da transação",
    "oldbalanceOrg": "Saldo da conta de origem",
    "oldbalanceDest": "Saldo da conta de destino",
    "amount_to_old_org": "Valor ÷ saldo de origem",
    "orig_balance_zero": "Conta de origem com saldo zero",
    "dest_balance_zero": "Conta de destino com saldo zero",
    "amount_equals_old_org": "Valor igual a TODO o saldo de origem",
}


def load_model(path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


def load_thresholds(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes: {missing}. Esperado: {REQUIRED_COLUMNS}")


def score_transactions(model, df: pd.DataFrame) -> np.ndarray:
    """Probabilidade de fraude (0 a 1) para cada linha."""
    check_columns(df)
    X = build_features(df, VARIANT)
    return model.predict_proba(X)[:, 1]


def contributions(model, df: pd.DataFrame) -> pd.DataFrame:
    """Contribuição de cada atributo para o score (log-odds; SHAP exato de árvores). Coluna ``bias`` = base."""
    check_columns(df)
    X = build_features(df, VARIANT)
    raw = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(raw, columns=list(X.columns) + ["bias"], index=df.index)


def top_reasons(contrib_row: pd.Series, x_row: pd.Series, k: int = 6) -> pd.DataFrame:
    """Os ``k`` atributos que mais pesaram (em módulo), com rótulo legível."""
    c = contrib_row.drop("bias")
    idx = c.abs().sort_values(ascending=False).index[:k]
    return pd.DataFrame({
        "chave": list(idx),
        "atributo": [FEATURE_LABELS.get(i, i) for i in idx],
        "contribuicao": c[idx].values,
        "valor": [x_row.get(i, np.nan) for i in idx],
    })


def level_names(score, t_auth, t_inv) -> list[str]:
    lv = decision_levels(score, t_auth, t_inv)
    return [LEVELS[i] for i in lv]


def logit(p, clip: float = 12.0):
    p = np.clip(np.asarray(p, dtype="float64"), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p / (1 - p)), -clip, clip)


def out_of_range(df: pd.DataFrame) -> np.ndarray:
    """True quando algum valor passa do máximo visto no PaySim (o score do modelo não é confiável)."""
    mask = np.zeros(len(df), dtype=bool)
    for col, lim in PAYSIM_MAX.items():
        mask |= df[col].astype("float64").values > lim
    return mask


def decide(score, df: pd.DataFrame, t_auth, t_inv) -> tuple[list[str], np.ndarray]:
    """Nível final: regra de precaução (fora da faixa -> investigar) e, no resto, os cortes do modelo."""
    lv = decision_levels(score, t_auth, t_inv)
    rule = out_of_range(df)
    lv = np.where(rule, 2, lv)
    return [LEVELS[i] for i in lv], rule
