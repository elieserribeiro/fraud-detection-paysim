"""Engenharia de atributos com quatro cenários (A a D), do mais fácil ao mais difícil.

O PaySim tem "atalhos" do simulador; os cenários medem o quanto cada modelo depende deles:

- A ``com_saldos``: todos os saldos + erros de saldo (``errorBalanceOrig``/``errorBalanceDest``).
- B ``sem_saldos_pos``: remove saldos pós-transação e erros derivados, mantendo só o que o banco
  sabe ANTES de aprovar a transação. É o cenário principal (realista).
- C ``sem_esvaziamento``: B sem o sinal de "esvaziar a conta" (valor = saldo, razão valor/saldo,
  saldo de origem zero). Obs.: modelos de árvore ainda podem reconstruir parte desse sinal a
  partir de ``amount_log`` e ``oldbalanceOrg``.
- D ``sem_esvaziamento_sem_tempo``: C também sem ``hour`` e ``day`` (o simulador gera fraude a
  ritmo constante, o que pode virar atalho).

Os mesmos métodos são usados no treino e no painel Streamlit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

_POST = ["newbalanceOrig", "newbalanceDest", "errorBalanceOrig", "errorBalanceDest",
         "orig_new_balance_zero"]
_DRAIN = ["amount_to_old_org", "orig_balance_zero", "amount_equals_old_org"]
_TIME = ["hour", "day"]

# nome da variante -> colunas removidas do conjunto completo
_DROPS = {
    "com_saldos": [],
    "sem_saldos_pos": _POST,
    "sem_esvaziamento": _POST + _DRAIN,
    "sem_esvaziamento_sem_tempo": _POST + _DRAIN + _TIME,
}
VARIANTS = tuple(_DROPS)
SCENARIOS = {"A": "com_saldos", "B": "sem_saldos_pos", "C": "sem_esvaziamento",
             "D": "sem_esvaziamento_sem_tempo"}

_RAW_NUMERIC = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]


def build_features(df: pd.DataFrame, variant: str = "sem_saldos_pos") -> pd.DataFrame:
    """Retorna apenas as colunas de entrada do modelo (sem o alvo)."""
    variant = SCENARIOS.get(variant, variant)  # aceita "A".."D"
    if variant not in VARIANTS:
        raise ValueError(f"variant deve ser um de {VARIANTS} ou {tuple(SCENARIOS)}")

    out = pd.DataFrame(index=df.index)

    # Tipo da transação (one-hot com categorias fixas -> mesmas colunas em treino e produção)
    for t in TYPES:
        out[f"type_{t}"] = (df["type"].astype(str) == t).astype("int8")

    amount = df["amount"].astype("float64")
    old_org = df["oldbalanceOrg"].astype("float64")
    old_dest = df["oldbalanceDest"].astype("float64")

    # Tempo (step = hora simulada)
    out["hour"] = (df["step"] % 24).astype("int8")
    out["day"] = (df["step"] // 24).astype("int16")

    # Valor e saldos conhecidos antes da transação
    out["amount_log"] = np.log1p(amount)
    out["oldbalanceOrg"] = old_org
    out["oldbalanceDest"] = old_dest
    out["amount_to_old_org"] = amount / (old_org + 1.0)
    out["orig_balance_zero"] = (old_org == 0).astype("int8")
    out["dest_balance_zero"] = (old_dest == 0).astype("int8")
    out["amount_equals_old_org"] = np.isclose(amount, old_org).astype("int8")

    if variant == "com_saldos":
        new_org = df["newbalanceOrig"].astype("float64")
        new_dest = df["newbalanceDest"].astype("float64")
        out["newbalanceOrig"] = new_org
        out["newbalanceDest"] = new_dest
        out["errorBalanceOrig"] = new_org + amount - old_org
        out["errorBalanceDest"] = old_dest + amount - new_dest
        out["orig_new_balance_zero"] = (new_org == 0).astype("int8")

    drops = [c for c in _DROPS[variant] if c in out.columns]
    return out.drop(columns=drops).astype("float32")


def feature_names(variant: str) -> list[str]:
    """Nomes das colunas geradas, na ordem de ``build_features``."""
    dummy = pd.DataFrame(
        {
            "step": [1], "type": ["TRANSFER"], "amount": [1.0],
            "oldbalanceOrg": [1.0], "newbalanceOrig": [0.0],
            "oldbalanceDest": [0.0], "newbalanceDest": [1.0],
        }
    )
    return list(build_features(dummy, variant).columns)
