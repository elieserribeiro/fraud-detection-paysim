"""API de scores de risco (FastAPI).

Rode com:  uvicorn api.main:app --reload
Documentação interativa: http://127.0.0.1:8000/docs
Lê o modelo e os cortes de ``app/data`` (ou da pasta em FRAUD_APP_DATA).
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.features import build_features  # noqa: E402
from fraud.scoring import VARIANT, contributions, load_model, load_thresholds, score_transactions, top_reasons  # noqa: E402
from fraud.thresholds import LEVELS, decision_levels  # noqa: E402


def data_dir() -> Path:
    return Path(os.environ.get("FRAUD_APP_DATA", ROOT / "app" / "data"))


@lru_cache(maxsize=1)
def bundle():
    d = data_dir()
    thr = load_thresholds(d / "thresholds.json")
    padrao = next(v for v in thr["thresholds"]["B"].values() if v["padrao"])
    t_auth = np.inf if padrao["t_auth"] is None else padrao["t_auth"]
    t_inv = np.inf if padrao["t_inv"] is None else padrao["t_inv"]
    return load_model(d / "xgboost_B.json"), float(t_auth), float(t_inv)


class Transacao(BaseModel):
    step: int = Field(ge=0, description="Hora da simulação (1 step = 1 hora)", examples=[230])
    type: Literal["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"] = Field(examples=["TRANSFER"])
    amount: float = Field(ge=0, examples=[181000.0])
    oldbalanceOrg: float = Field(ge=0, description="Saldo da conta de origem antes", examples=[181000.0])
    oldbalanceDest: float = Field(ge=0, description="Saldo da conta de destino antes", examples=[0.0])


class Motivo(BaseModel):
    atributo: str
    contribuicao: float


class Resultado(BaseModel):
    score: float = Field(description="Probabilidade estimada de fraude (0 a 1)")
    nivel: Literal["aprovar", "autenticar", "investigar"]
    motivos: list[Motivo]


app = FastAPI(title="API Antifraude PaySim", version="1.0.0",
              description="Score de risco e decisão em três níveis (aprovar, autenticar, investigar). Projeto de portfólio.")


def _avaliar(itens: list[Transacao], com_motivos: bool) -> list[Resultado]:
    model, t_auth, t_inv = bundle()
    df = pd.DataFrame([i.model_dump() for i in itens])
    s = score_transactions(model, df)
    niveis = decision_levels(s, t_auth, t_inv)
    motivos = [[] for _ in itens]
    if com_motivos:
        contrib = contributions(model, df)
        X = build_features(df, VARIANT)
        motivos = [[Motivo(atributo=a, contribuicao=round(float(c), 4))
                    for a, c in top_reasons(contrib.iloc[k], X.iloc[k], k=5)[["atributo", "contribuicao"]].values]
                   for k in range(len(df))]
    return [Resultado(score=float(s[k]), nivel=LEVELS[int(niveis[k])], motivos=motivos[k]) for k in range(len(df))]


@app.get("/health")
def health():
    _, t_auth, t_inv = bundle()
    return {"status": "ok", "modelo": "xgboost (cenário B)",
            "cortes": {"autenticar": t_auth if np.isfinite(t_auth) else None,
                       "investigar": t_inv if np.isfinite(t_inv) else None}}


@app.post("/score", response_model=Resultado)
def score(t: Transacao):
    return _avaliar([t], com_motivos=True)[0]


@app.post("/score/lote", response_model=list[Resultado])
def score_lote(itens: list[Transacao]):
    return _avaliar(itens, com_motivos=False)
