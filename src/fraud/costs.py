"""Impacto financeiro de uma regra "sinalizar se score >= corte" (dois níveis).

Premissas (editáveis em ``configs/config.yaml``):
- perda evitada = ``catch_rate`` x valor das fraudes sinalizadas;
- custo = ``investigation_cost`` por transação sinalizada (fraude ou não).

``legit_weight`` reescala os falsos positivos para a taxa de fraude real (0,129% no PaySim),
já que a taxa de fraude do conjunto de validação/teste é diferente da real.
A versão de três níveis (autenticação adicional) vem na Fase 6.
"""
from __future__ import annotations

import numpy as np

REAL_FRAUD_RATE = 0.001291  # 8.213 / 6.362.620


def legit_weight_for(y_true, real_rate: float = REAL_FRAUD_RATE) -> float:
    """Quantos falsos positivos "reais" cada falso positivo do conjunto representa."""
    p = float(np.mean(y_true))
    if p <= 0 or p >= 1:
        return 1.0
    return ((1 - real_rate) / real_rate) / ((1 - p) / p)


def cost_curve(y_true, amount, score, investigation_cost, catch_rate, legit_weight=1.0):
    """Benefício líquido para cada corte possível (um por valor distinto de score)."""
    y = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    score = np.asarray(score, dtype="float64")
    order = np.argsort(-score, kind="mergesort")
    s, yy, aa = score[order], y[order], amount[order]
    cum_val = np.cumsum(aa * yy)
    cum_tp = np.cumsum(yy)
    cum_fp = np.cumsum(~yy)
    last = np.r_[s[1:] != s[:-1], True]  # fim de cada grupo de empate
    net = catch_rate * cum_val[last] - investigation_cost * (cum_tp[last] + legit_weight * cum_fp[last])
    return s[last], net


def best_threshold(y_true, amount, score, investigation_cost, catch_rate, legit_weight=1.0) -> float:
    """Corte que maximiza o benefício líquido. ``inf`` se não vale a pena sinalizar nada."""
    thr, net = cost_curve(y_true, amount, score, investigation_cost, catch_rate, legit_weight)
    i = int(np.argmax(net))
    return float(thr[i]) if net[i] > 0 else float("inf")


def financial_at_threshold(y_true, amount, score, threshold, investigation_cost, catch_rate,
                           legit_weight=1.0) -> dict:
    y = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    flagged = np.asarray(score, dtype="float64") >= threshold
    tp = int((y & flagged).sum())
    fp = int((~y & flagged).sum())
    total = float(amount[y].sum())
    caught = float(amount[y & flagged].sum())
    saved = catch_rate * caught
    cost = investigation_cost * (tp + legit_weight * fp)
    return {
        "alertas": tp + fp,
        "fraudes_pegas": tp,
        "falsos_positivos": fp,
        "falsos_positivos_reponderados": legit_weight * fp,
        "valor_fraudado_total": total,
        "valor_poupado": saved,
        "valor_poupado_%": 100.0 * saved / total if total else 0.0,
        "custo_investigacao": cost,
        "beneficio_liquido": saved - cost,
    }
