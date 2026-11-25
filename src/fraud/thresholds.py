"""Cortes de risco em três níveis por custo esperado.

Regra de decisão sobre o score ``s`` (cortes ``t_auth <= t_inv``):
  s <  t_auth            -> aprovar automaticamente
  t_auth <= s < t_inv    -> pedir autenticação adicional
  s >= t_inv             -> encaminhar para investigação

Custo esperado total = custo operacional (autenticação/investigação por transação tratada)
                     + perda residual (valor das fraudes que não foram barradas).
- Fraude aprovada: perda de 100% do valor.
- Fraude autenticada: ``auth_catch`` do valor é evitado; o resto é perda.
- Fraude investigada: ``inv_catch`` do valor é evitado; o resto é perda.
``legit_weight`` reescala as transações legítimas para a taxa de fraude real (ver ``costs.py``).

O custo é separável em (corte de autenticação) + (corte de investigação), então o ótimo
exato, entre TODOS os cortes possíveis, sai em O(n) depois de ordenar os scores.
"""
from __future__ import annotations

import numpy as np

LEVELS = ("aprovar", "autenticar", "investigar")


def decision_levels(score, t_auth, t_inv) -> np.ndarray:
    """0 = aprovar, 1 = autenticar, 2 = investigar."""
    s = np.asarray(score, dtype="float64")
    out = np.zeros(len(s), dtype=np.int8)
    out[s >= t_auth] = 1
    out[s >= t_inv] = 2
    return out


def _weights(y, legit_weight, weights):
    """Peso de cada transação: 1 para fraude, ``legit_weight`` para legítima (ou ``weights`` explícito)."""
    if weights is not None:
        return np.asarray(weights, dtype="float64")
    return np.where(y, 1.0, float(legit_weight))


def evaluate_levels(y_true, amount, score, t_auth, t_inv, inv_cost, auth_cost, inv_catch, auth_catch,
                    legit_weight=1.0, weights=None) -> dict:
    y = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    w = _weights(y, legit_weight, weights)
    lv = decision_levels(score, t_auth, t_inv)
    total_val = float((amount * w)[y].sum())
    eq_total = float(w.sum())
    rows, saved, op_cost, residual = [], 0.0, 0.0, 0.0
    for k, (nome, unit_cost, catch) in enumerate(
            [("aprovar", 0.0, 0.0), ("autenticar", auth_cost, auth_catch), ("investigar", inv_cost, inv_catch)]):
        m = lv == k
        fr, lg = m & y, m & ~y
        eq = float(w[m].sum())
        val_f = float((amount * w)[fr].sum())
        c_op, c_res = unit_cost * eq, (1.0 - catch) * val_f
        saved += catch * val_f
        op_cost += c_op
        residual += c_res
        rows.append({"nivel": nome, "transacoes": int(m.sum()), "fraudes": int(fr.sum()),
                     "falsos_positivos": int(lg.sum()), "falsos_positivos_reponderados": float(w[lg].sum()),
                     "valor_fraude": val_f, "custo_operacional": c_op, "perda_residual": c_res,
                     "pct_transacoes_reais": 100.0 * eq / eq_total if eq_total else 0.0})
    return {"niveis": rows, "valor_fraudado_total": total_val, "valor_poupado": saved,
            "valor_poupado_%": 100.0 * saved / total_val if total_val else 0.0,
            "perda_residual": residual, "custo_operacional": op_cost,
            "custo_total": op_cost + residual, "beneficio_liquido": total_val - (op_cost + residual)}


def optimize_levels(y_true, amount, score, inv_cost, auth_cost, inv_catch, auth_catch, legit_weight=1.0,
                    weights=None) -> dict:
    """Cortes que minimizam o custo esperado. ``inf`` = nível nunca usado."""
    y = np.asarray(y_true).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    s = np.asarray(score, dtype="float64")
    w = _weights(y, legit_weight, weights)
    order = np.argsort(-s, kind="mergesort")
    s_d, fy, a_d, w_d = s[order], y[order], amount[order], w[order]
    cum_val = np.r_[0.0, np.cumsum(a_d * fy * w_d)]
    cum_w = np.r_[0.0, np.cumsum(w_d)]
    ends = np.flatnonzero(np.r_[s_d[1:] != s_d[:-1], True]) + 1   # nº de itens com score >= corte
    ks = np.r_[0, ends]
    thr = np.r_[np.inf, s_d[ends - 1]]
    FV, eq = cum_val[ks], cum_w[ks]
    total_val = float(cum_val[-1])

    # custo = f(i) + g(j), com j <= i (investigados são um subconjunto dos sinalizados)
    f = auth_cost * eq + (1.0 - auth_catch) * FV + (total_val - FV)
    g = (inv_cost - auth_cost) * eq + (auth_catch - inv_catch) * FV
    gmin = np.minimum.accumulate(g)
    idx = np.arange(len(g))
    best_j = np.maximum.accumulate(np.where(g == gmin, idx, 0))
    total = f + gmin
    i = int(np.argmin(total))
    j = int(best_j[i])
    two = f + g                                   # sem nível de autenticação (j == i)
    i2 = int(np.argmin(two))
    return {"t_auth": float(thr[i]), "t_inv": float(thr[j]), "t_two": float(thr[i2]),
            "beneficio_liquido": total_val - float(total[i]),
            "beneficio_dois_niveis": total_val - float(two[i2])}
