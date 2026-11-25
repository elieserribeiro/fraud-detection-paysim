"""Combinação (ensemble) dos scores dos modelos.

Os scores de cada modelo têm escalas diferentes. ``TailRank`` os leva a uma escala comum que
preserva a cauda superior (onde estão as fraudes): t = -log10(1 - ECDF_validação(score)).
Os pesos e o stacking são ajustados SÓ na validação; o teste nunca entra no ajuste.
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score


class TailRank:
    """Transformação monotônica por modelo, ajustada nos scores da validação."""

    def fit(self, scores) -> "TailRank":
        self.sorted_ = np.sort(np.asarray(scores, dtype="float64"))
        return self

    def transform(self, scores) -> np.ndarray:
        n = len(self.sorted_)
        u = np.searchsorted(self.sorted_, np.asarray(scores, dtype="float64"), side="right") / (n + 1)
        return -np.log10(1.0 - u + 1e-12)


def fit_tail_ranks(score_dict: dict) -> dict:
    return {k: TailRank().fit(v) for k, v in score_dict.items()}


def stack_matrix(score_dict: dict, ranks: dict, names: list[str]) -> np.ndarray:
    return np.column_stack([ranks[n].transform(score_dict[n]) for n in names])


def first_half_mask(step) -> np.ndarray:
    """Divide a validação em dois blocos temporais (True = primeira metade, por linhas)."""
    step = np.asarray(step)
    return step <= np.median(step)


def stack_fit_predict(Q_fit, y_fit, Q_pred) -> np.ndarray:
    lr = LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000).fit(Q_fit, y_fit)
    return lr.decision_function(Q_pred)


def stack_oof(Q, y, fold_mask) -> np.ndarray:
    """Stacking fora da amostra: ajusta em um bloco temporal e pontua o outro."""
    y = np.asarray(y)
    out = np.empty(len(y))
    for f in (True, False):
        pred = fold_mask == f
        out[pred] = stack_fit_predict(Q[~pred], y[~pred], Q[pred])
    return out


def weight_grid(n_models: int, steps: int = 4):
    """Pesos em múltiplos de 1/steps que somam 1 (ex.: 0, 0,25, 0,5, 0,75, 1)."""
    for combo in itertools.product(range(steps + 1), repeat=n_models):
        if sum(combo) == steps:
            yield np.array(combo, dtype=float) / steps


def best_weights(Q, y, steps: int = 4) -> np.ndarray:
    best, best_ap = None, -1.0
    for w in weight_grid(Q.shape[1], steps):
        ap = average_precision_score(y, Q @ w)
        if ap > best_ap:
            best, best_ap = w, ap
    return best


def weights_oof(Q, y, fold_mask, steps: int = 4) -> np.ndarray:
    """Média ponderada fora da amostra: escolhe pesos em um bloco e pontua o outro."""
    y = np.asarray(y)
    out = np.empty(len(y))
    for f in (True, False):
        pred = fold_mask == f
        w = best_weights(Q[~pred], y[~pred], steps)
        out[pred] = Q[pred] @ w
    return out
