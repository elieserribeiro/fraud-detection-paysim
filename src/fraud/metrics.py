"""Métricas de avaliação para fraude: PR-AUC, Precision, Recall, F1, taxa de falsos positivos
e recall em valor (quanto do dinheiro fraudado foi sinalizado)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def confusion_at(y_true, flagged) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    tp = int((y_true & flagged).sum())
    fp = int((~y_true & flagged).sum())
    fn = int((y_true & ~flagged).sum())
    tn = int((~y_true & ~flagged).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_at_threshold(y_true, score, threshold, amount=None) -> dict:
    """Métricas para um corte fixo: flagged = score >= threshold."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    flagged = score >= threshold
    c = confusion_at(y_true, flagged)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "flagged_pct": 100.0 * flagged.mean(),
        **c,
    }
    if amount is not None:
        amount = np.asarray(amount, dtype="float64")
        total = amount[y_true.astype(bool)].sum()
        out["value_recall"] = float(amount[y_true.astype(bool) & flagged].sum() / total) if total else 0.0
    return out


def best_f1_threshold(y_true, score) -> float:
    """Corte que maximiza o F1 (usado apenas como referência de baseline)."""
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def recall_at_precision(y_true, score, min_precision: float = 0.9) -> float:
    """Maior recall possível mantendo precision >= min_precision (0 se impossível)."""
    p, r, _ = precision_recall_curve(y_true, score)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def evaluate(y_true, score, amount=None, min_precision: float = 0.9) -> dict:
    """Resumo único: métricas independentes de corte + métricas no corte de melhor F1."""
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    out = {
        "prevalencia_%": 100.0 * float(y_true.mean()),
        "pr_auc": float(average_precision_score(y_true, score)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        f"recall@precision>={min_precision:.0%}": recall_at_precision(y_true, score, min_precision),
    }
    thr = best_f1_threshold(y_true, score)
    out.update({f"bestF1_{k}": v for k, v in metrics_at_threshold(y_true, score, thr, amount).items()})
    return out
