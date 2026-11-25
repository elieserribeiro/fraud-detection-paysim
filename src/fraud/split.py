"""Split temporal por `step`: treino / validação (thresholds) / teste."""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    df: pd.DataFrame,
    train_end: float = 0.60,
    val_end: float = 0.80,
    time_col: str = "step",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide pelo tempo, sem embaralhar.

    `train_end` e `val_end` são frações do intervalo de `step`. Treino: step <= corte1;
    validação: corte1 < step <= corte2; teste: step > corte2.
    A validação serve para calibrar/escolher thresholds; o teste só mede o resultado final.
    """
    if not 0 < train_end < val_end < 1:
        raise ValueError("Esperado 0 < train_end < val_end < 1")
    lo, hi = df[time_col].min(), df[time_col].max()
    span = hi - lo + 1
    cut1 = lo + int(np.floor(span * train_end)) - 1
    cut2 = lo + int(np.floor(span * val_end)) - 1
    train = df[df[time_col] <= cut1]
    val = df[(df[time_col] > cut1) & (df[time_col] <= cut2)]
    test = df[df[time_col] > cut2]
    return train, val, test


def make_split(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica o split escolhido em ``cfg['split']['mode']`` ("por_linhas" ou "por_steps")."""
    s = cfg["split"]
    mode = s.get("mode", "por_linhas")
    fn = {"por_linhas": temporal_split_by_rows, "por_steps": temporal_split}.get(mode)
    if fn is None:
        raise ValueError("split.mode deve ser 'por_linhas' ou 'por_steps'")
    return fn(df, s["train_end"], s["val_end"])


def temporal_split_by_rows(
    df: pd.DataFrame,
    train_end: float = 0.60,
    val_end: float = 0.80,
    time_col: str = "step",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split temporal cujos cortes são frações do NÚMERO DE TRANSAÇÕES (não do intervalo de steps).

    No PaySim o volume se concentra nos primeiros steps; este corte equilibra o tamanho dos
    conjuntos. Os cortes caem sempre em fronteiras de step, sem misturar passado e futuro.
    """
    if not 0 < train_end < val_end < 1:
        raise ValueError("Esperado 0 < train_end < val_end < 1")
    counts = df.groupby(time_col).size().sort_index()
    cum = counts.cumsum() / counts.sum()
    idx = cum.index
    cut1 = idx[min(int(cum.searchsorted(train_end)), len(idx) - 1)]
    cut2 = idx[min(int(cum.searchsorted(val_end)), len(idx) - 1)]
    if cut2 <= cut1:
        raise ValueError("Cortes colapsaram: dados concentrados demais em poucos steps")
    train = df[df[time_col] <= cut1]
    val = df[(df[time_col] > cut1) & (df[time_col] <= cut2)]
    test = df[df[time_col] > cut2]
    return train, val, test
