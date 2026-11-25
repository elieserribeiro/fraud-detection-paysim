"""Apoio aos modelos de anomalia (Isolation Forest e Autoencoder).

Decisões do projeto: treinar só com transações legítimas e só em TRANSFER e CASH_OUT
(único escopo onde existe fraude). As demais transações recebem o menor score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .features import build_features

SCOPE_TYPES = ["TRANSFER", "CASH_OUT"]


def in_scope(df: pd.DataFrame) -> np.ndarray:
    """Máscara booleana das transações em que o modelo de anomalia atua."""
    return df["type"].astype(str).isin(SCOPE_TYPES).to_numpy()


def signed_log1p(a):
    """log1p preservando o sinal (valores e saldos têm cauda muito longa)."""
    a = np.asarray(a, dtype="float64")
    return np.sign(a) * np.log1p(np.abs(a))


class AnomalyPrep:
    """Features -> log com sinal -> padronização -> corte em ±``clip``.

    Ajustado só nas legítimas de treino. O corte evita que uma coluna binária quase sempre zero
    (ex.: ``amount_equals_old_org``, desvio-padrão ~0,001) gere valores padronizados na casa de
    centenas e distorça a reconstrução inteira do Autoencoder (artefato numérico, não anomalia).
    """

    def __init__(self, variant: str, clip: float = 5.0):
        self.variant = variant
        self.clip = clip
        self.cols: list[str] = []
        self.scaler = StandardScaler()

    def fit(self, df_legit: pd.DataFrame) -> "AnomalyPrep":
        X = build_features(df_legit, self.variant)
        self.cols = [c for c in X.columns if X[c].std() > 0]  # descarta colunas constantes
        self.scaler.fit(signed_log1p(X[self.cols].to_numpy()))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = build_features(df, self.variant)[self.cols].to_numpy()
        Z = self.scaler.transform(signed_log1p(X))
        return np.clip(Z, -self.clip, self.clip).astype("float32")


def fill_out_of_scope(score_in_scope: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Devolve um score para todas as linhas; as fora de escopo ficam abaixo do menor score."""
    score_in_scope = np.asarray(score_in_scope, dtype="float64")
    out = np.full(len(mask), score_in_scope.min() - 1.0, dtype="float64")
    out[np.asarray(mask, dtype=bool)] = score_in_scope
    return out


def build_autoencoder(n_features: int, seed: int = 42):
    """Autoencoder denso simétrico com gargalo; o score é o erro de reconstrução."""
    import keras
    from keras import layers

    keras.utils.set_random_seed(seed)
    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(32, activation="relu")(inp)
    x = layers.Dense(16, activation="relu")(x)
    code = layers.Dense(max(2, n_features // 2), activation="relu")(x)
    x = layers.Dense(16, activation="relu")(code)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def reconstruction_error(model, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Erro quadrático médio por linha (quanto maior, mais anômalo)."""
    rec = model.predict(X, batch_size=batch_size, verbose=0)
    return np.mean((X - rec) ** 2, axis=1)
