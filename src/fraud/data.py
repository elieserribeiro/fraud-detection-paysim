"""Carga do PaySim: CSV -> Parquet (leitura rápida) e amostragem para desenvolvimento."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DTYPES = {
    "step": "int16",
    "type": "category",
    # float64: com float32 os "erros de saldo" (new + amount - old) ficam com ruído de arredondamento
    "amount": "float64",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

REQUIRED_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig",
    "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud",
]


def csv_to_parquet(csv_path: str | Path, parquet_path: str | Path) -> Path:
    """Converte o CSV bruto em Parquet com dtypes compactos."""
    df = pd.read_csv(csv_path, dtype=DTYPES)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(missing)}")
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def load_parquet(parquet_path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def dev_sample(df: pd.DataFrame, frac: float, target: str = "isFraud", seed: int = 42) -> pd.DataFrame:
    """Amostra as transações legítimas e mantém TODAS as fraudes (uso em desenvolvimento)."""
    fraud = df[df[target] == 1]
    legit = df[df[target] == 0].sample(frac=frac, random_state=seed)
    return pd.concat([fraud, legit]).sort_values("step").reset_index(drop=True)
