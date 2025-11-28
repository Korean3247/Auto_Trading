from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from offline_training.feature_engineering import build_feature_matrix


def make_features(df_prices: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    feats = build_feature_matrix(df_prices.reset_index(drop=False).rename(columns={"index": "timestamp"}))
    prices = df_prices.loc[feats.index, "close"].to_numpy()
    feature_only = feats.drop(columns=["close", "open", "high", "low", "volume", "timestamp"], errors="ignore")
    return prices, feature_only.to_numpy()


def save_feature_bundle(path: Path, prices: np.ndarray, features: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(features)
    df.insert(0, "close", prices)
    df.to_parquet(path, index=False)
    logger.info(f"Saved features to {path}")


def load_feature_bundle(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_parquet(path)
    prices = df["close"].to_numpy()
    feats = df.drop(columns=["close"]).to_numpy()
    return prices, feats
