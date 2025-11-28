from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_engineering import extract_feature_target


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx]


def _generate_synthetic(n: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    prices = 30000 + rng.normal(0, 50, size=n).cumsum()
    high = prices + rng.normal(5, 5, size=n)
    low = prices - rng.normal(5, 5, size=n)
    volume = rng.normal(1000, 200, size=n).clip(min=10)
    df = pd.DataFrame(
        {
            "open": prices + rng.normal(0, 5, size=n),
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        }
    )
    return df


def load_market_data(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    return _generate_synthetic()


def build_datasets(
    data_path: Path,
    train_split: float,
    val_split: float,
) -> Tuple[FeatureDataset, FeatureDataset, FeatureDataset]:
    df = load_market_data(data_path)
    features_df, target = extract_feature_target(df)
    features = features_df.to_numpy()
    targets = target.to_numpy()

    n = len(features)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_set = FeatureDataset(features[:train_end], targets[:train_end])
    val_set = FeatureDataset(features[train_end:val_end], targets[train_end:val_end])
    test_set = FeatureDataset(features[val_end:], targets[val_end:])
    return train_set, val_set, test_set
