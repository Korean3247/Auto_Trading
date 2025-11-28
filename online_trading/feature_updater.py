from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np
import pandas as pd

from offline_training.feature_engineering import build_feature_matrix
from online_trading.live_feed import Candle


class FeatureUpdater:
    """
    Maintains a rolling dataframe of recent candles and produces a latest feature vector.
    """

    def __init__(self, maxlen: int = 500):
        self.candles: Deque[Candle] = deque(maxlen=maxlen)

    def update(self, candle: Candle) -> Optional[np.ndarray]:
        self.candles.append(candle)
        if len(self.candles) < 60:  # need some history for indicators
            return None
        df = pd.DataFrame(
            [
                {
                    "timestamp": c.ts,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in self.candles
            ]
        )
        feats = build_feature_matrix(df)
        if feats.empty:
            return None
        latest = feats.drop(columns=["close", "open", "high", "low", "volume", "timestamp"], errors="ignore").iloc[-1]
        return latest.to_numpy(dtype=float)
