from __future__ import annotations

import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns, spread, volume deltas."""
    out = df.copy()
    for window in [1, 5, 15, 30]:
        out[f"logret_{window}"] = np.log(out["close"]).diff(window)
    for window in [1, 5]:
        out[f"spread_{window}"] = (out["high"] - out["low"]).rolling(window).mean()
        out[f"volume_delta_{window}"] = out["volume"].diff(window)
    return out


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi_14"] = RSIIndicator(close=out["close"], window=14).rsi()
    out["ema_7"] = EMAIndicator(close=out["close"], window=7).ema_indicator()
    out["ema_25"] = EMAIndicator(close=out["close"], window=25).ema_indicator()
    bb = BollingerBands(close=out["close"], window=20, window_dev=2)
    out["bb_high"] = bb.bollinger_hband()
    out["bb_low"] = bb.bollinger_lband()
    atr = AverageTrueRange(high=out["high"], low=out["low"], close=out["close"], window=14)
    out["atr_14"] = atr.average_true_range()
    out["volatility_30"] = out["close"].rolling(30).std()
    return out


def compute_futures_features(df: pd.DataFrame) -> pd.DataFrame:
    """Placeholder futures-specific features; expects optional columns funding_rate, open_interest."""
    out = df.copy()
    if "funding_rate" in out.columns:
        out["funding_rate_1h"] = out["funding_rate"].rolling(60).mean()
    if "open_interest" in out.columns:
        out["oi_change_15m"] = out["open_interest"].diff(15)
    if "long_short_ratio" in out.columns:
        out["long_short_ratio_smooth"] = out["long_short_ratio"].rolling(10).mean()
    return out


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline combining base, indicators, and futures extras."""
    features = compute_base_features(df)
    features = compute_indicators(features)
    features = compute_futures_features(features)
    features = features.dropna()
    return features


def extract_feature_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Simple supervised label: next-step direction (long/flat/short) using next close return.
    """
    feats = build_feature_matrix(df)
    forward_ret = feats["close"].pct_change().shift(-1)
    target = pd.cut(
        forward_ret,
        bins=[-np.inf, -0.0005, 0.0005, np.inf],
        labels=[0, 1, 2],  # short, flat, long
    )
    feats = feats.iloc[:-1]
    target = target.iloc[:-1]
    feature_only = feats.drop(columns=["close", "open", "high", "low", "volume", "timestamp"], errors="ignore")
    return feature_only, target.astype(int)
