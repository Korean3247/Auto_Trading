from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
from loguru import logger


def _load_from_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_history(
    symbol: str,
    start: datetime,
    end: datetime,
    timeframe: str = "1m",
    source: str = "binance",
    local_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load historical OHLCV from exchange or local file.
    """
    if local_path:
        df = _load_from_csv(local_path)
        if df is not None:
            return df
    if source != "binance":
        raise ValueError(f"Unsupported source {source}")
    exchange = ccxt.binance(
        {
            "enableRateLimit": True,
        }
    )
    since = int(start.timestamp() * 1000)
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
        if batch[-1][0] >= int(end.timestamp() * 1000):
            break
    if not rows:
        raise RuntimeError("No OHLCV fetched.")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    logger.info(f"Fetched {len(df)} rows of OHLCV for {symbol}")
    return df
