from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import ccxt
import pandas as pd
from ccxt.base.errors import ExchangeError, ExchangeNotAvailable
from loguru import logger


def _load_local(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _create_exchange(exchange_id: str):
    if not hasattr(ccxt, exchange_id):
        raise RuntimeError(f"Exchange {exchange_id} not found in ccxt")
    klass = getattr(ccxt, exchange_id)
    return klass({"enableRateLimit": True})


def _fetch_ohlcv(exchange, symbol: str, timeframe: str, start_ms: int, max_bars: int) -> list:
    rows = []
    since = start_ms
    while len(rows) < max_bars:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=min(1000, max_bars - len(rows)))
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + 1
    return rows


def download_ohlcv_to_parquet(cfg: dict) -> Path:
    ds = cfg["data_source"]
    out_path = Path(cfg["paths"]["data"])
    exchanges: Sequence[str] = [ds["exchange_id"]] + ds.get("fallback_exchanges", [])
    last_error: Optional[Exception] = None

    for ex_id in exchanges:
        try:
            exchange = _create_exchange(ex_id)
            start_ms = int(pd.Timestamp(ds["start_iso"]).timestamp() * 1000)
            rows = _fetch_ohlcv(exchange, ds["symbol"], ds["timeframe"], start_ms, ds["max_bars"])
            if not rows:
                raise RuntimeError(f"No OHLCV fetched from {ex_id}")
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            logger.info(f"[data] Downloaded {len(df)} rows from {ex_id} -> {out_path}")
            return out_path
        except (ExchangeNotAvailable, ExchangeError, RuntimeError) as e:
            last_error = e
            logger.error(f"[data] Exchange {ex_id} unavailable ({e}); trying fallback if any.")
            continue
    raise RuntimeError(f"Failed to download OHLCV from exchanges {exchanges}: {last_error}")


def load_ohlcv(cfg: dict) -> pd.DataFrame:
    path = Path(cfg["paths"]["data"])
    ds = cfg["data_source"]
    allow_synth = ds.get("allow_synthetic_fallback", False)
    auto_dl = ds.get("auto_download_on_missing", False)

    df = _load_local(path)
    if df is not None:
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp", drop=False)
        return df

    if auto_dl:
        downloaded = download_ohlcv_to_parquet(cfg)
        df = _load_local(downloaded)
        if df is not None:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                df = df.set_index("timestamp", drop=False)
            return df

    if allow_synth:
        logger.warning("[data] Using synthetic data because no real OHLCV available and fallback allowed.")
        import numpy as np

        rng = np.random.default_rng(seed=42)
        n = 5000
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
        df["timestamp"] = pd.date_range(start="2020-01-01", periods=len(df), freq="T", tz="UTC")
        df = df.set_index("timestamp", drop=False)
        return df

    raise RuntimeError(
        "[data] No OHLCV data found and auto_download_on_missing=False or download failed. "
        "Provide historical parquet or enable download/fallback in config."
    )
