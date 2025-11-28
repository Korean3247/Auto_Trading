from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import numpy as np
import websockets


@dataclass
class Candle:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class BinanceFeed:
    def __init__(self, symbol: str, url: str):
        self.symbol = symbol.lower()
        self.url = url

    async def stream(self) -> AsyncIterator[Candle]:
        stream_name = f"{self.symbol}@kline_1m"
        while True:
            try:
                async with websockets.connect(f"{self.url}/{stream_name}") as ws:
                    async for msg in ws:
                        data = json.loads(msg)
                        k = data.get("k", {})
                        if not k or not k.get("x"):  # only closed candles
                            continue
                        yield Candle(
                            ts=int(k["t"]) // 1000,
                            open=float(k["o"]),
                            high=float(k["h"]),
                            low=float(k["l"]),
                            close=float(k["c"]),
                            volume=float(k["v"]),
                        )
            except Exception:
                await asyncio.sleep(1)
                continue


class SimulatedFeed:
    def __init__(self, base_price: float = 30000.0, interval_sec: int = 1):
        self.price = base_price
        self.interval_sec = interval_sec

    async def stream(self) -> AsyncIterator[Candle]:
        while True:
            move = random.gauss(0, 10)
            high = self.price + abs(random.gauss(0, 5))
            low = self.price - abs(random.gauss(0, 5))
            close = self.price + move
            volume = abs(random.gauss(1000, 200))
            candle = Candle(
                ts=int(time.time()),
                open=self.price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
            self.price = close
            yield candle
            await asyncio.sleep(self.interval_sec)


def build_feed(source: str, symbol: str, websocket_url: str, interval_sec: int = 60) -> object:
    if source == "binance":
        return BinanceFeed(symbol, websocket_url)
    return SimulatedFeed(interval_sec=interval_sec)
