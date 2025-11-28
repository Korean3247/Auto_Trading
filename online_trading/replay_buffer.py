from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Transition:
    state: List[float]
    action: int
    reward: float
    next_state: List[float]
    timestamp: int


class ReplayBuffer:
    def __init__(self, path: Path, max_size: int = 200000):
        self.path = path
        self.max_size = max_size
        self.buffer: List[Transition] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        df = pd.read_parquet(self.path)
        self.buffer = [
            Transition(
                state=row["state"],
                action=int(row["action"]),
                reward=float(row["reward"]),
                next_state=row["next_state"],
                timestamp=int(row["timestamp"]),
            )
            for _, row in df.iterrows()
        ]

    def _maybe_trim(self) -> None:
        if len(self.buffer) > self.max_size:
            excess = len(self.buffer) - self.max_size
            self.buffer = self.buffer[excess:]

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)
        self._maybe_trim()

    def sample(self, batch_size: int) -> Optional[List[Transition]]:
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [
                {
                    "state": t.state,
                    "action": t.action,
                    "reward": t.reward,
                    "next_state": t.next_state,
                    "timestamp": t.timestamp,
                }
                for t in self.buffer
            ]
        )
        df.to_parquet(self.path, index=False)
