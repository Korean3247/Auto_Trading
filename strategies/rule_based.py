from __future__ import annotations

import numpy as np

from strategies.base import Strategy


class MovingAverageCrossStrategy(Strategy):
    """
    Simple rule-based strategy on two feature indices: fast vs slow signal.
    """

    def __init__(self, fast_idx: int, slow_idx: int, threshold: float = 0.0):
        self.fast_idx = fast_idx
        self.slow_idx = slow_idx
        self.threshold = threshold

    def reset(self) -> None:
        return None

    def act(self, obs) -> int:
        if obs is None or len(obs) <= max(self.fast_idx, self.slow_idx):
            return 0
        fast = obs[self.fast_idx]
        slow = obs[self.slow_idx]
        if fast - slow > self.threshold:
            return 1
        if slow - fast > self.threshold:
            return 2
        return 0
