from __future__ import annotations

import math
import random
from typing import Dict, Optional, Tuple

import numpy as np


class TradingEnv:
    """
    Minimal gym-style trading env using precomputed features and close prices.
    Actions: 0=flat, 1=long, 2=short
    Reward: delta equity (return) per step.
    """

    def __init__(
        self,
        price_array: np.ndarray,
        feature_array: np.ndarray,
        config: Dict,
        timestamps: Optional[np.ndarray] = None,
    ):
        assert len(price_array) == len(feature_array), "price and feature lengths must match"
        self.prices = price_array
        self.features = feature_array
        self.cfg = config
        self.timestamps = timestamps
        self.max_steps = config["rl"]["max_steps_per_episode"]
        self.initial_equity = config["rl"]["initial_equity"]
        self.fee_rate = config["rl"]["fee_rate"]
        self.slippage = config["rl"]["slippage"]
        self.reward_scale = config["rl"].get("reward_scale", 1.0)
        self.min_equity = config["rl"].get("min_equity", 1e-6)

        self.current_step = 0
        self.steps_in_episode = 0
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price = 0.0
        self.equity = self.initial_equity

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        if start_idx is None:
            upper = max(0, len(self.prices) - self.max_steps - 2)
            start_idx = random.randint(0, upper) if upper > 0 else 0
        self.current_step = start_idx
        self.steps_in_episode = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = self.initial_equity
        return self.features[self.current_step].astype(np.float32)

    def _apply_action(self, action: int, price: float) -> Tuple[float, float]:
        """
        Update position and compute transaction cost. Returns (fee_cost, new_entry_price).
        """
        target_pos = {0: -1, 1: 0, 2: 1}.get(action, 0)  # 0=short,1=flat,2=long
        fee_cost = 0.0
        new_entry = self.entry_price
        if target_pos != self.position:
            trade_notional = abs(target_pos - self.position) * price
            fee_cost = trade_notional * self.fee_rate
            new_entry = price if target_pos != 0 else 0.0
        self.position = target_pos
        return fee_cost, new_entry

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_step >= len(self.prices) - 2:
            return self.features[self.current_step].astype(np.float32), 0.0, True, {"reason": "data_end"}
        price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]

        # PnL from holding the previous position across the price move
        pnl = (next_price - price) * self.position

        fee_cost, new_entry = self._apply_action(action, next_price * (1 + math.copysign(self.slippage, action - 1)))
        prev_equity = self.equity
        self.equity += pnl - fee_cost
        self.entry_price = new_entry

        reward = self.reward_scale * (self.equity - prev_equity) / max(prev_equity, 1e-8)

        self.current_step += 1
        self.steps_in_episode += 1
        done = (
            self.steps_in_episode >= self.max_steps
            or self.current_step >= len(self.prices) - 2
            or self.equity <= self.min_equity
        )
        info = {
            "equity": self.equity,
            "pnl": pnl,
            "fee": fee_cost,
            "position": self.position,
            "step": self.current_step,
            "ts": self.timestamps[self.current_step] if self.timestamps is not None else None,
        }
        return self.features[self.current_step].astype(np.float32), float(reward), bool(done), info
