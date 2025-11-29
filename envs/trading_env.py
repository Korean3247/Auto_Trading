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
        self.max_steps = config["rl"].get("max_steps_per_episode") or config["rl"]["episode_length"]
        self.initial_equity = config["rl"]["initial_equity"]
        self.reward_scale = config["rl"].get("reward_scale", 1.0)
        self.min_equity = config["rl"].get("min_equity", 1e-6)

        self.action_mapping = config["rl"].get("action_mapping", [-1.0, 0.0, 1.0])
        self.comm = config.get("cost", {}).get("commission_bps", 0.0) / 1e4
        self.slip_bps = config.get("cost", {}).get("slippage_bps", 0.0) / 1e4
        self.spread_bps = config.get("cost", {}).get("spread_bps", 0.0) / 1e4

        self.current_step = 0
        self.steps_in_episode = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.equity = self.initial_equity
        self.bench_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.returns_window: list[float] = []

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        if start_idx is None:
            min_idx = self.cfg["rl"].get("min_start_index", 0)
            max_idx = self.cfg["rl"].get("max_start_index", len(self.prices) - self.max_steps - 2)
            start_idx = random.randint(min_idx, max_idx) if max_idx > min_idx else min_idx
        self.current_step = start_idx
        self.steps_in_episode = 0
        self.position = 0.0
        self.entry_price = 0.0
        self.equity = self.initial_equity
        self.bench_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.returns_window = []
        return self.features[self.current_step].astype(np.float32)

    def _apply_action(self, action: int, price: float) -> Tuple[float, float]:
        """
        Update position and compute transaction cost. Returns (fee_cost, new_entry_price).
        """
        target_pos = self.action_mapping[action] if action < len(self.action_mapping) else 0.0
        fee_cost = 0.0
        new_entry = self.entry_price
        if target_pos != self.position:
            trade_notional = abs(target_pos - self.position) * price
            fee_cost = trade_notional * self.comm
            new_entry = price * (1 + self.spread_bps * np.sign(target_pos)) if target_pos != 0 else 0.0
        self.position = target_pos
        return fee_cost, new_entry

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.current_step >= len(self.prices) - 2:
            return self.features[self.current_step].astype(np.float32), 0.0, True, {"reason": "data_end"}
        price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]

        # PnL from holding the previous position across the price move
        pnl = (next_price - price) * self.position

        # Execution cost
        exec_price = next_price * (1 + self.slip_bps * np.sign(action - 2))
        fee_cost, new_entry = self._apply_action(action, exec_price)
        prev_equity = self.equity
        self.equity += pnl - fee_cost
        self.entry_price = new_entry

        # Benchmark equity (always full long)
        bench_ret = (next_price - price) / max(price, 1e-8)
        self.bench_equity *= (1 + bench_ret)
        # Excess return reward
        policy_ret = (self.equity - prev_equity) / max(prev_equity, 1e-8)
        reward = policy_ret - self.cfg["rl"].get("excess_return_alpha", 1.0) * bench_ret

        # Risk penalties
        self.peak_equity = max(self.peak_equity, self.equity)
        dd = (self.peak_equity - self.equity) / max(self.peak_equity, 1e-8)
        dd_threshold = self.cfg["rl"].get("dd_threshold", 0.2)
        dd_penalty = self.cfg["rl"].get("dd_penalty_coeff", 0.0) * max(0.0, dd - dd_threshold)

        ret = policy_ret
        self.returns_window.append(ret)
        if len(self.returns_window) > self.cfg["rl"].get("vol_window", 50):
            self.returns_window.pop(0)
        vol_penalty = 0.0
        if self.returns_window:
            vol_penalty = self.cfg["rl"].get("vol_penalty_coeff", 0.0) * float(np.std(self.returns_window))

        action_mapping = self.cfg["rl"].get("action_mapping", [-1.0, 0.0, 1.0])
        target_pos = action_mapping[action] if action < len(action_mapping) else 0.0
        turnover_penalty = self.cfg["rl"].get("turnover_penalty_coeff", 0.0) * abs(self.position - target_pos)
        exposure_penalty = self.cfg["rl"].get("exposure_penalty_coeff", 0.0) * abs(self.position)

        reward = self.reward_scale * (reward - dd_penalty - vol_penalty - turnover_penalty - exposure_penalty)

        self.current_step += 1
        self.steps_in_episode += 1
        done = (
            self.steps_in_episode >= self.cfg["rl"].get("episode_length", self.max_steps)
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
            "bench_equity": self.bench_equity,
            "drawdown": dd,
        }
        return self.features[self.current_step].astype(np.float32), float(reward), bool(done), info
