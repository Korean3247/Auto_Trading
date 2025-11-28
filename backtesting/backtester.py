from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from envs.trading_env import TradingEnv
from offline_training.dataset import load_market_data
from offline_training.feature_engineering import build_feature_matrix
from risk.risk_manager import RiskManager
from strategies.rl_policy import RLStrategy
from strategies.rule_based import MovingAverageCrossStrategy


def build_strategy(cfg: dict, input_dim: int, ckpt_path: Path) -> object:
    stype = cfg["strategy"]["type"]
    if stype == "rl":
        return RLStrategy(ckpt_path, cfg, input_dim_override=input_dim, greedy=True)
    rb = cfg["strategy"]["rule_based"]
    return MovingAverageCrossStrategy(rb.get("fast_idx", 0), rb.get("slow_idx", 1), rb.get("threshold", 0.0))


@dataclass
class BacktestResult:
    equity_curve: List[float]
    actions: List[int]
    timestamps: List[Optional[object]]
    trades: List[Dict]
    metrics: Dict


class Backtester:
    def __init__(self, cfg: dict, start: Optional[pd.Timestamp] = None, end: Optional[pd.Timestamp] = None):
        self.cfg = cfg
        df = load_market_data(Path(cfg["paths"]["data"]), cfg)
        if start:
            df = df[df.index >= start]
        if end:
            df = df[df.index <= end]
        feats = build_feature_matrix(df.reset_index(drop=False).rename(columns={"index": "timestamp"}))
        prices = df.loc[feats.index, "close"].to_numpy()
        features = feats.drop(columns=["close", "open", "high", "low", "volume", "timestamp"], errors="ignore").to_numpy()
        timestamps = df.loc[feats.index].index.to_numpy()
        self.env = TradingEnv(prices, features, cfg, timestamps=timestamps)
        self.strategy = build_strategy(cfg, features.shape[1], Path(cfg["paths"].get("best_rl_policy", "")))
        self.risk = RiskManager(cfg)

    def run(self) -> BacktestResult:
        obs = self.env.reset(start_idx=0)
        self.strategy.reset()
        equity_curve = []
        actions = []
        timestamps = []
        trades: List[Dict] = []
        prev_equity = self.env.equity
        step = 0
        while True:
            raw_action = self.strategy.act(obs)
            safe_action = self.risk.check_action(raw_action, self.env.equity, {}, {"ts": self.env.timestamps[self.env.current_step] if hasattr(self.env, "timestamps") else None, "price": self.env.prices[self.env.current_step]})
            next_obs, reward, done, info = self.env.step(safe_action)
            equity_curve.append(info["equity"])
            actions.append(safe_action)
            timestamps.append(info.get("ts"))
            trades.append(
                {
                    "step": step,
                    "ts": info.get("ts"),
                    "action": safe_action,
                    "equity": info["equity"],
                    "pnl_step": info["equity"] - prev_equity,
                    "position": info["position"],
                }
            )
            prev_equity = info["equity"]
            obs = next_obs
            step += 1
            if done:
                break

        equity_arr = np.array(equity_curve)
        pnl = np.diff(np.insert(equity_arr, 0, self.env.initial_equity))
        returns = pnl / np.maximum(np.insert(equity_arr[:-1], 0, self.env.initial_equity), 1e-8)
        sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)) if len(returns) > 1 else 0.0
        max_dd = float(np.max(np.maximum.accumulate(equity_arr) - equity_arr)) if len(equity_arr) else 0.0
        metrics = {"final_equity": float(equity_arr[-1]) if len(equity_arr) else 0.0, "sharpe": sharpe, "max_drawdown": max_dd}
        return BacktestResult(equity_curve, actions, timestamps, trades, metrics)


def main(config_path: str, start: Optional[str], end: Optional[str]) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    res = Backtester(cfg, start_ts, end_ts).run()
    logger.info(res.metrics)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    args = parser.parse_args()
    main(args.config, args.start, args.end)
