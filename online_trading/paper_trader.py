from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger

from strategies.rl_policy import RLStrategy
from strategies.rule_based import MovingAverageCrossStrategy
from strategies.base import Strategy
from risk.risk_manager import RiskManager
from online_trading.replay_buffer import ReplayBuffer, Transition
from online_trading.feature_updater import FeatureUpdater
from online_trading.live_feed import build_feed
from online_trading.execution_engine import ExecutionEngine


@dataclass
class AccountState:
    balance_usdt: float
    position_size: float = 0.0  # in BTC
    entry_price: float = 0.0
    realized_pnl: float = 0.0

    def equity(self, mark_price: float) -> float:
        return self.balance_usdt + self.unrealized_pnl(mark_price)

    def unrealized_pnl(self, mark_price: float) -> float:
        return self.position_size * (mark_price - self.entry_price)


class PaperTrader:
    def __init__(self, cfg: dict, model, device: torch.device):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.state = AccountState(balance_usdt=cfg["paper_trading"]["starting_balance"])
        self.max_position_frac = cfg["paper_trading"]["max_position_size"]
        self.fee_taker = cfg["paper_trading"]["fee_taker"]
        self.slippage_bps = cfg["paper_trading"]["slippage_bps"]

    def _target_qty(self, action: int, price: float) -> float:
        equity = self.state.equity(price)
        frac = {0: -self.max_position_frac, 1: 0.0, 2: self.max_position_frac}.get(action, 0.0)
        target_notional = equity * frac
        return target_notional / price

    def step(self, action: int, price: float) -> AccountState:
        slipped_price = price * (1 + np.sign(action - 1) * self.slippage_bps / 10000)
        old_size = self.state.position_size
        target_size = self._target_qty(action, slipped_price)

        # Close existing position and realize PnL
        if old_size != 0.0:
            pnl_close = old_size * (slipped_price - self.state.entry_price)
            self.state.balance_usdt += pnl_close
            self.state.realized_pnl += pnl_close

        trade_notional = abs(target_size - old_size) * slipped_price
        fee = trade_notional * self.fee_taker
        self.state.balance_usdt -= fee

        self.state.position_size = target_size
        self.state.entry_price = slipped_price if target_size != 0 else 0.0
        return self.state


async def run_loop(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_rl = Path(cfg["paths"].get("best_rl_policy", cfg["paths"]["best_policy"]))
    ckpt = ckpt_rl if ckpt_rl.exists() else Path(cfg["paths"]["best_policy"])
    strategy: Optional[Strategy] = None
    risk = RiskManager(cfg)
    exec_engine = ExecutionEngine(cfg["execution"]["mode"], cfg, model=None, device=device)

    feed = build_feed(
        cfg["live_feed"]["source"],
        cfg["live_feed"]["symbol"],
        cfg["live_feed"]["websocket_url"],
        interval_sec=cfg["live_feed"].get("rest_poll_seconds", 60),
    )
    updater = FeatureUpdater()
    buffer = ReplayBuffer(Path(cfg["paths"]["replay_buffer"]), max_size=cfg["online_training"]["max_buffer_size"])
    log_path = Path(cfg["paths"]["log_dir"]) / "live" / "trades.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    prev_features = None
    prev_equity = exec_engine.paper.state.equity(0) if exec_engine.paper else 0.0

    async for candle in feed.stream():
        features = updater.update(candle)
        if features is None:
            continue
        if strategy is None:
            if cfg["strategy"]["type"] == "rl":
                strategy = RLStrategy(ckpt, cfg, device=device, input_dim_override=len(features), greedy=True)
            else:
                rb = cfg["strategy"]["rule_based"]
                strategy = MovingAverageCrossStrategy(rb.get("fast_idx", 0), rb.get("slow_idx", 1), rb.get("threshold", 0.0))
            strategy.reset()
        action = strategy.act(features)
        safe_action = risk.check_action(
            action,
            prev_equity,
            exec_engine.get_position(),
            {"ts": candle.ts, "price": candle.close},
        )
        exec_result = exec_engine.execute_action(safe_action, candle.close, ts=candle.ts)
        equity = exec_result.get("equity", prev_equity)
        reward = equity - prev_equity

        if prev_features is not None:
            buffer.add(
                Transition(
                    state=prev_features.tolist(),
                    action=safe_action,
                    reward=reward,
                    next_state=features.tolist(),
                    timestamp=candle.ts,
                )
            )
            buffer.save()

        prev_features = features
        prev_equity = equity

        log_row = {
            "ts": candle.ts,
            "price": candle.close,
            "action": safe_action,
            "equity": equity,
            "position": exec_result.get("position_size", 0.0),
            "balance": exec_result.get("balance", 0.0),
            "pnl_step": reward,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(log_row) + "\n")
        logger.info(log_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time paper trader loop.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    try:
        asyncio.run(run_loop(cfg))
    except KeyboardInterrupt:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()
