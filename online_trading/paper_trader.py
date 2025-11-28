from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from loguru import logger

from models.policy_base import PolicyConfig, load_policy, select_action
from online_trading.replay_buffer import ReplayBuffer, Transition
from online_trading.feature_updater import FeatureUpdater
from online_trading.live_feed import build_feed


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
    ckpt = Path(cfg["paths"]["best_policy"])
    model = None
    policy_cfg: Optional[PolicyConfig] = None
    trader = PaperTrader(cfg, None, device)  # model is set lazily

    feed = build_feed(
        cfg["live_feed"]["source"],
        cfg["live_feed"]["symbol"],
        cfg["live_feed"]["websocket_url"],
        interval_sec=cfg["live_feed"].get("rest_poll_seconds", 60),
    )
    updater = FeatureUpdater()
    buffer = ReplayBuffer(Path(cfg["paths"]["replay_buffer"]), max_size=cfg["online_training"]["max_buffer_size"])

    prev_features = None
    prev_equity = trader.state.equity(0)

    async for candle in feed.stream():
        features = updater.update(candle)
        if features is None:
            continue
        if model is None:
            policy_cfg = PolicyConfig(
                input_dim=len(features),
                hidden_sizes=cfg["model"]["hidden_sizes"],
                activation=cfg["model"]["activation"],
                action_space=cfg["model"].get("action_space", "discrete"),
            )
            model = load_policy(ckpt, policy_cfg, device=device)
            trader.model = model
        feat_tensor = torch.tensor(features, dtype=torch.float32)
        action, confidence = select_action(model, feat_tensor, device=device)
        state = trader.step(action, candle.close)
        equity = state.equity(candle.close)
        reward = equity - prev_equity

        if prev_features is not None:
            buffer.add(
                Transition(
                    state=prev_features.tolist(),
                    action=action,
                    reward=reward,
                    next_state=features.tolist(),
                    timestamp=candle.ts,
                )
            )
            buffer.save()

        prev_features = features
        prev_equity = equity

        logger.info(
            {
                "ts": candle.ts,
                "price": candle.close,
                "action": action,
                "confidence_or_size": confidence,
                "balance": state.balance_usdt,
                "unrealized": state.unrealized_pnl(candle.close),
                "equity": equity,
                "position": state.position_size,
            }
        )


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
