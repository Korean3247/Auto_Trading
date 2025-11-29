from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from loguru import logger

from models.policy_base import PolicyConfig, load_policy
from offline_training.dataset import load_price_and_features


def load_policy_with_infer(cfg: dict, ckpt: Path, input_dim: int) -> torch.nn.Module:
    policy_cfg = PolicyConfig(
        input_dim=input_dim,
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    return load_policy(ckpt, policy_cfg, device=torch.device("cpu"))


def evaluate_policy(cfg: dict, ckpt_path: Path, force_action: str | None = None) -> Dict:
    prices, features = load_price_and_features(Path(cfg["paths"]["data"]), cfg)
    if len(prices) < 2:
        raise RuntimeError("Not enough data for evaluation.")

    # Infer input_dim from checkpoint if config left null/0
    dummy_cfg = PolicyConfig(
        input_dim=features.shape[1],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    policy = load_policy(ckpt_path, dummy_cfg, device=torch.device("cpu"))
    policy.eval()

    initial_equity = cfg["paper_trading"]["starting_balance"]
    fee = cfg["rl"].get("fee_rate", cfg["paper_trading"]["fee_taker"])
    _ = cfg["rl"].get("slippage", cfg["paper_trading"]["slippage_bps"] / 10000)  # slippage placeholder

    equity = initial_equity
    max_equity = initial_equity
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    trades = 0
    wins = 0
    action_counts = {0: 0, 1: 0, 2: 0}
    equity_curve = []

    for t in range(1, len(prices)):
        obs = torch.tensor(features[t - 1], dtype=torch.float32).unsqueeze(0)
        if force_action:
            action = {"short": 0, "flat": 1, "long": 2}[force_action]
        else:
            with torch.no_grad():
                logits = policy(obs)
                action = int(torch.argmax(logits, dim=-1).item())
        action_counts[action] = action_counts.get(action, 0) + 1

        target_pos = {0: -1, 1: 0, 2: 1}.get(action, 0)  # 0=short,1=flat,2=long
        price_prev = prices[t - 1]
        price_now = prices[t]
        ret = (price_now - price_prev) / max(price_prev, 1e-8)

        # PnL from holding previous position
        pnl_hold = position * ret * equity

        # If position change, apply fee and reset entry
        fee_cost = 0.0
        if target_pos != position:
            trades += 1
            trade_notional = abs(target_pos - position) * price_now
            fee_cost = trade_notional * fee
            if position != 0:
                # Evaluate win/loss on closing leg
                if pnl_hold > 0:
                    wins += 1
            entry_price = price_now
        equity = equity + pnl_hold - fee_cost
        position = target_pos
        max_equity = max(max_equity, equity)
        equity_curve.append(equity)

    equity_arr = np.array(equity_curve)
    max_dd = float(np.max(np.maximum.accumulate(equity_arr) - equity_arr)) if len(equity_arr) else 0.0
    total_return = (equity - initial_equity) / initial_equity
    win_rate = wins / trades if trades > 0 else 0.0

    summary = {
        "initial_equity": initial_equity,
        "final_equity": equity,
        "total_return_pct": total_return * 100,
        "max_drawdown": max_dd,
        "total_trades": trades,
        "win_rate": win_rate,
        "action_distribution": action_counts,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL policy on historical data.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--force_action", type=str, choices=["short", "flat", "long"], help="Override policy with constant action to sanity-check environment/equity updates.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(args.checkpoint)
    summary = evaluate_policy(cfg, ckpt, force_action=args.force_action)

    logger.info("=== RL Policy Evaluation ===")
    logger.info(f"Initial equity : {summary['initial_equity']:.2f}")
    logger.info(f"Final equity   : {summary['final_equity']:.2f}")
    logger.info(f"Total return   : {summary['total_return_pct']:.2f}%")
    logger.info(f"Max drawdown   : {summary['max_drawdown']:.4f}")
    logger.info(f"Total trades   : {summary['total_trades']}")
    logger.info(f"Win rate       : {summary['win_rate']*100:.2f}%")
    logger.info(f"Action counts  : {summary['action_distribution']}")

    out_path = Path(cfg["paths"]["log_dir"]) / "rl_eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
