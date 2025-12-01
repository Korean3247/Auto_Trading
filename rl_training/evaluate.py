from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from loguru import logger

from envs.trading_env import TradingEnv
from models.policy_base import PolicyConfig, load_policy
from offline_training.dataset import load_price_and_features


def resolve_force_action(action_map: list[float], name: str) -> int:
    if name == "short":
        return int(np.argmin(action_map))
    if name == "long":
        return int(np.argmax(action_map))
    # flat -> closest to zero exposure
    return int(np.argmin(np.abs(np.array(action_map))))


def evaluate_policy(cfg: dict, ckpt_path: Path, force_action: Optional[str] = None, sample: bool = False, temperature: float = 1.0) -> Dict:
    prices, features = load_price_and_features(Path(cfg["paths"]["data"]), cfg)
    if len(prices) < 2:
        raise RuntimeError("Not enough data for evaluation.")

    action_map = cfg["rl"].get("action_mapping", [-1.0, 0.0, 1.0])
    policy_cfg = PolicyConfig(
        input_dim=features.shape[1],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
        action_dim=len(action_map),
    )
    policy = load_policy(ckpt_path, policy_cfg, device=torch.device("cpu"))
    policy.eval()

    env_cfg = cfg.copy()
    env_cfg["rl"]["max_steps_per_episode"] = len(prices)
    env = TradingEnv(prices, features, env_cfg)
    obs = env.reset(start_idx=0)

    action_counts = {i: 0 for i in range(len(action_map))}
    trades = 0
    wins = 0
    prev_pos = 0.0
    equity_curve = []
    returns = []
    bench_returns = []

    while True:
        if force_action:
            action_idx = resolve_force_action(action_map, force_action)
        else:
            with torch.no_grad():
                logits = policy(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                if sample:
                    probs = torch.softmax(logits / max(temperature, 1e-4), dim=-1).squeeze(0).cpu().numpy()
                    action_idx = int(np.random.choice(len(probs), p=probs))
                else:
                    action_idx = int(torch.argmax(logits, dim=-1).item())
        action_counts[action_idx] = action_counts.get(action_idx, 0) + 1

        next_obs, reward, done, info = env.step(action_idx)
        equity_curve.append(info["equity"])
        returns.append(info.get("policy_ret", reward))
        bench_returns.append(info.get("bench_ret", 0.0))

        if abs(info["position"] - prev_pos) > 1e-9:
            trades += 1
            if info.get("pnl", 0.0) > 0:
                wins += 1
        prev_pos = info["position"]
        obs = next_obs
        if done:
            break

    equity_arr = np.array(equity_curve)
    max_dd = float(np.max(np.maximum.accumulate(equity_arr) - equity_arr)) if len(equity_arr) else 0.0
    total_return = (equity_arr[-1] - equity_arr[0]) / equity_arr[0] if len(equity_arr) else 0.0
    win_rate = wins / trades if trades > 0 else 0.0
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(len(returns))) if returns else 0.0
    bench_ret_total = float(np.prod(np.array(bench_returns) + 1) - 1) if bench_returns else 0.0
    excess_ret_total = float(np.prod(np.array(returns) + 1) - 1) - bench_ret_total if returns else 0.0

    summary = {
        "initial_equity": float(equity_arr[0]) if len(equity_arr) else cfg["paper_trading"]["starting_balance"],
        "final_equity": float(equity_arr[-1]) if len(equity_arr) else cfg["paper_trading"]["starting_balance"],
        "total_return_pct": total_return * 100,
        "max_drawdown": max_dd,
        "total_trades": trades,
        "win_rate": win_rate,
        "sharpe": sharpe,
        "action_distribution": action_counts,
        "benchmark_return": bench_ret_total,
        "excess_return": excess_ret_total,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RL policy on historical data.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--force_action",
        type=str,
        choices=["short", "flat", "long"],
        help="Override policy with constant action to sanity-check environment/equity updates.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Use stochastic sampling (softmax) for actions instead of greedy argmax.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature when --sample is set (higher → more random).",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    ckpt = Path(args.checkpoint)
    summary = evaluate_policy(cfg, ckpt, force_action=args.force_action, sample=args.sample, temperature=args.temperature)

    logger.info("=== RL Policy Evaluation ===")
    logger.info(f"Initial equity : {summary['initial_equity']:.2f}")
    logger.info(f"Final equity   : {summary['final_equity']:.2f}")
    logger.info(f"Total return   : {summary['total_return_pct']:.2f}%")
    logger.info(f"Max drawdown   : {summary['max_drawdown']:.4f}")
    logger.info(f"Total trades   : {summary['total_trades']}")
    logger.info(f"Win rate       : {summary['win_rate']*100:.2f}%")
    logger.info(f"Sharpe         : {summary['sharpe']:.4f}")
    logger.info(f"Action counts  : {summary['action_distribution']}")

    out_path = Path(cfg["paths"]["log_dir"]) / "rl_eval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
