from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from loguru import logger

from models.policy_base import PolicyConfig, load_policy
from offline_training.dataset import load_market_data
from offline_training.feature_engineering import build_feature_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple backtest for policy model.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_backtest(cfg: dict, ckpt: Path) -> dict:
    df = load_market_data(Path(cfg["paths"]["data"]))
    feats = build_feature_matrix(df)
    returns = feats["close"].pct_change().shift(-1).fillna(0).to_numpy()
    feature_matrix = feats.drop(columns=["close", "open", "high", "low", "volume"], errors="ignore").to_numpy()

    policy_cfg = PolicyConfig(
        input_dim=feature_matrix.shape[1],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy(ckpt, policy_cfg, device=device)

    actions = []
    with torch.no_grad():
        x = torch.tensor(feature_matrix, dtype=torch.float32).to(device)
        logits = model(x)
        actions = torch.argmax(logits, dim=-1).cpu().numpy()

    # Map actions to positions: 0 short,1 flat,2 long
    pos = np.where(actions == 0, -1, np.where(actions == 2, 1, 0))
    pnl = pos * returns
    equity = (1 + pnl).cumprod()
    sharpe = np.mean(pnl) / (np.std(pnl) + 1e-8) * np.sqrt(1440)  # per minute to daily-ish
    max_dd = np.max(np.maximum.accumulate(equity) - equity)
    return {"sharpe": float(sharpe), "max_drawdown": float(max_dd), "final_equity": float(equity[-1])}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    ckpt = Path(args.checkpoint or cfg["paths"]["best_policy"])
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        return
    metrics = run_backtest(cfg, ckpt)
    logger.info(metrics)


if __name__ == "__main__":
    main()
