from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from loguru import logger

from models.policy_base import PolicyConfig, PolicyMLP, save_checkpoint
from online_trading.replay_buffer import ReplayBuffer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online fine-tuning from replay buffer.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def anchor_loss(model: PolicyMLP, anchor_state: dict, coef: float) -> torch.Tensor:
    loss = 0.0
    for name, param in model.named_parameters():
        loss = loss + F.mse_loss(param, anchor_state[name]) * coef
    return loss


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(Path(cfg["paths"]["replay_buffer"]), max_size=cfg["online_training"]["max_buffer_size"])
    if len(buffer) < cfg["online_training"]["batch_size"]:
        logger.error("Replay buffer too small; run paper trader first.")
        return

    sample_state = buffer.buffer[0].state
    policy_cfg = PolicyConfig(
        input_dim=len(sample_state),
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    ckpt = Path(args.checkpoint or cfg["paths"]["best_policy"])
    model = PolicyMLP(
        policy_cfg.input_dim,
        policy_cfg.hidden_sizes,
        policy_cfg.activation,
        policy_cfg.action_space,
    )
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        logger.info(f"Loaded checkpoint {ckpt}")
    model.to(device)

    anchor_state = {k: v.detach().clone().to(device) for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["online_training"]["learning_rate"],
        weight_decay=cfg["online_training"]["weight_decay"],
    )

    batch = buffer.sample(cfg["online_training"]["batch_size"])
    states = torch.tensor([t.state for t in batch], dtype=torch.float32, device=device)
    actions = torch.tensor([t.action for t in batch], dtype=torch.long, device=device)

    model.train()
    logits = model(states)
    loss = F.cross_entropy(logits, actions)
    loss = loss + anchor_loss(model, anchor_state, cfg["online_training"]["weight_anchor_coef"])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    version_name = datetime.utcnow().strftime("policy_v1_%Y%m%d_%H%M%S.pt")
    out_path = Path(cfg["paths"]["checkpoints_dir"]) / version_name
    save_checkpoint(model, out_path)
    logger.info(f"Saved online-updated checkpoint to {out_path}")


if __name__ == "__main__":
    main()
