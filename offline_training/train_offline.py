from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.utils.data import DataLoader

from models.policy_base import PolicyConfig, PolicyMLP, save_checkpoint
from offline_training.dataset import build_datasets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline training for BTC futures policy.")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--output", type=str, default=None, help="Override checkpoint output path")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    total = 0
    correct = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        pred = logits.argmax(dim=-1)
        correct += (pred == batch_y).sum().item()
        total += len(batch_x)
    return {"accuracy": correct / max(total, 1)}


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))

    set_seed(cfg["offline_training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, val_set, test_set = build_datasets(
        Path(cfg["paths"]["data"]),
        train_split=cfg["offline_training"]["train_split"],
        val_split=cfg["offline_training"]["val_split"],
    )
    input_dim = train_set.features.shape[1]
    policy_cfg = PolicyConfig(
        input_dim=input_dim,
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    model = PolicyMLP(
        input_dim=policy_cfg.input_dim,
        hidden_sizes=policy_cfg.hidden_sizes,
        activation=policy_cfg.activation,
        action_space=policy_cfg.action_space,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["offline_training"]["learning_rate"],
        weight_decay=cfg["offline_training"]["weight_decay"],
    )
    train_loader = DataLoader(train_set, batch_size=cfg["offline_training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg["offline_training"]["batch_size"])
    test_loader = DataLoader(test_set, batch_size=cfg["offline_training"]["batch_size"])

    best_val = 0.0
    best_path = Path(args.output or cfg["paths"]["best_policy"])
    best_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting offline training on {device} for {cfg['offline_training']['epochs']} epochs")
    for epoch in range(cfg["offline_training"]["epochs"]):
        loss = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch+1}: train_loss={loss:.4f}, val_acc={metrics['accuracy']:.4f}")
        if metrics["accuracy"] >= best_val:
            best_val = metrics["accuracy"]
            save_checkpoint(model, best_path)
            logger.info(f"Saved new best checkpoint to {best_path}")

    final_metrics = evaluate(model, test_loader, device)
    logger.info(f"Test accuracy={final_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
