from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from loguru import logger
from torch.distributions import Categorical

from envs.trading_env import TradingEnv
from models.policy_base import PolicyConfig, PolicyMLP, ValueMLP, load_actor_critic, save_checkpoint
from offline_training.dataset import load_price_and_features


@dataclass
class PPOConfig:
    total_timesteps: int
    rollout_length: int
    batch_size: int
    gamma: float
    gae_lambda: float
    clip_ratio: float
    actor_lr: float
    critic_lr: float
    value_coef: float
    entropy_coef: float
    max_grad_norm: float
    device: str


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.logprobs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: List[float] = []
        self.positions: List[float] = []
        self.entropies: List[float] = []
        self.probs: List[np.ndarray] = []

    def add(self, obs, action, logprob, reward, done, value, position, entropy=None, probs=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.positions.append(position)
        if entropy is not None:
            self.entropies.append(entropy)
        if probs is not None:
            self.probs.append(probs)

    def clear(self):
        self.__init__()


def compute_gae(rewards, dones, values, gamma, lam):
    advantages = []
    gae = 0.0
    values = values + [0.0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    return advantages, returns


def collect_rollout(env: TradingEnv, actor: PolicyMLP, critic: ValueMLP, cfg: Dict) -> Tuple[RolloutBuffer, Dict]:
    buffer = RolloutBuffer()
    obs = env.reset()
    device = torch.device(cfg["rl"]["device"] if torch.cuda.is_available() else "cpu")
    total_reward = 0.0

    for _ in range(cfg["rl"]["rollout_length"]):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = actor(obs_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            value = critic(obs_tensor)
            entropy = dist.entropy()
        probs = dist.probs
        next_obs, reward, done, info = env.step(int(action.item()))
        total_reward += reward
        buffer.add(
            obs,
            int(action.item()),
            float(logprob.item()),
            reward,
            done,
            float(value.item()),
            info.get("position", 0.0),
            float(entropy.item()) if entropy is not None else None,
            probs.squeeze(0).cpu().numpy() if probs is not None else None,
        )
        obs = next_obs
        if done:
            obs = env.reset()
    action_counts = {a: buffer.actions.count(a) for a in set(buffer.actions)}
    mean_probs = (
        np.mean(np.stack(buffer.probs, axis=0), axis=0).tolist()
        if buffer.probs
        else [0.0, 0.0, 0.0]
    )
    stats = {
        "total_reward": total_reward,
        "mean_abs_position": float(np.mean(np.abs(buffer.positions))) if buffer.positions else 0.0,
        "action_counts": action_counts,
        "mean_entropy": float(np.mean(buffer.entropies)) if buffer.entropies else 0.0,
        "mean_probs": mean_probs,
    }
    return buffer, stats


def ppo_update(
    actor: PolicyMLP,
    critic: ValueMLP,
    buffer: RolloutBuffer,
    ppo_cfg: PPOConfig,
    device: torch.device,
) -> Tuple[float, float]:
    advantages, returns = compute_gae(
        buffer.rewards, buffer.dones, buffer.values, gamma=ppo_cfg.gamma, lam=ppo_cfg.gae_lambda
    )
    obs_t = torch.tensor(np.array(buffer.obs), dtype=torch.float32, device=device)
    actions_t = torch.tensor(buffer.actions, dtype=torch.long, device=device)
    old_logprobs_t = torch.tensor(buffer.logprobs, dtype=torch.float32, device=device)
    advantages_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

    policy_losses = []
    value_losses = []
    entropy_terms = []

    for _ in range(max(len(buffer.obs) // ppo_cfg.batch_size, 1)):
        idx = torch.randperm(len(buffer.obs))[: ppo_cfg.batch_size]
        batch_obs = obs_t[idx]
        batch_actions = actions_t[idx]
        batch_old_logprobs = old_logprobs_t[idx]
        batch_adv = advantages_t[idx]
        batch_returns = returns_t[idx]

        logits = actor(batch_obs)
        dist = Categorical(logits=logits)
        new_logprobs = dist.log_prob(batch_actions)
        entropy = dist.entropy().mean()

        ratio = (new_logprobs - batch_old_logprobs).exp()
        clipped = torch.clamp(ratio, 1 - ppo_cfg.clip_ratio, 1 + ppo_cfg.clip_ratio) * batch_adv
        policy_loss = -(torch.min(ratio * batch_adv, clipped)).mean()

        values = critic(batch_obs)
        value_loss = F.mse_loss(values, batch_returns)

        actor.optimizer.zero_grad()  # type: ignore[attr-defined]
        critic.optimizer.zero_grad()  # type: ignore[attr-defined]
        (policy_loss + ppo_cfg.value_coef * value_loss - ppo_cfg.entropy_coef * entropy).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), ppo_cfg.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(critic.parameters(), ppo_cfg.max_grad_norm)
        actor.optimizer.step()  # type: ignore[attr-defined]
        critic.optimizer.step()  # type: ignore[attr-defined]

        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_terms.append(entropy.item())

    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def attach_optimizers(actor: PolicyMLP, critic: ValueMLP, cfg: Dict, device: torch.device):
    actor.to(device)
    critic.to(device)
    actor.train()
    critic.train()
    actor.optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["rl"]["actor_lr"])  # type: ignore[attr-defined]
    critic.optimizer = torch.optim.Adam(critic.parameters(), lr=cfg["rl"]["critic_lr"])  # type: ignore[attr-defined]


def main(config_path: str = "config/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    price_arr, feature_arr = load_price_and_features(Path(cfg["paths"]["data"]), cfg)
    policy_cfg = PolicyConfig(
        input_dim=feature_arr.shape[1],
        hidden_sizes=cfg["model"]["hidden_sizes"],
        activation=cfg["model"]["activation"],
        action_space=cfg["model"].get("action_space", "discrete"),
    )
    device = torch.device(cfg["rl"]["device"] if torch.cuda.is_available() else "cpu")
    env = TradingEnv(price_arr, feature_arr, cfg)

    actor, critic, _ = load_actor_critic(Path(cfg["paths"]["best_rl_policy"]), policy_cfg, device=device)
    attach_optimizers(actor, critic, cfg, device)

    ppo_cfg = PPOConfig(
        total_timesteps=cfg["rl"]["total_timesteps"],
        rollout_length=cfg["rl"]["rollout_length"],
        batch_size=cfg["rl"]["batch_size"],
        gamma=cfg["rl"]["gamma"],
        gae_lambda=cfg["rl"]["gae_lambda"],
        clip_ratio=cfg["rl"]["clip_ratio"],
        actor_lr=cfg["rl"]["actor_lr"],
        critic_lr=cfg["rl"]["critic_lr"],
        value_coef=cfg["rl"]["value_coef"],
        entropy_coef=cfg["rl"]["entropy_coef"],
        max_grad_norm=cfg["rl"]["max_grad_norm"],
        device=cfg["rl"]["device"],
    )

    timesteps = 0
    while timesteps < ppo_cfg.total_timesteps:
        buffer, stats = collect_rollout(env, actor, critic, cfg)
        policy_loss, value_loss = ppo_update(actor, critic, buffer, ppo_cfg, device)
        timesteps += cfg["rl"]["rollout_length"]
        logger.info(
            f"Timesteps={timesteps}, policy_loss={policy_loss:.4f}, value_loss={value_loss:.4f}, "
            f"equity={env.equity:.2f}, rollout_reward={stats['total_reward']:.6f}, "
            f"mean_abs_position={stats['mean_abs_position']:.4f}, action_counts={stats['action_counts']}, "
            f"mean_entropy={stats['mean_entropy']:.4f}, mean_probs={stats['mean_probs']}"
        )
        if timesteps % (cfg["rl"]["rollout_length"] * 10) == 0:
            save_checkpoint(
                actor,
                Path(cfg["paths"]["best_rl_policy"]),
                value_state=critic.state_dict(),
                config=cfg,
                extra={"timesteps": timesteps},
            )

    save_checkpoint(
        actor,
        Path(cfg["paths"]["best_rl_policy"]),
        value_state=critic.state_dict(),
        config=cfg,
        extra={"timesteps": timesteps},
    )
    logger.info(f"Training finished. Saved to {cfg['paths']['best_rl_policy']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
