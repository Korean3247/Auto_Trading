from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.distributions import Categorical

from models.policy_base import PolicyConfig, load_policy
from strategies.base import Strategy


class RLStrategy(Strategy):
    def __init__(
        self,
        checkpoint: Path,
        cfg: dict,
        device: Optional[torch.device] = None,
        greedy: bool = True,
        input_dim_override: Optional[int] = None,
    ):
        self.cfg = cfg
        self.checkpoint = checkpoint
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.greedy = greedy
        self.policy: Optional[torch.nn.Module] = None
        self.input_dim_override = input_dim_override

    def reset(self) -> None:
        if self.policy is not None:
            return
        policy_cfg = PolicyConfig(
            input_dim=self.input_dim_override or self.cfg["model"]["input_dim"] or 0,
            hidden_sizes=self.cfg["model"]["hidden_sizes"],
            activation=self.cfg["model"]["activation"],
            action_space=self.cfg["model"].get("action_space", "discrete"),
        )
        self.policy = load_policy(self.checkpoint, policy_cfg, device=self.device)

    def act(self, obs) -> int:
        if self.policy is None:
            self.input_dim_override = len(obs)
            self.reset()
        assert self.policy is not None
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.policy(x)
            dist = Categorical(logits=logits)
            if self.greedy:
                action = torch.argmax(dist.probs, dim=-1)
            else:
                action = dist.sample()
        return int(action.item())
