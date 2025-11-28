import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


def _activation(name: str) -> Callable[[], nn.Module]:
    name = name.lower()
    if name == "relu":
        return nn.ReLU
    if name == "gelu":
        return nn.GELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation: {name}")


def build_mlp(input_dim: int, hidden_sizes: Iterable[int], activation: str, output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(_activation(activation)())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class PolicyMLP(nn.Module):
    """
    Simple MLP policy that supports discrete logits or continuous action with tanh squashing.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: Iterable[int],
        activation: str = "relu",
        action_space: str = "discrete",
    ):
        super().__init__()
        self.action_space = action_space
        output_dim = 3 if action_space == "discrete" else 1
        self.net = build_mlp(input_dim, hidden_sizes, activation, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def act(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.action_space == "discrete":
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
            return action, probs
        value = torch.tanh(self.forward(x))
        return value, value


@dataclass
class PolicyConfig:
    input_dim: int
    hidden_sizes: List[int]
    activation: str
    action_space: str = "discrete"

    @classmethod
    def from_dict(cls, data: dict) -> "PolicyConfig":
        input_dim = data.get("input_dim")
        return cls(
            input_dim=int(input_dim) if input_dim is not None else 0,
            hidden_sizes=list(data.get("hidden_sizes", [])),
            activation=str(data.get("activation", "relu")),
            action_space=str(data.get("action_space", "discrete")),
        )


def load_policy(
    path: Path,
    config: PolicyConfig,
    device: Optional[torch.device] = None,
) -> PolicyMLP:
    device = device or torch.device("cpu")
    model = PolicyMLP(
        input_dim=config.input_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        action_space=config.action_space,
    )
    if path.exists():
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def select_action(model: PolicyMLP, features: torch.Tensor, device: Optional[torch.device] = None) -> Tuple[int, float]:
    """
    Features: 1D tensor of shape [feature_dim].
    Returns discrete action id and confidence/size.
    """
    device = device or torch.device("cpu")
    model.eval()
    with torch.no_grad():
        x = features.to(device).unsqueeze(0)
        if model.action_space == "discrete":
            logits = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            action = int(torch.argmax(probs).item())
            confidence = float(probs[action].item())
            return action, confidence
        value = torch.tanh(model(x)).squeeze(0)
        return int(math.copysign(1, value.item()) if abs(value.item()) > 1e-3 else 1), float(value.item())
