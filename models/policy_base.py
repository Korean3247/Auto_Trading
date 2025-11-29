import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

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
        action_dim: int = 3,
    ):
        super().__init__()
        self.action_space = action_space
        output_dim = action_dim if action_space == "discrete" else 1
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


class ValueMLP(nn.Module):
    """Critic network predicting state value."""

    def __init__(self, input_dim: int, hidden_sizes: Iterable[int], activation: str = "relu"):
        super().__init__()
        self.net = build_mlp(input_dim, hidden_sizes, activation, output_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


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


def _infer_input_dim_from_state(state: dict) -> int:
    for v in state.values():
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return v.shape[1]
    raise RuntimeError("Unable to infer input_dim from checkpoint state.")


def load_policy(
    path: Path,
    config: PolicyConfig,
    device: Optional[torch.device] = None,
) -> PolicyMLP:
    device = device or torch.device("cpu")
    state = torch.load(path, map_location=device, weights_only=False) if path.exists() else None
    state_dict = None
    if isinstance(state, dict) and "policy_state_dict" in state:
        state_dict = state["policy_state_dict"]
    elif state is not None:
        state_dict = state

    input_dim = config.input_dim
    if input_dim == 0 or input_dim is None:
        if state_dict is None:
            raise RuntimeError("Cannot infer input_dim without checkpoint state.")
        input_dim = _infer_input_dim_from_state(state_dict)

    model = PolicyMLP(
        input_dim=input_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        action_space=config.action_space,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_checkpoint(
    model: Union[nn.Module, dict],
    path: Path,
    value_state: Optional[dict] = None,
    config: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Save policy (and optionally value) states. If only a model is provided, save raw state_dict for backward compatibility.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict() if hasattr(model, "state_dict") else model
    if value_state is None and config is None and extra is None:
        torch.save(state_dict, path)
        return
    payload = {
        "policy_state_dict": state_dict,
        "value_state_dict": value_state,
        "config": config,
        "extra": extra,
    }
    torch.save(payload, path)


def load_actor_critic(
    path: Path,
    policy_cfg: PolicyConfig,
    device: Optional[torch.device] = None,
) -> Tuple[PolicyMLP, ValueMLP, Optional[dict]]:
    """
    Load actor/critic pair from a checkpoint payload; if absent, init fresh models.
    """
    device = device or torch.device("cpu")
    payload = torch.load(path, map_location=device) if path.exists() else None
    policy_state = None
    value_state = None
    if isinstance(payload, dict):
        policy_state = payload.get("policy_state_dict")
        value_state = payload.get("value_state_dict")
    elif payload is not None:
        policy_state = payload

    input_dim = policy_cfg.input_dim
    if (input_dim == 0 or input_dim is None) and policy_state is not None:
        input_dim = _infer_input_dim_from_state(policy_state)

    actor = PolicyMLP(
        input_dim=input_dim,
        hidden_sizes=policy_cfg.hidden_sizes,
        activation=policy_cfg.activation,
        action_space=policy_cfg.action_space,
        action_dim=3,
    ).to(device)
    critic = ValueMLP(
        input_dim=input_dim,
        hidden_sizes=policy_cfg.hidden_sizes,
        activation=policy_cfg.activation,
    ).to(device)
    if policy_state is not None:
        actor.load_state_dict(policy_state)
    if value_state is not None:
        critic.load_state_dict(value_state)
    actor.eval()
    critic.eval()
    return actor, critic, payload


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
