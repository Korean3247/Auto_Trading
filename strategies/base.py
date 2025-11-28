from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Strategy(ABC):
    """Common strategy interface."""

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def act(self, obs: Any) -> int:
        """Return discrete action id (0 flat, 1 long, 2 short)."""
        ...
