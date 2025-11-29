from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict


class RiskManager:
    """
    Applies simple risk rules on top of strategy actions.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.daily_start_equity = None
        self.max_equity = None
        self.cooldown_until: datetime | None = None

    def reset_day(self, equity: float) -> None:
        self.daily_start_equity = equity
        self.max_equity = equity
        self.cooldown_until = None

    def _in_cooldown(self, now: datetime) -> bool:
        return self.cooldown_until is not None and now < self.cooldown_until

    def check_action(self, action: int, equity: float, position: Dict, context: Dict) -> int:
        now = context.get("ts") or datetime.utcnow()
        if self.daily_start_equity is None:
            self.reset_day(equity)
        day_loss_limit = self.cfg["risk"]["max_daily_loss_pct"]
        dd_limit = self.cfg["risk"]["drawdown_stop_pct"]
        max_notional_pct = self.cfg["risk"]["max_position_notional_pct"]
        max_leverage = self.cfg["risk"]["max_leverage"]

        # Cooldown handling
        if self._in_cooldown(now):
            return 0

        # Update max equity
        self.max_equity = max(self.max_equity or equity, equity)

        # Daily loss check
        daily_ret = (equity - self.daily_start_equity) / max(self.daily_start_equity, 1e-8)
        if daily_ret <= -day_loss_limit:
            self.cooldown_until = now + timedelta(minutes=self.cfg["risk"]["cooldown_minutes"])
            return 0

        # Drawdown check
        dd = (self.max_equity - equity) / max(self.max_equity, 1e-8)
        if dd >= dd_limit:
            self.cooldown_until = now + timedelta(minutes=self.cfg["risk"]["cooldown_minutes"])
            return 0

        # Notional/lev check (approx)
        price = context.get("price", 0.0)
        target_pos = {0: -1, 1: 0, 2: 1}.get(action, 0)
        target_notional = abs(target_pos) * price
        if equity > 0:
            if target_notional / equity > max_notional_pct:
                target_pos = max_notional_pct * equity / price
        # Leverage check: if implied leverage exceeds max, flatten
        if price > 0 and equity > 0 and (target_notional / equity) > max_leverage:
            return 0

        return action

    def update(self, trade_result: Dict) -> None:
        # Placeholder: could aggregate PnL history here
        return None
