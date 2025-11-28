from __future__ import annotations

from datetime import datetime
from typing import Optional

from loguru import logger


class ExecutionEngine:
    """
    Unified execution interface for paper, testnet, and live modes.
    """

    def __init__(self, mode: str, cfg: dict, model=None, device=None):
        self.mode = mode
        self.cfg = cfg
        self.paper = None
        if mode == "paper":
            from online_trading.paper_trader import PaperTrader  # lazy import to avoid cycle

            self.paper = PaperTrader(cfg, model, device)
        # Placeholders for binance_testnet/live integration

    def get_position(self) -> dict:
        if self.mode == "paper" and self.paper:
            st = self.paper.state
            return {
                "size": st.position_size,
                "entry_price": st.entry_price,
                "equity": st.equity(st.entry_price or 0.0),
                "balance": st.balance_usdt,
            }
        return {}

    def execute_action(self, action: int, price: float, ts: Optional[datetime] = None) -> dict:
        if self.mode == "paper" and self.paper:
            state = self.paper.step(action, price)
            return {
                "price": price,
                "ts": ts,
                "position_size": state.position_size,
                "entry_price": state.entry_price,
                "balance": state.balance_usdt,
                "equity": state.equity(price),
            }
        logger.warning(f"Execution mode {self.mode} not implemented; skipping action.")
        return {"price": price, "ts": ts, "skipped": True}
