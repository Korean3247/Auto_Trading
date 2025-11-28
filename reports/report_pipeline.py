from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger

from llm.report_generator import generate


def load_logs(log_path: Path) -> List[dict]:
    if not log_path.exists():
        return []
    out = []
    with log_path.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def aggregate(logs: List[dict]) -> Dict:
    if not logs:
        return {}
    pnl = []
    equity = []
    top_trades = []
    for row in logs:
        pnl.append(row.get("pnl_step", 0))
        equity.append(row.get("equity", 0))
        top_trades.append(
            {
                "ts": row.get("ts"),
                "action": row.get("action"),
                "pnl": row.get("pnl_step", 0),
                "entry": row.get("price"),
                "exit": row.get("price"),
            }
        )
    equity_arr = np.array(equity)
    pnl_arr = np.array(pnl)
    returns = pnl_arr / np.maximum(np.insert(equity_arr[:-1], 0, equity_arr[0] if equity_arr.size else 1), 1e-8)
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)) if len(returns) else 0.0
    max_dd = float(np.max(np.maximum.accumulate(equity_arr) - equity_arr)) if len(equity_arr) else 0.0
    summary = {
        "date": datetime.utcnow().date().isoformat(),
        "trade_count": len(logs),
        "pnl": float(np.sum(pnl_arr)),
        "return_pct": float((equity_arr[-1] - equity_arr[0]) / equity_arr[0]) if len(equity_arr) else 0.0,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "top_trades": sorted(top_trades, key=lambda x: x["pnl"], reverse=True)[:5],
        "worst_trade": min(top_trades, key=lambda x: x["pnl"]) if top_trades else {},
    }
    return summary


async def run_report(log_dir: Path, llm_model: str, period: str = "daily") -> str:
    log_path = log_dir
    logs = load_logs(log_path)
    summary = aggregate(logs)
    if not summary:
        logger.warning("No logs to report.")
        return ""
    report = await generate(summary, llm_model=llm_model, period=period)
    return report
