from __future__ import annotations

import numpy as np


def compute_sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(1440))


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    return float(np.max(peak - equity_curve))
