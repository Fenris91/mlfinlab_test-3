from __future__ import annotations

import numpy as np


def compute_sharpe(returns: np.ndarray, periods_per_year: int = 365 * 24) -> float:
    """Compute annualized Sharpe ratio from return series."""
    if returns.size == 0:
        return 0.0
    std = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
    if std == 0.0:
        return float("inf") if float(np.mean(returns)) > 0 else 0.0
    return float(np.sqrt(periods_per_year) * np.mean(returns) / std)


def compute_metrics(returns: np.ndarray) -> dict[str, float]:
    """Compute total return and max drawdown from periodic returns."""
    if returns.size == 0:
        return {"total_return": 0.0, "max_drawdown": 0.0}
    equity = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(equity)
    drawdown = 1.0 - (equity / running_peak)
    return {
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": float(np.max(drawdown)),
    }
