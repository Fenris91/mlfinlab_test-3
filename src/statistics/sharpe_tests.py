from __future__ import annotations

import math
import random
from statistics import mean, stdev


def _validate_returns(returns: list[float]) -> list[float]:
    """Validate and normalize return inputs."""
    data = [float(value) for value in returns]
    if len(data) < 2:
        raise ValueError("returns must contain at least two observations.")
    return data


def annualized_sharpe_ratio(returns: list[float], periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from periodic returns.

    Args:
        returns: Sequence of arithmetic returns.
        periods_per_year: Number of return periods in one year.

    Returns:
        Annualized Sharpe ratio.

    Raises:
        ValueError: If periods are invalid or volatility is zero.
    """
    data = _validate_returns(returns)
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive.")

    std_ret = stdev(data)
    if std_ret == 0.0:
        raise ValueError("returns volatility is zero; Sharpe ratio is undefined.")
    return (mean(data) / std_ret) * math.sqrt(periods_per_year)


def sharpe_t_statistic(returns: list[float], periods_per_year: int = 252) -> float:
    """Compute t-statistic for the annualized Sharpe ratio estimate."""
    sharpe = annualized_sharpe_ratio(returns=returns, periods_per_year=periods_per_year)
    n_obs = float(len(returns))
    return sharpe * math.sqrt(n_obs / periods_per_year)


def bootstrap_sharpe_p_value(
    returns: list[float],
    periods_per_year: int = 252,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> float:
    """Estimate a one-sided p-value for Sharpe ratio > 0 via bootstrap.

    Args:
        returns: Sequence of arithmetic returns.
        periods_per_year: Number of return periods in one year.
        n_bootstrap: Number of bootstrap replications.
        seed: Seed for deterministic random sampling.

    Returns:
        Estimated p-value for the null hypothesis Sharpe <= 0.
    """
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    data = _validate_returns(returns)
    observed = annualized_sharpe_ratio(data, periods_per_year=periods_per_year)
    centered = [value - mean(data) for value in data]

    rng = random.Random(seed)
    exceed_count = 0
    for _ in range(n_bootstrap):
        sample = [rng.choice(centered) for _ in centered]
        sim_sharpe = annualized_sharpe_ratio(sample, periods_per_year=periods_per_year)
        if sim_sharpe >= observed:
            exceed_count += 1

    return exceed_count / n_bootstrap
