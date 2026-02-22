from __future__ import annotations

import math
import random

import pytest

from src.statistics.sharpe_tests import annualized_sharpe_ratio, bootstrap_sharpe_p_value, sharpe_t_statistic


def test_annualized_sharpe_ratio_positive_series() -> None:
    returns = [0.01, 0.02, -0.005, 0.015, 0.01]
    sharpe = annualized_sharpe_ratio(returns, periods_per_year=252)
    assert sharpe > 0.0


def test_sharpe_t_statistic_is_finite() -> None:
    returns = [0.001, -0.002, 0.003, 0.002, -0.001, 0.004]
    t_stat = sharpe_t_statistic(returns)
    assert math.isfinite(t_stat)


def test_bootstrap_sharpe_p_value_is_deterministic() -> None:
    rng = random.Random(42)
    returns = [rng.gauss(0.001, 0.01) for _ in range(80)]
    p_one = bootstrap_sharpe_p_value(returns, n_bootstrap=200, seed=42)
    p_two = bootstrap_sharpe_p_value(returns, n_bootstrap=200, seed=42)
    assert p_one == pytest.approx(p_two)
