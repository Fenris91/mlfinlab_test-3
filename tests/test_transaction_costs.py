from __future__ import annotations

import pytest

from src.costs.transaction_costs import linear_transaction_cost, total_round_trip_cost


def test_linear_transaction_cost() -> None:
    assert linear_transaction_cost(notional=100_000, bps=5) == pytest.approx(50.0)


def test_total_round_trip_cost() -> None:
    total = total_round_trip_cost(notional=200_000, entry_bps=2, exit_bps=3, fixed_fee=1.5)
    assert total == pytest.approx(101.5)


def test_linear_transaction_cost_invalid_input() -> None:
    with pytest.raises(ValueError):
        linear_transaction_cost(notional=-1.0, bps=2)
