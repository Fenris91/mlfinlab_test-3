from __future__ import annotations

import pytest

from src.risk.ruin_probability import probability_of_ruin


def test_probability_of_ruin_favored_game() -> None:
    value = probability_of_ruin(starting_capital=1000.0, loss_per_trade=100.0, win_probability=0.6)
    assert 0.0 < value < 1.0


def test_probability_of_ruin_unfavorable_game() -> None:
    value = probability_of_ruin(starting_capital=1000.0, loss_per_trade=100.0, win_probability=0.4)
    assert value == pytest.approx(1.0)


def test_probability_of_ruin_invalid_capital() -> None:
    with pytest.raises(ValueError):
        probability_of_ruin(starting_capital=0.0, loss_per_trade=100.0, win_probability=0.5)
