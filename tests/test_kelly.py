from __future__ import annotations

import pytest

from src.bet_sizing.kelly import capped_kelly_fraction, kelly_fraction


def test_kelly_fraction_matches_formula() -> None:
    result = kelly_fraction(win_probability=0.6, win_loss_ratio=1.5)
    assert result == pytest.approx(0.3333333333)


def test_capped_kelly_fraction_clips_value() -> None:
    result = capped_kelly_fraction(win_probability=0.8, win_loss_ratio=3.0, cap=0.4)
    assert result == pytest.approx(0.4)


def test_kelly_fraction_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError):
        kelly_fraction(win_probability=1.2, win_loss_ratio=1.0)
