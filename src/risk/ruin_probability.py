from __future__ import annotations

import math


def probability_of_ruin(starting_capital: float, loss_per_trade: float, win_probability: float) -> float:
    """Approximate gambler's ruin probability for repeated fixed-size trades.

    Args:
        starting_capital: Initial capital in currency units.
        loss_per_trade: Absolute loss amount when a trade loses.
        win_probability: Probability of a winning trade in [0, 1].

    Returns:
        Probability of eventual ruin under a simple biased random walk model.

    Raises:
        ValueError: If inputs are outside supported ranges.
    """
    if starting_capital <= 0.0:
        raise ValueError("starting_capital must be positive.")
    if loss_per_trade <= 0.0:
        raise ValueError("loss_per_trade must be positive.")
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be in [0, 1].")

    steps_to_ruin = int(math.ceil(starting_capital / loss_per_trade))
    q = 1.0 - win_probability

    if win_probability == 0.0:
        return 1.0
    if q == 0.0:
        return 0.0
    if win_probability <= q:
        return 1.0

    ratio = q / win_probability
    return float(ratio**steps_to_ruin)
