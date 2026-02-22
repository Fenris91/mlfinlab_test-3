from __future__ import annotations


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    """Compute the Kelly-optimal portfolio fraction.

    Args:
        win_probability: Probability of a winning outcome in [0, 1].
        win_loss_ratio: Ratio of average win to average loss, must be positive.

    Returns:
        The unconstrained Kelly fraction.

    Raises:
        ValueError: If input values are outside allowed ranges.
    """
    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be in [0, 1].")
    if win_loss_ratio <= 0.0:
        raise ValueError("win_loss_ratio must be positive.")

    loss_probability = 1.0 - win_probability
    return win_probability - (loss_probability / win_loss_ratio)


def capped_kelly_fraction(win_probability: float, win_loss_ratio: float, cap: float = 1.0) -> float:
    """Compute a Kelly fraction with optional symmetric cap.

    Args:
        win_probability: Probability of a winning outcome in [0, 1].
        win_loss_ratio: Ratio of average win to average loss, must be positive.
        cap: Absolute upper bound applied symmetrically to the Kelly fraction.

    Returns:
        The Kelly fraction clipped to [-cap, cap].

    Raises:
        ValueError: If cap is not positive.
    """
    if cap <= 0.0:
        raise ValueError("cap must be positive.")

    raw_fraction = kelly_fraction(win_probability=win_probability, win_loss_ratio=win_loss_ratio)
    return max(-cap, min(cap, raw_fraction))
