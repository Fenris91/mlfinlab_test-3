from __future__ import annotations


def linear_transaction_cost(notional: float, bps: float) -> float:
    """Calculate linear trading cost from notional and basis points.

    Args:
        notional: Trade notional value in currency units.
        bps: Cost in basis points (1 bps = 0.01%).

    Returns:
        Transaction cost in currency units.

    Raises:
        ValueError: If notional is negative or bps is negative.
    """
    if notional < 0.0:
        raise ValueError("notional must be non-negative.")
    if bps < 0.0:
        raise ValueError("bps must be non-negative.")
    return notional * (bps / 10_000.0)


def total_round_trip_cost(notional: float, entry_bps: float, exit_bps: float, fixed_fee: float = 0.0) -> float:
    """Compute total round-trip transaction cost.

    Args:
        notional: Trade notional value in currency units.
        entry_bps: Entry leg cost in basis points.
        exit_bps: Exit leg cost in basis points.
        fixed_fee: Additional fixed fee charged per round trip.

    Returns:
        Total round-trip trading cost.

    Raises:
        ValueError: If fixed fee is negative.
    """
    if fixed_fee < 0.0:
        raise ValueError("fixed_fee must be non-negative.")
    return linear_transaction_cost(notional, entry_bps) + linear_transaction_cost(notional, exit_bps) + fixed_fee
