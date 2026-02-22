from __future__ import annotations

from itertools import combinations


def combinatorial_purged_splits(
    n_samples: int,
    n_groups: int,
    n_test_groups: int,
    embargo: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Generate combinatorial purged train/test splits over contiguous groups.

    Args:
        n_samples: Total number of observations.
        n_groups: Number of contiguous groups used to define test blocks.
        n_test_groups: Number of groups selected for each test set.
        embargo: Number of samples removed around each test block from train set.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if n_groups <= 1:
        raise ValueError("n_groups must be greater than one.")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError("n_test_groups must be in [1, n_groups).")
    if embargo < 0:
        raise ValueError("embargo must be non-negative.")

    indices = list(range(n_samples))
    group_sizes = [n_samples // n_groups] * n_groups
    for i in range(n_samples % n_groups):
        group_sizes[i] += 1

    groups: list[list[int]] = []
    cursor = 0
    for size in group_sizes:
        groups.append(indices[cursor : cursor + size])
        cursor += size

    splits: list[tuple[list[int], list[int]]] = []
    for test_group_idx in combinations(range(n_groups), n_test_groups):
        test_indices = [idx for group_id in test_group_idx for idx in groups[group_id]]
        blocked = set(test_indices)

        for idx in test_indices:
            for offset in range(-embargo, embargo + 1):
                blocked_idx = idx + offset
                if 0 <= blocked_idx < n_samples:
                    blocked.add(blocked_idx)

        train_indices = [idx for idx in indices if idx not in blocked]
        splits.append((train_indices, test_indices))

    return splits
