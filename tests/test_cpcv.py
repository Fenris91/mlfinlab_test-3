from __future__ import annotations

import pytest

from src.cross_validation.cpcv import combinatorial_purged_splits


def test_combinatorial_split_count() -> None:
    splits = combinatorial_purged_splits(n_samples=20, n_groups=5, n_test_groups=2, embargo=0)
    assert len(splits) == 10


def test_combinatorial_split_has_disjoint_sets() -> None:
    splits = combinatorial_purged_splits(n_samples=30, n_groups=6, n_test_groups=2, embargo=1)
    train, test = splits[0]
    assert set(train).isdisjoint(test)


def test_combinatorial_split_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        combinatorial_purged_splits(n_samples=10, n_groups=1, n_test_groups=1)
