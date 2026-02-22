from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts import monte_carlo


def test_fold_bootstrap_deterministic() -> None:
    folds = np.array([0.01, -0.02, 0.015, 0.005, -0.001])
    a = monte_carlo.bootstrap_fold_metrics(folds, n_bootstrap=1000, seed=7)
    b = monte_carlo.bootstrap_fold_metrics(folds, n_bootstrap=1000, seed=7)
    assert a == b


def test_fold_bootstrap_ci_contains_observed() -> None:
    folds = np.array([0.01, -0.02, 0.015, 0.005, -0.001, 0.02, 0.012])
    result = monte_carlo.bootstrap_fold_metrics(folds, n_bootstrap=2000, seed=42)
    observed = result["sharpe"]["observed"]
    low, high = result["sharpe"]["ci99"]
    assert low <= observed <= high


def test_trade_bootstrap_known_distribution() -> None:
    trades = np.full(87, 0.01)
    result = monte_carlo.bootstrap_trade_sharpe(trades, n_bootstrap=1000, seed=42)
    assert result["p_sharpe_lt_zero"] == 0.0


def test_ruin_probability_losing_strategy() -> None:
    folds = np.full(42, -0.01)
    result = monte_carlo.bootstrap_fold_metrics(folds, n_bootstrap=1000, seed=42)
    assert result["ruin_probability"] >= 0.99


def test_deflated_sharpe_formula() -> None:
    result = monte_carlo.deflated_sharpe(1.2, n_strategies=6, total_trials=300)
    expected_haircut = max(0.0, 1.0 - 0.05 * np.sqrt(np.log(300) / 6))
    assert np.isclose(result["haircut_factor"], expected_haircut)
    assert np.isclose(result["deflated_sharpe"], 1.2 * expected_haircut)


def test_output_json_structure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = monte_carlo.run_analysis(
        strategy="xgb-full",
        symbol="BTC/USDT",
        n_bootstrap=100,
        seed=42,
        n_strategies=6,
        total_trials=300,
    )
    out_dir = Path("results") / "xgb-full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "monte_carlo.json"
    out_path.write_text(json.dumps(result), encoding="utf-8")

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert "fold_bootstrap" in saved
    assert "trade_bootstrap" in saved
    assert "deflated_sharpe" in saved
    assert "sharpe" in saved["fold_bootstrap"]
