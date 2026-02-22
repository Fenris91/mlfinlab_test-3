from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.data import funding_rates


class DummyExchange:
    def __init__(self, pages):
        self.pages = list(pages)
        self.calls = []

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        self.calls.append({"symbol": symbol, "since": since, "limit": limit})
        if not self.pages:
            return []
        return self.pages.pop(0)


def _install_ccxt(monkeypatch, exchange):
    fake_ccxt = types.SimpleNamespace(binance=lambda _kwargs: exchange)
    monkeypatch.setitem(sys.modules, "ccxt", fake_ccxt)


def test_download_funding_rates_returns_correct_columns(monkeypatch):
    exchange = DummyExchange(
        [
            [
                {
                    "timestamp": 1700000000000,
                    "symbol": "BTC/USDT:USDT",
                    "fundingRate": 0.0001,
                    "markPrice": 50000,
                }
            ]
        ]
    )
    _install_ccxt(monkeypatch, exchange)

    df = funding_rates.download_funding_rates("BTC/USDT:USDT", since="2024-01-01")

    assert list(df.columns) == ["symbol", "funding_rate", "mark_price"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_download_funding_rates_auto_appends_futures_suffix(monkeypatch):
    exchange = DummyExchange([[]])
    _install_ccxt(monkeypatch, exchange)

    funding_rates.download_funding_rates("BTC/USDT")

    assert exchange.calls[0]["symbol"] == "BTC/USDT:USDT"


def test_download_all_universe_sequential_with_progress(monkeypatch, tmp_path):
    calls = []

    def fake_download(symbol, since=None, limit=1000, exchange_id="binance"):
        calls.append(symbol)
        idx = pd.DatetimeIndex(["2024-01-01"], tz="UTC")
        return pd.DataFrame(
            {"symbol": [funding_rates._normalize_symbol(symbol)], "funding_rate": [0.1], "mark_price": [1]},
            index=idx,
        )

    monkeypatch.setattr(funding_rates, "download_funding_rates", fake_download)

    funding_rates.download_all_universe(["BTC/USDT", "ETH/USDT"], cache_dir=str(tmp_path))

    assert calls == ["BTC/USDT", "ETH/USDT"]


def test_cache_append_mode(monkeypatch, tmp_path):
    path = tmp_path / "BTC_USDT_funding.parquet"
    idx = pd.DatetimeIndex(["2024-01-01 00:00:00+00:00", "2024-01-01 08:00:00+00:00"])
    initial = pd.DataFrame(
        {"symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"], "funding_rate": [0.1, 0.2], "mark_price": [1, 1]},
        index=idx,
    )
    initial.to_parquet(path)

    def fake_download(symbol, since=None, limit=1000, exchange_id="binance"):
        new_idx = pd.DatetimeIndex(["2024-01-01 08:00:00+00:00", "2024-01-01 16:00:00+00:00"])
        return pd.DataFrame(
            {"symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"], "funding_rate": [0.25, 0.3], "mark_price": [1, 1]},
            index=new_idx,
        )

    monkeypatch.setattr(funding_rates, "download_funding_rates", fake_download)
    funding_rates.download_all_universe(["BTC/USDT"], cache_dir=str(tmp_path))

    combined = pd.read_parquet(path)
    assert len(combined) == 3
    assert combined.index.is_unique


def test_cache_round_trip(tmp_path):
    idx = pd.DatetimeIndex(["2024-01-01 00:00:00+00:00"]) 
    original = pd.DataFrame(
        {"symbol": ["BTC/USDT:USDT"], "funding_rate": [0.1], "mark_price": [1]}, index=idx
    )
    target = tmp_path / "BTC_USDT_funding.parquet"
    original.to_parquet(target)

    loaded = funding_rates.load_funding_rates("BTC/USDT", cache_dir=str(tmp_path))
    pd.testing.assert_frame_equal(original, loaded)


def test_load_funding_rates_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        funding_rates.load_funding_rates("BTC/USDT", cache_dir=str(tmp_path))


def test_build_funding_features_output_shape():
    idx = pd.date_range("2024-01-01", periods=240, freq="h", tz="UTC")
    funding_idx = pd.date_range("2024-01-01", periods=30, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.001] * len(funding_idx)}, index=funding_idx)
    ohlcv_df = pd.DataFrame({"close": range(len(idx))}, index=idx)

    features = funding_rates.build_funding_features(funding_df, ohlcv_df)

    assert features.shape == (len(ohlcv_df), 7)
    assert list(features.columns) == [
        "funding_rate",
        "funding_rate_ma_24",
        "funding_rate_ma_168",
        "funding_cumulative_24h",
        "funding_zscore",
        "funding_positive_streak",
        "funding_flip",
    ]


def test_build_funding_features_forward_fill():
    ohlcv_idx = pd.date_range("2024-01-01", periods=16, freq="h", tz="UTC")
    funding_idx = pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.1, 0.2]}, index=funding_idx)
    ohlcv_df = pd.DataFrame({"close": range(16)}, index=ohlcv_idx)

    features = funding_rates.build_funding_features(funding_df, ohlcv_df)

    assert features["funding_rate"].iloc[1:8].notna().all()


def test_build_funding_features_zscore_window():
    idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.001 * i for i in range(200)]}, index=idx)
    ohlcv_df = pd.DataFrame({"close": range(200)}, index=idx)

    features = funding_rates.build_funding_features(funding_df, ohlcv_df)

    assert features["funding_zscore"].iloc[:167].isna().all()
    assert pd.notna(features["funding_zscore"].iloc[167])


def test_build_funding_features_positive_streak():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.1, 0.2, -0.1, 0.3, 0.4]}, index=idx)
    ohlcv_df = pd.DataFrame({"close": range(5)}, index=idx)

    features = funding_rates.build_funding_features(funding_df, ohlcv_df)

    assert list(features["funding_positive_streak"]) == [1, 2, 0, 1, 2]


def test_build_funding_features_flip_detection():
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    funding_df = pd.DataFrame({"funding_rate": [0.1, -0.2, -0.3, 0.1, 0.2]}, index=idx)
    ohlcv_df = pd.DataFrame({"close": range(5)}, index=idx)

    features = funding_rates.build_funding_features(funding_df, ohlcv_df)

    assert list(features["funding_flip"]) == [0, 1, 0, 1, 0]


def test_cli_download(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        funding_rates,
        "download_all_universe",
        lambda symbols, days, cache_dir, exchange_id="binance": {symbol: 1 for symbol in symbols},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "funding_rates",
            "download",
            "--symbols",
            "BTC/USDT",
            "ETH/USDT",
            "--cache-dir",
            str(tmp_path),
        ],
    )

    funding_rates.main()
    out = capsys.readouterr().out
    assert "BTC/USDT" in out and "ETH/USDT" in out


def test_cli_download_universe(monkeypatch, tmp_path):
    config = tmp_path / "asset_universe.yaml"
    config.write_text("assets:\n  - symbol: BTC/USDT\n  - symbol: ETH/USDT\n", encoding="utf-8")

    called = {}

    def fake_download(symbols, days, cache_dir, exchange_id="binance"):
        called["symbols"] = symbols
        return {symbol: 1 for symbol in symbols}

    monkeypatch.setattr(funding_rates, "download_all_universe", fake_download)
    monkeypatch.setattr(
        sys,
        "argv",
        ["funding_rates", "download-universe", "--config", str(config), "--cache-dir", str(tmp_path)],
    )

    funding_rates.main()
    assert called["symbols"] == ["BTC/USDT", "ETH/USDT"]
