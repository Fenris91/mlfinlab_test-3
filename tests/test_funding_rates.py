from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import pytest

from src.data import funding_rates as fr


class DummyExchange:
    def __init__(self, pages=None):
        self.pages = pages or []
        self.calls = []

    def fetch_funding_rate_history(self, symbol, since=None, limit=1000):
        self.calls.append((symbol, since, limit))
        if self.pages:
            return self.pages.pop(0)
        return []


@pytest.fixture
def sample_rows():
    return [
        {"timestamp": 1700000000000, "symbol": "BTC/USDT:USDT", "fundingRate": 0.0001, "markPrice": 40000},
        {"timestamp": 1700028800000, "symbol": "BTC/USDT:USDT", "fundingRate": 0.0002, "markPrice": 40100},
    ]


def test_download_funding_rates_returns_correct_columns(monkeypatch, sample_rows):
    ex = DummyExchange(pages=[sample_rows])

    class DummyCcxt:
        @staticmethod
        def binance(_):
            return ex

    monkeypatch.setitem(sys.modules, "ccxt", DummyCcxt)
    df = fr.download_funding_rates("BTC/USDT:USDT")
    assert list(df.columns) == ["symbol", "funding_rate", "mark_price"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_download_funding_rates_auto_appends_futures_suffix(monkeypatch, sample_rows):
    ex = DummyExchange(pages=[sample_rows])

    class DummyCcxt:
        @staticmethod
        def binance(_):
            return ex

    monkeypatch.setitem(sys.modules, "ccxt", DummyCcxt)
    fr.download_funding_rates("BTC/USDT")
    assert ex.calls[0][0] == "BTC/USDT:USDT"


def test_download_all_universe_sequential_with_progress(monkeypatch):
    calls = []

    def fake_download(symbol, since, limit, exchange_id):
        calls.append(symbol)
        idx = pd.to_datetime(["2024-01-01"], utc=True)
        return pd.DataFrame({"symbol": [symbol], "funding_rate": [0.1], "mark_price": [1]}, index=idx)

    monkeypatch.setattr(fr, "download_funding_rates", fake_download)
    stats = fr.download_all_universe(["BTC/USDT", "ETH/USDT"], cache_dir="/tmp/funding-tests")
    assert calls == ["BTC/USDT", "ETH/USDT"]
    assert set(stats.keys()) == {"BTC/USDT", "ETH/USDT"}


def test_cache_append_mode(monkeypatch, tmp_path):
    cache = tmp_path / "funding"
    idx = pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 08:00:00"], utc=True)
    existing = pd.DataFrame(
        {"symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"], "funding_rate": [0.1, 0.2], "mark_price": [1, 1]},
        index=idx,
    )
    existing.to_parquet(cache / "BTC_USDT_funding.parquet") if cache.exists() else None
    cache.mkdir(parents=True, exist_ok=True)
    existing.to_parquet(cache / "BTC_USDT_funding.parquet")

    def fake_download(symbol, since, limit, exchange_id):
        new_idx = pd.to_datetime(["2024-01-01 08:00:00", "2024-01-01 16:00:00"], utc=True)
        return pd.DataFrame(
            {"symbol": ["BTC/USDT:USDT", "BTC/USDT:USDT"], "funding_rate": [0.3, 0.4], "mark_price": [1, 1]},
            index=new_idx,
        )

    monkeypatch.setattr(fr, "download_funding_rates", fake_download)
    stats = fr.download_all_universe(["BTC/USDT"], cache_dir=str(cache))
    loaded = pd.read_parquet(cache / "BTC_USDT_funding.parquet")
    assert len(loaded) == 3
    assert stats["BTC/USDT"] == 1


def test_cache_round_trip(tmp_path):
    cache = tmp_path / "funding"
    cache.mkdir()
    idx = pd.to_datetime(["2024-01-01"], utc=True)
    expected = pd.DataFrame({"symbol": ["BTC/USDT:USDT"], "funding_rate": [0.1], "mark_price": [1]}, index=idx)
    expected.to_parquet(cache / "BTC_USDT_funding.parquet")
    loaded = fr.load_funding_rates("BTC/USDT", cache_dir=str(cache))
    pd.testing.assert_frame_equal(expected, loaded)


def test_load_funding_rates_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fr.load_funding_rates("BTC/USDT", cache_dir=str(tmp_path))


def test_build_funding_features_output_shape():
    idx_f = pd.date_range("2024-01-01", periods=5, freq="8h", tz="UTC")
    idx_o = pd.date_range("2024-01-01", periods=40, freq="1h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.1, 0.2, 0.1, -0.1, -0.2]}, index=idx_f)
    ohlcv = pd.DataFrame({"close": range(40)}, index=idx_o)
    out = fr.build_funding_features(funding, ohlcv)
    assert out.index.equals(ohlcv.index)
    assert out.shape[1] == 7


def test_build_funding_features_forward_fill():
    idx_f = pd.date_range("2024-01-01", periods=2, freq="8h", tz="UTC")
    idx_o = pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.1, 0.2]}, index=idx_f)
    out = fr.build_funding_features(funding, pd.DataFrame(index=idx_o))
    assert out.loc[idx_o[1:7], "funding_rate"].isna().sum() == 0


def test_build_funding_features_zscore_window():
    idx = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.01] * 200}, index=idx)
    out = fr.build_funding_features(funding, pd.DataFrame(index=idx))
    assert out["funding_zscore"].iloc[:167].isna().all()


def test_build_funding_features_positive_streak():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.1, 0.2, -0.1, 0.3]}, index=idx)
    out = fr.build_funding_features(funding, pd.DataFrame(index=idx))
    assert out["funding_positive_streak"].tolist() == [1, 2, 0, 1]


def test_build_funding_features_flip_detection():
    idx = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    funding = pd.DataFrame({"funding_rate": [0.1, 0.2, -0.1, -0.2]}, index=idx)
    out = fr.build_funding_features(funding, pd.DataFrame(index=idx))
    assert out["funding_flip"].tolist() == [0, 0, 1, 0]


def test_cli_download(monkeypatch, tmp_path):
    called = {}

    def fake_download_all_universe(symbols, days, exchange_id, cache_dir):
        called["symbols"] = symbols
        return {symbols[0]: 1}

    monkeypatch.setattr(fr, "download_all_universe", fake_download_all_universe)
    monkeypatch.setattr(sys, "argv", ["funding_rates", "download", "--symbols", "BTC/USDT", "--cache-dir", str(tmp_path)])
    fr.main()
    assert called["symbols"] == ["BTC/USDT"]


def test_cli_download_universe(monkeypatch, tmp_path):
    cfg = tmp_path / "assets.yaml"
    cfg.write_text("assets:\n  - symbol: BTC/USDT\n  - symbol: ETH/USDT\n", encoding="utf-8")
    called = {}

    def fake_download_all_universe(symbols, days, exchange_id, cache_dir):
        called["symbols"] = symbols
        return {s: 1 for s in symbols}

    monkeypatch.setattr(fr, "download_all_universe", fake_download_all_universe)
    monkeypatch.setattr(sys, "argv", ["funding_rates", "download-universe", "--config", str(cfg), "--cache-dir", str(tmp_path)])
    fr.main()
    assert called["symbols"] == ["BTC/USDT", "ETH/USDT"]
