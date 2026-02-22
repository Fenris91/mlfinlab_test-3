from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


def _normalize_symbol(symbol: str) -> str:
    """Normalize a spot-style symbol into a ccxt futures symbol when needed."""
    if ":" in symbol:
        return symbol
    if symbol.endswith("/USDT"):
        return f"{symbol}:USDT"
    return symbol


def _safe_symbol(symbol: str) -> str:
    """Convert a trading symbol into its safe file-name representation."""
    return _normalize_symbol(symbol).replace(":USDT", "").replace("/", "_")


def _cache_path(symbol: str, cache_dir: str) -> Path:
    """Build funding cache path for a symbol."""
    return Path(cache_dir) / f"{_safe_symbol(symbol)}_funding.parquet"


def _build_exchange(exchange_id: str) -> Any:
    """Instantiate a ccxt futures exchange client."""
    ccxt = importlib.import_module("ccxt")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"options": {"defaultType": "future"}})


def download_funding_rates(
    symbol: str,
    since: str | None = None,
    limit: int = 1000,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Download historical funding rates for a single symbol via ccxt.

    Args:
        symbol: Trading pair in ccxt futures format (e.g. "BTC/USDT:USDT").
            If given as "BTC/USDT", automatically appends ":USDT".
        since: ISO-8601 start date string (e.g. "2022-01-01"). If None,
            fetches from earliest available.
        limit: Page size per API request.
        exchange_id: CCXT exchange identifier.

    Returns:
        DataFrame with UTC DatetimeIndex and columns
        [symbol, funding_rate, mark_price].
    """
    normalized_symbol = _normalize_symbol(symbol)
    exchange = _build_exchange(exchange_id)

    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000) if since else None

    rows: list[dict[str, Any]] = []
    next_since = since_ms

    while True:
        page = exchange.fetch_funding_rate_history(normalized_symbol, since=next_since, limit=limit)
        if not page:
            break
        rows.extend(page)
        if len(page) < limit:
            break

        next_since = int(page[-1]["timestamp"]) + 1
        time.sleep(1.0)

    if not rows:
        return pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])

    frame = pd.DataFrame(rows)
    frame = frame.rename(columns={"fundingRate": "funding_rate", "markPrice": "mark_price"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame[["symbol", "funding_rate", "mark_price"]]


def load_funding_rates(symbol: str, cache_dir: str = "data_cache/funding") -> pd.DataFrame:
    """Load cached funding rates from Parquet."""
    path = _cache_path(symbol, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"Funding cache file not found: {path}")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    return frame.sort_index()


def download_all_universe(
    symbols: list[str],
    days: int = 1460,
    exchange_id: str = "binance",
    cache_dir: str = "data_cache/funding",
) -> dict[str, int]:
    """Batch-download funding rates for multiple symbols with append-mode caching."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    start_since = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=days)).date().isoformat()

    downloaded: dict[str, int] = {}
    for symbol in tqdm(symbols, desc="Funding download"):
        path = _cache_path(symbol, cache_dir)
        existing = pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])
        since = start_since

        if path.exists():
            existing = load_funding_rates(symbol, cache_dir)
            if not existing.empty:
                since = (existing.index.max() + pd.Timedelta(milliseconds=1)).date().isoformat()

        fresh = download_funding_rates(symbol=symbol, since=since, exchange_id=exchange_id)
        combined = pd.concat([existing, fresh]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(path)
        downloaded[symbol] = max(0, len(combined) - len(existing))

    return downloaded


def build_funding_features(funding_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Build funding-rate features aligned to an OHLCV DatetimeIndex."""
    aligned_rate = funding_df["funding_rate"].sort_index().reindex(ohlcv_df.index, method="ffill")

    features = pd.DataFrame(index=ohlcv_df.index)
    features["funding_rate"] = aligned_rate
    features["funding_rate_ma_24"] = aligned_rate.rolling(24).mean()
    features["funding_rate_ma_168"] = aligned_rate.rolling(168).mean()
    features["funding_cumulative_24h"] = aligned_rate.rolling(24).sum()

    rolling_mean = aligned_rate.rolling(168).mean()
    rolling_std = aligned_rate.rolling(168).std()
    features["funding_zscore"] = (aligned_rate - rolling_mean) / rolling_std

    positive = aligned_rate > 0
    streak = pd.Series(0, index=aligned_rate.index, dtype="int64")
    run = 0
    for i, is_positive in enumerate(positive):
        if bool(is_positive):
            run += 1
            streak.iloc[i] = run
        else:
            run = 0
            streak.iloc[i] = 0
    features["funding_positive_streak"] = streak

    prev = aligned_rate.shift(1)
    features["funding_flip"] = ((aligned_rate * prev) < 0).astype("int64")

    return features


def _extract_symbols_from_config(config_path: str) -> list[str]:
    """Extract asset symbols from an asset universe YAML config file."""
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return [item["symbol"] for item in config.get("assets", []) if "symbol" in item]


def _list_cached(cache_dir: str) -> list[str]:
    """List symbols available in funding cache directory."""
    path = Path(cache_dir)
    if not path.exists():
        return []
    return sorted(item.name for item in path.glob("*_funding.parquet"))


def main() -> None:
    """Command-line entrypoint for funding rates downloader utilities."""
    parser = argparse.ArgumentParser(description="Funding rates utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download", help="Download a list of symbols")
    download_parser.add_argument("--symbols", nargs="+", required=True)
    download_parser.add_argument("--days", type=int, default=1460)
    download_parser.add_argument("--cache-dir", default="data_cache/funding")

    universe_parser = subparsers.add_parser("download-universe", help="Download from universe YAML")
    universe_parser.add_argument("--config", required=True)
    universe_parser.add_argument("--days", type=int, default=1460)
    universe_parser.add_argument("--cache-dir", default="data_cache/funding")

    list_parser = subparsers.add_parser("list", help="List cached funding files")
    list_parser.add_argument("--cache-dir", default="data_cache/funding")

    args = parser.parse_args()

    if args.command == "download":
        stats = download_all_universe(args.symbols, days=args.days, cache_dir=args.cache_dir)
        for symbol, n_rows in stats.items():
            print(f"{symbol}: {n_rows} new rows")
        return

    if args.command == "download-universe":
        symbols = _extract_symbols_from_config(args.config)
        stats = download_all_universe(symbols, days=args.days, cache_dir=args.cache_dir)
        for symbol, n_rows in stats.items():
            print(f"{symbol}: {n_rows} new rows")
        return

    for item in _list_cached(args.cache_dir):
        print(item)


if __name__ == "__main__":
    main()
