from __future__ import annotations

import argparse
import importlib
import time
from datetime import datetime, timedelta, timezone
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
    """Return the funding cache file path for a symbol."""
    return Path(cache_dir) / f"{_safe_symbol(symbol)}_funding.parquet"


def _build_exchange(exchange_id: str) -> Any:
    """Instantiate a ccxt futures exchange client."""
    ccxt = importlib.import_module("ccxt")
    exchange_cls = getattr(ccxt, exchange_id)
    return exchange_cls({"options": {"defaultType": "future"}})


def _to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert ccxt funding payload rows to the canonical dataframe shape."""
    if not rows:
        return pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])

    df = pd.DataFrame(rows)
    df = df.rename(columns={"fundingRate": "funding_rate", "markPrice": "mark_price"})
    df = df[["timestamp", "symbol", "funding_rate", "mark_price"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    return df


def download_funding_rates(
    symbol: str,
    since: str | None = None,
    limit: int = 1000,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """Download historical funding rates for a single symbol via ccxt."""
    normalized = _normalize_symbol(symbol)
    exchange = _build_exchange(exchange_id)

    since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000) if since else None

    rows: list[dict[str, Any]] = []
    next_since = since_ms
    while True:
        page = exchange.fetch_funding_rate_history(normalized, since=next_since, limit=limit)
        if not page:
            break
        rows.extend(page)
        if len(page) < limit:
            break
        next_since = int(page[-1]["timestamp"]) + 1
        time.sleep(1.0)

    df = _to_dataframe(rows)
    if not df.empty:
        df = df[~df.index.duplicated(keep="last")]
    return df


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
    default_since = datetime.now(timezone.utc) - timedelta(days=days)

    stats: dict[str, int] = {}
    for symbol in tqdm(symbols, desc="Funding", unit="symbol"):
        cache_path = _cache_path(symbol, cache_dir)
        existing = pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])
        if cache_path.exists():
            existing = load_funding_rates(symbol, cache_dir)

        if existing.empty:
            since = default_since.isoformat()
        else:
            latest = pd.Timestamp(existing.index.max())
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            else:
                latest = latest.tz_convert("UTC")
            since = (latest + pd.Timedelta(milliseconds=1)).isoformat()

        new_data = download_funding_rates(
            symbol=symbol,
            since=since,
            limit=1000,
            exchange_id=exchange_id,
        )

        if existing.empty:
            combined = new_data
        else:
            combined = pd.concat([existing, new_data]).sort_index()
            combined = combined[~combined.index.duplicated(keep="last")]

        combined.to_parquet(cache_path)
        stats[symbol] = len(combined) - len(existing)

    return stats


def build_funding_features(funding_df: pd.DataFrame, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Build funding-rate features aligned to an OHLCV DatetimeIndex."""
    target_index = ohlcv_df.index
    aligned = funding_df[["funding_rate"]].copy().reindex(target_index).ffill()

    out = pd.DataFrame(index=target_index)
    out["funding_rate"] = aligned["funding_rate"]
    out["funding_rate_ma_24"] = out["funding_rate"].rolling(24).mean()
    out["funding_rate_ma_168"] = out["funding_rate"].rolling(168).mean()
    out["funding_cumulative_24h"] = out["funding_rate"].rolling(24).sum()

    mean_168 = out["funding_rate"].rolling(168).mean()
    std_168 = out["funding_rate"].rolling(168).std()
    out["funding_zscore"] = (out["funding_rate"] - mean_168) / std_168

    positive = out["funding_rate"] > 0
    streak = pd.Series(0, index=out.index, dtype="int64")
    run = 0
    for i, is_pos in enumerate(positive):
        if is_pos:
            run += 1
        else:
            run = 0
        streak.iat[i] = run
    out["funding_positive_streak"] = streak

    sign = out["funding_rate"].fillna(0.0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    out["funding_flip"] = (sign != sign.shift(1)).fillna(False).astype(int)
    out.loc[out.index[0], "funding_flip"] = 0

    return out


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Funding rate downloader and feature builder.")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="Download funding data for explicit symbols.")
    dl.add_argument("--symbols", nargs="+", required=True)
    dl.add_argument("--days", type=int, default=1460)
    dl.add_argument("--cache-dir", default="data_cache/funding")
    dl.add_argument("--exchange-id", default="binance")

    dlu = sub.add_parser("download-universe", help="Download funding data for config universe.")
    dlu.add_argument("--config", default="configs/asset_universe.yaml")
    dlu.add_argument("--days", type=int, default=1460)
    dlu.add_argument("--cache-dir", default="data_cache/funding")
    dlu.add_argument("--exchange-id", default="binance")

    ls_cmd = sub.add_parser("list", help="List locally cached funding files.")
    ls_cmd.add_argument("--cache-dir", default="data_cache/funding")
    return parser.parse_args()


def main() -> None:
    """Run funding-rate CLI commands."""
    args = _parse_args()

    if args.command == "download":
        result = download_all_universe(
            symbols=args.symbols,
            days=args.days,
            exchange_id=args.exchange_id,
            cache_dir=args.cache_dir,
        )
        for symbol, count in result.items():
            print(f"{symbol}: +{count} rows")
        return

    if args.command == "download-universe":
        symbols = _extract_symbols_from_config(args.config)
        result = download_all_universe(
            symbols=symbols,
            days=args.days,
            exchange_id=args.exchange_id,
            cache_dir=args.cache_dir,
        )
        for symbol, count in result.items():
            print(f"{symbol}: +{count} rows")
        return

    if args.command == "list":
        for path in sorted(Path(args.cache_dir).glob("*_funding.parquet")):
            print(path.name)


if __name__ == "__main__":
    main()
