from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


def _normalize_symbol(symbol: str) -> str:
    """Normalize an input spot/perp symbol to ccxt USDT perpetual format."""
    return symbol if ":" in symbol else f"{symbol}:USDT"


def _cache_file(symbol: str, cache_dir: str) -> Path:
    """Return the funding cache file path for a symbol."""
    base_symbol = _normalize_symbol(symbol).replace(":USDT", "")
    safe = base_symbol.replace("/", "_")
    return Path(cache_dir) / f"{safe}_funding.parquet"


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
    """Download historical funding rates for a single symbol via ccxt.

    Args:
        symbol: Trading pair in ccxt futures format (e.g. "BTC/USDT:USDT").
            If given as "BTC/USDT", automatically append ":USDT" for futures.
        since: ISO-8601 start date string (e.g. "2022-01-01"). If None,
            fetches from the earliest available.
        limit: Page size per API request (max 1000 for Binance).
        exchange_id: CCXT exchange identifier (default "binance").

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns:
        [symbol, funding_rate, mark_price]
    """
    import ccxt

    normalized = _normalize_symbol(symbol)
    exchange_cls = getattr(ccxt, exchange_id)
    exchange = exchange_cls({"options": {"defaultType": "future"}})

    since_ms = None
    if since is not None:
        since_ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)

    rows: list[dict[str, Any]] = []
    current_since = since_ms
    while True:
        batch = exchange.fetch_funding_rate_history(normalized, since=current_since, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
        current_since = int(batch[-1]["timestamp"]) + 1
        time.sleep(1.0)

    df = _to_dataframe(rows)
    if not df.empty:
        df = df[~df.index.duplicated(keep="last")]
    return df


def load_funding_rates(
    symbol: str,
    cache_dir: str = "data_cache/funding",
) -> pd.DataFrame:
    """Load cached funding rates from Parquet.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT").
        cache_dir: Cache directory path.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns:
        [symbol, funding_rate, mark_price]

    Raises:
        FileNotFoundError: If no cache file exists for this symbol.
    """
    path = _cache_file(symbol, cache_dir)
    if not path.exists():
        raise FileNotFoundError(f"No funding cache found for {symbol}: {path}")
    return pd.read_parquet(path)


def download_all_universe(
    symbols: list[str],
    days: int = 1460,
    exchange_id: str = "binance",
    cache_dir: str = "data_cache/funding",
) -> dict[str, int]:
    """Batch-download funding rates for multiple symbols.

    Downloads sequentially to respect rate limits. Appends to existing
    cache files (does not re-download data that is already cached).
    Shows progress bar via tqdm.

    Args:
        symbols: List of symbols in ccxt format.
        days: Number of days of history to fetch.
        exchange_id: CCXT exchange identifier.
        cache_dir: Directory for Parquet cache files.

    Returns:
        Dict mapping symbol to number of new records downloaded.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    default_since = datetime.now(timezone.utc) - timedelta(days=days)

    stats: dict[str, int] = {}
    for symbol in tqdm(symbols, desc="Funding", unit="symbol"):
        cache_path = _cache_file(symbol, cache_dir)
        existing = pd.DataFrame(columns=["symbol", "funding_rate", "mark_price"])
        if cache_path.exists():
            existing = pd.read_parquet(cache_path)

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


def build_funding_features(
    funding_df: pd.DataFrame,
    ohlcv_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build funding rate features aligned to an OHLCV DatetimeIndex.

    Resamples 8h funding data to 1h via forward-fill, then computes:

    - funding_rate: raw rate, forward-filled from 8h to 1h
    - funding_rate_ma_24: 24-bar simple moving average
    - funding_rate_ma_168: 168-bar (1 week) simple moving average
    - funding_cumulative_24h: rolling 24-bar sum (proxy for crowding cost)
    - funding_zscore: z-score over 168-bar window
    - funding_positive_streak: consecutive hours of positive funding (resets on sign change)
    - funding_flip: binary flag (1 if funding changed sign this bar, else 0)

    Args:
        funding_df: Raw funding rate DataFrame with DatetimeIndex and
            'funding_rate' column (8h native frequency).
        ohlcv_df: OHLCV DataFrame with 1h DatetimeIndex to align to.

    Returns:
        DataFrame indexed identically to ohlcv_df with the 7 feature columns.
        Missing values at the start (from rolling windows) are NaN.
    """
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
        with open(args.config, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        symbols = [entry["symbol"] for entry in config.get("assets", [])]
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
