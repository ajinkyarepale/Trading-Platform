"""
backtest/data.py
─────────────────
Unified data-fetching layer.

Supported sources:
  • yfinance  – default, free
  • alphavantage – requires API key
  • csv       – load from local CSV files
  • sqlite    – read from the cache database

The module automatically:
  – Caches fetched data to SQLite to avoid redundant API calls.
  – Aligns timestamps across symbols.
  – Handles missing bars (forward-fill with configurable limit).
  – Applies adjustments for splits/dividends (yfinance "auto_adjust=True").
"""

from __future__ import annotations

import logging
import sqlite3
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Literal

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Valid timeframes and their pandas freq equivalents
TIMEFRAME_MAP: Dict[str, str] = {
    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "4h": "4h", "1d": "1D", "1w": "1W", "1mo": "1ME",
}

DataSource = Literal["yfinance", "alphavantage", "csv", "sqlite"]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(symbol: str, timeframe: str, start: str, end: str) -> str:
    return f"{symbol}_{timeframe}_{start}_{end}"


class DataCache:
    """SQLite-backed OHLCV cache."""

    def __init__(self, cache_path: Path):
        self.path = cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS ohlcv_cache (
                    cache_key TEXT PRIMARY KEY,
                    symbol TEXT,
                    timeframe TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    data TEXT,        -- CSV string
                    fetched_at TEXT DEFAULT (datetime('now'))
                )"""
            )

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT data FROM ohlcv_cache WHERE cache_key=?", (key,)
            ).fetchone()
        if row:
            return pd.read_csv(StringIO(row[0]), index_col=0, parse_dates=True)
        return None

    def set(self, key: str, symbol: str, timeframe: str,
            start: str, end: str, df: pd.DataFrame):
        csv_str = df.to_csv()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT OR REPLACE INTO ohlcv_cache
                   (cache_key, symbol, timeframe, start_date, end_date, data)
                   VALUES (?,?,?,?,?,?)""",
                (key, symbol, timeframe, start, end, csv_str),
            )


# ── Fetchers ──────────────────────────────────────────────────────────────────

def _fetch_yfinance(symbol: str, start: str, end: str, timeframe: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance not installed. Run: pip install yfinance")

    interval = timeframe  # yfinance uses same keys
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval=interval, auto_adjust=True)
    if df.empty:
        raise ValueError(f"yfinance returned no data for {symbol}")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("UTC").tz_localize(None)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    return df


def _fetch_alphavantage(symbol: str, start: str, end: str,
                         timeframe: str, api_key: str) -> pd.DataFrame:
    import requests

    func_map = {
        "1d": "TIME_SERIES_DAILY_ADJUSTED",
        "1w": "TIME_SERIES_WEEKLY_ADJUSTED",
        "1mo": "TIME_SERIES_MONTHLY_ADJUSTED",
    }
    intraday_map = {"1m": "1min", "5m": "5min", "15m": "15min",
                    "30m": "30min", "1h": "60min"}

    if timeframe in func_map:
        function = func_map[timeframe]
        params = {"function": function, "symbol": symbol,
                  "outputsize": "full", "apikey": api_key}
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        data = resp.json()
        key = [k for k in data if "Time Series" in k][0]
        df = pd.DataFrame(data[key]).T
        df.index = pd.to_datetime(df.index)
        col_map = {c: c.split(". ")[1] for c in df.columns}
        df = df.rename(columns=col_map).astype(float)
        df = df.rename(columns={"adjusted close": "close",
                                  "open": "open", "high": "high",
                                  "low": "low", "volume": "volume"})
    elif timeframe in intraday_map:
        iv = intraday_map[timeframe]
        params = {"function": "TIME_SERIES_INTRADAY", "symbol": symbol,
                  "interval": iv, "outputsize": "full", "apikey": api_key}
        resp = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        data = resp.json()
        key = [k for k in data if "Time Series" in k][0]
        df = pd.DataFrame(data[key]).T.astype(float)
        df.index = pd.to_datetime(df.index)
        df.columns = ["open", "high", "low", "close", "volume"]
    else:
        raise ValueError(f"Unsupported timeframe for Alpha Vantage: {timeframe}")

    df = df.sort_index()
    df = df.loc[start:end]
    return df[["open", "high", "low", "close", "volume"]]


def _fetch_csv(symbol: str, start: str, end: str,
               timeframe: str, csv_dir: Path) -> pd.DataFrame:
    """Load OHLCV from a CSV file named <csv_dir>/<symbol>_<timeframe>.csv"""
    path = csv_dir / f"{symbol}_{timeframe}.csv"
    if not path.exists():
        path = csv_dir / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    df = df.loc[start:end]
    return df[["open", "high", "low", "close", "volume"]]


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_data(
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "1d",
    source: DataSource = "yfinance",
    use_cache: bool = True,
    cache_path: Optional[Path] = None,
    alpha_vantage_key: Optional[str] = None,
    csv_dir: Optional[Path] = None,
    fill_missing: bool = True,
    fill_limit: int = 5,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for *symbol*.

    Returns a DataFrame with columns: open, high, low, close, volume
    indexed by UTC datetime (tz-naive).
    """
    if timeframe not in TIMEFRAME_MAP:
        raise ValueError(f"timeframe must be one of {list(TIMEFRAME_MAP)}")

    cache: Optional[DataCache] = None
    if use_cache and cache_path:
        cache = DataCache(cache_path)
        key = _cache_key(symbol, timeframe, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s [%s]", symbol, timeframe)
            return cached

    logger.info("Fetching %s from %s [%s – %s] via %s", symbol, start, end, timeframe, source)

    if source == "yfinance":
        df = _fetch_yfinance(symbol, start, end, timeframe)
    elif source == "alphavantage":
        if not alpha_vantage_key:
            raise ValueError("alpha_vantage_key is required for AlphaVantage source")
        df = _fetch_alphavantage(symbol, start, end, timeframe, alpha_vantage_key)
    elif source == "csv":
        if not csv_dir:
            raise ValueError("csv_dir is required for CSV source")
        df = _fetch_csv(symbol, start, end, timeframe, csv_dir)
    else:
        raise ValueError(f"Unknown source: {source}")

    # Standardise
    df = df.sort_index()
    if fill_missing:
        expected_freq = TIMEFRAME_MAP[timeframe]
        idx = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
        df = df.reindex(idx).ffill(limit=fill_limit)
        df = df.dropna()

    if cache:
        cache.set(key, symbol, timeframe, start, end, df)

    return df


def fetch_multi(
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = "1d",
    price_col: str = "close",
    **kwargs,
) -> pd.DataFrame:
    """
    Fetch multiple symbols and return a single DataFrame of *price_col* values,
    aligned to a common index.
    """
    frames: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            df = fetch_data(sym, start, end, timeframe, **kwargs)
            frames[sym] = df[price_col]
        except Exception as exc:
            logger.warning("Could not fetch %s: %s", sym, exc)

    if not frames:
        raise ValueError("No data fetched for any symbol")

    result = pd.DataFrame(frames)
    # Forward-fill small gaps then drop rows with ANY missing
    result = result.ffill(limit=3).dropna()
    return result
