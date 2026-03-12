"""
ml_strategies/features.py
──────────────────────────
Technical indicator feature engineering for ML strategies.
Generates 20+ features without TA-Lib dependency.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Builds a feature matrix from OHLCV data.

    All features are forward-safe (use only past data).
    """

    def __init__(self, feature_list: Optional[List[str]] = None):
        self.feature_list = feature_list or self._default_features()

    @staticmethod
    def _default_features() -> List[str]:
        return [
            "ret_1d", "ret_5d", "ret_10d", "ret_20d",
            "vol_5d", "vol_10d", "vol_20d",
            "rsi_14", "rsi_7",
            "macd", "macd_signal",
            "bb_upper", "bb_lower", "bb_width",
            "atr_14", "atr_ratio",
            "sma_ratio_10_50", "sma_ratio_20_200",
            "high_low_ratio", "close_range_pct",
            "volume_ratio",
        ]

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of features aligned to *data*'s index."""
        feat: dict = {}
        c = data["close"]
        h = data["high"]
        l = data["low"]
        v = data.get("volume", pd.Series(1, index=data.index))

        # Returns
        for n in [1, 5, 10, 20]:
            feat[f"ret_{n}d"] = c.pct_change(n)

        # Volatility (rolling std of returns)
        for n in [5, 10, 20]:
            feat[f"vol_{n}d"] = c.pct_change().rolling(n).std(ddof=1)

        # RSI
        for p in [7, 14]:
            feat[f"rsi_{p}"] = self._rsi(c, p)

        # MACD
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        feat["macd"] = macd / c
        feat["macd_signal"] = (macd.ewm(span=9, adjust=False).mean()) / c

        # Bollinger Bands
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std(ddof=1)
        feat["bb_upper"] = (sma20 + 2 * std20) / c - 1
        feat["bb_lower"] = (sma20 - 2 * std20) / c - 1
        feat["bb_width"] = (4 * std20) / sma20

        # ATR
        atr = self._atr(data, 14)
        feat["atr_14"] = atr
        feat["atr_ratio"] = atr / c

        # SMA ratios
        for fast, slow in [(10, 50), (20, 200)]:
            sma_f = c.rolling(fast).mean()
            sma_s = c.rolling(slow).mean()
            feat[f"sma_ratio_{fast}_{slow}"] = sma_f / sma_s - 1

        # Price-bar features
        feat["high_low_ratio"] = h / l - 1
        feat["close_range_pct"] = (c - l) / (h - l + 1e-10)

        # Volume ratio
        feat["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)

        df = pd.DataFrame(feat, index=data.index)
        return df[self.feature_list].dropna()

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(data: pd.DataFrame, period: int) -> pd.Series:
        h, l, c = data["high"], data["low"], data["close"]
        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
