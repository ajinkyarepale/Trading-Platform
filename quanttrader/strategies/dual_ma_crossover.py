"""
strategies/dual_ma_crossover.py
────────────────────────────────
Dual moving average crossover with optional volatility filter.

Logic:
  - BUY  when fast MA crosses above slow MA  AND  (optionally) volatility < vol_threshold
  - SELL when fast MA crosses below slow MA
  - ma_type: 'sma' | 'ema'
  - vol_filter: if True, skip signals when ATR/Close > vol_threshold
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata


def _ma(series: pd.Series, period: int, kind: str) -> pd.Series:
    if kind == "ema":
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(period).mean()


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = data["high"], data["low"], data["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class DualMACrossover(Strategy):
    metadata = StrategyMetadata(
        name="DualMACrossover",
        version="1.0.0",
        author="QuantTrader",
        description="SMA/EMA crossover with volatility filter.",
        tags=["trend-following", "moving-average"],
        param_schema={
            "fast_period":    (int,   2,   100,  10),
            "slow_period":    (int,   5,   500,  50),
            "ma_type":        (str,   None, None, "sma"),
            "vol_filter":     (bool,  None, None, False),
            "vol_threshold":  (float, 0.001, 0.2, 0.03),
            "atr_period":     (int,   5,    50,   14),
        },
    )

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> pd.Series:
        params = self.validate_params(params)

        if isinstance(data, dict):
            data = next(iter(data.values()))

        fast = _ma(data["close"], params["fast_period"], params["ma_type"])
        slow = _ma(data["close"], params["slow_period"], params["ma_type"])

        signal = pd.Series(0, index=data.index, dtype=int)

        # Vol filter
        if params["vol_filter"]:
            atr = _atr(data, params["atr_period"])
            vol_ratio = atr / data["close"]
            low_vol = vol_ratio < params["vol_threshold"]
        else:
            low_vol = pd.Series(True, index=data.index)

        cross_up   = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        cross_down = (fast < slow) & (fast.shift(1) >= slow.shift(1))

        position = 0
        for i in range(len(data)):
            if cross_up.iloc[i] and low_vol.iloc[i]:
                position = 1
            elif cross_down.iloc[i]:
                position = -1
            signal.iloc[i] = position

        return signal
