"""
strategies/breakout.py
───────────────────────
Donchian channel breakout with ATR-based stop loss.

Logic:
  - Upper channel = highest high over N bars.
  - Lower channel = lowest low over N bars.
  - BUY  when close breaks above upper channel.
  - SELL when close breaks below lower channel.
  - Stop loss = entry_price ∓ atr_mult × ATR(atr_period).
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = data["high"], data["low"], data["close"]
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


class BreakoutStrategy(Strategy):
    metadata = StrategyMetadata(
        name="BreakoutStrategy",
        version="1.0.0",
        author="QuantTrader",
        description="Donchian channel breakout with ATR stop loss.",
        tags=["breakout", "trend-following", "donchian"],
        param_schema={
            "channel_period": (int,   5,   200, 20),
            "atr_period":     (int,   5,   50,  14),
            "atr_mult":       (float, 0.5, 5.0, 2.0),
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

        ch  = params["channel_period"]
        atr = _atr(data, params["atr_period"])

        upper = data["high"].rolling(ch).max().shift(1)
        lower = data["low"].rolling(ch).min().shift(1)

        signal   = pd.Series(0, index=data.index, dtype=int)
        position = 0
        stop     = np.nan
        entry_px = np.nan

        for i in range(ch, len(data)):
            close = data["close"].iloc[i]

            # Check stop loss
            if position == 1 and close < stop:
                position = 0
                stop = np.nan
            elif position == -1 and close > stop:
                position = 0
                stop = np.nan

            # Entry signals
            if position == 0:
                if close > upper.iloc[i]:
                    position = 1
                    entry_px = close
                    stop = close - params["atr_mult"] * atr.iloc[i]
                elif close < lower.iloc[i]:
                    position = -1
                    entry_px = close
                    stop = close + params["atr_mult"] * atr.iloc[i]

            signal.iloc[i] = position

        return signal
