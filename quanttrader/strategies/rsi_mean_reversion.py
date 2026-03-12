"""
strategies/rsi_mean_reversion.py
──────────────────────────────────
RSI-based mean reversion.

Logic:
  - Compute RSI(rsi_period)
  - Go LONG  when RSI < oversold  (default 30)
  - Go SHORT when RSI > overbought (default 70)
  - Exit when RSI crosses back through exit_rsi (default 50)
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class RSIMeanReversion(Strategy):
    metadata = StrategyMetadata(
        name="RSIMeanReversion",
        version="1.0.0",
        author="QuantTrader",
        description="RSI overbought/oversold mean reversion.",
        tags=["mean-reversion", "oscillator", "rsi"],
        param_schema={
            "rsi_period":   (int,   2,  100, 14),
            "oversold":     (float, 5,  45,  30.0),
            "overbought":   (float, 55, 95,  70.0),
            "exit_rsi":     (float, 40, 60,  50.0),
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

        rsi = _rsi(data["close"], params["rsi_period"])
        oversold    = params["oversold"]
        overbought  = params["overbought"]
        exit_level  = params["exit_rsi"]

        signal = pd.Series(0, index=data.index, dtype=int)
        position = 0

        for i in range(len(rsi)):
            r = rsi.iloc[i]
            if np.isnan(r):
                continue
            if position == 0:
                if r < oversold:
                    position = 1
                elif r > overbought:
                    position = -1
            elif position == 1 and r > exit_level:
                position = 0
            elif position == -1 and r < exit_level:
                position = 0
            signal.iloc[i] = position

        return signal
