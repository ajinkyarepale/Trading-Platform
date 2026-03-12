"""
strategies/zscore_mean_reversion.py
─────────────────────────────────────
Classic Z-score mean-reversion strategy.

Logic:
  - Compute a rolling z-score of the close price (or spread for pairs).
  - Go LONG when z < -entry_z  (price is cheap vs history)
  - Go SHORT when z > +entry_z  (price is expensive vs history)
  - Exit when |z| < exit_z

Works well on mean-reverting instruments (e.g., FX crosses, ETF spreads).
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata


class ZScoreMeanReversion(Strategy):
    metadata = StrategyMetadata(
        name="ZScoreMeanReversion",
        version="1.1.0",
        author="QuantTrader",
        description="Rolling z-score mean-reversion on price series.",
        tags=["mean-reversion", "statistical"],
        param_schema={
            # name: (type, min, max, default)
            "lookback":  (int,   10,  500,  60),
            "entry_z":   (float, 0.5, 5.0,  2.0),
            "exit_z":    (float, 0.0, 3.0,  0.5),
        },
    )

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> pd.Series:
        params = self.validate_params(params)
        lookback = params["lookback"]
        entry_z  = params["entry_z"]
        exit_z   = params["exit_z"]

        if isinstance(data, dict):
            # Use the first symbol's close if multiple passed
            data = next(iter(data.values()))

        price = data["close"]
        rolling_mean = price.rolling(lookback).mean()
        rolling_std  = price.rolling(lookback).std(ddof=1)
        zscore = (price - rolling_mean) / rolling_std.replace(0, np.nan)

        signal = pd.Series(0, index=data.index, dtype=int)
        position = 0

        for i in range(lookback, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -entry_z:
                    position = 1    # long
                elif z > entry_z:
                    position = -1   # short
            elif position == 1 and z > -exit_z:
                position = 0
            elif position == -1 and z < exit_z:
                position = 0
            signal.iloc[i] = position

        return signal
