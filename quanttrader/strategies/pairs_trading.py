"""
strategies/pairs_trading.py
────────────────────────────
Statistical arbitrage pairs trading.

Logic:
  1. Estimate hedge ratio (β) via OLS on training window.
  2. Compute spread = leg1 - β * leg2.
  3. Apply z-score mean reversion on the spread.
  4. LONG spread (buy leg1, sell leg2) when z < -entry_z.
  5. SHORT spread (sell leg1, buy leg2) when z > +entry_z.

Supports cointegration check (Engle-Granger test) during on_start().
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata

logger = logging.getLogger(__name__)


def _engle_granger_coint(y: pd.Series, x: pd.Series) -> float:
    """Return p-value of Engle-Granger cointegration test."""
    try:
        from statsmodels.tsa.stattools import coint
        _, pvalue, _ = coint(y, x)
        return float(pvalue)
    except ImportError:
        logger.warning("statsmodels not installed; skipping cointegration test.")
        return 0.05


def _hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """OLS hedge ratio: y = β·x + ε  →  return β."""
    x_mat = np.column_stack([x.values, np.ones(len(x))])
    result = np.linalg.lstsq(x_mat, y.values, rcond=None)
    return float(result[0][0])


class PairsTrading(Strategy):
    """
    Multi-symbol strategy: expects data = {"LEG1": df1, "LEG2": df2}.
    Returns a DataFrame with columns LEG1 and LEG2 holding +1/-1 signals.
    """

    metadata = StrategyMetadata(
        name="PairsTrading",
        version="1.1.0",
        author="QuantTrader",
        description="Z-score spread mean-reversion for correlated pairs.",
        tags=["pairs", "mean-reversion", "statistical-arb"],
        param_schema={
            "lookback":       (int,   20,  500,  60),
            "entry_z":        (float, 0.5, 5.0,  2.0),
            "exit_z":         (float, 0.0, 3.0,  0.5),
            "hedge_window":   (int,   20,  500,  60),
            "coint_pvalue":   (float, 0.01, 0.2, 0.05),
            "check_coint":    (bool,  None, None, True),
        },
    )

    def on_start(self, data, params):
        if not isinstance(data, dict) or len(data) < 2:
            raise ValueError("PairsTrading requires data = {'LEG1': df1, 'LEG2': df2}")
        if params.get("check_coint", True):
            syms = list(data.keys())
            y = data[syms[0]]["close"]
            x = data[syms[1]]["close"]
            pval = _engle_granger_coint(y, x)
            if pval > params.get("coint_pvalue", 0.05):
                logger.warning("Pair may not be cointegrated (p=%.3f). Proceed with caution.", pval)
            else:
                logger.info("Cointegration confirmed (p=%.3f).", pval)

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ) -> pd.DataFrame:
        params = self.validate_params(params)
        syms = list(data.keys())
        leg1_sym, leg2_sym = syms[0], syms[1]

        y = data[leg1_sym]["close"]
        x = data[leg2_sym]["close"]
        common_idx = y.index.intersection(x.index)
        y, x = y.loc[common_idx], x.loc[common_idx]

        lookback     = params["lookback"]
        hedge_window = params["hedge_window"]
        entry_z      = params["entry_z"]
        exit_z       = params["exit_z"]

        # Rolling hedge ratio and spread
        hedge_ratios = pd.Series(np.nan, index=common_idx)
        for i in range(hedge_window, len(y)):
            sl = slice(i - hedge_window, i)
            hedge_ratios.iloc[i] = _hedge_ratio(y.iloc[sl], x.iloc[sl])

        spread = y - hedge_ratios * x
        roll_mean = spread.rolling(lookback).mean()
        roll_std  = spread.rolling(lookback).std(ddof=1)
        zscore = (spread - roll_mean) / roll_std.replace(0, np.nan)

        sig1 = pd.Series(0, index=common_idx, dtype=int)
        sig2 = pd.Series(0, index=common_idx, dtype=int)
        position = 0

        start = max(lookback, hedge_window)
        for i in range(start, len(zscore)):
            z = zscore.iloc[i]
            if np.isnan(z):
                continue
            if position == 0:
                if z < -entry_z:
                    position = 1    # long spread: buy leg1, sell leg2
                elif z > entry_z:
                    position = -1   # short spread: sell leg1, buy leg2
            elif position == 1 and z > -exit_z:
                position = 0
            elif position == -1 and z < exit_z:
                position = 0
            sig1.iloc[i] = position
            sig2.iloc[i] = -position   # opposite leg

        return pd.DataFrame({leg1_sym: sig1, leg2_sym: sig2})
