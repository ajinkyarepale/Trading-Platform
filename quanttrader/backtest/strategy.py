"""
backtest/strategy.py
─────────────────────
Abstract base class for all trading strategies.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StrategyMetadata:
    name: str
    version: str = "1.0.0"
    author: str = "unknown"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    # Parameter schema: {param_name: (type, min, max, default)}
    param_schema: Dict[str, tuple] = field(default_factory=dict)


class Strategy(abc.ABC):
    """
    Base strategy class.

    Sub-classes must:
      1. Set a ``metadata`` class attribute (StrategyMetadata).
      2. Implement ``generate_signals(data, params)``.

    Signals:
      • Single-symbol: return pd.Series with values in {-1, 0, 1}.
      • Multi-symbol:  return pd.DataFrame, one column per symbol.

    Values:
      -1  → short
       0  → flat / close any open position
       1  → long
    """

    metadata: StrategyMetadata = StrategyMetadata(name="BaseStrategy")

    # ── Lifecycle hooks (optional overrides) ─────────────────────────────────

    def on_start(self, data: pd.DataFrame, params: Dict[str, Any]) -> None:
        """Called once before the backtest loop starts."""

    def on_end(self, data: pd.DataFrame, params: Dict[str, Any]) -> None:
        """Called once after the backtest loop ends."""

    # ── Core interface ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Given historical OHLCV data and a parameter dictionary,
        return a signal series / DataFrame aligned to *data*'s index.
        """

    # ── Validation helpers ────────────────────────────────────────────────────

    def validate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate *params* against ``metadata.param_schema``.
        Missing params are filled with schema defaults.
        Raises ValueError for out-of-range values.
        """
        schema = self.metadata.param_schema
        result: Dict[str, Any] = {}
        for name, (dtype, lo, hi, default) in schema.items():
            val = params.get(name, default)
            try:
                val = dtype(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Param '{name}' must be {dtype.__name__}: {exc}") from exc
            if lo is not None and val < lo:
                raise ValueError(f"Param '{name}' must be >= {lo}, got {val}")
            if hi is not None and val > hi:
                raise ValueError(f"Param '{name}' must be <= {hi}, got {val}")
            result[name] = val
        # Pass through any extra params not in schema
        for k, v in params.items():
            if k not in result:
                result[k] = v
        return result

    def __repr__(self) -> str:
        return f"<Strategy: {self.metadata.name} v{self.metadata.version}>"
