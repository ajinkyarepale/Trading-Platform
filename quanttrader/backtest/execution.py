"""
backtest/execution.py
──────────────────────
Simulates order fills, position sizing, and cost accounting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionSizing(str, Enum):
    FIXED = "fixed"          # fixed number of units
    PERCENT = "percent"      # percent of equity
    KELLY = "kelly"          # fractional Kelly (requires win_rate + avg_win/loss)


@dataclass
class InstrumentCosts:
    spread: float = 0.0
    commission_pct: float = 0.001
    commission_fixed: float = 0.0
    slippage_pct: float = 0.0005
    margin_rate: float = 1.0
    lot_size: float = 1.0
    min_lot: float = 1.0


@dataclass
class Trade:
    symbol: str
    entry_time: Any
    exit_time: Any
    direction: int          # 1 = long, -1 = short
    entry_price: float
    exit_price: float
    qty: float
    pnl: float              # in cash, after costs
    pnl_pct: float
    costs: float
    bars: int
    entry_signal: int
    exit_signal: int


def apply_slippage(price: float, direction: int, slippage_pct: float,
                   order_type: OrderType = OrderType.MARKET) -> float:
    """Return fill price after slippage. Buys get worse (higher), sells too."""
    if order_type == OrderType.LIMIT:
        return price  # assume exact fill at limit price
    return price * (1 + direction * slippage_pct)


def apply_spread(price: float, direction: int, spread: float) -> float:
    """Half-spread cost applied on entry."""
    return price + direction * spread / 2


def calc_commission(value: float, costs: InstrumentCosts) -> float:
    return abs(value) * costs.commission_pct + costs.commission_fixed


def calc_position_size(
    equity: float,
    price: float,
    sizing: PositionSizing,
    sizing_param: float,
    costs: InstrumentCosts,
    kelly_win_rate: float = 0.5,
    kelly_avg_win: float = 1.0,
    kelly_avg_loss: float = 1.0,
) -> float:
    """Return position size in units, respecting min_lot and lot_size."""
    if sizing == PositionSizing.FIXED:
        units = sizing_param
    elif sizing == PositionSizing.PERCENT:
        notional = equity * sizing_param
        units = notional / price if price > 0 else 0
    elif sizing == PositionSizing.KELLY:
        b = kelly_avg_win / kelly_avg_loss if kelly_avg_loss > 0 else 1
        p = kelly_win_rate
        q = 1 - p
        kelly_f = (b * p - q) / b if b > 0 else 0
        kelly_f = max(0, min(kelly_f, sizing_param))  # sizing_param = max fraction
        notional = equity * kelly_f
        units = notional / price if price > 0 else 0
    else:
        units = sizing_param

    # Round down to nearest lot_size
    if costs.lot_size > 0:
        units = np.floor(units / costs.lot_size) * costs.lot_size
    return max(units, 0)
