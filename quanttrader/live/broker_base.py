"""
live/broker_base.py
====================
Abstract Broker interface that all live connectors must implement.

All connectors (Alpaca, IBKR, MT5, Binance) implement this interface
so the rest of the system can be broker-agnostic.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class AccountInfo:
    """Broker account information."""
    account_id: str
    balance: float
    equity: float
    margin_used: float = 0.0
    margin_available: float = 0.0
    unrealised_pnl: float = 0.0
    realised_pnl: float = 0.0
    currency: str = "USD"
    is_paper: bool = True


@dataclass
class Position:
    """An open position."""
    symbol: str
    units: float          # Positive=long, negative=short
    entry_price: float
    current_price: float
    unrealised_pnl: float = 0.0
    side: str = "long"    # "long" or "short"


@dataclass
class Order:
    """An order (placed or historical)."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    units: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_price: Optional[float] = None
    filled_units: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Broker(abc.ABC):
    """
    Abstract broker interface.

    All live connectors must implement this interface so that
    the live trading engine and risk manager can be broker-agnostic.
    """

    def __init__(self, paper: bool = True) -> None:
        self.paper = paper
        self._connected = False

    # ── Connection lifecycle ────────────────────────────────────────────────

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection to the broker. Returns True on success."""
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ── Account ─────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account information."""
        ...

    # ── Orders ──────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        units: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Place an order. Returns Order with status."""
        ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order. Returns True if cancelled."""
        ...

    @abc.abstractmethod
    def get_open_orders(self) -> List[Order]:
        """Return all open/pending orders."""
        ...

    # ── Positions ────────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_positions(self) -> List[Position]:
        """Return all open positions."""
        ...

    def get_position(self, symbol: str) -> Optional[Position]:
        """Return position for a specific symbol (or None)."""
        for pos in self.get_positions():
            if pos.symbol == symbol:
                return pos
        return None

    def close_position(self, symbol: str) -> Optional[Order]:
        """Close the entire position for a symbol."""
        pos = self.get_position(symbol)
        if pos is None:
            return None
        side = OrderSide.SELL if pos.units > 0 else OrderSide.BUY
        return self.place_order(symbol, side, abs(pos.units))

    # ── Market Data ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def get_bars(
        self,
        symbol: str,
        timeframe: str,
        count: int = 100,
    ) -> pd.DataFrame:
        """Return recent OHLCV bars."""
        ...

    @abc.abstractmethod
    def get_quote(self, symbol: str) -> Dict[str, float]:
        """Return current bid/ask/last for a symbol."""
        ...
