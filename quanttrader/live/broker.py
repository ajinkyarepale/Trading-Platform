"""
live/broker.py
───────────────
Abstract Broker interface and PaperBroker implementation.

All real connectors (Alpaca, IBKR, MT5, CCXT) implement the same interface,
making it trivial to swap brokers without changing strategy code.
"""

from __future__ import annotations

import abc
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Order:
    order_id: str
    symbol: str
    side: str          # "buy" | "sell"
    qty: float
    order_type: str    # "market" | "limit"
    limit_price: Optional[float] = None
    status: str = "pending"
    filled_price: Optional[float] = None
    filled_qty: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None


@dataclass
class Position:
    symbol: str
    qty: float          # positive = long, negative = short
    avg_price: float
    current_price: float = 0.0

    @property
    def unrealised_pnl(self) -> float:
        return self.qty * (self.current_price - self.avg_price)

    @property
    def market_value(self) -> float:
        return abs(self.qty) * self.current_price


@dataclass
class AccountInfo:
    cash: float
    equity: float
    buying_power: float
    margin_used: float = 0.0
    currency: str = "USD"


class Broker(abc.ABC):
    """Common interface for all broker connectors."""

    @abc.abstractmethod
    def connect(self) -> bool:
        """Establish connection. Returns True on success."""

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """Return current account state."""

    @abc.abstractmethod
    def place_order(
        self, symbol: str, side: str, qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        """Submit an order. Returns Order object."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""

    @abc.abstractmethod
    def get_positions(self) -> List[Position]:
        """Return all open positions."""

    @abc.abstractmethod
    def get_bars(
        self, symbol: str, timeframe: str, limit: int = 200
    ) -> pd.DataFrame:
        """Return recent OHLCV bars."""

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Clean up connection."""


class PaperBroker(Broker):
    """
    In-process paper trading broker.
    Suitable for strategy testing without real money.
    """

    def __init__(self, initial_capital: float = 100_000.0,
                 slippage_pct: float = 0.0005):
        self.initial_capital = initial_capital
        self.slippage_pct = slippage_pct
        self._cash = initial_capital
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._prices: Dict[str, float] = {}
        self._connected = False

    # ── Interface ─────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        self._connected = True
        logger.info("PaperBroker connected (capital=%.2f)", self._cash)
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("PaperBroker disconnected.")

    def get_account(self) -> AccountInfo:
        equity = self._cash + sum(
            p.unrealised_pnl for p in self._positions.values()
        )
        return AccountInfo(cash=self._cash, equity=equity,
                           buying_power=self._cash)

    def place_order(
        self, symbol: str, side: str, qty: float,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> Order:
        order_id = str(uuid.uuid4())[:8]
        price = self._prices.get(symbol, 0.0)
        if price == 0.0:
            raise ValueError(f"No price available for {symbol}. Call update_price() first.")

        # Simulate slippage
        slip = price * self.slippage_pct
        fill_price = price + slip if side == "buy" else price - slip

        cost = fill_price * qty
        if side == "buy":
            if cost > self._cash:
                raise ValueError(f"Insufficient funds: need {cost:.2f}, have {self._cash:.2f}")
            self._cash -= cost
            if symbol in self._positions:
                pos = self._positions[symbol]
                new_qty = pos.qty + qty
                new_avg = (pos.qty * pos.avg_price + qty * fill_price) / new_qty
                self._positions[symbol] = Position(symbol, new_qty, new_avg, fill_price)
            else:
                self._positions[symbol] = Position(symbol, qty, fill_price, fill_price)
        else:  # sell
            if symbol not in self._positions:
                raise ValueError(f"No position in {symbol} to sell.")
            pos = self._positions[symbol]
            pnl = (fill_price - pos.avg_price) * qty
            self._cash += cost + pnl
            remaining = pos.qty - qty
            if remaining <= 0.001:
                del self._positions[symbol]
            else:
                self._positions[symbol] = Position(symbol, remaining,
                                                    pos.avg_price, fill_price)

        order = Order(
            order_id=order_id, symbol=symbol, side=side, qty=qty,
            order_type=order_type, limit_price=limit_price,
            status="filled", filled_price=fill_price,
            filled_qty=qty, filled_at=datetime.utcnow(),
        )
        self._orders[order_id] = order
        logger.info("FILL: %s %s %.4f @ %.4f", side.upper(), symbol, qty, fill_price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = "cancelled"
            return True
        return False

    def get_positions(self) -> List[Position]:
        return list(self._positions.values())

    def get_bars(self, symbol: str, timeframe: str = "1d",
                 limit: int = 200) -> pd.DataFrame:
        """Fetch bars via yfinance for paper trading purposes."""
        import yfinance as yf
        import pandas as pd
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{limit}d", interval=timeframe)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]

    def update_price(self, symbol: str, price: float) -> None:
        """Update current market price for a symbol."""
        self._prices[symbol] = price
        if symbol in self._positions:
            self._positions[symbol].current_price = price

    @property
    def equity(self) -> float:
        return self.get_account().equity


# ── Alpaca connector stub ─────────────────────────────────────────────────────

class AlpacaBroker(Broker):
    """
    Alpaca Markets connector (US stocks & crypto).
    Requires: pip install alpaca-py
    Set env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self._client = None

    def connect(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient
            self._client = TradingClient(
                self.api_key, self.secret_key, paper=self.paper
            )
            logger.info("Alpaca connected (paper=%s)", self.paper)
            return True
        except Exception as exc:
            logger.error("Alpaca connection failed: %s", exc)
            return False

    def get_account(self) -> AccountInfo:
        acct = self._client.get_account()
        return AccountInfo(
            cash=float(acct.cash),
            equity=float(acct.equity),
            buying_power=float(acct.buying_power),
        )

    def place_order(self, symbol, side, qty, order_type="market", limit_price=None) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        s = OrderSide.BUY if side == "buy" else OrderSide.SELL
        if order_type == "market":
            req = MarketOrderRequest(symbol=symbol, qty=qty, side=s,
                                     time_in_force=TimeInForce.DAY)
        else:
            req = LimitOrderRequest(symbol=symbol, qty=qty, side=s,
                                    limit_price=limit_price,
                                    time_in_force=TimeInForce.DAY)
        o = self._client.submit_order(req)
        return Order(order_id=str(o.id), symbol=symbol, side=side,
                     qty=qty, order_type=order_type, limit_price=limit_price,
                     status=str(o.status))

    def cancel_order(self, order_id: str) -> bool:
        self._client.cancel_order_by_id(order_id)
        return True

    def get_positions(self) -> List[Position]:
        raw = self._client.get_all_positions()
        return [Position(p.symbol, float(p.qty), float(p.avg_entry_price),
                         float(p.current_price)) for p in raw]

    def get_bars(self, symbol: str, timeframe: str = "1d",
                 limit: int = 200) -> pd.DataFrame:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        tf_map = {"1d": TimeFrame.Day, "1h": TimeFrame.Hour, "1m": TimeFrame.Minute}
        tf = tf_map.get(timeframe, TimeFrame.Day)
        req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=tf,
                               limit=limit)
        bars = client.get_stock_bars(req).df
        bars.columns = [c.lower() for c in bars.columns]
        return bars

    def disconnect(self) -> None:
        self._client = None
