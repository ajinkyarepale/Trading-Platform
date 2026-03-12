"""
live/paper_broker.py
=====================
Paper trading broker that simulates fills using yfinance data.
Used for testing live trading logic without real money.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from live.broker_base import (
    AccountInfo, Broker, Order, OrderSide, OrderStatus,
    OrderType, Position
)
from backtest.data import DataFetcher
from utils.logger import get_logger

log = get_logger(__name__)


class PaperBroker(Broker):
    """
    Simulated paper trading broker.

    Fills market orders immediately at current price + slippage.
    State is stored in memory (not persisted between restarts).
    """

    def __init__(
        self,
        initial_balance: float = 100_000.0,
        slippage_pct: float = 0.0001,
        commission_pct: float = 0.001,
    ) -> None:
        super().__init__(paper=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct

        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._fetcher = DataFetcher(cache_enabled=True)

    def connect(self) -> bool:
        self._connected = True
        log.info("PaperBroker connected (paper trading mode)")
        return True

    def disconnect(self) -> None:
        self._connected = False

    def get_account(self) -> AccountInfo:
        unrealised = sum(p.unrealised_pnl for p in self._positions.values())
        return AccountInfo(
            account_id="PAPER-001",
            balance=self.balance,
            equity=self.balance + unrealised,
            unrealised_pnl=unrealised,
            is_paper=True,
        )

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        units: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        order_id = str(uuid.uuid4())[:8]
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            units=units,
            limit_price=limit_price,
            stop_price=stop_price,
            timestamp=datetime.utcnow(),
        )

        if order_type == OrderType.MARKET:
            self._fill_market_order(order)
        else:
            order.status = OrderStatus.PENDING
            log.info(f"Limit order placed: {order_id} | {side.value} {units} {symbol} @ {limit_price}")

        self._orders[order_id] = order
        return order

    def _fill_market_order(self, order: Order) -> None:
        """Simulate immediate fill for market orders."""
        price = self._get_current_price(order.symbol)
        if price is None:
            order.status = OrderStatus.REJECTED
            return

        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        commission = fill_price * order.units * self.commission_pct
        trade_value = fill_price * order.units

        if order.side == OrderSide.BUY:
            if trade_value + commission > self.balance:
                order.status = OrderStatus.REJECTED
                log.warning(f"Rejected {order.order_id}: insufficient balance")
                return
            self.balance -= trade_value + commission
            # Update position
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                new_units = pos.units + order.units
                if new_units == 0:
                    del self._positions[order.symbol]
                else:
                    avg_price = (pos.units * pos.entry_price + order.units * fill_price) / new_units
                    pos.units = new_units
                    pos.entry_price = avg_price
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    units=order.units,
                    entry_price=fill_price,
                    current_price=fill_price,
                    side="long",
                )
        else:  # SELL
            self.balance += trade_value - commission
            if order.symbol in self._positions:
                pos = self._positions[order.symbol]
                pos.units -= order.units
                if abs(pos.units) < 1e-9:
                    del self._positions[order.symbol]

        order.status = OrderStatus.FILLED
        order.filled_price = fill_price
        order.filled_units = order.units
        log.info(f"Filled {order.order_id}: {order.side.value} {order.units} {order.symbol} @ {fill_price:.4f} (comm={commission:.2f})")

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._orders:
            self._orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders.values() if o.status == OrderStatus.PENDING]

    def get_positions(self) -> List[Position]:
        # Update unrealised PnL
        for sym, pos in self._positions.items():
            price = self._get_current_price(sym)
            if price:
                pos.current_price = price
                pos.unrealised_pnl = (price - pos.entry_price) * pos.units
        return list(self._positions.values())

    def get_bars(self, symbol: str, timeframe: str = "1d", count: int = 100) -> pd.DataFrame:
        from datetime import timedelta
        end = datetime.utcnow().strftime("%Y-%m-%d")
        # Approximate start date
        days = count * 1.5
        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            df = self._fetcher.fetch(symbol, start, end, timeframe)
            return df.tail(count)
        except Exception as e:
            log.error(f"Failed to get bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Dict[str, float]:
        price = self._get_current_price(symbol) or 0.0
        spread = price * 0.0001
        return {"bid": price - spread, "ask": price + spread, "last": price}

    def _get_current_price(self, symbol: str) -> Optional[float]:
        try:
            bars = self.get_bars(symbol, "1d", count=5)
            if bars.empty:
                return None
            return float(bars["close"].iloc[-1])
        except Exception:
            return None
