"""
live/trader.py
==============
Live trading engine with APScheduler integration.

Handles:
  - Strategy execution at scheduled intervals
  - Order placement with risk checks
  - State persistence
  - Event logging
"""
from __future__ import annotations

import json
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.strategy import Strategy
from live.broker_base import Broker, OrderSide, OrderType
from live.risk import RiskManager
from utils.logger import get_logger

log = get_logger(__name__)

STATE_FILE = Path("data/live_state.pkl")
EVENTS_DB = Path("data/live_events.db")


class LiveTrader:
    """
    Live trading engine.

    Runs a strategy on a schedule, manages positions with risk checks,
    and persists state for crash recovery.

    Example
    -------
    >>> broker = PaperBroker(initial_balance=100_000)
    >>> broker.connect()
    >>> trader = LiveTrader(strategy, broker, symbols=["AAPL"], timeframe="1d")
    >>> trader.start(schedule_cron="0 16 * * 1-5")  # Daily at 4pm on weekdays
    """

    def __init__(
        self,
        strategy: Strategy,
        broker: Broker,
        symbols: List[str],
        timeframe: str = "1d",
        lookback_bars: int = 200,
        state_file: Path = STATE_FILE,
        events_db: Path = EVENTS_DB,
        risk_manager: Optional[RiskManager] = None,
    ) -> None:
        self.strategy = strategy
        self.broker = broker
        self.symbols = symbols
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.state_file = state_file
        self.events_db = events_db

        self.risk_manager = risk_manager or RiskManager(broker)
        self._state: Dict[str, Any] = {}
        self._scheduler = None

        self._init_events_db()
        self._load_state()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(
        self,
        schedule_cron: Optional[str] = None,
        run_immediately: bool = True,
    ) -> None:
        """
        Start the live trader.

        Args:
            schedule_cron  : Cron expression for scheduled runs (e.g., "0 16 * * 1-5").
                             If None, runs once and exits.
            run_immediately: Run one iteration immediately on start.
        """
        if not self.broker.is_connected:
            raise RuntimeError("Broker not connected. Call broker.connect() first.")

        if run_immediately:
            self._run_iteration()

        if schedule_cron:
            try:
                from apscheduler.schedulers.blocking import BlockingScheduler
                from apscheduler.triggers.cron import CronTrigger
            except ImportError:
                raise ImportError("apscheduler not installed. Run: pip install apscheduler")

            self._scheduler = BlockingScheduler(timezone="UTC")
            trigger = CronTrigger.from_crontab(schedule_cron, timezone="UTC")
            self._scheduler.add_job(self._run_iteration, trigger=trigger, id="strategy_run")
            log.info(f"Scheduler started with cron: {schedule_cron}")
            self._scheduler.start()

    def stop(self) -> None:
        """Stop the scheduler."""
        if self._scheduler and self._scheduler.running:
            self._scheduler.shutdown()
        log.info("LiveTrader stopped")

    # ── Core iteration ────────────────────────────────────────────────────────

    def _run_iteration(self) -> None:
        """Single strategy execution cycle."""
        log.info(f"=== Strategy iteration: {datetime.utcnow().isoformat()} ===")
        try:
            # 1. Fetch latest data
            data_dict = self._fetch_data()
            if not data_dict:
                log.warning("No data fetched – skipping iteration")
                return

            # 2. Generate signals
            if len(self.symbols) == 1:
                signals = self.strategy.run(data_dict[self.symbols[0]])
                signal_dict = {self.symbols[0]: float(signals.iloc[-1]) if not signals.empty else 0}
            else:
                signals_df = self.strategy.run(data_dict)
                signal_dict = {sym: float(signals_df[sym].iloc[-1])
                               for sym in self.symbols if sym in signals_df.columns}

            log.info(f"Signals: {signal_dict}")
            self._log_event("signals", signal_dict)

            # 3. Execute orders
            for symbol, signal in signal_dict.items():
                self._execute_signal(symbol, signal)

            # 4. Save state
            self._state["last_signals"] = signal_dict
            self._state["last_run"] = datetime.utcnow().isoformat()
            self._save_state()

        except Exception as e:
            log.error(f"Iteration error: {e}", exc_info=True)
            self._log_event("error", {"message": str(e)})

    def _execute_signal(self, symbol: str, signal: float) -> None:
        """Execute a signal: place orders based on current vs desired position."""
        current_pos = self.broker.get_position(symbol)
        current_units = current_pos.units if current_pos else 0.0

        account = self.broker.get_account()
        equity = account.equity

        # Determine desired position
        if signal > 0:
            desired_units = (equity * 0.95) / self._get_price(symbol)  # 95% of capital
        elif signal < 0:
            desired_units = -(equity * 0.95) / self._get_price(symbol)
        else:
            desired_units = 0.0

        delta = desired_units - current_units

        if abs(delta) < 0.01:  # Negligible change
            return

        price = self._get_price(symbol)
        side = OrderSide.BUY if delta > 0 else OrderSide.SELL

        # Risk check
        if not self.risk_manager.check_order(symbol, side.value, abs(delta), price):
            log.warning(f"Risk check failed for {symbol} {side.value} {abs(delta):.2f}")
            return

        order = self.broker.place_order(
            symbol=symbol,
            side=side,
            units=abs(delta),
            order_type=OrderType.MARKET,
        )
        self._log_event("order", {
            "symbol": symbol,
            "side": side.value,
            "units": abs(delta),
            "price": price,
            "order_id": order.order_id,
            "status": order.status.value,
        })

    def _fetch_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch market data for all symbols."""
        result = {}
        for sym in self.symbols:
            df = self.broker.get_bars(sym, self.timeframe, self.lookback_bars)
            if not df.empty:
                result[sym] = df
        return result

    def _get_price(self, symbol: str) -> float:
        quote = self.broker.get_quote(symbol)
        return quote.get("last", 0.0) or quote.get("ask", 1.0)

    # ── State persistence ─────────────────────────────────────────────────────

    def _save_state(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "wb") as f:
            pickle.dump(self._state, f)

    def _load_state(self) -> None:
        if self.state_file.exists():
            with open(self.state_file, "rb") as f:
                self._state = pickle.load(f)
            log.info(f"Loaded state from {self.state_file}")

    # ── Event logging ─────────────────────────────────────────────────────────

    def _init_events_db(self) -> None:
        self.events_db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.events_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    data      TEXT
                )
            """)
            conn.commit()

    def _log_event(self, event_type: str, data: Any) -> None:
        with sqlite3.connect(self.events_db) as conn:
            conn.execute(
                "INSERT INTO events (timestamp, event_type, data) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), event_type, json.dumps(data, default=str)),
            )
            conn.commit()
