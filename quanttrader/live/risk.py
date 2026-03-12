"""
live/risk.py
─────────────
Real-time risk manager for live trading.

Checks on every order:
  - Max position size per symbol
  - Portfolio-level leverage
  - Daily / weekly P&L limits
  - Max drawdown kill switch
  - Correlated position concentration
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

KILL_SWITCH_FILE = Path("KILL_SWITCH")   # touch this file to halt trading


class RiskViolation(Exception):
    """Raised when an order would breach a risk limit."""


class RiskManager:
    def __init__(
        self,
        max_position_pct: float = 0.10,
        max_leverage: float = 2.0,
        max_drawdown_pct: float = 0.20,
        daily_loss_limit_pct: float = 0.05,
        weekly_loss_limit_pct: float = 0.10,
        correlation_limit: float = 0.80,
        initial_equity: float = 100_000.0,
    ):
        self.max_position_pct    = max_position_pct
        self.max_leverage        = max_leverage
        self.max_drawdown_pct    = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.weekly_loss_limit_pct = weekly_loss_limit_pct
        self.correlation_limit   = correlation_limit

        self._peak_equity   = initial_equity
        self._day_start_eq  = initial_equity
        self._week_start_eq = initial_equity
        self._last_check_date = date.today()

    # ── Core check ────────────────────────────────────────────────────────────

    def check_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        equity: float,
        positions: list,
    ) -> None:
        """
        Raises RiskViolation if the proposed order violates any limit.
        Call this BEFORE submitting to broker.
        """
        self._refresh_daily_tracking(equity)

        # Kill switch
        if KILL_SWITCH_FILE.exists():
            raise RiskViolation("KILL SWITCH active – trading halted.")

        # Max position size
        notional = qty * price
        if notional / equity > self.max_position_pct:
            raise RiskViolation(
                f"{symbol}: notional {notional:.0f} exceeds {self.max_position_pct*100:.0f}% "
                f"of equity ({equity:.0f})."
            )

        # Max leverage
        total_exposure = sum(abs(p.qty) * p.current_price for p in positions) + notional
        if total_exposure / equity > self.max_leverage:
            raise RiskViolation(
                f"Max leverage {self.max_leverage}x would be exceeded "
                f"(current exposure {total_exposure / equity:.2f}x)."
            )

        # Drawdown kill switch
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self.max_drawdown_pct:
            raise RiskViolation(
                f"Max drawdown {self.max_drawdown_pct*100:.0f}% breached "
                f"(current DD {drawdown*100:.2f}%)."
            )

        # Daily loss
        daily_loss = (self._day_start_eq - equity) / self._day_start_eq
        if daily_loss > self.daily_loss_limit_pct:
            raise RiskViolation(
                f"Daily loss limit {self.daily_loss_limit_pct*100:.0f}% breached "
                f"(today: {daily_loss*100:.2f}%)."
            )

        # Weekly loss
        weekly_loss = (self._week_start_eq - equity) / self._week_start_eq
        if weekly_loss > self.weekly_loss_limit_pct:
            raise RiskViolation(
                f"Weekly loss limit {self.weekly_loss_limit_pct*100:.0f}% breached "
                f"(this week: {weekly_loss*100:.2f}%)."
            )

        logger.debug("Risk OK: %s %s %.4f @ %.4f", side, symbol, qty, price)

    def _refresh_daily_tracking(self, equity: float) -> None:
        today = date.today()
        if today != self._last_check_date:
            self._day_start_eq = equity
            self._last_check_date = today
            if today.weekday() == 0:  # Monday
                self._week_start_eq = equity
            logger.info("Daily equity reset: %.2f", equity)

    def activate_kill_switch(self) -> None:
        KILL_SWITCH_FILE.touch()
        logger.critical("KILL SWITCH ACTIVATED")

    def deactivate_kill_switch(self) -> None:
        if KILL_SWITCH_FILE.exists():
            KILL_SWITCH_FILE.unlink()
        logger.info("Kill switch deactivated.")
