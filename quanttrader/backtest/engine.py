"""
backtest/engine.py
───────────────────
Event-driven backtesting engine.

Features:
  - Single and multi-symbol backtests
  - Configurable slippage, spread, commission
  - Multiple position sizing modes
  - Short selling support
  - Per-bar portfolio tracking
  - Trade log generation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .strategy import Strategy
from .execution import (
    InstrumentCosts, Trade, OrderType, PositionSizing,
    apply_slippage, apply_spread, calc_commission, calc_position_size,
)
from .metrics import compute_metrics
from config.settings import load_instrument_config

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    sizing: PositionSizing = PositionSizing.PERCENT
    sizing_param: float = 0.95          # 95% of equity per trade
    order_type: OrderType = OrderType.MARKET
    allow_short: bool = True
    use_margin: bool = False
    benchmark_data: Optional[pd.Series] = None  # for benchmark comparison
    risk_free: float = 0.0
    freq: int = 252                     # trading periods per year
    # Costs override (if None, load from instruments.yaml)
    costs_override: Optional[InstrumentCosts] = None


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: Union[str, List[str]]
    params: Dict[str, Any]
    start: str
    end: str
    timeframe: str
    equity: pd.Series
    positions: pd.DataFrame          # one column per symbol
    trades: List[Trade]
    metrics: Dict[str, Any]
    signals: pd.DataFrame


class BacktestEngine:
    """
    Single-entry point for running a backtest.

    Usage:
        engine = BacktestEngine(config)
        result = engine.run(strategy, data, params)
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        strategy: Strategy,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Optional[Dict[str, Any]] = None,
        symbol: str = "UNKNOWN",
        timeframe: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a full backtest for *strategy* on *data*.

        Parameters
        ----------
        strategy : Strategy subclass instance
        data     : Single DataFrame (OHLCV) or dict {symbol: DataFrame}
        params   : Strategy parameter dict
        symbol   : Symbol name for single-symbol backtest
        """
        params = params or {}
        params = strategy.validate_params(params)

        # Determine if multi-symbol
        if isinstance(data, dict):
            symbols = list(data.keys())
            multi = True
        else:
            symbols = [symbol]
            multi = False

        # Generate signals
        strategy.on_start(data, params)
        raw_signals = strategy.generate_signals(data, params)
        strategy.on_end(data, params)

        # Normalise to DataFrame
        if isinstance(raw_signals, pd.Series):
            sig_df = raw_signals.to_frame(name=symbols[0])
        else:
            sig_df = raw_signals

        # Simulate
        if multi:
            result = self._run_multi(strategy, data, sig_df, symbols, params, timeframe, start, end)
        else:
            df_single = data if isinstance(data, pd.DataFrame) else data[symbols[0]]
            result = self._run_single(strategy, df_single, sig_df[symbols[0]], symbol, params, timeframe, start, end)

        return result

    # ── Single-symbol simulation ──────────────────────────────────────────────

    def _run_single(
        self, strategy: Strategy, data: pd.DataFrame,
        signals: pd.Series, symbol: str,
        params: dict, timeframe: str, start, end,
    ) -> BacktestResult:
        cfg = self.config
        costs_cfg = load_instrument_config(symbol)
        costs = cfg.costs_override or InstrumentCosts(**{
            k: costs_cfg[k] for k in InstrumentCosts.__dataclass_fields__ if k in costs_cfg
        })

        capital = cfg.initial_capital
        position = 0.0         # units held
        entry_price = 0.0
        entry_time = None
        entry_signal = 0
        entry_bar = 0

        equity_vals = []
        trades: List[Trade] = []
        pos_series = []

        for i, (ts, row) in enumerate(data.iterrows()):
            close = float(row["close"])
            sig = int(signals.get(ts, 0)) if ts in signals.index else 0

            # Mark-to-market equity
            unrealised = position * (close - entry_price) if position != 0 else 0
            equity_vals.append(capital + unrealised)
            pos_series.append(position)

            # ── Close existing position ───────────────────────────────────────
            if position != 0 and (sig != np.sign(position) or sig == 0):
                exit_price = apply_spread(close, -int(np.sign(position)), costs.spread)
                exit_price = apply_slippage(exit_price, -int(np.sign(position)),
                                            costs.slippage_pct, cfg.order_type)
                comm = calc_commission(abs(position) * exit_price, costs)
                raw_pnl = position * (exit_price - entry_price)
                net_pnl = raw_pnl - comm
                capital += net_pnl + position * entry_price  # return cost basis

                trades.append(Trade(
                    symbol=symbol, entry_time=entry_time, exit_time=ts,
                    direction=int(np.sign(position)),
                    entry_price=entry_price, exit_price=exit_price,
                    qty=abs(position), pnl=net_pnl,
                    pnl_pct=net_pnl / (abs(position) * entry_price) if entry_price else 0,
                    costs=comm, bars=i - entry_bar,
                    entry_signal=entry_signal, exit_signal=sig,
                ))
                position = 0.0

            # ── Open new position ─────────────────────────────────────────────
            if sig != 0 and position == 0:
                if sig == -1 and not cfg.allow_short:
                    continue
                buy_price = apply_spread(close, sig, costs.spread)
                buy_price = apply_slippage(buy_price, sig, costs.slippage_pct, cfg.order_type)
                units = calc_position_size(capital, buy_price, cfg.sizing,
                                           cfg.sizing_param, costs)
                if units > 0:
                    comm = calc_commission(units * buy_price, costs)
                    capital -= units * buy_price + comm
                    position = units * sig   # negative for short
                    entry_price = buy_price
                    entry_time = ts
                    entry_signal = sig
                    entry_bar = i

        equity = pd.Series(equity_vals, index=data.index, name="equity")
        positions = pd.DataFrame({symbol: pos_series}, index=data.index)
        metrics = compute_metrics(
            equity, [t.__dict__ for t in trades],
            cfg.benchmark_data, cfg.risk_free, cfg.freq,
        )

        return BacktestResult(
            strategy_name=strategy.metadata.name,
            symbol=symbol,
            params=params,
            start=str(start or data.index[0].date()),
            end=str(end or data.index[-1].date()),
            timeframe=timeframe,
            equity=equity,
            positions=positions,
            trades=trades,
            metrics=metrics,
            signals=signals.to_frame(name=symbol),
        )

    # ── Multi-symbol simulation ───────────────────────────────────────────────

    def _run_multi(
        self, strategy: Strategy, data: Dict[str, pd.DataFrame],
        signals: pd.DataFrame, symbols: List[str],
        params: dict, timeframe: str, start, end,
    ) -> BacktestResult:
        """
        Equal-weight multi-symbol backtest.
        Each symbol receives 1/N of capital per signal.
        """
        cfg = self.config
        n = len(symbols)
        alloc_pct = (cfg.sizing_param if cfg.sizing == PositionSizing.PERCENT else 1.0) / n

        # Run per-symbol backtests with reduced allocation
        sub_results = []
        for sym in symbols:
            sub_cfg = BacktestConfig(
                initial_capital=cfg.initial_capital / n,
                sizing=cfg.sizing,
                sizing_param=alloc_pct * n,  # adjusted back
                order_type=cfg.order_type,
                allow_short=cfg.allow_short,
                risk_free=cfg.risk_free,
                freq=cfg.freq,
            )
            sub_engine = BacktestEngine(sub_cfg)
            sub_res = sub_engine._run_single(
                strategy, data[sym], signals[sym], sym, params, timeframe, start, end
            )
            sub_results.append(sub_res)

        # Combine equity curves
        combined_eq = sum(r.equity for r in sub_results)
        combined_eq.name = "equity"
        all_trades = [t for r in sub_results for t in r.trades]
        positions = pd.concat([r.positions for r in sub_results], axis=1)

        metrics = compute_metrics(
            combined_eq, [t.__dict__ for t in all_trades],
            cfg.benchmark_data, cfg.risk_free, cfg.freq,
        )

        return BacktestResult(
            strategy_name=strategy.metadata.name,
            symbol=symbols,
            params=params,
            start=str(start or combined_eq.index[0].date()),
            end=str(end or combined_eq.index[-1].date()),
            timeframe=timeframe,
            equity=combined_eq,
            positions=positions,
            trades=all_trades,
            metrics=metrics,
            signals=signals,
        )
