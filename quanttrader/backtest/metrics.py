"""
backtest/metrics.py
────────────────────
Comprehensive performance metrics for a backtest equity curve.

All functions accept:
  equity    – pd.Series of portfolio value (index = DatetimeIndex)
  returns   – pd.Series of period returns (computed if not provided)
  benchmark – optional pd.Series of benchmark values
  trades    – optional list of trade dicts for trade-level stats
  risk_free – annualised risk-free rate (default 0.0)
  freq      – number of periods per year (252 = daily, 252*6.5*60 = per-minute …)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd


# ── Helpers ───────────────────────────────────────────────────────────────────

def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _ann_factor(freq: int) -> float:
    return float(freq)


def _drawdown_series(equity: pd.Series) -> pd.Series:
    roll_max = equity.cummax()
    return (equity - roll_max) / roll_max


# ── Individual metrics ────────────────────────────────────────────────────────

def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, freq: int = 252) -> float:
    n_years = len(equity) / freq
    if n_years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.0, freq: int = 252) -> float:
    r = _returns(equity)
    if r.std() == 0:
        return 0.0
    rf_per_period = risk_free / freq
    excess = r - rf_per_period
    return float(np.sqrt(freq) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(equity: pd.Series, risk_free: float = 0.0, freq: int = 252) -> float:
    r = _returns(equity)
    rf_per_period = risk_free / freq
    excess = r - rf_per_period
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(np.sqrt(freq) * excess.mean() / downside.std(ddof=1))


def max_drawdown(equity: pd.Series) -> float:
    dd = _drawdown_series(equity)
    return float(dd.min())


def calmar_ratio(equity: pd.Series, freq: int = 252) -> float:
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, freq) / mdd)


def downside_deviation(equity: pd.Series, freq: int = 252) -> float:
    r = _returns(equity)
    negative = r[r < 0]
    if len(negative) == 0:
        return 0.0
    return float(np.sqrt(freq) * negative.std(ddof=1))


def value_at_risk(equity: pd.Series, confidence: float = 0.95) -> float:
    r = _returns(equity)
    return float(np.percentile(r, (1 - confidence) * 100))


def conditional_var(equity: pd.Series, confidence: float = 0.95) -> float:
    r = _returns(equity)
    var = value_at_risk(equity, confidence)
    tail = r[r <= var]
    return float(tail.mean()) if len(tail) > 0 else var


def ulcer_index(equity: pd.Series) -> float:
    """Ulcer Index: RMS of drawdown depths (penalises long/deep drawdowns)."""
    dd_pct = _drawdown_series(equity) * 100  # convert to percentage
    return float(np.sqrt((dd_pct ** 2).mean()))


def win_rate(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    wins = sum(1 for t in trades if t.get("pnl", 0) > 0)
    return wins / len(trades)


def profit_factor(trades: List[Dict[str, Any]]) -> float:
    gains = sum(t["pnl"] for t in trades if t.get("pnl", 0) > 0)
    losses = abs(sum(t["pnl"] for t in trades if t.get("pnl", 0) < 0))
    return float(gains / losses) if losses > 0 else float("inf")


def average_trade(trades: List[Dict[str, Any]]) -> float:
    if not trades:
        return 0.0
    return float(np.mean([t.get("pnl", 0) for t in trades]))


def max_consecutive_losses(trades: List[Dict[str, Any]]) -> int:
    if not trades:
        return 0
    max_consec = cur = 0
    for t in trades:
        if t.get("pnl", 0) < 0:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0
    return max_consec


def avg_holding_period(trades: List[Dict[str, Any]]) -> Optional[float]:
    periods = [t.get("bars", None) for t in trades if t.get("bars") is not None]
    return float(np.mean(periods)) if periods else None


# ── Rolling metrics ───────────────────────────────────────────────────────────

def rolling_sharpe(equity: pd.Series, window: int = 63, freq: int = 252) -> pd.Series:
    r = _returns(equity)
    mean = r.rolling(window).mean()
    std = r.rolling(window).std(ddof=1)
    return (np.sqrt(freq) * mean / std).rename("rolling_sharpe")


def rolling_drawdown(equity: pd.Series) -> pd.Series:
    return _drawdown_series(equity).rename("rolling_drawdown")


# ── Benchmark comparison ──────────────────────────────────────────────────────

def information_ratio(equity: pd.Series, benchmark: pd.Series, freq: int = 252) -> float:
    r = _returns(equity)
    b = _returns(benchmark).reindex(r.index).fillna(0)
    active = r - b
    if active.std() == 0:
        return 0.0
    return float(np.sqrt(freq) * active.mean() / active.std(ddof=1))


def beta(equity: pd.Series, benchmark: pd.Series) -> float:
    r = _returns(equity)
    b = _returns(benchmark).reindex(r.index).fillna(0)
    cov = np.cov(r, b, ddof=1)
    if cov[1, 1] == 0:
        return 0.0
    return float(cov[0, 1] / cov[1, 1])


def alpha(equity: pd.Series, benchmark: pd.Series,
          risk_free: float = 0.0, freq: int = 252) -> float:
    b = beta(equity, benchmark)
    port_ret = cagr(equity, freq)
    bench_ret = cagr(benchmark.reindex(equity.index).ffill(), freq)
    return float(port_ret - risk_free - b * (bench_ret - risk_free))


# ── Master compute function ───────────────────────────────────────────────────

def compute_metrics(
    equity: pd.Series,
    trades: Optional[List[Dict[str, Any]]] = None,
    benchmark: Optional[pd.Series] = None,
    risk_free: float = 0.0,
    freq: int = 252,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """
    Compute all performance metrics and return as a flat dictionary.
    """
    trades = trades or []
    m: Dict[str, Any] = {}

    # Return-based
    m["total_return_pct"] = round(total_return(equity) * 100, 4)
    m["cagr_pct"] = round(cagr(equity, freq) * 100, 4)
    m["sharpe"] = round(sharpe_ratio(equity, risk_free, freq), 4)
    m["sortino"] = round(sortino_ratio(equity, risk_free, freq), 4)
    m["calmar"] = round(calmar_ratio(equity, freq), 4)
    m["max_drawdown_pct"] = round(max_drawdown(equity) * 100, 4)
    m["downside_dev"] = round(downside_deviation(equity, freq), 6)
    m["var_95_pct"] = round(value_at_risk(equity, confidence) * 100, 4)
    m["cvar_95_pct"] = round(conditional_var(equity, confidence) * 100, 4)
    m["ulcer_index"] = round(ulcer_index(equity), 4)

    # Volatility
    r = _returns(equity)
    m["volatility_ann_pct"] = round(r.std(ddof=1) * np.sqrt(freq) * 100, 4)

    # Trade-level
    m["n_trades"] = len(trades)
    m["win_rate_pct"] = round(win_rate(trades) * 100, 2)
    m["profit_factor"] = round(profit_factor(trades), 4)
    m["avg_trade"] = round(average_trade(trades), 4)
    m["max_consec_losses"] = max_consecutive_losses(trades)
    hp = avg_holding_period(trades)
    m["avg_holding_bars"] = round(hp, 1) if hp is not None else None

    # Benchmark
    if benchmark is not None:
        m["information_ratio"] = round(information_ratio(equity, benchmark, freq), 4)
        m["beta"] = round(beta(equity, benchmark), 4)
        m["alpha_ann_pct"] = round(alpha(equity, benchmark, risk_free, freq) * 100, 4)
        bh = total_return(benchmark.reindex(equity.index).ffill())
        m["benchmark_return_pct"] = round(bh * 100, 4)
        m["excess_return_pct"] = round(m["total_return_pct"] - m["benchmark_return_pct"], 4)

    return m
