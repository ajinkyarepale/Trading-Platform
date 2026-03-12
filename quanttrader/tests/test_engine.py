"""
tests/test_engine.py
====================
Unit tests for the backtesting engine and strategies.
"""
import numpy as np
import pandas as pd
import pytest

from backtest.engine import Backtester, BacktestResult
from backtest.strategy import Strategy, StrategyMetadata


# ── Minimal test strategies ───────────────────────────────────────────────────

class AlwaysLong(Strategy):
    """Always returns a long signal."""
    metadata = StrategyMetadata(name="AlwaysLong")

    def generate_signals(self, data, params):
        return pd.Series(1.0, index=data.index)


class AlwaysFlat(Strategy):
    """Never trades."""
    metadata = StrategyMetadata(name="AlwaysFlat")

    def generate_signals(self, data, params):
        return pd.Series(0.0, index=data.index)


class AlwaysShort(Strategy):
    """Always returns a short signal."""
    metadata = StrategyMetadata(name="AlwaysShort")

    def generate_signals(self, data, params):
        return pd.Series(-1.0, index=data.index)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trending_data():
    """Consistently trending up price series."""
    idx = pd.date_range("2020-01-01", periods=250, freq="B")
    close = 100 + np.arange(250) * 0.5
    df = pd.DataFrame({
        "open": close - 0.2,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": 1_000_000,
    }, index=idx)
    return df


@pytest.fixture
def flat_data():
    """Flat price (no trend)."""
    idx = pd.date_range("2020-01-01", periods=100, freq="B")
    close = np.full(100, 100.0)
    df = pd.DataFrame({
        "open": close, "high": close + 0.01,
        "low": close - 0.01, "close": close, "volume": 100_000,
    }, index=idx)
    return df


# ── Backtester tests ─────────────────────────────────────────────────────────

def test_always_long_equity_grows(trending_data):
    bt = Backtester(initial_capital=10_000)
    result = bt.run(AlwaysLong(), trending_data)
    assert result.equity_curve.iloc[-1] > result.equity_curve.iloc[0]


def test_always_flat_equity_constant(flat_data):
    bt = Backtester(initial_capital=10_000)
    result = bt.run(AlwaysFlat(), flat_data)
    # No trades, equity should stay near initial
    assert abs(result.equity_curve.iloc[-1] - 10_000) < 1


def test_always_flat_no_trades(flat_data):
    bt = Backtester(initial_capital=10_000)
    result = bt.run(AlwaysFlat(), flat_data)
    assert len(result.trades) == 0


def test_backtest_result_type(trending_data):
    bt = Backtester(initial_capital=50_000)
    result = bt.run(AlwaysLong(), trending_data)
    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve, pd.Series)
    assert isinstance(result.metrics, dict)
    assert isinstance(result.trades, pd.DataFrame)


def test_no_short_flag(trending_data):
    """With no_short=True, short signals should be ignored."""
    bt = Backtester(initial_capital=10_000, allow_short=False)
    result = bt.run(AlwaysShort(), trending_data)
    assert len(result.trades) == 0


def test_metrics_computed(trending_data):
    bt = Backtester()
    result = bt.run(AlwaysLong(), trending_data)
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics
    assert "total_return" in result.metrics


def test_equity_curve_length(trending_data):
    bt = Backtester()
    result = bt.run(AlwaysLong(), trending_data)
    assert len(result.equity_curve) == len(trending_data)


def test_initial_capital_respected(trending_data):
    capital = 75_000
    bt = Backtester(initial_capital=capital)
    result = bt.run(AlwaysLong(), trending_data)
    assert result.equity_curve.iloc[0] == capital


# ── Strategy parameter validation ────────────────────────────────────────────

def test_strategy_default_params():
    from strategies.zscore_mean_reversion import ZScoreMeanReversion
    s = ZScoreMeanReversion()
    assert s.params["lookback"] == 20
    assert s.params["entry_z"] == 2.0


def test_strategy_param_override():
    from strategies.zscore_mean_reversion import ZScoreMeanReversion
    s = ZScoreMeanReversion(lookback=30, entry_z=1.5)
    assert s.params["lookback"] == 30
    assert s.params["entry_z"] == 1.5


def test_strategy_param_validation_error():
    from strategies.zscore_mean_reversion import ZScoreMeanReversion
    with pytest.raises(ValueError):
        ZScoreMeanReversion(entry_z=10.0)  # max is 4.0
