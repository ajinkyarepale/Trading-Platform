"""tests/test_backtest.py – Integration tests for the backtest engine."""

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine, BacktestConfig
from backtest.strategy import Strategy, StrategyMetadata


class AlwaysLongStrategy(Strategy):
    metadata = StrategyMetadata(name="AlwaysLong")

    def generate_signals(self, data, params):
        return pd.Series(1, index=data.index, dtype=int)


class AlwaysShortStrategy(Strategy):
    metadata = StrategyMetadata(name="AlwaysShort")

    def generate_signals(self, data, params):
        return pd.Series(-1, index=data.index, dtype=int)


class NoTradeStrategy(Strategy):
    metadata = StrategyMetadata(name="NoTrade")

    def generate_signals(self, data, params):
        return pd.Series(0, index=data.index, dtype=int)


def make_data(n=500, trend=0.0003, seed=0):
    rng = np.random.default_rng(seed)
    ret = rng.normal(trend, 0.012, n)
    close = 100 * np.cumprod(1 + ret)
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low  = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": high, "low": low, "close": close,
        "volume": np.ones(n) * 1_000_000,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))


@pytest.fixture
def up_trending_data():
    return make_data(n=500, trend=0.001)


@pytest.fixture
def down_trending_data():
    return make_data(n=500, trend=-0.001)


class TestBacktestEngine:
    def test_always_long_on_uptrend(self, up_trending_data):
        engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
        result = engine.run(AlwaysLongStrategy(), up_trending_data,
                            symbol="TEST", timeframe="1d")
        assert result.metrics["total_return_pct"] > 0

    def test_no_trade_strategy(self, up_trending_data):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(NoTradeStrategy(), up_trending_data,
                            symbol="TEST", timeframe="1d")
        assert result.metrics["n_trades"] == 0
        # Equity should stay flat (just initial capital)
        assert result.equity.iloc[-1] == pytest.approx(100_000, rel=0.001)

    def test_equity_length_matches_data(self, up_trending_data):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(AlwaysLongStrategy(), up_trending_data,
                            symbol="TEST", timeframe="1d")
        assert len(result.equity) == len(up_trending_data)

    def test_short_disabled(self, down_trending_data):
        cfg = BacktestConfig(allow_short=False)
        engine = BacktestEngine(cfg)
        result = engine.run(AlwaysShortStrategy(), down_trending_data,
                            symbol="TEST", timeframe="1d")
        # No short positions opened → trades should be 0
        assert result.metrics["n_trades"] == 0

    def test_result_has_required_attributes(self, up_trending_data):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(AlwaysLongStrategy(), up_trending_data,
                            symbol="TEST", timeframe="1d")
        for attr in ["equity", "trades", "metrics", "positions", "signals"]:
            assert hasattr(result, attr)
