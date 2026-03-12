"""tests/test_data.py – Tests for data fetching and caching."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from backtest.data import DataCache, fetch_multi, _cache_key


class TestDataCache:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.cache = DataCache(Path(self.tmpdir) / "test_cache.db")

    def make_df(self, n=10):
        idx = pd.date_range("2022-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "volume": 1_000_000,
        }, index=idx)

    def test_cache_miss(self):
        result = self.cache.get("nonexistent_key")
        assert result is None

    def test_cache_set_and_get(self):
        df = self.make_df()
        key = "TEST_1d_2022-01-01_2022-01-14"
        self.cache.set(key, "TEST", "1d", "2022-01-01", "2022-01-14", df)
        result = self.cache.get(key)
        assert result is not None
        assert len(result) == len(df)
        assert list(result.columns) == list(df.columns)

    def test_cache_key_unique(self):
        k1 = _cache_key("AAPL", "1d", "2020-01-01", "2021-01-01")
        k2 = _cache_key("AAPL", "1h", "2020-01-01", "2021-01-01")
        k3 = _cache_key("GOOGL","1d", "2020-01-01", "2021-01-01")
        assert k1 != k2
        assert k1 != k3
        assert k2 != k3


class TestStrategySignals:
    """Smoke tests to ensure strategies return valid signal ranges."""

    def make_data(self, n=300):
        import numpy as np
        rng = np.random.default_rng(0)
        close = 100 * (1 + rng.normal(0.0002, 0.01, n)).cumprod()
        high = close * 1.005
        low  = close * 0.995
        return pd.DataFrame({
            "open": close, "high": high, "low": low,
            "close": close, "volume": 1e6,
        }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    def _test_strategy(self, strategy_cls, params=None):
        from backtest.engine import BacktestEngine, BacktestConfig
        data = self.make_data()
        strat = strategy_cls()
        engine = BacktestEngine(BacktestConfig(allow_short=True))
        result = engine.run(strat, data, params or {}, symbol="TEST")
        signals = result.signals.iloc[:, 0]
        assert set(signals.unique()).issubset({-1, 0, 1})
        assert len(result.equity) == len(data)

    def test_zscore(self):
        from strategies.zscore_mean_reversion import ZScoreMeanReversion
        self._test_strategy(ZScoreMeanReversion, {"lookback": 30})

    def test_dual_ma(self):
        from strategies.dual_ma_crossover import DualMACrossover
        self._test_strategy(DualMACrossover, {"fast_period": 10, "slow_period": 40})

    def test_rsi(self):
        from strategies.rsi_mean_reversion import RSIMeanReversion
        self._test_strategy(RSIMeanReversion, {"rsi_period": 14})

    def test_breakout(self):
        from strategies.breakout import BreakoutStrategy
        self._test_strategy(BreakoutStrategy, {"channel_period": 20})
