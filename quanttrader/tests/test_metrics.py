"""tests/test_metrics.py – Unit tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    total_return, cagr, sharpe_ratio, sortino_ratio,
    max_drawdown, calmar_ratio, compute_metrics,
    win_rate, profit_factor, value_at_risk, conditional_var,
)


def make_equity(n=252, drift=0.0003, vol=0.01, seed=42):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    equity = pd.Series(
        100_000 * np.cumprod(1 + r),
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )
    return equity


@pytest.fixture
def flat_equity():
    return pd.Series(
        [100_000] * 252,
        index=pd.date_range("2022-01-01", periods=252, freq="B"),
    )


@pytest.fixture
def growing_equity():
    return make_equity(252, drift=0.0005)


@pytest.fixture
def declining_equity():
    return make_equity(252, drift=-0.0005)


class TestBasicMetrics:
    def test_total_return_flat(self, flat_equity):
        assert total_return(flat_equity) == pytest.approx(0.0, abs=1e-6)

    def test_total_return_growing(self, growing_equity):
        assert total_return(growing_equity) > 0

    def test_total_return_declining(self, declining_equity):
        assert total_return(declining_equity) < 0

    def test_cagr_positive(self, growing_equity):
        assert cagr(growing_equity) > 0

    def test_sharpe_flat(self, flat_equity):
        assert sharpe_ratio(flat_equity) == pytest.approx(0.0, abs=1e-3)

    def test_sharpe_positive(self, growing_equity):
        s = sharpe_ratio(growing_equity)
        assert isinstance(s, float)

    def test_max_drawdown_no_decline(self, flat_equity):
        assert max_drawdown(flat_equity) == pytest.approx(0.0, abs=1e-6)

    def test_max_drawdown_negative(self):
        eq = pd.Series([100, 90, 80, 95, 100],
                       index=pd.date_range("2022-01-01", periods=5))
        assert max_drawdown(eq) == pytest.approx(-0.2, abs=1e-6)

    def test_sortino_zero_downside(self, flat_equity):
        assert sortino_ratio(flat_equity) == pytest.approx(0.0, abs=1e-3)


class TestTradeMetrics:
    def make_trades(self, n_win=60, n_loss=40):
        trades = [{"pnl": 100} for _ in range(n_win)] + \
                 [{"pnl": -80} for _ in range(n_loss)]
        return trades

    def test_win_rate(self):
        trades = self.make_trades(6, 4)
        assert win_rate(trades) == pytest.approx(0.6)

    def test_profit_factor(self):
        trades = [{"pnl": 100}, {"pnl": 100}, {"pnl": -50}]
        assert profit_factor(trades) == pytest.approx(4.0)

    def test_empty_trades(self):
        assert win_rate([]) == 0.0
        assert profit_factor([]) == float("inf")


class TestRiskMetrics:
    def test_var_negative(self, declining_equity):
        var = value_at_risk(declining_equity)
        assert var < 0

    def test_cvar_le_var(self, declining_equity):
        var = value_at_risk(declining_equity)
        cvar = conditional_var(declining_equity)
        assert cvar <= var


class TestComputeMetrics:
    def test_compute_metrics_keys(self, growing_equity):
        m = compute_metrics(growing_equity)
        required = ["total_return_pct", "cagr_pct", "sharpe", "sortino",
                    "max_drawdown_pct", "calmar", "n_trades"]
        for key in required:
            assert key in m, f"Missing metric: {key}"

    def test_compute_metrics_trades(self, growing_equity):
        trades = [{"pnl": 100, "bars": 5}, {"pnl": -50, "bars": 3}]
        m = compute_metrics(growing_equity, trades)
        assert m["n_trades"] == 2
        assert m["win_rate_pct"] == 50.0
