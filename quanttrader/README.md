# QuantTrader 🚀

> A production-ready, modular quantitative trading platform built with Python 3.10+.
> Covers everything from backtesting and parameter optimisation to live execution and HTML reporting.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Project Structure](#project-structure)
3. [Components](#components)
4. [CLI Reference](#cli-reference)
5. [Adding a New Strategy](#adding-a-new-strategy)
6. [Running Optimization](#running-optimization)
7. [Live Trading](#live-trading)
8. [Deploying on VPS](#deploying-on-vps)
9. [Configuration](#configuration)

---

## Quick Start

```bash
# 1. Clone / unzip the project
cd quanttrader

# 2. Create a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and edit the .env file
cp .env.example .env

# 5. Create required directories
python -c "from config.settings import settings; settings.ensure_dirs()"

# 6. Run your first backtest
python cli.py run backtest \
    --strategy ZScoreMeanReversion \
    --symbol SPY \
    --from 2018-01-01 \
    --to 2024-01-01 \
    --params '{"lookback":60,"entry_z":2.0,"exit_z":0.5}' \
    --save --report

# 7. View the leaderboard
python cli.py dashboard leaderboard --metric sharpe --top 10
```

---

## Project Structure

```
quanttrader/
├── backtest/               ← Core backtesting engine
│   ├── data.py             ← Multi-source OHLCV fetcher + SQLite cache
│   ├── strategy.py         ← Abstract Strategy base class
│   ├── engine.py           ← Event-driven simulation engine
│   ├── execution.py        ← Order fills, slippage, position sizing
│   └── metrics.py          ← 20+ performance metrics
├── experiments/            ← SQLAlchemy experiment database
│   └── database.py         ← log/query/export experiments
├── optimization/           ← Parameter search
│   ├── grid_search.py      ← Parallel exhaustive grid search
│   ├── random_search.py    ← Random sampling
│   ├── bayesian.py         ← Optuna-based TPE
│   └── walk_forward.py     ← Walk-forward out-of-sample analysis
├── ml_strategies/          ← Machine learning strategies
│   ├── features.py         ← 20-feature technical indicator engineering
│   └── classification.py   ← Random Forest direction prediction
├── live/                   ← Live trading infrastructure
│   ├── broker.py           ← Broker interface + PaperBroker + AlpacaBroker
│   ├── risk.py             ← Real-time risk manager
│   ├── scheduler.py        ← APScheduler-based job runner
│   └── state.py            ← Persistent strategy state
├── dashboard/              ← Reporting & visualisation
│   ├── leaderboard.py      ← Rich CLI leaderboard
│   └── report.py           ← Interactive Plotly HTML reports
├── strategies/             ← Built-in example strategies
│   ├── zscore_mean_reversion.py
│   ├── dual_ma_crossover.py
│   ├── rsi_mean_reversion.py
│   ├── pairs_trading.py
│   └── breakout.py
├── config/
│   ├── instruments.yaml    ← Per-instrument costs and metadata
│   ├── live_config.yaml    ← Live trading config
│   └── settings.py         ← Pydantic settings + env loading
├── deploy/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── trading-bot.service ← Systemd service file
├── tests/                  ← pytest test suite
├── notebooks/              ← Jupyter exploration notebooks
├── cli.py                  ← Click-based CLI entry point
└── requirements.txt
```

---

## Components

### 1. Backtesting Engine

```python
from backtest.data import fetch_data
from backtest.engine import BacktestEngine, BacktestConfig
from strategies import ZScoreMeanReversion

# Fetch data
df = fetch_data("SPY", "2018-01-01", "2024-01-01", "1d")

# Configure and run
engine = BacktestEngine(BacktestConfig(
    initial_capital=100_000,
    sizing_param=0.95,        # 95% of equity per trade
    allow_short=True,
))
result = engine.run(
    ZScoreMeanReversion(),
    df,
    params={"lookback": 60, "entry_z": 2.0, "exit_z": 0.5},
    symbol="SPY",
)

print(result.metrics)
```

### 2. Multi-Symbol / Pairs

```python
from backtest.data import fetch_data
from strategies import PairsTrading

data = {
    "EWA": fetch_data("EWA", "2018-01-01", "2024-01-01"),
    "EWC": fetch_data("EWC", "2018-01-01", "2024-01-01"),
}
result = engine.run(PairsTrading(), data, params={}, symbol="EWA_EWC")
```

### 3. Experiment Database

```python
from experiments.database import log_experiment, get_experiments

# Save
eid = log_experiment(result, notes="Testing lookback=60", tags="mean-reversion,spy")

# Query
df = get_experiments(strategy_name="ZScore")
print(df[["id", "sharpe", "total_return_pct", "max_drawdown_pct"]])
```

### 4. Parameter Optimisation

```python
from optimization.grid_search import grid_search
from backtest.engine import BacktestEngine, BacktestConfig

def objective(params):
    result = engine.run(strategy, df, params, symbol="SPY")
    return result.metrics["sharpe"]

results = grid_search(objective, {
    "lookback": [30, 60, 90, 120],
    "entry_z":  [1.5, 2.0, 2.5],
    "exit_z":   [0.25, 0.5, 1.0],
})
print(results.head(5))
```

### 5. Walk-Forward Analysis

```python
from optimization.walk_forward import walk_forward
from optimization.grid_search import grid_search

def optimize_fn(train_df):
    def obj(p): return engine.run(strat, train_df, p, symbol="SPY").metrics["sharpe"]
    res = grid_search(obj, {"lookback":[30,60,90], "entry_z":[1.5,2.0,2.5]})
    best = res.iloc[0]
    return {k: best[k] for k in ["lookback","entry_z","exit_z"]}, best["score"]

def evaluate_fn(val_df, params):
    r = engine.run(strat, val_df, params, symbol="SPY")
    return r.metrics["sharpe"], r.metrics, r.equity

wf_result = walk_forward(full_df, optimize_fn, evaluate_fn, n_windows=5)
print(wf_result.summary_df)
print("OOS Sharpe:", wf_result.oos_metrics["sharpe"])
```

---

## CLI Reference

```bash
# ── Data ──────────────────────────────────────────────────────────
trade data fetch --symbol AAPL --from 2020-01-01 --to 2024-01-01 --timeframe 1d
trade data fetch --symbol BTC-USD --timeframe 1h --output btc.csv

# ── Backtest ──────────────────────────────────────────────────────
trade run backtest --strategy ZScoreMeanReversion --symbol SPY
trade run backtest --strategy DualMACrossover --symbol QQQ \
    --params '{"fast_period":10,"slow_period":50}' --save --report

# ── Optimization ──────────────────────────────────────────────────
trade run optimization --strategy ZScoreMeanReversion --symbol SPY \
    --method bayesian --n-trials 100 --metric sharpe \
    --param-grid '{"lookback":["int",20,120],"entry_z":["float",1.0,3.0]}'

# ── Leaderboard ───────────────────────────────────────────────────
trade dashboard leaderboard --metric sharpe --top 20
trade dashboard leaderboard --strategy ZScore --export top.csv

# ── HTML Report ───────────────────────────────────────────────────
trade dashboard report --id 42 --open

# ── Live Trading ──────────────────────────────────────────────────
trade live start --config config/live_config.yaml --paper
```

---

## Adding a New Strategy

1. Create `strategies/my_strategy.py`:

```python
from backtest.strategy import Strategy, StrategyMetadata
import pandas as pd

class MyStrategy(Strategy):
    metadata = StrategyMetadata(
        name="MyStrategy",
        version="1.0.0",
        param_schema={
            "period": (int, 5, 200, 20),
        },
    )

    def generate_signals(self, data, params):
        params = self.validate_params(params)
        # ... compute signals ...
        return pd.Series(signal, index=data.index, dtype=int)
```

2. Register it in `strategies/__init__.py`:
```python
from .my_strategy import MyStrategy
```

3. Run it:
```bash
trade run backtest --strategy MyStrategy --symbol SPY
```

---

## Live Trading

1. Set up broker credentials in `.env`.
2. Edit `config/live_config.yaml` with your strategies, symbols, and schedule.
3. Start in paper mode:
```bash
trade live start --config config/live_config.yaml --paper
```

### Paper Broker (no credentials needed)
```python
from live.broker import PaperBroker
from live.risk import RiskManager

broker = PaperBroker(initial_capital=100_000)
broker.connect()
broker.update_price("AAPL", 185.50)
order = broker.place_order("AAPL", "buy", 10)
print(broker.equity)
```

---

## Running Tests

```bash
pytest                              # all tests
pytest tests/test_metrics.py -v    # specific file
pytest --cov=backtest --cov-report=html   # with coverage
```

---

## Deploying on VPS

### Docker
```bash
cd deploy
docker-compose up -d quanttrader
docker logs quanttrader -f
```

### Systemd
```bash
sudo cp deploy/trading-bot.service /etc/systemd/system/
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo journalctl -u trading-bot -f
```

---

## Configuration

All settings live in `config/settings.py` and can be overridden via:
- Environment variables: `DATA__ALPHA_VANTAGE_KEY=...`
- `.env` file (copy from `.env.example`)
- Direct `Settings()` instantiation in code

Per-instrument costs are in `config/instruments.yaml`.

---

## Built-in Strategies

| Strategy | Type | Key Params |
|---|---|---|
| `ZScoreMeanReversion` | Mean-reversion | lookback, entry_z, exit_z |
| `DualMACrossover` | Trend-following | fast_period, slow_period, ma_type |
| `RSIMeanReversion` | Mean-reversion | rsi_period, oversold, overbought |
| `PairsTrading` | Stat-arb | lookback, entry_z, hedge_window |
| `BreakoutStrategy` | Breakout | channel_period, atr_period, atr_mult |
| `RandomForestStrategy` | ML | n_estimators, max_depth, retrain_every |

---

## License

MIT – use freely, trade responsibly. Past performance does not guarantee future results.


