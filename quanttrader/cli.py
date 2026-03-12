#!/usr/bin/env python3
"""
cli.py
───────
QuantTrader command-line interface.

Commands:
  trade data fetch         – Download & cache historical data
  trade run backtest       – Run a single backtest
  trade run optimization   – Run parameter optimisation
  trade live start         – Start live trading loop
  trade dashboard lb       – Print experiment leaderboard
  trade dashboard report   – Generate HTML report for an experiment
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

# ── CLI Root ──────────────────────────────────────────────────────────────────

@click.group()
@click.option("--log-level", default="INFO", help="Logging level.")
def trade(log_level):
    """QuantTrader – Quantitative Trading Platform"""
    setup_logging(log_level)


# ── Data ──────────────────────────────────────────────────────────────────────

@trade.group()
def data():
    """Data management commands."""

@data.command("fetch")
@click.option("--symbol", "-s", required=True, help="Ticker symbol, e.g. AAPL or EURUSD=X")
@click.option("--from",   "start", default="2020-01-01", show_default=True)
@click.option("--to",     "end",   default=None, help="End date (default: today)")
@click.option("--timeframe", "-t", default="1d", show_default=True,
              type=click.Choice(["1m","5m","15m","30m","1h","4h","1d","1w","1mo"]))
@click.option("--source", default="yfinance",
              type=click.Choice(["yfinance","alphavantage","csv"]))
@click.option("--output", "-o", default=None, help="CSV output path")
def fetch_data(symbol, start, end, timeframe, source, output):
    """Fetch historical OHLCV data and optionally save to CSV."""
    from datetime import date
    from backtest.data import fetch_data as _fetch
    from config.settings import settings

    end = end or str(date.today())
    click.echo(f"Fetching {symbol} [{timeframe}] from {start} to {end} via {source}…")

    df = _fetch(
        symbol, start, end, timeframe, source,
        use_cache=True, cache_path=settings.data.cache_dir / "cache.db",
    )
    click.echo(f"  → {len(df)} bars fetched.")
    click.echo(df.tail(5).to_string())

    if output:
        df.to_csv(output)
        click.echo(f"  → Saved to {output}")


# ── Run ───────────────────────────────────────────────────────────────────────

@trade.group("run")
def run():
    """Backtesting & optimisation."""


@run.command("backtest")
@click.option("--strategy", "-s", required=True,
              help="Strategy class name (e.g. ZScoreMeanReversion)")
@click.option("--symbol",   "-S", default="SPY", show_default=True)
@click.option("--from",     "start", default="2018-01-01", show_default=True)
@click.option("--to",       "end",   default=None)
@click.option("--timeframe","-t", default="1d", show_default=True)
@click.option("--params",   "-p", default="{}", help='JSON params, e.g. \'{"lookback":60}\'')
@click.option("--capital",  "-c", default=100_000.0, show_default=True, type=float)
@click.option("--report",   "-r", is_flag=True, help="Generate HTML report")
@click.option("--save",     is_flag=True, help="Save to experiment DB")
@click.option("--notes",    default="", help="Notes for experiment log")
@click.option("--tags",     default="", help="Comma-separated tags")
def run_backtest(strategy, symbol, start, end, timeframe, params,
                 capital, report, save, notes, tags):
    """Run a single backtest and print metrics."""
    from datetime import date
    from backtest.data import fetch_data
    from backtest.engine import BacktestEngine, BacktestConfig
    from config.settings import settings
    import importlib, strategies as strat_module

    end = end or str(date.today())
    params_dict = json.loads(params)

    # Load strategy
    strat_cls = getattr(strat_module, strategy, None)
    if strat_cls is None:
        click.echo(f"Strategy '{strategy}' not found in strategies/. Available:")
        for name in dir(strat_module):
            if not name.startswith("_"):
                click.echo(f"  {name}")
        sys.exit(1)

    strat = strat_cls()

    # Fetch data
    click.echo(f"Fetching {symbol}…")
    df = fetch_data(
        symbol, start, end, timeframe,
        use_cache=True, cache_path=settings.data.cache_dir / "cache.db",
    )

    # Run
    engine = BacktestEngine(BacktestConfig(initial_capital=capital))
    result = engine.run(strat, df, params_dict, symbol=symbol, timeframe=timeframe,
                        start=start, end=end)

    # Print metrics
    click.echo("\n" + "─" * 60)
    click.echo(f"  {result.strategy_name}  |  {symbol}  |  {start} → {end}")
    click.echo("─" * 60)
    for k, v in result.metrics.items():
        click.echo(f"  {k:<30} {v}")
    click.echo("─" * 60)

    if save:
        from experiments.database import log_experiment
        eid = log_experiment(result, notes=notes, tags=tags)
        click.echo(f"\n✓ Saved as experiment #{eid}")

    if report:
        from dashboard.report import generate_html_report
        settings.reports_dir.mkdir(parents=True, exist_ok=True)
        rpath = settings.reports_dir / f"{strategy}_{symbol}_{end}.html"
        generate_html_report(result, output_path=str(rpath), open_browser=True)
        click.echo(f"\n✓ Report: {rpath}")


@run.command("optimization")
@click.option("--strategy", "-s", required=True)
@click.option("--symbol",   "-S", default="SPY", show_default=True)
@click.option("--from",     "start", default="2018-01-01", show_default=True)
@click.option("--to",       "end",   default=None)
@click.option("--timeframe","-t", default="1d", show_default=True)
@click.option("--method",   "-m", default="grid",
              type=click.Choice(["grid","random","bayesian","walkforward"]))
@click.option("--metric",   default="sharpe", show_default=True)
@click.option("--n-trials", default=50, show_default=True, type=int)
@click.option("--param-grid", default=None,
              help="JSON param grid, e.g. '{\"lookback\":[30,60,90]}'")
def run_optimization(strategy, symbol, start, end, timeframe, method,
                     metric, n_trials, param_grid):
    """Run parameter optimisation."""
    from datetime import date
    import strategies as strat_module
    from backtest.data import fetch_data
    from backtest.engine import BacktestEngine, BacktestConfig
    from config.settings import settings

    end = end or str(date.today())

    strat_cls = getattr(strat_module, strategy)
    strat = strat_cls()

    click.echo(f"Fetching {symbol}…")
    df = fetch_data(symbol, start, end, timeframe,
                    use_cache=True, cache_path=settings.data.cache_dir / "cache.db")

    def run_fn(params):
        engine = BacktestEngine(BacktestConfig())
        result = engine.run(strat, df, params, symbol=symbol, timeframe=timeframe)
        return result.metrics.get(metric, 0) or 0

    pg = json.loads(param_grid) if param_grid else {}

    if method == "grid":
        from optimization.grid_search import grid_search
        results_df = grid_search(run_fn, pg)
    elif method == "random":
        from optimization.random_search import random_search
        results_df = random_search(run_fn, pg, n_iter=n_trials)
    elif method == "bayesian":
        from optimization.bayesian import bayesian_optimize
        results_df = bayesian_optimize(run_fn, pg, n_trials=n_trials)
    else:
        click.echo("Walk-forward requires custom setup. See docs/walk_forward.md")
        return

    click.echo(f"\nTop 10 results by {metric}:")
    click.echo(results_df.head(10).to_string(index=False))


# ── Dashboard ─────────────────────────────────────────────────────────────────

@trade.group()
def dashboard():
    """Reporting and visualisation."""


@dashboard.command("leaderboard")
@click.option("--metric",   "-m", default="sharpe", show_default=True)
@click.option("--top",      "-n", default=20, show_default=True, type=int)
@click.option("--strategy", default=None, help="Filter by strategy name")
@click.option("--symbol",   default=None)
@click.option("--tags",     default=None)
@click.option("--export",   default=None, help="CSV export path")
def leaderboard(metric, top, strategy, symbol, tags, export):
    """Print experiment leaderboard."""
    from experiments.database import get_experiments
    from dashboard.leaderboard import print_leaderboard

    df = get_experiments(strategy_name=strategy, symbols=symbol, tags=tags)
    print_leaderboard(df, metric=metric, top=top, export_path=export)


@dashboard.command("report")
@click.option("--id", "exp_id", required=True, type=int, help="Experiment ID")
@click.option("--output", "-o", default=None)
@click.option("--open", "open_browser", is_flag=True)
def make_report(exp_id, output, open_browser):
    """Generate HTML report for an experiment."""
    from experiments.database import ExperimentDB
    db = ExperimentDB()
    # Reconstruct a lightweight result object
    df = db.get_experiments()
    row = df[df["id"] == exp_id]
    if row.empty:
        click.echo(f"Experiment #{exp_id} not found.")
        sys.exit(1)
    equity = db.get_equity_curve(exp_id)
    if equity is None:
        click.echo("No equity curve stored for this experiment.")
        sys.exit(1)

    from types import SimpleNamespace
    result = SimpleNamespace(
        strategy_name=row["strategy_name"].iloc[0],
        symbol=row["symbols"].iloc[0],
        timeframe=row["timeframe"].iloc[0],
        start=row["date_range_start"].iloc[0],
        end=row["date_range_end"].iloc[0],
        equity=equity,
        metrics={k: v for k, v in row.iloc[0].items() if k not in
                 ("id","strategy_name","symbols","timeframe",
                  "date_range_start","date_range_end","notes","tags","timestamp")},
        trades=[],
        params={},
    )
    from dashboard.report import generate_html_report
    from config.settings import settings
    out = output or str(settings.reports_dir / f"experiment_{exp_id}.html")
    generate_html_report(result, output_path=out, open_browser=open_browser)
    click.echo(f"Report saved: {out}")


# ── Live ──────────────────────────────────────────────────────────────────────

@trade.group()
def live():
    """Live trading commands."""


@live.command("start")
@click.option("--config", "-c", required=True, help="Path to live_config.yaml")
@click.option("--paper", is_flag=True, default=True, help="Paper trading mode")
def live_start(config, paper):
    """Start the live trading engine."""
    import yaml
    cfg_path = Path(config)
    if not cfg_path.exists():
        click.echo(f"Config not found: {config}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    click.echo(f"Starting live trading (paper={paper}) with config: {config}")
    click.echo("  Strategies: " + str(cfg.get("strategies", [])))
    click.echo("  Symbols:    " + str(cfg.get("symbols", [])))
    click.echo("\n[Live engine stub – implement live/runner.py to complete]")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trade()
