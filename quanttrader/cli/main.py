"""
cli/main.py
===========
QuantTrader command-line interface.

Usage:
  trade --help
  trade run backtest --strategy zscore --symbols AAPL --from 2020-01-01 --to 2023-12-31
  trade run optimization --strategy ma_crossover --method grid
  trade live start --paper
  trade dashboard leaderboard --metric sharpe --top 20
  trade data fetch --symbol AAPL --from 2020-01-01 --to 2024-01-01
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

# ── Registry of available strategies ────────────────────────────────────────
STRATEGY_REGISTRY = {
    "zscore": ("strategies.zscore_mean_reversion", "ZScoreMeanReversion"),
    "ma_crossover": ("strategies.dual_ma_crossover", "DualMACrossover"),
    "rsi": ("strategies.rsi_mean_reversion", "RSIMeanReversion"),
    "breakout": ("strategies.breakout", "DonchianBreakout"),
    "pairs": ("strategies.pairs_trading", "PairsTrading"),
    "random_forest": ("ml_strategies.ml_strategy", "RandomForestStrategy"),
    "xgboost": ("ml_strategies.ml_strategy", "XGBoostStrategy"),
}


def _load_strategy(name: str):
    """Dynamically load a strategy class."""
    if name not in STRATEGY_REGISTRY:
        click.echo(f"Unknown strategy: {name!r}. Available: {list(STRATEGY_REGISTRY.keys())}", err=True)
        sys.exit(1)
    module_path, class_name = STRATEGY_REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


# ── Top-level group ──────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0", prog_name="QuantTrader")
def cli():
    """
    QuantTrader – Professional Quantitative Trading Platform.

    \b
    Commands:
      run           Backtest or optimize a strategy
      live          Live trading operations
      dashboard     View results and reports
      data          Fetch and manage market data

    Run 'trade COMMAND --help' for command-specific help.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# trade run
# ─────────────────────────────────────────────────────────────────────────────

@cli.group()
def run():
    """Run backtests and optimizations."""
    pass


@run.command("backtest")
@click.option("--strategy", "-s", required=True,
              type=click.Choice(list(STRATEGY_REGISTRY.keys())),
              help="Strategy name")
@click.option("--symbols", "-sym", required=True,
              help="Comma-separated list of symbols (e.g., AAPL,MSFT)")
@click.option("--from", "start_date", default="2020-01-01",
              help="Start date YYYY-MM-DD")
@click.option("--to", "end_date", default=datetime.today().strftime("%Y-%m-%d"),
              help="End date YYYY-MM-DD")
@click.option("--timeframe", "-tf", default="1d",
              type=click.Choice(["1m","5m","15m","1h","1d"]),
              help="Data timeframe")
@click.option("--params", "-p", default="{}",
              help='JSON params, e.g. \'{"lookback": 30, "entry_z": 2.0}\'')
@click.option("--capital", "-c", default=100_000.0, type=float,
              help="Initial capital")
@click.option("--no-short", is_flag=True, help="Disable short selling")
@click.option("--report", "-r", is_flag=True, help="Generate HTML report")
@click.option("--save", is_flag=True, help="Save results to experiment database")
@click.option("--notes", default="", help="Notes for experiment record")
def backtest_cmd(strategy, symbols, start_date, end_date, timeframe, params,
                 capital, no_short, report, save, notes):
    """Run a backtest for a strategy."""
    from backtest.data import DataFetcher
    from backtest.engine import Backtester

    StrategyClass = _load_strategy(strategy)
    try:
        strategy_params = json.loads(params)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON params: {e}", err=True)
        sys.exit(1)

    symbol_list = [s.strip() for s in symbols.split(",")]

    click.echo(f"\n🔍 Running backtest: {strategy.upper()}")
    click.echo(f"   Symbols  : {symbol_list}")
    click.echo(f"   Period   : {start_date} → {end_date}")
    click.echo(f"   Timeframe: {timeframe}")
    click.echo(f"   Capital  : ${capital:,.0f}")
    click.echo(f"   Params   : {strategy_params}\n")

    fetcher = DataFetcher()
    data_dict = {}
    for sym in symbol_list:
        click.echo(f"   Fetching {sym}...")
        try:
            data_dict[sym] = fetcher.fetch(sym, start_date, end_date, timeframe)
        except Exception as e:
            click.echo(f"   ⚠ Failed to fetch {sym}: {e}", err=True)

    if not data_dict:
        click.echo("❌ No data fetched. Exiting.", err=True)
        sys.exit(1)

    strat = StrategyClass(**strategy_params)
    bt = Backtester(initial_capital=capital, allow_short=not no_short)

    if len(symbol_list) == 1:
        sym = symbol_list[0]
        strat._symbol = sym
        result = bt.run(strat, data_dict[sym])
    else:
        from backtest.engine import PortfolioBacktester
        pbt = PortfolioBacktester(initial_capital=capital)
        result = pbt.run(strat, data_dict)

    _print_metrics(result.metrics)

    if save:
        from experiments.database import ExperimentDB
        db = ExperimentDB()
        exp_id = db.log_experiment(
            result,
            symbols=symbol_list,
            timeframe=timeframe,
            date_range=(start_date, end_date),
            notes=notes,
        )
        click.echo(f"\n✅ Saved as experiment #{exp_id}")

    if report:
        from dashboard.report import generate_html_report
        report_path = generate_html_report(
            equity_curve=result.equity_curve,
            trades=result.trades,
            metrics=result.metrics,
            strategy_name=strat.metadata.name,
            params=result.params,
        )
        click.echo(f"\n📊 Report: {report_path}")


@run.command("optimization")
@click.option("--strategy", "-s", required=True,
              type=click.Choice(list(STRATEGY_REGISTRY.keys())))
@click.option("--symbols", "-sym", required=True)
@click.option("--from", "start_date", default="2018-01-01")
@click.option("--to", "end_date", default="2023-12-31")
@click.option("--timeframe", "-tf", default="1d",
              type=click.Choice(["1m","5m","15m","1h","1d"]))
@click.option("--method", "-m", default="grid",
              type=click.Choice(["grid", "random", "bayesian"]))
@click.option("--n-trials", default=50, type=int, help="Trials for random/bayesian")
@click.option("--objective", "-obj", default="sharpe_ratio",
              help="Metric to optimise")
@click.option("--jobs", "-j", default=-1, type=int, help="Parallel jobs (-1=all cores)")
@click.option("--capital", "-c", default=100_000.0, type=float)
def optimization_cmd(strategy, symbols, start_date, end_date, timeframe,
                     method, n_trials, objective, jobs, capital):
    """Optimize strategy parameters."""
    from backtest.data import DataFetcher
    from optimization.optimizer import GridSearchOptimizer, RandomSearchOptimizer, BayesianOptimizer

    StrategyClass = _load_strategy(strategy)
    symbol_list = [s.strip() for s in symbols.split(",")]
    sym = symbol_list[0]

    click.echo(f"\n🔬 Optimising {strategy.upper()} on {sym}")
    click.echo(f"   Method   : {method} | Objective: {objective}\n")

    fetcher = DataFetcher()
    data = fetcher.fetch(sym, start_date, end_date, timeframe)

    strat_instance = StrategyClass()
    param_grid = strat_instance.get_param_grid()

    bt_kwargs = {"initial_capital": capital}

    if method == "grid":
        opt = GridSearchOptimizer(objective=objective, n_jobs=jobs, backtester_kwargs=bt_kwargs)
        result = opt.run(StrategyClass, data, param_grid)
    elif method == "random":
        opt = RandomSearchOptimizer(n_iter=n_trials, objective=objective,
                                    n_jobs=jobs, backtester_kwargs=bt_kwargs)
        result = opt.run(StrategyClass, data, param_grid)
    else:
        # Bayesian
        param_space = {}
        for k, vals in param_grid.items():
            if vals and isinstance(vals[0], int):
                param_space[k] = ("int", int(min(vals)), int(max(vals)))
            elif vals:
                param_space[k] = ("float", float(min(vals)), float(max(vals)))
        from optimization.optimizer import BayesianOptimizer
        opt = BayesianOptimizer(n_trials=n_trials, objective=objective, backtester_kwargs=bt_kwargs)
        result = opt.run(StrategyClass, data, param_space)

    click.echo(f"\n🏆 Best {objective}: {result.best_score:.4f}")
    click.echo(f"   Best params: {result.best_params}")
    click.echo(f"\nTop 10 results:")
    top10 = result.all_results.head(10)[["score"] + list(result.best_params.keys())]
    click.echo(top10.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# trade live
# ─────────────────────────────────────────────────────────────────────────────

@cli.group()
def live():
    """Live trading operations."""
    pass


@live.command("start")
@click.option("--strategy", "-s", required=True,
              type=click.Choice(list(STRATEGY_REGISTRY.keys())))
@click.option("--symbols", "-sym", required=True)
@click.option("--params", "-p", default="{}")
@click.option("--paper", is_flag=True, default=True, help="Use paper trading")
@click.option("--capital", "-c", default=100_000.0, type=float)
@click.option("--cron", default=None,
              help="Cron schedule (e.g., '0 16 * * 1-5'). None=run once.")
@click.option("--timeframe", "-tf", default="1d")
def live_start(strategy, symbols, params, paper, capital, cron, timeframe):
    """Start live trading."""
    from live.paper_broker import PaperBroker
    from live.trader import LiveTrader
    from live.risk import RiskManager

    StrategyClass = _load_strategy(strategy)
    symbol_list = [s.strip() for s in symbols.split(",")]
    strategy_params = json.loads(params)

    click.echo(f"\n🚀 Starting {'PAPER' if paper else 'LIVE'} trading")
    click.echo(f"   Strategy : {strategy}")
    click.echo(f"   Symbols  : {symbol_list}")

    strat = StrategyClass(**strategy_params)
    broker = PaperBroker(initial_balance=capital)
    broker.connect()

    risk = RiskManager(broker)
    trader = LiveTrader(
        strategy=strat,
        broker=broker,
        symbols=symbol_list,
        timeframe=timeframe,
        risk_manager=risk,
    )

    try:
        trader.start(schedule_cron=cron, run_immediately=True)
    except KeyboardInterrupt:
        click.echo("\n⏹ Stopped by user")
        trader.stop()


@live.command("status")
def live_status():
    """Show live trading status."""
    from live.paper_broker import PaperBroker
    broker = PaperBroker()
    broker.connect()
    account = broker.get_account()
    click.echo(f"\n💰 Account: ${account.equity:,.2f} (P&L: ${account.unrealised_pnl:+,.2f})")
    positions = broker.get_positions()
    if positions:
        click.echo("\nOpen Positions:")
        for p in positions:
            click.echo(f"  {p.symbol:10} {p.units:+.2f} @ {p.entry_price:.4f}"
                       f"  P&L: ${p.unrealised_pnl:+,.2f}")
    else:
        click.echo("No open positions")


# ─────────────────────────────────────────────────────────────────────────────
# trade dashboard
# ─────────────────────────────────────────────────────────────────────────────

@cli.group()
def dashboard():
    """View results and reports."""
    pass


@dashboard.command("leaderboard")
@click.option("--metric", "-m", default="sharpe_ratio",
              type=click.Choice(["sharpe_ratio", "total_return", "calmar_ratio",
                                 "cagr", "win_rate", "profit_factor"]),
              help="Sort metric")
@click.option("--top", "-n", default=20, type=int, help="Number of rows")
@click.option("--strategy", default=None, help="Filter by strategy name")
@click.option("--min-sharpe", default=None, type=float)
@click.option("--export", default=None, type=click.Path(),
              help="Export to CSV/JSON path")
def leaderboard_cmd(metric, top, strategy, min_sharpe, export):
    """Show experiment leaderboard."""
    from experiments.database import ExperimentDB
    from dashboard.leaderboard import print_leaderboard

    db = ExperimentDB()
    df = db.get_experiments(
        strategy_name=strategy,
        min_sharpe=min_sharpe,
    )

    if df.empty:
        click.echo("📭 No experiments found. Run a backtest with --save first.")
        return

    print_leaderboard(df, metric=metric, top=top)

    if export:
        path = Path(export)
        if path.suffix == ".json":
            df.head(top).to_json(path, orient="records", indent=2)
        else:
            df.head(top).to_csv(path, index=False)
        click.echo(f"\n📤 Exported to {path}")


@dashboard.command("report")
@click.option("--id", "exp_id", required=True, type=int, help="Experiment ID")
@click.option("--output", "-o", default=None, type=click.Path())
def report_cmd(exp_id, output):
    """Generate HTML report for an experiment."""
    from experiments.database import ExperimentDB
    from dashboard.report import generate_html_report

    db = ExperimentDB()
    exp = db.get_experiment(exp_id)
    if exp is None:
        click.echo(f"❌ Experiment #{exp_id} not found.", err=True)
        sys.exit(1)

    output_path = Path(output) if output else None
    report_path = generate_html_report(
        equity_curve=exp["equity_curve"],
        trades=exp["trades"],
        metrics=exp["metrics"],
        strategy_name=exp["strategy_name"],
        params=exp["params"],
        output_path=output_path,
    )
    click.echo(f"📊 Report generated: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# trade data
# ─────────────────────────────────────────────────────────────────────────────

@cli.group()
def data():
    """Fetch and manage market data."""
    pass


@data.command("fetch")
@click.option("--symbol", "-sym", required=True, help="Ticker symbol")
@click.option("--from", "start_date", default="2020-01-01")
@click.option("--to", "end_date", default=datetime.today().strftime("%Y-%m-%d"))
@click.option("--timeframe", "-tf", default="1d",
              type=click.Choice(["1m","5m","15m","1h","1d"]))
@click.option("--source", default="yfinance",
              type=click.Choice(["yfinance", "alphavantage", "csv"]))
@click.option("--output", "-o", default=None, type=click.Path(),
              help="Save to CSV file")
def data_fetch(symbol, start_date, end_date, timeframe, source, output):
    """Fetch market data for a symbol."""
    from backtest.data import DataFetcher

    click.echo(f"\n📡 Fetching {symbol} {timeframe} [{start_date} → {end_date}]...")
    fetcher = DataFetcher(source=source)
    df = fetcher.fetch(symbol, start_date, end_date, timeframe)

    click.echo(f"✅ {len(df)} bars fetched")
    click.echo(f"\nFirst 5 rows:\n{df.head()}")
    click.echo(f"\nLast 5 rows:\n{df.tail()}")

    if output:
        df.to_csv(output)
        click.echo(f"\n💾 Saved to {output}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_metrics(metrics: dict) -> None:
    """Pretty-print key metrics."""
    click.echo("\n" + "="*50)
    click.echo("📈 PERFORMANCE METRICS")
    click.echo("="*50)

    display = [
        ("Total Return",    "total_return",    "{:.1%}"),
        ("CAGR",            "cagr",            "{:.1%}"),
        ("Annualised Vol",  "annualised_volatility", "{:.1%}"),
        ("Sharpe Ratio",    "sharpe_ratio",    "{:.3f}"),
        ("Sortino Ratio",   "sortino_ratio",   "{:.3f}"),
        ("Calmar Ratio",    "calmar_ratio",    "{:.3f}"),
        ("Max Drawdown",    "max_drawdown",    "{:.1%}"),
        ("Win Rate",        "win_rate",        "{:.1%}"),
        ("Profit Factor",   "profit_factor",   "{:.2f}"),
        ("Num Trades",      "num_trades",      "{:.0f}"),
        ("VaR 95%",         "var_95",          "{:.2%}"),
        ("CVaR 95%",        "cvar_95",         "{:.2%}"),
    ]

    for label, key, fmt in display:
        val = metrics.get(key)
        if val is None:
            continue
        try:
            formatted = fmt.format(float(val))
            # Colour coding
            if key in ("total_return", "cagr", "sharpe_ratio", "sortino_ratio"):
                color = "green" if float(val) > 0 else "red"
            elif key == "max_drawdown":
                color = "red"
            else:
                color = None

            if color:
                click.echo(f"  {label:22}: {click.style(formatted, fg=color)}")
            else:
                click.echo(f"  {label:22}: {formatted}")
        except Exception:
            click.echo(f"  {label:22}: {val}")

    click.echo("="*50 + "\n")


if __name__ == "__main__":
    cli()
