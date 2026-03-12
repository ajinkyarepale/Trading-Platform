"""
notebooks/quickstart.py
────────────────────────
Full platform demo – run with: python notebooks/quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings; warnings.filterwarnings("ignore")

from config.settings import settings
settings.ensure_dirs()

print("=" * 60)
print("  QuantTrader Quickstart Demo")
print("=" * 60)

# 1. Fetch Data
print("\n[1] Fetching SPY data...")
from backtest.data import fetch_data
df = fetch_data("SPY", "2018-01-01", "2024-01-01", "1d",
                use_cache=True, cache_path=settings.data.cache_dir / "cache.db")
print(f"  {len(df)} bars | {df.index[0].date()} → {df.index[-1].date()}")

# 2. Backtest
print("\n[2] Running Z-Score backtest...")
from backtest.engine import BacktestEngine, BacktestConfig
from strategies.zscore_mean_reversion import ZScoreMeanReversion
engine = BacktestEngine(BacktestConfig(initial_capital=100_000))
result = engine.run(ZScoreMeanReversion(), df,
                    {"lookback":60,"entry_z":2.0,"exit_z":0.5}, symbol="SPY")
for k, v in result.metrics.items():
    if v is not None: print(f"  {k:<30} {v}")

# 3. Save & Leaderboard
print("\n[3] Saving experiment...")
from experiments.database import ExperimentDB
db = ExperimentDB()
eid = db.log_experiment(result, notes="quickstart demo")
print(f"  Saved as experiment #{eid}")

from dashboard.leaderboard import print_leaderboard
print_leaderboard(db.get_experiments(), metric="sharpe", top=10)

# 4. Report
print("\n[4] Generating HTML report...")
from dashboard.report import generate_html_report
rp = settings.reports_dir / "quickstart_demo.html"
generate_html_report(result, output_path=str(rp))
print(f"  Report: {rp}")

print("\n✓ Done! Open the HTML report in your browser.")
