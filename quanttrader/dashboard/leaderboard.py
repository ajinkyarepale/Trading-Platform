"""
dashboard/leaderboard.py
─────────────────────────
CLI leaderboard – ranked table of experiment results.
"""

from __future__ import annotations

import sys
from typing import List, Optional

import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
except ImportError:
    RICH = False


def print_leaderboard(
    df: pd.DataFrame,
    metric: str = "sharpe",
    top: int = 20,
    columns: Optional[List[str]] = None,
    export_path: Optional[str] = None,
    ascending: bool = False,
) -> None:
    """
    Print a ranked leaderboard of experiments to stdout.

    Parameters
    ----------
    df         : DataFrame from ExperimentDB.get_experiments()
    metric     : Column to sort by
    top        : Show only the top N rows
    columns    : Columns to display (default: sensible subset)
    export_path: If set, also write to this CSV path
    ascending  : Sort direction
    """
    if df.empty:
        print("No experiments found.")
        return

    default_cols = [
        "id", "strategy_name", "symbols", "timeframe",
        "date_range_start", "date_range_end",
        "total_return_pct", "cagr_pct", "sharpe", "sortino",
        "max_drawdown_pct", "calmar", "n_trades", "win_rate_pct",
    ]

    if metric not in df.columns:
        print(f"Warning: metric '{metric}' not found. Available: {list(df.columns)}")
        metric = "sharpe" if "sharpe" in df.columns else df.columns[0]

    display_cols = [c for c in (columns or default_cols) if c in df.columns]
    ranked = df.sort_values(metric, ascending=ascending).head(top)

    if export_path:
        ranked[display_cols].to_csv(export_path, index=False)
        print(f"Exported to {export_path}")

    if RICH:
        console = Console()
        table = Table(
            title=f"🏆 Leaderboard  (sorted by {metric}, top {top})",
            show_header=True, header_style="bold cyan",
        )

        for col in display_cols:
            table.add_column(col, justify="right" if ranked[col].dtype != "object" else "left")

        for _, row in ranked[display_cols].iterrows():
            table.add_row(*[_fmt(row[c], c) for c in display_cols])

        console.print(table)
    else:
        print(ranked[display_cols].to_string(index=False))


def _fmt(val, col: str) -> str:
    if pd.isna(val):
        return "—"
    if "pct" in col or col in ("cagr_pct", "total_return_pct"):
        return f"{val:.2f}%"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)
