"""
optimization/walk_forward.py
─────────────────────────────
Walk-forward (out-of-sample) analysis.

Splits data into training/validation windows, optimises on train,
evaluates on validation, and aggregates OOS performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WFWindow:
    window_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    best_params: Dict[str, Any]
    train_score: float
    val_score: float
    val_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WFResult:
    windows: List[WFWindow]
    oos_equity: pd.Series        # concatenated OOS equity curves
    oos_metrics: Dict[str, Any]  # aggregate metrics over all OOS periods

    @property
    def summary_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "window": w.window_id,
                "train": f"{w.train_start} → {w.train_end}",
                "val": f"{w.val_start} → {w.val_end}",
                "train_score": w.train_score,
                "val_score": w.val_score,
                **{f"param_{k}": v for k, v in w.best_params.items()},
            }
            for w in self.windows
        ])


def walk_forward(
    data: pd.DataFrame,
    optimize_fn: Callable[[pd.DataFrame, Optional[pd.DataFrame]], Tuple[Dict, float]],
    evaluate_fn: Callable[[pd.DataFrame, Dict], Tuple[float, Dict, pd.Series]],
    n_windows: int = 5,
    train_ratio: float = 0.7,
    mode: str = "rolling",   # "rolling" | "expanding"
    min_train_bars: int = 252,
) -> WFResult:
    """
    Walk-forward analysis.

    Parameters
    ----------
    data        : Full OHLCV DataFrame.
    optimize_fn : fn(train_df) → (best_params, best_score).
    evaluate_fn : fn(val_df, params) → (score, metrics_dict, equity_series).
    n_windows   : Number of train/val folds.
    train_ratio : Fraction of each window dedicated to training.
    mode        : 'rolling' (fixed window) or 'expanding' (anchor at start).
    """
    n = len(data)
    window_size = n // n_windows
    windows: List[WFWindow] = []
    oos_equity_parts: List[pd.Series] = []

    for i in range(n_windows):
        if mode == "expanding":
            train_start_idx = 0
        else:
            train_start_idx = i * window_size

        val_end_idx = min((i + 1) * window_size + int(window_size * (1 - train_ratio)), n)
        train_end_idx = train_start_idx + int(window_size * train_ratio)
        val_start_idx = train_end_idx

        if train_end_idx - train_start_idx < min_train_bars:
            logger.warning("Window %d: not enough training data, skipping", i)
            continue
        if val_end_idx <= val_start_idx:
            break

        train_df = data.iloc[train_start_idx:train_end_idx]
        val_df   = data.iloc[val_start_idx:val_end_idx]

        logger.info(
            "Window %d | Train: %s → %s | Val: %s → %s",
            i,
            train_df.index[0].date(), train_df.index[-1].date(),
            val_df.index[0].date(),  val_df.index[-1].date(),
        )

        # Optimise on training window
        best_params, train_score = optimize_fn(train_df)

        # Evaluate on validation window
        val_score, val_metrics, val_equity = evaluate_fn(val_df, best_params)

        oos_equity_parts.append(val_equity)
        windows.append(WFWindow(
            window_id=i,
            train_start=str(train_df.index[0].date()),
            train_end=str(train_df.index[-1].date()),
            val_start=str(val_df.index[0].date()),
            val_end=str(val_df.index[-1].date()),
            best_params=best_params,
            train_score=train_score,
            val_score=val_score,
            val_metrics=val_metrics,
        ))

    if not oos_equity_parts:
        raise ValueError("No valid walk-forward windows produced data.")

    # Chain equity curves (re-base each segment to the previous endpoint)
    chained_parts = [oos_equity_parts[0]]
    for part in oos_equity_parts[1:]:
        scale = chained_parts[-1].iloc[-1] / part.iloc[0]
        chained_parts.append(part * scale)
    oos_equity = pd.concat(chained_parts)
    oos_equity.name = "equity"

    from backtest.metrics import compute_metrics
    oos_metrics = compute_metrics(oos_equity)

    logger.info(
        "Walk-forward complete | %d windows | OOS Sharpe: %.3f | OOS Return: %.2f%%",
        len(windows), oos_metrics.get("sharpe", 0), oos_metrics.get("total_return_pct", 0),
    )

    return WFResult(windows=windows, oos_equity=oos_equity, oos_metrics=oos_metrics)
