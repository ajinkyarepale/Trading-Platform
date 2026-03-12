"""
optimization/optimizer.py
=========================
Parameter optimization for trading strategies.

Supports:
  - Grid Search (parallel)
  - Random Search
  - Bayesian Optimization (optuna)
  - Walk-Forward Analysis
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from backtest.engine import Backtester
from backtest.strategy import Strategy
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result Container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """Container for optimization output."""
    best_params: Dict[str, Any]
    best_score: float
    all_results: pd.DataFrame        # All param combinations and scores
    objective: str = "sharpe_ratio"
    method: str = "grid"

    def __repr__(self) -> str:
        return (
            f"OptimizationResult({self.method} | "
            f"best_{self.objective}={self.best_score:.4f} | "
            f"params={self.best_params})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Worker function (must be module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_params(args: Tuple) -> Dict[str, Any]:
    """Evaluate a single parameter combination. Used by multiprocessing."""
    strategy_class, data, params, bt_kwargs, objective = args
    try:
        strategy = strategy_class(**params)
        bt = Backtester(**bt_kwargs)
        result = bt.run(strategy, data, params={})
        score = result.metrics.get(objective, float("-inf"))
        return {"params": params, "score": score, **{f"metric_{k}": v for k, v in result.metrics.items()}}
    except Exception as e:
        return {"params": params, "score": float("-inf"), "error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Grid Search
# ─────────────────────────────────────────────────────────────────────────────

class GridSearchOptimizer:
    """
    Exhaustive grid search over all parameter combinations.

    Example
    -------
    >>> opt = GridSearchOptimizer(n_jobs=4, objective="sharpe_ratio")
    >>> result = opt.run(
    ...     strategy_class=MACrossover,
    ...     data=df,
    ...     param_grid={"fast_period": [5, 10, 20], "slow_period": [50, 100, 200]},
    ... )
    """

    def __init__(
        self,
        objective: str = "sharpe_ratio",
        n_jobs: int = -1,
        backtester_kwargs: Optional[Dict] = None,
    ) -> None:
        self.objective = objective
        self.n_jobs = cpu_count() if n_jobs == -1 else max(1, n_jobs)
        self.bt_kwargs = backtester_kwargs or {}

    def run(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
    ) -> OptimizationResult:
        """
        Run grid search.

        Args:
            strategy_class: Uninstantiated Strategy subclass.
            data          : OHLCV DataFrame.
            param_grid    : {param_name: [values_to_try]}.

        Returns:
            OptimizationResult with best params and full result table.
        """
        keys = list(param_grid.keys())
        combos = list(itertools.product(*[param_grid[k] for k in keys]))
        param_dicts = [dict(zip(keys, c)) for c in combos]

        log.info(f"Grid search: {len(param_dicts)} combinations, {self.n_jobs} workers")

        args_list = [
            (strategy_class, data, p, self.bt_kwargs, self.objective)
            for p in param_dicts
        ]

        if self.n_jobs == 1:
            results = [_evaluate_params(a) for a in args_list]
        else:
            with Pool(self.n_jobs) as pool:
                results = pool.map(_evaluate_params, args_list)

        return self._build_result(results, "grid")

    def _build_result(self, raw: List[Dict], method: str) -> OptimizationResult:
        rows = []
        for r in raw:
            row = {**r["params"], "score": r["score"]}
            for k, v in r.items():
                if k.startswith("metric_"):
                    row[k[7:]] = v
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
        best = df.iloc[0]
        best_params = {k: best[k] for k in raw[0]["params"].keys()}
        return OptimizationResult(
            best_params=best_params,
            best_score=float(best["score"]),
            all_results=df,
            objective=self.objective,
            method=method,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Random Search
# ─────────────────────────────────────────────────────────────────────────────

class RandomSearchOptimizer(GridSearchOptimizer):
    """
    Random sampling of the parameter space.

    Example
    -------
    >>> opt = RandomSearchOptimizer(n_iter=50, objective="calmar_ratio")
    >>> result = opt.run(strategy_class=MACrossover, data=df, param_grid={...})
    """

    def __init__(
        self,
        n_iter: int = 50,
        objective: str = "sharpe_ratio",
        n_jobs: int = -1,
        seed: int = 42,
        backtester_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(objective=objective, n_jobs=n_jobs, backtester_kwargs=backtester_kwargs)
        self.n_iter = n_iter
        self.seed = seed

    def run(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
    ) -> OptimizationResult:
        random.seed(self.seed)
        keys = list(param_grid.keys())
        sampled = []
        for _ in range(self.n_iter):
            p = {k: random.choice(param_grid[k]) for k in keys}
            sampled.append(p)

        log.info(f"Random search: {self.n_iter} samples, {self.n_jobs} workers")

        args_list = [(strategy_class, data, p, self.bt_kwargs, self.objective) for p in sampled]

        if self.n_jobs == 1:
            results = [_evaluate_params(a) for a in args_list]
        else:
            with Pool(self.n_jobs) as pool:
                results = pool.map(_evaluate_params, args_list)

        return self._build_result(results, "random")


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian Optimization (optuna)
# ─────────────────────────────────────────────────────────────────────────────

class BayesianOptimizer:
    """
    Bayesian optimization using Optuna.

    Example
    -------
    >>> opt = BayesianOptimizer(n_trials=100, objective="sharpe_ratio")
    >>> result = opt.run(
    ...     strategy_class=MACrossover,
    ...     data=df,
    ...     param_space={
    ...         "fast_period": ("int", 2, 50),
    ...         "slow_period": ("int", 20, 200),
    ...         "threshold": ("float", 0.0, 0.05),
    ...     },
    ... )
    """

    def __init__(
        self,
        n_trials: int = 100,
        objective: str = "sharpe_ratio",
        direction: str = "maximize",
        backtester_kwargs: Optional[Dict] = None,
        seed: int = 42,
    ) -> None:
        self.n_trials = n_trials
        self.objective = objective
        self.direction = direction
        self.bt_kwargs = backtester_kwargs or {}
        self.seed = seed

    def run(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_space: Dict[str, Tuple],
    ) -> OptimizationResult:
        """
        Run Bayesian optimization.

        Args:
            strategy_class: Uninstantiated Strategy subclass.
            data          : OHLCV DataFrame.
            param_space   : {name: ("int"|"float"|"categorical", lo, hi) or ("categorical", [choices])}.

        Returns:
            OptimizationResult.
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna not installed. Run: pip install optuna")

        all_results = []
        objective_str = self.objective

        def _objective(trial: "optuna.Trial") -> float:
            params = {}
            for name, spec in param_space.items():
                if spec[0] == "int":
                    params[name] = trial.suggest_int(name, spec[1], spec[2])
                elif spec[0] == "float":
                    params[name] = trial.suggest_float(name, spec[1], spec[2])
                elif spec[0] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec[1])
                else:
                    params[name] = trial.suggest_float(name, spec[1], spec[2])

            try:
                strategy = strategy_class(**params)
                bt = Backtester(**self.bt_kwargs)
                result = bt.run(strategy, data, params={})
                score = result.metrics.get(objective_str, float("-inf"))
                all_results.append({"params": params, "score": score, **result.metrics})
                return score
            except Exception:
                return float("-inf")

        sampler = optuna.samplers.TPESampler(seed=self.seed)
        study = optuna.create_study(direction=self.direction, sampler=sampler)
        study.optimize(_objective, n_trials=self.n_trials, show_progress_bar=False)

        log.info(f"Bayesian opt: best {self.objective} = {study.best_value:.4f}")

        rows = []
        for r in all_results:
            row = {**r["params"], "score": r["score"]}
            for k, v in r.items():
                if k not in ("params", "score"):
                    row[k] = v
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

        return OptimizationResult(
            best_params=study.best_params,
            best_score=study.best_value,
            all_results=df,
            objective=self.objective,
            method="bayesian",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: Dict[str, Any]
    train_score: float
    test_score: float
    test_result: Any  # BacktestResult


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis to detect in-sample vs out-of-sample performance.

    Uses either rolling or expanding training window.

    Example
    -------
    >>> wfa = WalkForwardAnalyzer(
    ...     n_windows=5,
    ...     train_pct=0.7,
    ...     window_type="rolling",
    ...     optimizer=GridSearchOptimizer(),
    ... )
    >>> windows, oos_equity = wfa.run(MACrossover, data, param_grid={...})
    """

    def __init__(
        self,
        n_windows: int = 5,
        train_pct: float = 0.7,
        window_type: Literal["rolling", "expanding"] = "expanding",
        optimizer: Optional[GridSearchOptimizer] = None,
        backtester_kwargs: Optional[Dict] = None,
    ) -> None:
        self.n_windows = n_windows
        self.train_pct = train_pct
        self.window_type = window_type
        self.optimizer = optimizer or GridSearchOptimizer(n_jobs=1)
        self.bt_kwargs = backtester_kwargs or {}

    def run(
        self,
        strategy_class: type,
        data: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
    ) -> Tuple[List[WalkForwardWindow], pd.Series]:
        """
        Run walk-forward analysis.

        Returns:
            (windows, combined_oos_equity)
        """
        n = len(data)
        window_size = n // (self.n_windows + 1)
        windows = []
        oos_equities = []

        for i in range(self.n_windows):
            # Training period
            if self.window_type == "expanding":
                train_end_idx = int(window_size * (i + 1) * self.train_pct / self.train_pct)
                train_end_idx = min(train_end_idx, int(n * self.train_pct))
                train_data = data.iloc[: window_size * (i + 1)]
            else:  # rolling
                start_idx = i * window_size
                train_end_idx = start_idx + int(window_size * self.train_pct)
                train_data = data.iloc[start_idx:train_end_idx]

            # Test period
            test_start_idx = len(train_data)
            test_end_idx = min(test_start_idx + window_size, n)
            test_data = data.iloc[test_start_idx:test_end_idx]

            if len(train_data) < 50 or len(test_data) < 10:
                continue

            log.info(f"WFA window {i+1}/{self.n_windows}: "
                     f"train={len(train_data)} bars, test={len(test_data)} bars")

            # Optimize on training
            train_result = self.optimizer.run(strategy_class, train_data, param_grid)
            best_params = train_result.best_params

            # Evaluate on test (out-of-sample)
            strategy = strategy_class(**best_params)
            bt = Backtester(**self.bt_kwargs)
            test_result = bt.run(strategy, test_data)
            test_score = test_result.metrics.get(self.optimizer.objective, 0.0)

            window = WalkForwardWindow(
                window_id=i + 1,
                train_start=train_data.index[0],
                train_end=train_data.index[-1],
                test_start=test_data.index[0],
                test_end=test_data.index[-1],
                best_params=best_params,
                train_score=train_result.best_score,
                test_score=test_score,
                test_result=test_result,
            )
            windows.append(window)
            oos_equities.append(test_result.equity_curve)

        # Stitch OOS equity curves
        combined_oos = _stitch_equity_curves(oos_equities) if oos_equities else pd.Series(dtype=float)

        return windows, combined_oos

    def summary(self, windows: List[WalkForwardWindow]) -> pd.DataFrame:
        """Return summary DataFrame of all windows."""
        rows = []
        for w in windows:
            rows.append({
                "window": w.window_id,
                "train_start": w.train_start.date(),
                "train_end": w.train_end.date(),
                "test_start": w.test_start.date(),
                "test_end": w.test_end.date(),
                "train_score": w.train_score,
                "test_score": w.test_score,
                "efficiency": w.test_score / w.train_score if w.train_score != 0 else 0,
                **w.best_params,
            })
        return pd.DataFrame(rows)


def _stitch_equity_curves(curves: List[pd.Series]) -> pd.Series:
    """Stitch OOS equity curves into a continuous series."""
    if not curves:
        return pd.Series(dtype=float)
    result = curves[0].copy()
    for curve in curves[1:]:
        if result.empty:
            result = curve.copy()
            continue
        scale = result.iloc[-1] / curve.iloc[0]
        scaled = curve * scale
        result = pd.concat([result, scaled.iloc[1:]])
    return result.sort_index()
