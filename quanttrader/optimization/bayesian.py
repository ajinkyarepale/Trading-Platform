"""optimization/bayesian.py – Optuna-based Bayesian optimisation."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def bayesian_optimize(
    run_fn: Callable[[Dict[str, Any]], float],
    param_spec: Dict[str, Any],
    n_trials: int = 100,
    maximize: bool = True,
    study_name: str = "quanttrader",
    n_jobs: int = 1,
    timeout: float | None = None,
) -> pd.DataFrame:
    """
    Bayesian optimisation using Optuna's TPE sampler.

    param_spec format (Optuna-style):
    {
        "fast_period": ("int", 5, 50),
        "slow_period": ("int", 20, 200),
        "threshold":   ("float", 0.0, 1.0),
        "method":      ("categorical", ["ema", "sma"]),
    }
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna not installed. Run: pip install optuna")

    direction = "maximize" if maximize else "minimize"

    def objective(trial: "optuna.Trial") -> float:
        params: Dict[str, Any] = {}
        for name, spec in param_spec.items():
            kind = spec[0]
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2],
                                                  step=spec[3] if len(spec) > 3 else 1)
            elif kind == "float":
                log = spec[3] if len(spec) > 3 else False
                params[name] = trial.suggest_float(name, spec[1], spec[2], log=log)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
            else:
                raise ValueError(f"Unknown param type: {kind}")
        try:
            return run_fn(params)
        except Exception as exc:
            logger.warning("Trial failed: %s", exc)
            raise optuna.exceptions.TrialPruned()

    study = optuna.create_study(direction=direction, study_name=study_name)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, timeout=timeout,
                   show_progress_bar=True)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.rename(columns={"value": "score"})
    trials_df = trials_df.sort_values("score", ascending=not maximize)
    logger.info("Best score: %.4f  params: %s",
                study.best_value, study.best_params)
    return trials_df
