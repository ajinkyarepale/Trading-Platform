"""
optimization/grid_search.py
────────────────────────────
Parallel exhaustive grid search over parameter combinations.
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

logger = logging.getLogger(__name__)


def _evaluate(args: Tuple) -> Tuple[Dict, float]:
    """Worker function – runs one parameter combination."""
    run_fn, params = args
    try:
        score = run_fn(params)
    except Exception as exc:
        logger.warning("Params %s failed: %s", params, exc)
        score = float("-inf")
    return params, score


def grid_search(
    run_fn: Callable[[Dict[str, Any]], float],
    param_grid: Dict[str, Iterable],
    n_jobs: int = -1,
    maximize: bool = True,
) -> pd.DataFrame:
    """
    Exhaustive grid search.

    Parameters
    ----------
    run_fn     : Function that accepts a param dict and returns a scalar score.
    param_grid : {'param_name': [val1, val2, ...], ...}
    n_jobs     : Parallel workers (-1 = all CPUs).
    maximize   : True → sort descending by score.

    Returns
    -------
    DataFrame sorted by score (best first).
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))
    param_dicts = [{k: v for k, v in zip(keys, combo)} for combo in combos]

    logger.info("Grid search: %d combinations on %d workers",
                len(param_dicts), n_jobs if n_jobs > 0 else mp.cpu_count())

    n_workers = n_jobs if n_jobs > 0 else mp.cpu_count()
    args = [(run_fn, p) for p in param_dicts]

    with mp.Pool(n_workers) as pool:
        raw = list(tqdm(pool.imap(_evaluate, args), total=len(args),
                        desc="Grid Search"))

    results = [{"score": score, **params} for params, score in raw]
    df = pd.DataFrame(results).sort_values("score", ascending=not maximize)
    df = df.reset_index(drop=True)
    logger.info("Best score: %.4f  params: %s", df["score"].iloc[0],
                {k: df[k].iloc[0] for k in keys})
    return df
