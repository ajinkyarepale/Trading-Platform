"""optimization/random_search.py – Random parameter sampling."""

from __future__ import annotations

import logging
import random
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

ParamSpec = Dict[str, Union[List, Tuple]]   # list → discrete, tuple → (lo, hi, type)


def _sample(spec: ParamSpec, rng: random.Random) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, vals in spec.items():
        if isinstance(vals, (list, tuple)) and len(vals) == 3 and isinstance(vals[2], type):
            lo, hi, dtype = vals
            if dtype == int:
                params[name] = rng.randint(int(lo), int(hi))
            else:
                params[name] = dtype(rng.uniform(float(lo), float(hi)))
        elif isinstance(vals, (list, tuple)):
            params[name] = rng.choice(list(vals))
        else:
            params[name] = vals
    return params


def random_search(
    run_fn: Callable[[Dict[str, Any]], float],
    param_spec: ParamSpec,
    n_iter: int = 50,
    maximize: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random search over *param_spec*.

    param_spec values can be:
      - A list/tuple of choices: [0.1, 0.2, 0.5]
      - A range tuple: (lo, hi, type)  e.g. (5, 50, int)
    """
    rng = random.Random(seed)
    results = []

    for i in range(n_iter):
        params = _sample(param_spec, rng)
        try:
            score = run_fn(params)
        except Exception as exc:
            logger.warning("Iteration %d failed: %s", i, exc)
            score = float("-inf")
        results.append({"score": score, **params})
        logger.debug("Iter %d/%d  score=%.4f  params=%s", i + 1, n_iter, score, params)

    df = pd.DataFrame(results).sort_values("score", ascending=not maximize)
    return df.reset_index(drop=True)
