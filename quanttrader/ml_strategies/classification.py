"""
ml_strategies/classification.py
─────────────────────────────────
Random Forest direction-prediction strategy.

Features:  20 technical indicators (FeatureEngineer).
Target:    1 if next-bar return > threshold, else -1.
Training:  Walk-forward – re-train every `retrain_every` bars.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from backtest.strategy import Strategy, StrategyMetadata
from .features import FeatureEngineer

logger = logging.getLogger(__name__)


class RandomForestStrategy(Strategy):
    metadata = StrategyMetadata(
        name="RandomForestStrategy",
        version="1.0.0",
        author="QuantTrader",
        description="Random Forest next-day direction prediction.",
        tags=["ml", "classification", "random-forest"],
        param_schema={
            "n_estimators":   (int,   10,  500,  100),
            "max_depth":      (int,   2,   20,   5),
            "min_train_bars": (int,   100, 2000, 252),
            "retrain_every":  (int,   20,  500,  63),
            "return_threshold": (float, 0.0, 0.05, 0.001),
            "pred_threshold": (float, 0.5, 0.9,  0.55),
        },
    )

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir
        self._models: list = []
        self._scaler: Optional[StandardScaler] = None
        self.feature_eng = FeatureEngineer()

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> pd.Series:
        params = self.validate_params(params)

        if isinstance(data, dict):
            data = next(iter(data.values()))

        features = self.feature_eng.transform(data)
        prices = data["close"].reindex(features.index)

        n_est    = params["n_estimators"]
        max_d    = params["max_depth"]
        min_tr   = params["min_train_bars"]
        retrain  = params["retrain_every"]
        ret_thr  = params["return_threshold"]
        pred_thr = params["pred_threshold"]

        signal = pd.Series(0, index=features.index, dtype=int)
        model: Optional[RandomForestClassifier] = None
        scaler = StandardScaler()

        for i in range(min_tr, len(features)):
            # Re-train periodically
            if (i - min_tr) % retrain == 0:
                X_train = features.iloc[:i].values
                y_fwd = prices.iloc[1:i+1].values / prices.iloc[:i].values - 1
                y_train = np.where(y_fwd > ret_thr, 1, -1)

                # Remove NaN rows
                mask = ~np.isnan(X_train).any(axis=1)
                X_train, y_train = X_train[mask], y_train[mask]

                if len(np.unique(y_train)) < 2:
                    continue

                scaler.fit(X_train)
                X_scaled = scaler.transform(X_train)

                model = RandomForestClassifier(
                    n_estimators=n_est, max_depth=max_d,
                    n_jobs=-1, random_state=42,
                )
                model.fit(X_scaled, y_train)

            if model is None:
                continue

            X_cur = features.iloc[i:i+1].values
            if np.isnan(X_cur).any():
                continue

            proba = model.predict_proba(scaler.transform(X_cur))[0]
            classes = model.classes_

            # Map probabilities to signals
            prob_long  = proba[list(classes).index(1)]  if 1  in classes else 0
            prob_short = proba[list(classes).index(-1)] if -1 in classes else 0

            if prob_long >= pred_thr:
                signal.iloc[i] = 1
            elif prob_short >= pred_thr:
                signal.iloc[i] = -1

        return signal.reindex(data.index, fill_value=0)

    def save(self, path: Path) -> None:
        """Persist trained models and scaler."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": self._models, "scaler": self._scaler}, path / "rf_model.pkl")
        logger.info("Model saved to %s", path)

    def load(self, path: Path) -> None:
        """Load previously saved models."""
        pkg = joblib.load(Path(path) / "rf_model.pkl")
        self._models  = pkg["models"]
        self._scaler  = pkg["scaler"]
        logger.info("Model loaded from %s", path)
