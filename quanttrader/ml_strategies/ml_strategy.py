"""
ml_strategies/ml_strategy.py
==============================
ML-based strategy base classes and concrete implementations.

Base classes:
  - MLClassificationStrategy : predict direction (up/down/flat).
  - MLRegressionStrategy     : predict future return.

Implementations:
  - RandomForestStrategy     : RF classifier predicting next-day direction.
"""
from __future__ import annotations

import abc
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from backtest.strategy import Strategy, StrategyMetadata, ParamSpec
from ml_strategies.features import build_features
from utils.logger import get_logger

log = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Base ML Strategy
# ─────────────────────────────────────────────────────────────────────────────

class MLStrategyBase(Strategy, abc.ABC):
    """
    Abstract base for ML strategies.

    Handles the training/prediction cycle, walk-forward validation,
    and model persistence. Subclasses implement `build_model` and
    optionally `build_features`.
    """

    metadata = StrategyMetadata(name="ML Base", tags=["ml"])

    # ML-specific params
    train_window = ParamSpec(default=252, min_val=60, max_val=1000, dtype=int,
                             description="Training window in bars")
    retrain_freq = ParamSpec(default=63, min_val=1, max_val=252, dtype=int,
                             description="Retrain frequency in bars")
    prediction_horizon = ParamSpec(default=1, min_val=1, max_val=20, dtype=int,
                                   description="Bars ahead to predict")
    min_confidence = ParamSpec(default=0.6, min_val=0.5, max_val=1.0, dtype=float,
                               description="Min prediction probability to act")
    n_features = ParamSpec(default=20, min_val=5, max_val=50, dtype=int)

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._model = None
        self._feature_cols: List[str] = []
        self._is_trained = False

    # ── Abstract interface ────────────────────────────────────────────────────

    @abc.abstractmethod
    def build_model(self, params: Dict[str, Any]) -> Any:
        """Return a scikit-learn compatible estimator."""
        ...

    def build_features_for_data(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Build feature matrix. Override for custom features."""
        return build_features(df)

    # ── Core generate_signals ─────────────────────────────────────────────────

    def generate_signals(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        params: Dict[str, Any],
    ) -> pd.Series:
        df = data if isinstance(data, pd.DataFrame) else list(data.values())[0]

        train_window = int(params["train_window"])
        retrain_freq = int(params["retrain_freq"])
        horizon = int(params["prediction_horizon"])

        features = self.build_features_for_data(df, params)
        target = _build_direction_target(df["close"], horizon)

        signal = pd.Series(0.0, index=df.index)
        last_train_idx = 0

        for i in range(train_window, len(df)):
            # Retrain check
            if i - last_train_idx >= retrain_freq or self._model is None:
                X_train, y_train = self._prepare_training_data(
                    features, target, start=max(0, i - train_window), end=i
                )
                if len(X_train) < 30 or len(y_train.unique()) < 2:
                    continue
                self._model = self.build_model(params)
                try:
                    self._model.fit(X_train, y_train)
                    self._is_trained = True
                    last_train_idx = i
                    log.debug(f"ML model retrained at bar {i}")
                except Exception as e:
                    log.warning(f"Model training failed at bar {i}: {e}")
                    continue

            if not self._is_trained:
                continue

            # Predict
            X_pred = features.iloc[i:i+1][self._feature_cols].fillna(0)
            if X_pred.empty or X_pred.isna().all().all():
                continue

            try:
                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba(X_pred)[0]
                    classes = self._model.classes_
                    up_prob = proba[list(classes).index(1)] if 1 in classes else 0.5
                    down_prob = proba[list(classes).index(-1)] if -1 in classes else 0.5

                    min_conf = float(params["min_confidence"])
                    if up_prob >= min_conf:
                        signal.iloc[i] = 1.0
                    elif down_prob >= min_conf:
                        signal.iloc[i] = -1.0
                else:
                    pred = self._model.predict(X_pred)[0]
                    signal.iloc[i] = float(pred)
            except Exception as e:
                log.debug(f"Prediction failed at bar {i}: {e}")

        return signal

    def _prepare_training_data(
        self, features: pd.DataFrame, target: pd.Series, start: int, end: int
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Slice and clean training data."""
        X = features.iloc[start:end].copy()
        y = target.iloc[start:end].copy()

        # Align
        valid = X.index.intersection(y.index)
        X = X.loc[valid]
        y = y.loc[valid]

        # Drop rows with NaN
        mask = X.notna().all(axis=1) & y.notna()
        X = X[mask]
        y = y[mask]

        # Store feature columns for prediction
        self._feature_cols = list(X.columns)
        return X, y

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_model(self, path: Path) -> None:
        """Save trained model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self._model, "feature_cols": self._feature_cols}, f)
        log.info(f"Model saved to {path}")

    def load_model(self, path: Path) -> None:
        """Load a previously trained model."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_cols = data["feature_cols"]
        self._is_trained = True
        log.info(f"Model loaded from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest Strategy
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestStrategy(MLStrategyBase):
    """
    Random Forest Classifier predicting next-day price direction.

    Uses 20 features by default (technical indicators + returns).
    Walk-forward retraining to avoid look-ahead bias.

    Parameters
    ----------
    n_estimators   : Number of trees.
    max_depth      : Max tree depth.
    min_samples    : Min samples per leaf.
    + all MLStrategyBase params
    """

    metadata = StrategyMetadata(
        name="Random Forest",
        version="1.0.0",
        author="QuantTrader",
        description="RF classifier for next-day direction prediction.",
        tags=["ml", "classification", "random_forest"],
    )

    n_estimators = ParamSpec(default=100, min_val=10, max_val=500, dtype=int)
    max_depth = ParamSpec(default=5, min_val=2, max_val=20, dtype=int)
    min_samples = ParamSpec(default=20, min_val=5, max_val=100, dtype=int)

    def build_model(self, params: Dict[str, Any]) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_leaf=int(params["min_samples"]),
            n_jobs=1,
            random_state=42,
        )

    def get_feature_importances(self) -> Optional[pd.Series]:
        """Return feature importances if model is trained."""
        if self._model is None or not self._feature_cols:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self._feature_cols,
        ).sort_values(ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost Strategy
# ─────────────────────────────────────────────────────────────────────────────

class XGBoostStrategy(MLStrategyBase):
    """XGBoost classifier for direction prediction."""

    metadata = StrategyMetadata(
        name="XGBoost Direction",
        version="1.0.0",
        author="QuantTrader",
        description="XGBoost classifier predicting price direction.",
        tags=["ml", "classification", "xgboost"],
    )

    n_estimators = ParamSpec(default=100, min_val=10, max_val=500, dtype=int)
    max_depth = ParamSpec(default=4, min_val=2, max_val=10, dtype=int)
    learning_rate = ParamSpec(default=0.1, min_val=0.01, max_val=0.5, dtype=float)

    def build_model(self, params: Dict[str, Any]) -> Any:
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                learning_rate=float(params["learning_rate"]),
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
        except ImportError:
            log.warning("xgboost not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=100, random_state=42)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Build target
# ─────────────────────────────────────────────────────────────────────────────

def _build_direction_target(close: pd.Series, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """
    Build a direction classification target.

    Returns +1 (up), -1 (down), 0 (flat) based on future return.
    """
    future_ret = close.shift(-horizon).pct_change(horizon)
    target = pd.Series(0, index=close.index)
    target[future_ret > threshold] = 1
    target[future_ret < -threshold] = -1
    return target.shift(horizon)  # Align with current bar
