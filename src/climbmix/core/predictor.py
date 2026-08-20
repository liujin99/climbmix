"""
Predictive model for mixture weight → performance mapping.

Separated from iterative_bootstrapper.py for modularity.
Supports LightGBM with paper-specified hyperparameters.
"""

import time
import warnings
import numpy as np
import numpy.typing as npt
from typing import List, Optional

from climbmix.core.types import MixtureConfig, PredictorConfig


class LightGBMPredictor:
    def __init__(self, num_clusters: int, config: Optional[PredictorConfig] = None):
        self.num_clusters = num_clusters
        self.config = config or PredictorConfig()
        self._model = None
        self._is_fitted = False

    def _compute_colsample(self) -> float:
        """
        Scale colsample_bytree with num_clusters (search parameter count).

        Paper's K_enhanced=21 features. With more clusters (more features),
        subsample features more aggressively to prevent overfitting.
        With fewer clusters, use more features per tree.

        Formula: min(1.0, max(0.3, 20 / num_clusters))
        - 21 clusters → ~0.95 (nearly all features, as in paper)
        - 50 clusters → 0.4
        - 10 clusters → 1.0
        """
        return min(1.0, max(0.3, 20.0 / self.num_clusters))

    def fit(
        self,
        configs: List[MixtureConfig],
        losses: npt.NDArray[np.float64],
        val_configs: Optional[List[MixtureConfig]] = None,
        val_losses: Optional[npt.NDArray[np.float64]] = None,
    ):
        import lightgbm as lgb

        X = np.array([c.flatten() for c in configs])
        y = np.array(losses)

        valid_mask = np.isfinite(y)
        if not np.all(valid_mask):
            n_invalid = int((~valid_mask).sum())
            print(f"[Predictor] Filtering {n_invalid} non-finite losses")
            X = X[valid_mask]
            y = y[valid_mask]

        n_samples, n_features = X.shape
        adj = self.config.get_adjusted_params(n_samples, n_features)
        max_depth = adj["max_depth"]
        min_samples_leaf = adj["min_samples_leaf"]

        if self.config.auto_adjust:
            print(f"[Predictor] Auto-adjusted: max_depth={max_depth}, "
                  f"min_samples_leaf={min_samples_leaf} "
                  f"(N={n_samples}, k={n_features})")

        lgb_params = {
            "n_estimators": self.config.n_estimators,
            "learning_rate": self.config.learning_rate,
            "max_depth": max_depth,
            "num_leaves": min(15, 2 ** max_depth - 1),
            "min_child_samples": min_samples_leaf,
            "reg_alpha": self.config.l1_reg,
            "reg_lambda": self.config.l2_reg,
            "subsample": 1.0,
            "colsample_bytree": self._compute_colsample(),
            "random_state": 42,
            "verbose": -1,
        }

        self._model = lgb.LGBMRegressor(**lgb_params)

        if val_configs is not None and val_losses is not None:
            X_val = np.array([c.flatten() for c in val_configs])
            y_val = np.array(val_losses)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(
                    X, y,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(
                        stopping_rounds=self.config.early_stopping_rounds,
                        verbose=False,
                    )],
                )
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(X, y)

        self._is_fitted = True
        train_r2 = float(self._model.score(X, y))
        print(f"[Predictor] Trained on {len(X)} configs, R\u00b2={train_r2:.4f}")
        return self

    def predict(self, configs: List[MixtureConfig]) -> npt.NDArray[np.float64]:
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted")
        X = np.array([c.flatten() for c in configs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self._model.predict(X)

    def predict_and_rank(self, configs: List[MixtureConfig]) -> List[int]:
        predicted = self.predict(configs)
        return np.argsort(predicted).tolist()

    def score(self, configs: List[MixtureConfig], losses: npt.NDArray[np.float64]) -> float:
        if not self._is_fitted:
            raise RuntimeError("Predictor not fitted")
        X = np.array([c.flatten() for c in configs])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(self._model.score(X, losses))


PREDICTOR_REGISTRY = {
    "lightgbm": LightGBMPredictor,
}


def get_predictor(method: str, num_clusters: int, config: Optional[PredictorConfig] = None):
    if method not in PREDICTOR_REGISTRY:
        raise ValueError(
            f"Unknown predictor method '{method}'. "
            f"Available: {list(PREDICTOR_REGISTRY.keys())}"
        )
    return PREDICTOR_REGISTRY[method](num_clusters, config)
