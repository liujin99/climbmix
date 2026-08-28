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
        # Held-out metrics from the early-stopping split (paper D.10 reports
        # held-out Spearman; set by fit() when a validation set is given).
        self.val_r2_: Optional[float] = None
        self.val_spearman_: Optional[float] = None

    @staticmethod
    def _rankdata(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Average ranks (tied values share the mean of their positions),
        1-based — same convention as scipy.stats.rankdata."""
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        ranks = np.empty(len(x), dtype=np.float64)
        sx = x[order]
        i = 0
        n = len(x)
        while i < n:
            j = i
            while j + 1 < n and sx[j + 1] == sx[i]:
                j += 1
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
            i = j + 1
        return ranks

    @classmethod
    def _spearman(cls, a: npt.NDArray[np.float64], b: npt.NDArray[np.float64]) -> float:
        """Spearman rank correlation (numpy only, no scipy dependency).

        Pearson correlation on average ranks — identical to
        scipy.stats.spearmanr including ties. NaN when <2 points or when
        either side is constant (zero rank variance).
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if len(a) < 2:
            return float("nan")
        ra = cls._rankdata(a)
        rb = cls._rankdata(b)
        ra -= ra.mean()
        rb -= rb.mean()
        denom = np.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
        if denom == 0.0:
            return float("nan")
        return float((ra * rb).sum() / denom)

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
            best_iter = getattr(self._model, "best_iteration_", None)
            if best_iter is not None and best_iter < self.config.n_estimators:
                print(f"[Predictor] Early stopping: best_iteration="
                      f"{best_iter}/{self.config.n_estimators} trees")
            else:
                print(f"[Predictor] No early stopping: used all "
                      f"{self.config.n_estimators} trees")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._model.fit(X, y)

        self._is_fitted = True
        train_r2 = float(self._model.score(X, y))
        print(f"[Predictor] Trained on {len(X)} configs, train R\u00b2={train_r2:.4f}")
        if val_configs is not None and val_losses is not None:
            # Held-out metrics on the early-stopping split. The train R²
            # above is optimistic by construction (≤35 points, 500 trees);
            # these are the honest numbers (paper D.10 metric = Spearman).
            self.val_r2_ = float(self._model.score(X_val, y_val))
            val_pred = self._model.predict(X_val)
            self.val_spearman_ = self._spearman(val_pred, y_val)
            print(f"[Predictor] val R\u00b2={self.val_r2_:.4f}, "
                  f"val Spearman={self.val_spearman_:.4f} (n={len(y_val)})")
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
