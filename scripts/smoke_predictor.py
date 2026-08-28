#!/usr/bin/env python3
"""Standalone predictor smoke test (no NPU, ~seconds).

Exercises the exact production predictor path that speedrun CANNOT cover:
accumulated N >= 10 -> held-out val split -> LightGBM fit with
lgb.early_stopping callback -> val R2 / val Spearman. That path depends on
the server's lightgbm version (>= 4.0 callbacks API), so run this once on
any new machine before a large-scale search:

    python3 scripts/smoke_predictor.py

Exit 0 = predictor path healthy on this machine.
"""
import sys
import warnings

sys.path.insert(0, "src")

import numpy as np

from climbmix.core.types import MixtureConfig, MixtureWeights, PredictorConfig
from climbmix.core.predictor import LightGBMPredictor


def main() -> int:
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(0)
    K = 10
    N = 50

    # Sparse latent quality (3 clusters carry the signal) — learnable by
    # shallow trees under the production predictor config (auto-adjusted
    # depth/leaf + L1/L2 + early stopping). Loss-like target: lower=better.
    quality = np.zeros(K)
    quality[[1, 4, 7]] = [1.0, -1.0, 0.8]

    def make_configs(n, seed):
        r = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            w = r.dirichlet(np.full(K, 2.0))
            out.append(MixtureConfig(mixture_weights=MixtureWeights(weights=w)))
        return out

    def targets_of(configs, noise_rng):
        return np.array([
            -2.0 * float(np.dot(c.mixture_weights.weights, quality))
            + noise_rng.normal(0, 0.03)
            for c in configs
        ])

    train_cfg = make_configs(N, 1)
    train_y = targets_of(train_cfg, np.random.default_rng(2))
    val_cfg = make_configs(15, 3)
    val_y = targets_of(val_cfg, np.random.default_rng(4))

    pred = LightGBMPredictor(K, PredictorConfig())
    pred.fit(train_cfg, train_y, val_configs=val_cfg, val_losses=val_y)

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            ok = False

    check("val_r2_ set", pred.val_r2_ is not None, f"= {pred.val_r2_}")
    check("val_spearman_ set", pred.val_spearman_ is not None,
          f"= {pred.val_spearman_}")
    check("val_spearman sane",
          pred.val_spearman_ is not None and -1.0 <= pred.val_spearman_ <= 1.0)
    check("learned signal (rho > 0.5)",
          pred.val_spearman_ is not None and pred.val_spearman_ > 0.5,
          f"= {pred.val_spearman_}")
    preds = pred.predict(val_cfg)
    check("predict() works", len(preds) == len(val_cfg))
    best_iter = getattr(pred._model, "best_iteration_", None)
    print(f"  info best_iteration_ = {best_iter} "
          f"(None = no early stopping fired, also fine)")

    print("SMOKE " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
