"""
CLIMB Step 2.2: Iterative bootstrapping for mixture weight search.

Refactored: Predictor is now injected via config.predictor.method,
using the predictor registry from core/predictor.py.
"""

import time
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any, Tuple

from climbmix.core.types import (
    MixtureConfig,
    MixtureWeights,
    ProxyResult,
    IterationResult,
    CLIMBConfig,
)
from climbmix.core.dirichlet_sampler import DirichletSampler
from climbmix.core.predictor import get_predictor


class IterativeBootstrapper:
    def __init__(
        self,
        config: CLIMBConfig,
        cluster_token_counts: npt.NDArray[np.int64],
        cluster_labels: npt.NDArray[np.int64],
    ):
        self.config = config
        self.cluster_labels = cluster_labels
        self.num_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

        if len(cluster_token_counts) != self.num_clusters:
            per_doc_token_counts = cluster_token_counts
            per_cluster_tokens = np.zeros(self.num_clusters, dtype=np.int64)
            for c in np.unique(cluster_labels[cluster_labels >= 0]):
                mask = cluster_labels == c
                per_cluster_tokens[int(c)] = int(per_doc_token_counts[mask].sum())
            self.cluster_token_counts = per_cluster_tokens
        else:
            self.cluster_token_counts = cluster_token_counts

        self._sampler = DirichletSampler(
            self.num_clusters, self.cluster_token_counts, config,
        )

        self._accumulated_configs: List[MixtureConfig] = []
        self._accumulated_losses: List[float] = []
        self._iteration_results: List[IterationResult] = []
        self._predictor: Optional[Any] = None
        self._candidate_pool: List[MixtureConfig] = []

    def run_iteration(
        self,
        iteration: int,
        n_configs: int,
        proxy_runner: Optional[Any] = None,
    ) -> IterationResult:
        print(f"\n{'=' * 70}")
        print(f"  CLIMB Iteration {iteration}")
        print(f"  Configs: {n_configs}")
        print(f"{'=' * 70}")

        t0 = time.time()

        if iteration == 1 or self._predictor is None:
            print(f"[Iter {iteration}] Sampling {n_configs} configs from Dirichlet")
            new_configs = self._sampler.sample_batch(n_configs)
        else:
            print(f"[Iter {iteration}] Predictor-guided sampling")
            pool_size = max(n_configs * 20, 500)
            pool_configs = self._sampler.sample_batch(pool_size)

            existing_flats = set()
            for c in self._accumulated_configs:
                existing_flats.add(tuple(np.round(c.flatten(), 4)))

            novel_configs = [
                c for c in pool_configs
                if tuple(np.round(c.flatten(), 4)) not in existing_flats
            ]

            if len(novel_configs) == 0:
                novel_configs = pool_configs

            ranked_indices = self._predictor.predict_and_rank(novel_configs)
            top_n = min(self.config.search.sample_from_top_m * 3, len(novel_configs))
            top_configs = [novel_configs[i] for i in ranked_indices[:top_n]]

            new_configs = self._sampler.sample_from_top_n(
                top_configs, m=n_configs,
            )

        for i, c in enumerate(new_configs):
            c.config_id = len(self._accumulated_configs) + i

        losses: List[float] = []
        trained_configs: List[MixtureConfig] = []

        if proxy_runner is not None:
            print(f"[Iter {iteration}] Training {len(new_configs)} proxy models")
            results = proxy_runner.run_batch(new_configs, self.cluster_labels, self.cluster_token_counts)
            for r in results:
                losses.append(r.validation_loss)
                trained_configs.append(r.mixture_config)
                self._accumulated_configs.append(r.mixture_config)
                self._accumulated_losses.append(r.validation_loss)
        else:
            print(f"[Iter {iteration}] No proxy runner, using random losses (for testing)")
            rng = np.random.default_rng(iteration * 1000)
            for c in new_configs:
                loss = rng.uniform(2.5, 4.5)
                losses.append(loss)
                trained_configs.append(c)
                self._accumulated_configs.append(c)
                self._accumulated_losses.append(loss)

        losses_arr = np.array(losses, dtype=np.float64)

        # Subroutine 2: Predictor fitting
        print(f"[Iter {iteration}] Training predictor on {len(self._accumulated_configs)} accumulated configs")

        all_losses = np.array(self._accumulated_losses, dtype=np.float64)
        valid_mask = np.isfinite(all_losses)

        train_configs = [c for c, v in zip(self._accumulated_configs, valid_mask) if v]
        train_losses = all_losses[valid_mask]

        n_total = len(train_configs)
        if n_total < 10:
            val_configs_split = None
            val_losses_split = None
        else:
            n_val = max(5, int(n_total * 0.2))
            rng_split = np.random.default_rng(42)
            indices = rng_split.permutation(n_total)
            train_idx = indices[:n_total - n_val]
            val_idx = indices[n_total - n_val:]

            train_configs_split = [train_configs[i] for i in train_idx]
            train_losses_split = train_losses[train_idx]
            val_configs_split = [train_configs[i] for i in val_idx]
            val_losses_split = train_losses[val_idx]
            train_configs = train_configs_split
            train_losses = train_losses_split

        predictor = get_predictor(
            self.config.predictor.method,
            self.num_clusters,
            self.config.predictor,
        )
        predictor.fit(
            train_configs, train_losses,
            val_configs=val_configs_split,
            val_losses=val_losses_split,
        )

        self._predictor = predictor

        best_idx = int(np.argmin(losses_arr))
        best_config = trained_configs[best_idx]
        best_loss = float(losses_arr[best_idx])

        iter_result = IterationResult(
            iteration=iteration,
            n_configs=n_configs,
            n_trained=len(trained_configs),
            predictor=predictor,
            predictor_r2=None,
            best_config=best_config,
            best_loss=best_loss,
            all_configs=trained_configs,
            all_losses=losses_arr,
        )

        if n_total >= 10:
            iter_result.predictor_r2 = float(predictor.score(
                val_configs_split, val_losses_split,
            ))

        self._iteration_results.append(iter_result)

        elapsed = time.time() - t0
        print(f"\n[Iter {iteration}] Complete in {elapsed:.1f}s")
        print(f"  Best loss: {best_loss:.4f}")
        if iter_result.predictor_r2 is not None:
            print(f"  Predictor R\u00b2: {iter_result.predictor_r2:.4f}")

        return iter_result

    def search_optimal(self, proxy_runner: Optional[Any] = None) -> Tuple[MixtureConfig, List[IterationResult]]:
        print("\n" + "=" * 70)
        print("  CLIMB Iterative Bootstrapping Search")
        print(f"  Iterations: {self.config.search.num_iterations}")
        print(f"  Configs per iteration: {self.config.search.configs_per_iter}")
        print("=" * 70)

        t0 = time.time()

        for k in range(self.config.search.num_iterations):
            n_configs = self.config.search.configs_per_iter[k]
            self.run_iteration(k + 1, n_configs, proxy_runner)

        if self._predictor is None:
            best_idx = int(np.argmin(self._accumulated_losses))
            optimal = self._accumulated_configs[best_idx]
        else:
            final_pool = self._sampler.sample_batch(10000)
            predictions = self._predictor.predict(final_pool)
            best_idx = int(np.argmin(predictions))
            optimal = final_pool[best_idx]

        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"  CLIMB Search Complete ({elapsed:.1f}s)")
        print(f"  Optimal mixture weights:")
        for i, w in enumerate(optimal.mixture_weights.weights):
            print(f"    C{i}: {w:.4f}")
        print(f"{'=' * 70}")

        return optimal, self._iteration_results

    @property
    def predictor(self):
        return self._predictor

    @property
    def iteration_results(self) -> List[IterationResult]:
        return self._iteration_results
