"""
CLIMB Step 2.2: Iterative bootstrapping for mixture weight search.

Paper-aligned changes:
  1. Uses ProxyResult.score for ranking (accuracy=maximize, loss=minimize)
  2. Final search samples from full design space with refinement
  3. Predictor-guided sampling uses Dirichlet exploration around top-N
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
        self.metric_direction = config.metric_direction

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
        self._accumulated_scores: List[float] = []
        self._iteration_results: List[IterationResult] = []
        self._predictor: Optional[Any] = None

    def _is_better(self, score_a: float, score_b: float) -> bool:
        if self.metric_direction == "maximize":
            return score_a > score_b
        return score_a < score_b

    def _best_index(self, scores: npt.NDArray[np.float64]) -> int:
        if self.metric_direction == "maximize":
            return int(np.argmax(scores))
        return int(np.argmin(scores))

    def run_iteration(
        self,
        iteration: int,
        n_configs: int,
        proxy_runner: Optional[Any] = None,
    ) -> IterationResult:
        print(f"\n{'=' * 70}")
        print(f"  CLIMB Iteration {iteration}")
        print(f"  Configs: {n_configs}, metric: {self.metric_direction}")
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
            if self.metric_direction == "maximize":
                ranked_indices = ranked_indices[::-1]

            top_n = min(self.config.search.sample_from_top_m * 3, len(novel_configs))
            top_configs = [novel_configs[i] for i in ranked_indices[:top_n]]

            new_configs = self._sampler.sample_from_top_n(
                top_configs, m=n_configs,
            )

        for i, c in enumerate(new_configs):
            c.config_id = len(self._accumulated_configs) + i

        scores: List[float] = []
        trained_configs: List[MixtureConfig] = []

        if proxy_runner is not None:
            print(f"[Iter {iteration}] Training {len(new_configs)} proxy models")
            results = proxy_runner.run_batch(new_configs, self.cluster_labels, self.cluster_token_counts)
            for r in results:
                s = r.score
                scores.append(s)
                trained_configs.append(r.mixture_config)
                self._accumulated_configs.append(r.mixture_config)
                self._accumulated_scores.append(s)
        else:
            print(f"[Iter {iteration}] No proxy runner, using random scores (for testing)")
            rng = np.random.default_rng(iteration * 1000)
            for c in new_configs:
                if self.metric_direction == "maximize":
                    s = rng.uniform(0.2, 0.6)
                else:
                    s = rng.uniform(2.5, 4.5)
                scores.append(s)
                trained_configs.append(c)
                self._accumulated_configs.append(c)
                self._accumulated_scores.append(s)

        scores_arr = np.array(scores, dtype=np.float64)

        print(f"[Iter {iteration}] Training predictor on {len(self._accumulated_configs)} accumulated configs")

        all_scores = np.array(self._accumulated_scores, dtype=np.float64)

        if self.metric_direction == "minimize":
            predictor_targets = all_scores
        else:
            predictor_targets = -all_scores

        valid_mask = np.isfinite(predictor_targets)

        train_configs = [c for c, v in zip(self._accumulated_configs, valid_mask) if v]
        train_targets = predictor_targets[valid_mask]

        n_total = len(train_configs)
        if n_total < 10:
            val_configs_split = None
            val_targets_split = None
        else:
            n_val = max(5, int(n_total * 0.2))
            rng_split = np.random.default_rng(42)
            indices = rng_split.permutation(n_total)
            train_idx = indices[:n_total - n_val]
            val_idx = indices[n_total - n_val:]

            train_configs_split = [train_configs[i] for i in train_idx]
            train_targets_split = train_targets[train_idx]
            val_configs_split = [train_configs[i] for i in val_idx]
            val_targets_split = train_targets[val_idx]
            train_configs = train_configs_split
            train_targets = train_targets_split

        predictor = get_predictor(
            self.config.predictor.method,
            self.num_clusters,
            self.config.predictor,
        )
        predictor.fit(
            train_configs, train_targets,
            val_configs=val_configs_split,
            val_losses=val_targets_split,
        )

        self._predictor = predictor

        best_idx = self._best_index(scores_arr)
        best_config = trained_configs[best_idx]
        best_score = float(scores_arr[best_idx])
        best_loss = None

        iter_result = IterationResult(
            iteration=iteration,
            n_configs=n_configs,
            n_trained=len(trained_configs),
            predictor=predictor,
            predictor_r2=None,
            best_config=best_config,
            best_score=best_score,
            all_configs=trained_configs,
            all_scores=scores_arr,
        )

        if n_total >= 10 and val_configs_split is not None:
            iter_result.predictor_r2 = float(predictor.score(
                val_configs_split, val_targets_split,
            ))

        self._iteration_results.append(iter_result)

        elapsed = time.time() - t0
        print(f"\n[Iter {iteration}] Complete in {elapsed:.1f}s")
        print(f"  Best score: {best_score:.4f}")
        if iter_result.predictor_r2 is not None:
            print(f"  Predictor R\u00b2: {iter_result.predictor_r2:.4f}")

        return iter_result

    def search_optimal(self, proxy_runner: Optional[Any] = None) -> Tuple[MixtureConfig, List[IterationResult]]:
        """
        Paper (Section 3.3): "selects the best configuration predicted by
        the final predictor from the full design space A."

        Implementation: exhaustive search over the design space by
        sampling a large pool from multiple Dirichlet concentrations,
        then refining around the top predictions.
        """
        print("\n" + "=" * 70)
        print("  CLIMB Iterative Bootstrapping Search")
        print(f"  Iterations: {self.config.search.num_iterations}")
        print(f"  Configs per iteration: {self.config.search.configs_per_iter}")
        print(f"  Metric direction: {self.metric_direction}")
        print("=" * 70)

        t0 = time.time()

        for k in range(self.config.search.num_iterations):
            n_configs = self.config.search.configs_per_iter[k]
            self.run_iteration(k + 1, n_configs, proxy_runner)

        if self._predictor is None:
            best_idx = self._best_index(np.array(self._accumulated_scores))
            optimal = self._accumulated_configs[best_idx]
        else:
            optimal = self._search_full_design_space()

        elapsed = time.time() - t0
        print(f"\n{'=' * 70}")
        print(f"  CLIMB Search Complete ({elapsed:.1f}s)")
        print(f"  Optimal mixture weights:")
        for i, w in enumerate(optimal.mixture_weights.weights):
            print(f"    C{i}: {w:.4f}")
        print(f"{'=' * 70}")

        return optimal, self._iteration_results

    def _search_full_design_space(self) -> MixtureConfig:
        """
        Search the full design space A for the optimal configuration.

        Paper: selects best from the full design space using the final predictor.
        Strategy: sample large pools at multiple concentration levels (wide
        exploration + focused search), evaluate with predictor, then refine
        around the top predictions.
        """
        concentrations = [1.0, 5.0, 10.0, 50.0]
        pool_per_conc = 25000
        total_pool_size = len(concentrations) * pool_per_conc

        print(f"[Search] Sampling {total_pool_size} candidates from "
              f"{len(concentrations)} concentration levels")

        all_candidates = []
        for conc in concentrations:
            sampler = DirichletSampler(
                self.num_clusters,
                self.cluster_token_counts,
                self.config,
                seed=42 + int(conc),
            )
            sampler._concentration = np.full(
                self.num_clusters, conc, dtype=np.float64
            )
            batch = sampler.sample_batch(pool_per_conc)
            all_candidates.extend(batch)

        predictions = self._predictor.predict(all_candidates)
        if self.metric_direction == "maximize":
            predictions = -predictions

        best_pool_idx = int(np.argmin(predictions))
        best_from_pool = all_candidates[best_pool_idx]

        print(f"[Search] Refining around top prediction")

        refinement_sampler = DirichletSampler(
            self.num_clusters,
            self.cluster_token_counts,
            self.config,
            seed=999,
        )
        refinement_candidates = refinement_sampler.sample_from_top_n(
            [best_from_pool], m=5000,
            exploration_concentration=50.0,
        )

        refinement_predictions = self._predictor.predict(refinement_candidates)
        if self.metric_direction == "maximize":
            refinement_predictions = -refinement_predictions

        best_refine_idx = int(np.argmin(refinement_predictions))
        optimal = refinement_candidates[best_refine_idx]

        print(f"[Search] Full design space search: "
              f"{total_pool_size + 5000} candidates evaluated")

        return optimal

    @property
    def predictor(self):
        return self._predictor

    @property
    def iteration_results(self) -> List[IterationResult]:
        return self._iteration_results
