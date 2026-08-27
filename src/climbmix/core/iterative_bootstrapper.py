"""
CLIMB Step 2.2: Iterative bootstrapping for mixture weight search.

Paper-aligned changes:
  1. Uses ProxyResult.score for ranking (accuracy=maximize, loss=minimize)
  2. Final search samples from full design space with refinement
  3. Predictor-guided sampling uses Dirichlet exploration around top-N
"""

import time
import json
import os
import inspect
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Any, Tuple

from climbmix.core.types import (
    MixtureConfig,
    MixtureWeights,
    ProxyResult,
    IterationResult,
    CLIMBConfig,
    BENCHMARK_SIZES,
)
from climbmix.core.dirichlet_sampler import DirichletSampler
from climbmix.core.predictor import get_predictor
from climbmix.utils.io_utils import atomic_write_json, load_json_state


class IterativeBootstrapper:
    def __init__(
        self,
        config: CLIMBConfig,
        cluster_token_counts: npt.NDArray[np.int64],
        cluster_labels: npt.NDArray[np.int64],
        state_path: Optional[str] = None,
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
        self._accumulated_per_benchmark: List[Tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]] = []
        self._iteration_results: List[IterationResult] = []
        self._predictor: Optional[Any] = None
        self.w_floor = config.search.w_floor
        self.state_path = state_path
        self._last_completed_iter = 0
        # Configs of the iteration currently being trained (written to state
        # BEFORE the experiments run so a mid-iteration crash restores the
        # exact same configs instead of re-sampling different ones).
        self._pending: Optional[Dict[str, Any]] = None

    def _is_better(self, score_a: float, score_b: float) -> bool:
        if self.metric_direction == "maximize":
            return score_a > score_b
        return score_a < score_b

    def _best_index(self, scores: npt.NDArray[np.float64]) -> int:
        """Index of the best FINITE score; 0 if none (caller guards).

        NaN-safe: np.argmax/argmin on NaN-contaminated arrays return a NaN
        slot (NaN 'wins' comparisons), which would present a failed
        experiment as the iteration's best config.
        """
        scores = np.asarray(scores, dtype=np.float64)
        if scores.size == 0:
            return 0
        finite = np.isfinite(scores)
        if not finite.any():
            return 0
        if self.metric_direction == "maximize":
            return int(np.argmax(np.where(finite, scores, -np.inf)))
        return int(np.argmin(np.where(finite, scores, np.inf)))

    def _save_state(self):
        if not self.state_path:
            return
        state = {
            "last_completed_iter": self._last_completed_iter,
            "n_clusters": self.num_clusters,
            "accumulated_scores": self._accumulated_scores,
            "accumulated_configs": [
                {"weights": c.mixture_weights.weights.tolist(), "config_id": c.config_id}
                for c in self._accumulated_configs
            ],
            "accumulated_per_benchmark": [
                {"acc": d[0], "nll": d[1]}
                for d in self._accumulated_per_benchmark
            ],
            "pending": self._pending,
        }
        atomic_write_json(self.state_path, state)
        pend = ""
        if self._pending:
            pend = (f", pending iter {self._pending['iteration']} "
                    f"({len(self._pending['configs'])} configs)")
        print(f"[Search] State saved → {self.state_path} (iter {self._last_completed_iter}, "
              f"{len(self._accumulated_configs)} configs{pend})")

    def _load_state(self) -> int:
        if not self.state_path or not os.path.exists(self.state_path):
            return 0
        state = load_json_state(self.state_path)
        if state is None:
            print(f"[Search] State file unreadable (crashed mid-write?), starting fresh: "
                  f"{self.state_path}")
            return 0
        try:
            saved_configs = [
                MixtureConfig(
                    mixture_weights=MixtureWeights(weights=np.array(c["weights"], dtype=np.float64)),
                    config_id=c["config_id"],
                )
                for c in state["accumulated_configs"]
            ]
        except (KeyError, TypeError, ValueError):
            print(f"[Search] State file malformed, starting fresh: {self.state_path}")
            return 0

        # Defense: if the cluster count changed since this state was written
        # (recomputed cluster cache, edited data, or a run_climb.py invocation
        # bypassing the shell fingerprint), old config vectors have the wrong
        # dimension for the current search space — mixing them with new ones
        # would silently corrupt the predictor. Discard and start fresh.
        state_n_clusters = state.get("n_clusters")
        weights_k = len(saved_configs[0].mixture_weights.weights) if saved_configs else None
        if weights_k is None:
            pending_configs = (state.get("pending") or {}).get("configs") or []
            if pending_configs:
                weights_k = len(pending_configs[0])
        if (state_n_clusters is not None and weights_k is not None
                and state_n_clusters != weights_k):
            print(f"[Search] State file inconsistent (n_clusters={state_n_clusters} "
                  f"but config weights have K={weights_k}), starting fresh: {self.state_path}")
            return 0
        stale_k = state_n_clusters if state_n_clusters is not None else weights_k
        if stale_k is not None and stale_k != self.num_clusters:
            print(f"[Search] WARNING: n_clusters changed {stale_k} -> {self.num_clusters} "
                  f"since this state was written — discarding stale search state "
                  f"and starting fresh: {self.state_path}")
            return 0

        try:
            self._accumulated_scores = state["accumulated_scores"]
            self._accumulated_configs = saved_configs
            self._accumulated_per_benchmark = [
                (d["acc"], d["nll"]) for d in state["accumulated_per_benchmark"]
            ]
            self._last_completed_iter = state["last_completed_iter"]
            self._pending = state.get("pending")
        except (KeyError, TypeError, ValueError):
            print(f"[Search] State file malformed, starting fresh: {self.state_path}")
            return 0
        print(f"[Search] State loaded ← {self.state_path} (iter {self._last_completed_iter}, "
              f"{len(self._accumulated_configs)} configs)")
        return self._last_completed_iter

    def _compute_scores(self) -> npt.NDArray[np.float64]:
        """Compute SNR-weighted scores for all accumulated configs.

        Per-benchmark: z-score accuracy and NLL, combine with SNR weight.
        sigma2_noise = 0.25 / K (worst-case binomial variance).
        f = 1 - noise/between (can be negative when noise > between).
        w = max(w_floor, min(1, max(0, (1+f)/2))) — clamped to [w_floor, 1].

        Unmeasured entries (failed experiments -> empty per-task dicts,
        unparseable eval CSVs -> None) contribute NaN, never 0.0: a
        fabricated 0.0 poisons every z-score, the SNR weight, and the
        predictor, because 0.0 looks like a real measurement (observed in
        speedrun 2026-08-26 17:35). A config's score is the mean over the
        benchmarks it was actually measured on; NaN if it was measured on
        none. NaN scores are dropped by predictor training (isfinite) and
        skipped by best-config selection.
        """
        N = len(self._accumulated_per_benchmark)
        if N == 0:
            return np.array([], dtype=np.float64)

        benchmarks = self.config.val_tasks
        score_sum = np.zeros(N, dtype=np.float64)
        score_cnt = np.zeros(N, dtype=np.int64)

        for b in benchmarks:
            accs = np.array([
                (d[0] or {}).get(b, np.nan) for d in self._accumulated_per_benchmark
            ], dtype=np.float64)
            nlls = np.array([
                (d[1] or {}).get(b, np.nan) for d in self._accumulated_per_benchmark
            ], dtype=np.float64)

            valid = np.isfinite(accs) & np.isfinite(nlls)
            n_valid = int(valid.sum())
            if n_valid == 0:
                print(f"    {b}: no valid measurements, skipping")
                continue

            acc_z = (accs - np.nanmean(accs)) / (np.nanstd(accs) + 1e-12)
            nll_z = -(nlls - np.nanmean(nlls)) / (np.nanstd(nlls) + 1e-12)

            K = BENCHMARK_SIZES.get(b, 1000)
            sigma2_noise = 0.25 / K
            sigma2_between = float(accs[valid].var()) + 1e-12
            f = 1.0 - sigma2_noise / sigma2_between
            w = max(self.w_floor, min(1.0, max(0.0, (1.0 + f) / 2.0)))

            score_sum += np.where(valid, w * acc_z + (1.0 - w) * nll_z, 0.0)
            score_cnt += valid

            extra = f", {N - n_valid} unmeasured" if n_valid < N else ""
            print(f"    {b}: w={w:.3f} (f={f:.3f}, noise={sigma2_noise:.6f}, "
                  f"between={sigma2_between:.6f}{extra})")

        scores = np.full(N, np.nan, dtype=np.float64)
        measured = score_cnt > 0
        scores[measured] = score_sum[measured] / score_cnt[measured]
        return scores

    def _refit_predictor(self) -> Tuple[npt.NDArray[np.float64], Optional[List[Any]], Optional[npt.NDArray[np.float64]]]:
        """(Re)train the predictor on all accumulated configs.

        Used both at the end of every iteration and after a resume, so a
        resumed run continues exactly like an uninterrupted one:
          - the next iteration uses predictor-guided sampling (not plain
            Dirichlet), matching the never-interrupted behavior;
          - a run interrupted AFTER the last iteration still selects the
            final mixture via the full-design-space search (paper section
            3.3) instead of falling back to the accumulated best.

        Returns (all_scores, val_configs, val_targets).
        """
        print(f"[Predictor] Computing SNR-weighted scores on "
              f"{len(self._accumulated_per_benchmark)} accumulated configs")
        all_scores = self._compute_scores()
        self._accumulated_scores = all_scores.tolist()

        if not self._accumulated_configs:
            return all_scores, None, None

        if self.metric_direction == "minimize":
            predictor_targets = all_scores
        else:
            predictor_targets = -all_scores

        valid_mask = np.isfinite(predictor_targets)

        train_configs = [c for c, v in zip(self._accumulated_configs, valid_mask) if v]
        train_targets = predictor_targets[valid_mask]

        print(f"[Predictor] Training predictor on {len(train_configs)} valid "
              f"of {len(self._accumulated_configs)} accumulated configs")

        if not train_configs:
            print("[Predictor] WARNING: no config has a finite score (all "
                  "failed or unmeasured) — predictor not trained")
            self._predictor = None
            return all_scores, None, None

        n_total = len(train_configs)
        val_configs_split = None
        val_targets_split = None
        if n_total >= 10:
            n_val = max(5, int(n_total * 0.2))
            rng_split = np.random.default_rng(42)
            indices = rng_split.permutation(n_total)
            train_idx = indices[:n_total - n_val]
            val_idx = indices[n_total - n_val:]

            val_configs_split = [train_configs[i] for i in val_idx]
            val_targets_split = train_targets[val_idx]
            train_configs = [train_configs[i] for i in train_idx]
            train_targets = train_targets[train_idx]

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
        return all_scores, val_configs_split, val_targets_split

    def _reconstruct_iteration_results(self) -> List[IterationResult]:
        """Rebuild per-iteration results from accumulated state after a resume,
        so the final report/summary still covers pre-crash iterations."""
        results: List[IterationResult] = []
        offset = 0
        scores = (np.array(self._accumulated_scores, dtype=np.float64)
                  if self._accumulated_scores else np.array([], dtype=np.float64))
        for k, n in enumerate(self.config.search.configs_per_iter):
            if offset + n > len(self._accumulated_configs):
                break
            chunk_configs = self._accumulated_configs[offset:offset + n]
            chunk_scores = scores[offset:offset + n]
            best_idx = self._best_index(chunk_scores) if len(chunk_scores) else 0
            results.append(IterationResult(
                iteration=k + 1,
                n_configs=n,
                n_trained=n,
                predictor=self._predictor,
                predictor_r2=None,
                best_config=chunk_configs[best_idx] if chunk_configs else None,
                best_score=float(chunk_scores[best_idx]) if len(chunk_scores) else None,
                all_configs=chunk_configs,
                all_scores=chunk_scores,
            ))
            offset += n
        return results

    @staticmethod
    def _raise_if_all_failed(results, iteration: int) -> None:
        """Fail-fast when every experiment of a batch errored.

        A batch whose experiments run on different NPUs with independent
        configs cannot fail 100% for data reasons — an all-error batch means
        the environment is broken (leaked NPU memory, missing checkpoint,
        vanished data dir, ...). Fitting the predictor on that garbage and
        'completing' the search anyway (observed in speedrun 2026-08-26 17:35:
        7/7 mid_train failures silently scored 0.0) wastes hours and produces
        a confident-looking but meaningless result. The pending-configs state
        saved before the batch makes the resume re-run exactly these configs.
        """
        if results and all(r.metadata.get("error") for r in results):
            raise RuntimeError(
                f"All {len(results)} proxy experiments of iteration {iteration} "
                f"failed — environment failure? Check result/*/exp_*/mid_train.log "
                f"and 'npu-smi info' (leaked HBM shows as used memory with no "
                f"process). Resume re-runs this iteration.")

    def run_iteration(
        self,
        iteration: int,
        n_configs: int,
        proxy_runner: Optional[Any] = None,
        preset_configs: Optional[List[MixtureConfig]] = None,
    ) -> IterationResult:
        print(f"\n{'=' * 70}")
        print(f"  CLIMB Iteration {iteration}")
        print(f"  Configs: {n_configs}, metric: {self.metric_direction}")
        print(f"{'=' * 70}")

        t0 = time.time()

        if preset_configs is not None:
            new_configs = list(preset_configs)
            print(f"[Iter {iteration}] Restored {len(new_configs)} pending configs from saved state")
        elif iteration == 1 or self._predictor is None:
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

        # Persist this iteration's configs BEFORE running the experiments: if
        # we crash mid-iteration, the restart restores exactly these configs
        # (not a fresh Dirichlet/predictor-guided sample), which is what lets
        # the proxy runner match and reuse exp_XXXX/meta.json results.
        if self.state_path:
            self._pending = {
                "iteration": iteration,
                "configs": [c.mixture_weights.weights.tolist() for c in new_configs],
            }
            self._save_state()

        trained_configs: List[MixtureConfig] = []

        if proxy_runner is not None:
            print(f"[Iter {iteration}] Training {len(new_configs)} proxy models")
            # Global experiment ids (== config ids) so exp dirs never collide
            # across iterations and meta.json reuse can match them.
            run_kwargs = {}
            try:
                sig = inspect.signature(proxy_runner.run_batch)
                if "experiment_id_base" in sig.parameters:
                    run_kwargs["experiment_id_base"] = len(self._accumulated_configs)
            except (ValueError, TypeError):
                pass
            results = proxy_runner.run_batch(new_configs, **run_kwargs)
            self._raise_if_all_failed(results, iteration)
            n_err = sum(1 for r in results if r.metadata.get("error"))
            if n_err:
                print(f"[Iter {iteration}] WARNING: {n_err}/{len(results)} experiments "
                      f"failed — scored NaN (excluded from predictor training and "
                      f"best-config selection; no meta.json written, so a resume "
                      f"of this iteration re-runs them)")
            for r in results:
                trained_configs.append(r.mixture_config)
                self._accumulated_configs.append(r.mixture_config)
                self._accumulated_per_benchmark.append(
                    (r.per_task_accuracies, r.per_task_nlls)
                )
        else:
            print(f"[Iter {iteration}] No proxy runner, using random scores (for testing)")
            rng = np.random.default_rng(iteration * 1000)
            for c in new_configs:
                trained_configs.append(c)
                self._accumulated_configs.append(c)
                fake_acc = {b: float(rng.uniform(0.0, 0.1)) for b in self.config.val_tasks}
                fake_nll = {b: float(rng.uniform(2.0, 4.0)) for b in self.config.val_tasks}
                self._accumulated_per_benchmark.append((fake_acc, fake_nll))

        all_scores, val_configs_split, val_targets_split = self._refit_predictor()

        n_new = len(trained_configs)
        scores_arr = all_scores[-n_new:] if n_new > 0 else np.array([], dtype=np.float64)

        if not np.isfinite(scores_arr).any():
            # Mirrors _raise_if_all_failed for the no-error-but-unmeasured
            # case (e.g. base_eval exited 0 without a parseable CSV): an
            # iteration with zero measurable scores must not advance the
            # search on garbage.
            raise RuntimeError(
                f"Iteration {iteration}: no experiment produced a measurable "
                f"score (all failed or eval output unparseable). Check "
                f"result/*/exp_*/mid_train.log and eval.log; the pending-config "
                f"state makes resume re-run exactly this iteration.")

        best_idx = self._best_index(scores_arr)
        best_config = trained_configs[best_idx]
        best_score = float(scores_arr[best_idx])
        best_loss = None

        iter_result = IterationResult(
            iteration=iteration,
            n_configs=n_configs,
            n_trained=len(trained_configs),
            predictor=self._predictor,
            predictor_r2=None,
            best_config=best_config,
            best_score=best_score,
            all_configs=trained_configs,
            all_scores=scores_arr,
        )

        if val_configs_split is not None:
            iter_result.predictor_r2 = float(self._predictor.score(
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

        start_iter = 1
        resumed_pending: Optional[Dict[str, Any]] = None
        if self.state_path:
            loaded_iter = self._load_state()
            resumed_pending = self._pending
            if loaded_iter > 0:
                start_iter = loaded_iter + 1
                self._iteration_results = self._reconstruct_iteration_results()
                print(f"[Search] Resuming from iteration {start_iter}")
                # Refit the predictor from accumulated data so the resumed run
                # behaves exactly like an uninterrupted one: predictor-guided
                # sampling for the next iteration, and full-design-space
                # selection even if all iterations had already finished.
                if self._accumulated_configs:
                    self._refit_predictor()

        for k in range(start_iter - 1, self.config.search.num_iterations):
            n_configs = self.config.search.configs_per_iter[k]
            preset = None
            if resumed_pending is not None and resumed_pending.get("iteration") == k + 1:
                preset = [
                    MixtureConfig(
                        mixture_weights=MixtureWeights(
                            weights=np.array(w, dtype=np.float64)),
                    )
                    for w in resumed_pending["configs"]
                ]
                n_configs = len(preset)
                resumed_pending = None
            self.run_iteration(k + 1, n_configs, proxy_runner, preset_configs=preset)
            self._last_completed_iter = k + 1
            self._pending = None
            self._save_state()

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
