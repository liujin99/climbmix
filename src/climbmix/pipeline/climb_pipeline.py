"""
CLIMB pipeline — main entry point.

Refactored for strategy injection:
  - Discovery strategy: embedding_cluster (via config)
  - Quality filter strategy: none/doc_level/cluster_level/doc_and_cluster (via config)
  - Predictor strategy: lightgbm (via config)

Pipeline stages:
  0. Load data (auto-scan data_dir for all parquet files)
  1. Cluster discovery + quality filtering
  2. Iterative bootstrapping search
  3. Final data selection using optimal mixture weights
  4. Save outputs and generate report
"""

import json
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from climbmix.core.types import (
    CLIMBConfig,
    MixtureConfig,
    MixtureWeights,
    IterationResult,
    ClusterInfo,
    ProxyResult,
)
from climbmix.core.discovery import get_discovery
from climbmix.core.quality_filter import get_filter
from climbmix.core.iterative_bootstrapper import IterativeBootstrapper
from climbmix.sampling.data_selector import select_data_by_mixture, compute_mixture_dataset_stats
from climbmix.utils.token_estimate import estimate_tokens_from_text
from climbmix.utils.io_utils import atomic_savez, atomic_write_json, atomic_write_parquet
from climbmix.utils.embed_cache import pool_embedding_cache_key


class CLIMBPipeline:
    def __init__(self, config: CLIMBConfig):
        self.config = config

    def run(
        self,
        data_dir: Optional[str] = None,
        texts: Optional[List[str]] = None,
        cluster_labels: Optional[np.ndarray] = None,
        token_counts: Optional[np.ndarray] = None,
        quality_scores: Optional[np.ndarray] = None,
        metadata_manager: Optional[Any] = None,
        proxy_runner: Optional[Any] = None,
        target_runner: Optional[Any] = None,
        val_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        cluster_cache_dir: Optional[str] = None,
        resume_search: bool = False,
    ) -> Dict[str, Any]:
        data_dir = data_dir or self.config.data_dir
        output_dir = output_dir or self.config.output_dir
        cluster_cache_dir = cluster_cache_dir or output_dir
        os.makedirs(output_dir, exist_ok=True)

        if self.config.schema_path:
            from climbmix.data.column_schema import DatasetSchema
            schema = DatasetSchema.from_yaml(self.config.schema_path)
            adj_threshold = schema.prune_threshold
            if adj_threshold != self.config.discovery.prune_threshold:
                print(f"[Pipeline] Override prune_threshold: "
                      f"{self.config.discovery.prune_threshold} → {adj_threshold}")
                self.config.discovery.prune_threshold = adj_threshold
        elif self.config.quality_config_path:
            import yaml
            with open(self.config.quality_config_path) as f:
                qc = yaml.safe_load(f)
            adj_threshold = float(qc.get("prune_threshold", self.config.discovery.prune_threshold))
            if adj_threshold != self.config.discovery.prune_threshold:
                print(f"[Pipeline] Override prune_threshold: "
                      f"{self.config.discovery.prune_threshold} → {adj_threshold}")
                self.config.discovery.prune_threshold = adj_threshold

        t_start = time.time()
        stage_times: Dict[str, float] = {}

        print("\n" + "=" * 70)
        print("  Nemotron-CLIMB Pipeline")
        print(f"  Discovery: {self.config.discovery.method}")
        print(f"  Filter: {self.config.filtering.method}")
        print(f"  Proxy: {self.config.proxy.model_tag} ({self.config.proxy.scaling_M:.1f}M scaling)")
        print(f"  Output: {output_dir}")
        print("=" * 70)

        # Stage 0: Load data
        _t = time.time()
        texts_loaded, cluster_labels, quality_scores, token_counts, mm = \
            self._stage0_load(
                data_dir, texts, cluster_labels, quality_scores,
                token_counts, metadata_manager, output_dir,
            )
        stage_times["stage0_load"] = time.time() - _t

        # Stage 1: Cluster discovery (cacheable)
        cluster_cache_npz = os.path.join(cluster_cache_dir, "cluster_cache.npz")
        cluster_cache_json = os.path.join(cluster_cache_dir, "cluster_info_cache.json")
        embedding_cache_dir = self._pool_embedding_cache_dir(data_dir)
        if embedding_cache_dir:
            print(f"[Stage 1] Pool-level embedding/kmeans cache: {embedding_cache_dir}")
        cluster_cache_ok = False
        if os.path.exists(cluster_cache_npz) and os.path.exists(cluster_cache_json):
            try:
                cache = np.load(cluster_cache_npz, allow_pickle=False)
                final_labels = cache["final_labels"]
                cluster_info = self._load_cluster_cache(cluster_cache_json)
                cluster_cache_ok = len(cluster_info) > 0
            except Exception as e:
                print(f"[Stage 1] Cluster cache unreadable ({e}), recomputing")
        if cluster_cache_ok:
            _t = time.time()
            print("[Stage 1] Loading cached clusters...")
            num_clusters = len(cluster_info)
            print(f"[Stage 1] {num_clusters} clusters (from cache), {len(final_labels):,} documents")
            self._print_cluster_sizes(cluster_info)
            stage_times["stage1_discovery"] = time.time() - _t
        else:
            _t = time.time()
            discovery = get_discovery(self.config.discovery.method, self.config.discovery)
            if embedding_cache_dir:
                # Serialize concurrent runs over the SAME pool: the second
                # run waits for the first to finish writing the pool cache,
                # then reuses it instead of racing a partial write.
                from climbmix.utils.io_utils import file_lock
                lock_path = os.path.join(embedding_cache_dir, ".embed.lock")
                with file_lock(lock_path):
                    cluster_info, final_labels = discovery.discover(
                        texts=texts_loaded,
                        cluster_labels=cluster_labels,
                        quality_scores=quality_scores,
                        token_counts=token_counts,
                        metadata_manager=mm,
                        cache_dir=cluster_cache_dir,
                        embedding_cache_dir=embedding_cache_dir,
                    )
            else:
                cluster_info, final_labels = discovery.discover(
                    texts=texts_loaded,
                    cluster_labels=cluster_labels,
                    quality_scores=quality_scores,
                    token_counts=token_counts,
                    metadata_manager=mm,
                    cache_dir=cluster_cache_dir,
                )
            num_clusters = len(cluster_info)
            print(f"[Stage 1] {num_clusters} clusters, {len(final_labels):,} documents")
            self._print_cluster_sizes(cluster_info)
            stage_times["stage1_discovery"] = time.time() - _t
            self._save_cluster_cache(cluster_cache_npz, cluster_cache_json, final_labels, cluster_info)
            print(f"[Stage 1] Cached → {cluster_cache_npz}")

        # Stage 2: Quality filtering (after clusters are known)
        _t = time.time()
        quality_filter = get_filter(self.config.filtering.method)
        filtered_labels, cluster_quality = quality_filter.filter(
            final_labels, quality_scores, self.config.filtering,
        )
        stage_times["stage2_filter"] = time.time() - _t

        # Get cluster token counts for Dirichlet
        cluster_token_counts = np.array(
            [c.num_tokens for c in cluster_info], dtype=np.int64
        )

        # Stage 3: Iterative bootstrapping search
        _t = time.time()
        print("\n[Stage 3] Running iterative bootstrapping search")

        state_path = os.path.join(output_dir, "search_state.json") if resume_search else None
        bootstrapper = IterativeBootstrapper(
            self.config, cluster_token_counts, filtered_labels,
            state_path=state_path,
        )

        if proxy_runner is not None and hasattr(proxy_runner, '__init__'):
            proxy_runner.cluster_labels = filtered_labels
            proxy_runner.token_counts = token_counts
            proxy_runner.metadata_manager = mm

        optimal_weights, iter_results = bootstrapper.search_optimal(proxy_runner)
        stage_times["stage3_search"] = time.time() - _t

        # Stage 4: Final data selection
        _t = time.time()
        print("\n[Stage 4] Selecting data with optimal mixture weights")

        selected_indices, sampling_probs = select_data_by_mixture(
            filtered_labels,
            optimal_weights.mixture_weights,
            token_counts,
            target_tokens=self.config.target.target_tokens,
        )

        stats = compute_mixture_dataset_stats(
            filtered_labels, selected_indices, cluster_info,
            optimal_weights.mixture_weights,
        )
        stage_times["stage4_selection"] = time.time() - _t

        # Stage 5: Save outputs
        _t = time.time()
        print("\n[Stage 5] Saving outputs")
        search_extras = {
            "predictor_eval": bootstrapper.predictor_eval,
            "online_eval": bootstrapper.online_eval,
            "pruning_history": bootstrapper.pruning_history,
        }
        self._save_outputs(
            output_dir, optimal_weights, iter_results,
            cluster_info, final_labels, selected_indices,
            token_counts, texts_loaded, mm,
            stats, stage_times, t_start, search_extras,
        )
        stage_times["stage5_save"] = time.time() - _t

        # Stage 6: Target training (d28) with optimal mixture
        target_result = None
        if target_runner is not None:
            _t = time.time()
            print("\n[Stage 6] Target training with optimal mixture")

            if hasattr(target_runner, 'cluster_labels'):
                target_runner.cluster_labels = filtered_labels
                target_runner.token_counts = token_counts
                target_runner.metadata_manager = mm

            target_result = target_runner.run(
                optimal_weights.mixture_weights,
                selected_indices,
                output_dir,
            )
            stage_times["stage6_target"] = time.time() - _t

        elapsed = time.time() - t_start
        print(f"\n{'=' * 70}")
        print(f"  Pipeline Complete! ({elapsed:.1f}s)")
        print(f"  Optimal mixture weights:")
        for i, w in enumerate(optimal_weights.mixture_weights.weights):
            label = cluster_info[i].label if i < len(cluster_info) else f"C{i}"
            print(f"    {label}: {w:.4f}")
        print(f"  Selected: {len(selected_indices):,} documents")
        if target_result:
            print(f"  Target stem_metric: {target_result.get('stem_metric', 'N/A')}")
        print(f"  Output: {output_dir}/")
        print(f"{'=' * 70}")

        return {
            "optimal_weights": optimal_weights,
            "iteration_results": iter_results,
            "cluster_info": cluster_info,
            "selected_indices": selected_indices,
            "stats": stats,
            "target_result": target_result,
            "elapsed_seconds": elapsed,
            "stage_times": stage_times,
        }

    def _stage0_load(
        self, data_dir, texts, cluster_labels, quality_scores,
        token_counts, metadata_manager, output_dir=None,
    ):
        mm = None
        if metadata_manager is not None:
            mm = metadata_manager
            cluster_labels = mm.cluster_labels
            quality_scores = mm.quality_scores
            token_counts = mm.estimate_token_counts()
            texts = None
            print(f"[Stage 0] Loaded via ShardMetadataManager: {mm.num_docs:,} docs")
            return texts, cluster_labels, quality_scores, token_counts, mm

        if texts is not None:
            if token_counts is None:
                token_counts = np.array(
                    [estimate_tokens_from_text(t) for t in texts], dtype=np.int64
                )
            print(f"[Stage 0] Using pre-loaded texts: {len(texts):,} docs")
            return texts, cluster_labels, quality_scores, token_counts, mm

        if data_dir is not None:
            from climbmix.data.metadata_manager import ShardMetadataManager
            from climbmix.data.column_schema import DatasetSchema
            if self.config.schema_path:
                schema = DatasetSchema.from_yaml(self.config.schema_path)
            else:
                schema = DatasetSchema.from_yaml("config/schema_stem.yaml")
            mm = ShardMetadataManager(data_dir, schema=schema, cache_dir=data_dir)
            cluster_labels = mm.cluster_labels
            quality_scores = mm.quality_scores
            token_counts = mm.estimate_token_counts()
            texts = None
            print(f"[Stage 0] Loaded from data_dir: {mm.num_docs:,} docs")
            return texts, cluster_labels, quality_scores, token_counts, mm

        raise ValueError("Must provide data_dir, texts, or metadata_manager")

    def _pool_embedding_cache_dir(self, data_dir: str) -> Optional[str]:
        """Content-keyed stable cache dir for pool-level artifacts.

        Key = sha256(shard name+size manifest, embedding model, truncate len,
        and — for subsampled runs — the sample size, which determines WHICH
        docs get embedded). NOT keyed by K_enhanced/K_max/merge_distance/
        prune_threshold: those change the merge stage only, which must reuse
        the (expensive) embeddings and K-means results. The sample size is
        only hashed when > 0 so full-pool (streaming) keys stay stable.
        Empty config → None (legacy behavior).

        The key formula lives in utils/embed_cache.py (shared with
        scripts/embed_merge.py so the multi-node merge lands at exactly
        the path this method reads).
        """
        if not self.config.embedding_cache_dir:
            return None
        if not os.path.isdir(data_dir):
            return None
        disc = self.config.discovery
        shards = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        )
        if not shards:
            return None
        key = pool_embedding_cache_key(
            ((name, os.path.getsize(os.path.join(data_dir, name)))
             for name in shards),
            disc.embedding_model,
            disc.embedding_truncate_len,
            disc.embedding_sample_size,
        )
        return os.path.join(self.config.embedding_cache_dir, key)

    @staticmethod
    def _print_cluster_sizes(cluster_info, top_n=20):
        """Log per-cluster doc/token distribution (observation only).

        Paper §2.1 has no cluster-size filtering: small clusters are handled
        implicitly via token-count-weighted Dirichlet init and distance merging.
        This log just makes the size distribution visible for inspection.
        """
        if not cluster_info:
            return
        docs = [c.num_docs for c in cluster_info]
        toks = [c.num_tokens for c in cluster_info]
        order = sorted(range(len(cluster_info)), key=lambda i: toks[i], reverse=True)
        print(f"[Stage 1] Cluster sizes: {len(cluster_info)} clusters, "
              f"docs min/median/max = {min(docs):,}/{sorted(docs)[len(docs)//2]:,}/{max(docs):,}, "
              f"tokens total = {sum(toks):,}")
        show = order[:top_n] if len(order) > top_n else order
        lines = ", ".join(
            f"C{cluster_info[i].cluster_id}:{toks[i]:,}t/{docs[i]:,}d" for i in show
        )
        print(f"[Stage 1] Top clusters (tokens): {lines}"
              + (f" ... +{len(order) - top_n} more" if len(order) > top_n else ""))

    @staticmethod
    def _save_cluster_cache(npz_path: str, json_path: str, labels: np.ndarray, cluster_info):
        atomic_savez(npz_path, final_labels=labels)
        data = [
            {
                "cluster_id": c.cluster_id,
                "centroid": c.centroid.tolist(),
                "num_docs": c.num_docs,
                "num_tokens": c.num_tokens,
                "label": c.label,
                "quality_score": c.quality_score,
            }
            for c in cluster_info
        ]
        atomic_write_json(json_path, data)

    @staticmethod
    def _load_cluster_cache(json_path: str):
        from climbmix.core.types import ClusterInfo
        with open(json_path) as f:
            data = json.load(f)
        return [
            ClusterInfo(
                cluster_id=d["cluster_id"],
                centroid=np.array(d["centroid"], dtype=np.float64),
                num_docs=d["num_docs"],
                num_tokens=d["num_tokens"],
                label=d["label"],
                quality_score=d["quality_score"],
            )
            for d in data
        ]

    def _save_outputs(
        self,
        output_dir, optimal_weights, iter_results,
        cluster_info, cluster_labels, selected_indices,
        token_counts, texts, metadata_manager,
        stats, stage_times, t_start, search_extras=None,
    ):
        weights_dict = optimal_weights.mixture_weights.to_dict(
            cluster_labels=[c.label for c in cluster_info]
        )
        weights_path = os.path.join(output_dir, "optimal_mixture_weights.json")
        atomic_write_json(weights_path, weights_dict)
        print(f"[Save] Optimal weights -> {weights_path}")

        elapsed = time.time() - t_start
        summary = {
            "config": {
                "discovery_method": self.config.discovery.method,
                "filter_method": self.config.filtering.method,
                "proxy_size": self.config.proxy.model_tag,
                "K_enhanced": self.config.discovery.K_enhanced,
                "num_iterations": self.config.search.num_iterations,
                "configs_per_iter": self.config.search.configs_per_iter,
            },
            "metrics": {
                "optimal_weights": weights_dict,
                "num_clusters": len(cluster_info),
                "num_selected_docs": len(selected_indices),
            },
            "iteration_summary": [
                {
                    "iteration": r.iteration,
                    "n_configs": r.n_configs,
                    "n_trained": r.n_trained,
                    "best_score": r.best_score,
                    "predictor_r2": r.predictor_r2,
                    "predictor_spearman": r.predictor_spearman,
                    "online_spearman": r.online_spearman,
                }
                for r in iter_results
            ],
            "elapsed_seconds": elapsed,
            "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
        }
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        atomic_write_json(summary_path, summary, indent=2,
                          default=lambda x: float(x) if isinstance(x, np.floating) else x)

        sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
        if metadata_manager is not None:
            sampled_texts = metadata_manager.read_texts(selected_indices)
        elif texts is not None:
            sampled_texts = [texts[i] for i in selected_indices]
        else:
            sampled_texts = []

        if sampled_texts:
            atomic_write_parquet(
                sampled_path,
                pd.DataFrame({
                    "text": sampled_texts,
                    "doc_id": selected_indices.tolist(),
                    "cluster": cluster_labels[selected_indices].tolist(),
                }),
            )
            print(f"[Save] Sampled dataset ({len(selected_indices)} docs) -> {sampled_path}")

        cluster_path = os.path.join(output_dir, "cluster_info.json")
        cluster_json = [
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "num_docs": c.num_docs,
                "num_tokens": c.num_tokens,
                "centroid_norm": float(np.linalg.norm(c.centroid)),
            }
            for c in cluster_info
        ]
        atomic_write_json(cluster_path, cluster_json, indent=2)

        from climbmix.pipeline.report_generator import (
            generate_markdown_report,
            generate_distribution_chart,
        )

        chart_path = generate_distribution_chart(output_dir, cluster_info, stats)
        print(f"[Save] Domain distribution chart -> {chart_path}")

        report_path = generate_markdown_report(
            output_dir, self.config, cluster_info,
            optimal_weights, iter_results, stats,
            stage_times, elapsed,
            search_state=search_extras,
        )
        print(f"[Save] Markdown report -> {report_path}")
