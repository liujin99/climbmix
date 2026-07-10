"""
CLIMB pipeline — main entry point.

Refactored for strategy injection:
  - Discovery strategy: fdc_labels or embedding_cluster (via config)
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
        val_data_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        data_dir = data_dir or self.config.data_dir
        output_dir = output_dir or self.config.output_dir
        os.makedirs(output_dir, exist_ok=True)

        t_start = time.time()
        stage_times: Dict[str, float] = {}

        print("\n" + "=" * 70)
        print("  Nemotron-CLIMB Pipeline")
        print(f"  Discovery: {self.config.discovery.method}")
        print(f"  Filter: {self.config.filtering.method}")
        print(f"  Proxy: {self.config.proxy.model_size}")
        print(f"  Output: {output_dir}")
        print("=" * 70)

        # Stage 0: Load data
        _t = time.time()
        texts_loaded, cluster_labels, quality_scores, token_counts, mm = \
            self._stage0_load(
                data_dir, texts, cluster_labels, quality_scores,
                token_counts, metadata_manager,
            )
        stage_times["stage0_load"] = time.time() - _t

        # Stage 1: Quality filtering
        _t = time.time()
        quality_filter = get_filter(self.config.filtering.method)
        filtered_labels, cluster_quality = quality_filter.filter(
            cluster_labels, quality_scores, self.config.filtering,
        )
        stage_times["stage1_filter"] = time.time() - _t

        # Stage 2: Cluster discovery
        _t = time.time()
        discovery = get_discovery(self.config.discovery.method, self.config.discovery)
        cluster_info, final_labels = discovery.discover(
            texts=texts_loaded,
            cluster_labels=filtered_labels,
            quality_scores=quality_scores,
            token_counts=token_counts,
            metadata_manager=mm,
        )
        num_clusters = len(cluster_info)
        print(f"[Stage 2] {num_clusters} clusters, {len(final_labels):,} documents")
        stage_times["stage2_discovery"] = time.time() - _t

        # Get cluster token counts for Dirichlet
        cluster_token_counts = np.array(
            [c.num_tokens for c in cluster_info], dtype=np.int64
        )

        # Stage 3: Iterative bootstrapping search
        _t = time.time()
        print("\n[Stage 3] Running iterative bootstrapping search")

        bootstrapper = IterativeBootstrapper(
            self.config, cluster_token_counts, final_labels,
        )

        optimal_weights, iter_results = bootstrapper.search_optimal(proxy_runner)
        stage_times["stage3_search"] = time.time() - _t

        # Stage 4: Final data selection
        _t = time.time()
        print("\n[Stage 4] Selecting data with optimal mixture weights")

        selected_indices, sampling_probs = select_data_by_mixture(
            final_labels,
            optimal_weights.mixture_weights,
            token_counts,
        )

        stats = compute_mixture_dataset_stats(
            final_labels, selected_indices, cluster_info,
            optimal_weights.mixture_weights,
        )
        stage_times["stage4_selection"] = time.time() - _t

        # Stage 5: Save outputs
        _t = time.time()
        print("\n[Stage 5] Saving outputs")
        self._save_outputs(
            output_dir, optimal_weights, iter_results,
            cluster_info, final_labels, selected_indices,
            token_counts, texts_loaded, mm,
            stats, stage_times, t_start,
        )
        stage_times["stage5_save"] = time.time() - _t

        elapsed = time.time() - t_start
        print(f"\n{'=' * 70}")
        print(f"  Pipeline Complete! ({elapsed:.1f}s)")
        print(f"  Optimal mixture weights:")
        for i, w in enumerate(optimal_weights.mixture_weights.weights):
            label = cluster_info[i].label if i < len(cluster_info) else f"C{i}"
            print(f"    {label}: {w:.4f}")
        print(f"  Selected: {len(selected_indices):,} documents")
        print(f"  Output: {output_dir}/")
        print(f"{'=' * 70}")

        return {
            "optimal_weights": optimal_weights,
            "iteration_results": iter_results,
            "cluster_info": cluster_info,
            "selected_indices": selected_indices,
            "stats": stats,
            "elapsed_seconds": elapsed,
            "stage_times": stage_times,
        }

    def _stage0_load(
        self, data_dir, texts, cluster_labels, quality_scores,
        token_counts, metadata_manager,
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
            mm = ShardMetadataManager(data_dir)
            cluster_labels = mm.cluster_labels
            quality_scores = mm.quality_scores
            token_counts = mm.estimate_token_counts()
            texts = None
            print(f"[Stage 0] Loaded from data_dir: {mm.num_docs:,} docs")
            return texts, cluster_labels, quality_scores, token_counts, mm

        raise ValueError("Must provide data_dir, texts, or metadata_manager")

    def _save_outputs(
        self,
        output_dir, optimal_weights, iter_results,
        cluster_info, cluster_labels, selected_indices,
        token_counts, texts, metadata_manager,
        stats, stage_times, t_start,
    ):
        weights_dict = optimal_weights.mixture_weights.to_dict(
            cluster_labels=[c.label for c in cluster_info]
        )
        weights_path = os.path.join(output_dir, "optimal_mixture_weights.json")
        with open(weights_path, "w") as f:
            json.dump(weights_dict, f, indent=2)
        print(f"[Save] Optimal weights -> {weights_path}")

        elapsed = time.time() - t_start
        summary = {
            "config": {
                "discovery_method": self.config.discovery.method,
                "filter_method": self.config.filtering.method,
                "proxy_size": self.config.proxy.model_size,
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
                    "best_loss": r.best_loss,
                    "predictor_r2": r.predictor_r2,
                }
                for r in iter_results
            ],
            "elapsed_seconds": elapsed,
            "stage_times": {k: round(v, 1) for k, v in stage_times.items()},
        }
        summary_path = os.path.join(output_dir, "pipeline_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, np.floating) else x)

        sampled_path = os.path.join(output_dir, "sampled_dataset.parquet")
        if metadata_manager is not None:
            sampled_texts = metadata_manager.read_texts(selected_indices)
        elif texts is not None:
            sampled_texts = [texts[i] for i in selected_indices]
        else:
            sampled_texts = []

        if sampled_texts:
            pd.DataFrame({
                "text": sampled_texts,
                "doc_id": selected_indices.tolist(),
                "cluster": cluster_labels[selected_indices].tolist(),
            }).to_parquet(sampled_path, index=False)
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
        with open(cluster_path, "w") as f:
            json.dump(cluster_json, f, indent=2)

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
        )
        print(f"[Save] Markdown report -> {report_path}")
