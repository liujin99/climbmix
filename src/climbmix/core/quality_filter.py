"""
Quality filtering strategies.

Provides a registry to select between different quality filtering
methods via config:
  - "none": no filtering
  - "doc_level": per-document quality threshold
  - "cluster_level": per-cluster average quality pruning
  - "doc_and_cluster": document-level filter then cluster-level pruning
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt

from climbmix.core.types import QualityFilterConfig
from climbmix.core.protocols import QualityFilter


class NoFilter:
    def filter(
        self,
        cluster_labels: npt.NDArray[np.int64],
        quality_scores: Optional[npt.NDArray[np.float64]],
        config: QualityFilterConfig,
    ) -> Tuple[npt.NDArray[np.int64], Dict[int, float]]:
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        cluster_quality = {int(c): 5.0 for c in unique_clusters}
        return cluster_labels, cluster_quality


class DocLevelFilter:
    def filter(
        self,
        cluster_labels: npt.NDArray[np.int64],
        quality_scores: Optional[npt.NDArray[np.float64]],
        config: QualityFilterConfig,
    ) -> Tuple[npt.NDArray[np.int64], Dict[int, float]]:
        if quality_scores is None:
            print("[DocLevelFilter] No quality scores, skipping")
            return cluster_labels, {int(c): 5.0 for c in np.unique(cluster_labels[cluster_labels >= 0])}

        n_docs = len(cluster_labels)
        n_criteria = quality_scores.shape[1] if quality_scores.ndim > 1 else 1

        if n_criteria >= 3:
            english_idx = min(2, n_criteria - 1)
            english_scores = quality_scores[:, english_idx] if quality_scores.ndim > 1 else quality_scores
            english_mask = english_scores >= config.doc_english_min

            composite_scores = quality_scores.mean(axis=1) if quality_scores.ndim > 1 else quality_scores
            composite_mask = composite_scores >= config.doc_composite_min

            keep_mask = english_mask & composite_mask
        else:
            composite_scores = quality_scores.mean(axis=1) if quality_scores.ndim > 1 else quality_scores
            keep_mask = composite_scores >= config.doc_composite_min

        filtered_labels = cluster_labels.copy()
        filtered_labels[~keep_mask] = -1

        n_removed = int((filtered_labels == -1).sum()) - int((cluster_labels == -1).sum())
        print(f"[DocLevelFilter] Removed {n_removed} docs "
              f"(english<{config.doc_english_min} or composite<{config.doc_composite_min})")

        unique_clusters = np.unique(filtered_labels[filtered_labels >= 0])
        cluster_quality: Dict[int, float] = {}
        for c in unique_clusters:
            mask = filtered_labels == c
            cluster_quality[int(c)] = float(quality_scores[mask].mean())

        return filtered_labels, cluster_quality


class ClusterLevelFilter:
    def filter(
        self,
        cluster_labels: npt.NDArray[np.int64],
        quality_scores: Optional[npt.NDArray[np.float64]],
        config: QualityFilterConfig,
    ) -> Tuple[npt.NDArray[np.int64], Dict[int, float]]:
        from climbmix.core.cluster_merge import compute_cluster_quality, prune_clusters

        cluster_quality = compute_cluster_quality(
            cluster_labels, quality_scores,
            prune_threshold=config.cluster_avg_threshold,
        )

        n_before = len(np.unique(cluster_labels[cluster_labels >= 0]))

        if quality_scores is not None and cluster_quality:
            centroids = np.zeros((n_before, 1), dtype=np.float32)
            pruned_labels, _, _ = prune_clusters(
                cluster_labels, centroids, cluster_quality,
                threshold=config.cluster_avg_threshold,
            )
            n_after = len(np.unique(pruned_labels[pruned_labels >= 0]))
            print(f"[ClusterLevelFilter] Pruned {n_before - n_after}/{n_before} clusters "
                  f"(avg_quality<{config.cluster_avg_threshold})")
            return pruned_labels, cluster_quality

        return cluster_labels, cluster_quality


class DocAndClusterFilter:
    def filter(
        self,
        cluster_labels: npt.NDArray[np.int64],
        quality_scores: Optional[npt.NDArray[np.float64]],
        config: QualityFilterConfig,
    ) -> Tuple[npt.NDArray[np.int64], Dict[int, float]]:
        doc_filter = DocLevelFilter()
        filtered_labels, cluster_quality = doc_filter.filter(cluster_labels, quality_scores, config)

        cluster_filter = ClusterLevelFilter()
        final_labels, final_quality = cluster_filter.filter(filtered_labels, quality_scores, config)

        return final_labels, final_quality


FILTER_REGISTRY: Dict[str, QualityFilter] = {
    "none": NoFilter(),
    "doc_level": DocLevelFilter(),
    "cluster_level": ClusterLevelFilter(),
    "doc_and_cluster": DocAndClusterFilter(),
}


def get_filter(method: str) -> QualityFilter:
    if method not in FILTER_REGISTRY:
        raise ValueError(
            f"Unknown filter method '{method}'. "
            f"Available: {list(FILTER_REGISTRY.keys())}"
        )
    return FILTER_REGISTRY[method]
