"""
Cluster discovery strategies.

Provides a registry to select between different cluster discovery
methods via config:
  - "embedding_cluster": embed + K-means + prune + merge
  - "quality_cluster": cluster by domain + quality scores (no embedding, no text reading)
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt

from climbmix.core.types import ClusterInfo, ClusterDiscoveryConfig
from climbmix.core.protocols import ClusterDiscovery


class EmbeddingClusterDiscovery:
    def __init__(self, config: Optional[ClusterDiscoveryConfig] = None):
        self._config = config or ClusterDiscoveryConfig()

    def discover(
        self,
        texts: Optional[list],
        cluster_labels: Optional[npt.NDArray[np.int64]],
        quality_scores: Optional[npt.NDArray[np.float64]],
        token_counts: Optional[npt.NDArray[np.int64]],
        metadata_manager: Optional[object],
        cache_dir: Optional[str] = None,
    ) -> Tuple[list, npt.NDArray[np.int64]]:
        from climbmix.core.cluster_merge import preprocess_pipeline

        config = self._config
        device = config.embedding_device

        sample_size = config.embedding_sample_size
        if sample_size and sample_size > 0 and metadata_manager is not None:
            n_total = metadata_manager.num_docs
            if n_total > sample_size:
                rng = np.random.default_rng(42)
                sample_indices = rng.choice(n_total, size=sample_size, replace=False)
                sample_indices.sort()
                print(f"[EmbeddingCluster] Subsampling {sample_size:,} / {n_total:,} docs for embedding")
                loaded_texts = metadata_manager.read_texts(sample_indices)

                if token_counts is None:
                    from climbmix.utils.token_estimate import estimate_tokens_from_text
                    token_counts_sample = np.array(
                        [estimate_tokens_from_text(t) for t in loaded_texts], dtype=np.int64
                    )
                else:
                    token_counts_sample = token_counts[sample_indices]

                if quality_scores is not None:
                    quality_sample = quality_scores[sample_indices]
                else:
                    quality_sample = None

                cluster_info, sample_labels = preprocess_pipeline(
                    loaded_texts,
                    quality_scores=quality_sample,
                    token_counts=token_counts_sample,
                    embedding_model=config.embedding_model,
                    K_init=config.K_init,
                    K_enhanced=config.K_enhanced,
                    prune_threshold=config.prune_threshold,
                    merge_distance=config.merge_distance,
                    embedding_cache=os.path.join(cache_dir, "embedding_cache.npz") if cache_dir else None,
                    device=device,
                )

                final_labels = self._assign_remaining_by_domain(
                    sample_indices, sample_labels, cluster_labels, n_total,
                )

                if token_counts is not None:
                    cluster_info = self._rebuild_cluster_info(final_labels, cluster_info, token_counts)

                return cluster_info, final_labels

        if texts is not None:
            loaded_texts = texts
        elif metadata_manager is not None:
            print("[EmbeddingCluster] Full dataset mode: streaming embedding (no text pre-load)")
        else:
            raise ValueError("EmbeddingCluster discovery requires texts or metadata_manager")

        if token_counts is None:
            if metadata_manager is not None:
                token_counts = metadata_manager.estimate_token_counts()
            elif texts is not None:
                from climbmix.utils.token_estimate import estimate_tokens_from_text
                token_counts = np.array(
                    [estimate_tokens_from_text(t) for t in loaded_texts], dtype=np.int64
                )

        cluster_info, final_labels = preprocess_pipeline(
            texts=loaded_texts if texts is not None or metadata_manager is None else None,
            metadata_manager=metadata_manager if texts is None else None,
            quality_scores=quality_scores,
            token_counts=token_counts,
            embedding_model=config.embedding_model,
            K_init=config.K_init,
            K_enhanced=config.K_enhanced,
            prune_threshold=config.prune_threshold,
            merge_distance=config.merge_distance,
            embedding_cache=os.path.join(cache_dir, "embedding_cache.npz") if cache_dir else None,
            device=device,
            embedding_truncate_len=config.embedding_truncate_len,
        )

        return cluster_info, final_labels

    @staticmethod
    def _assign_remaining_by_domain(
        sample_indices: npt.NDArray[np.int64],
        sample_labels: npt.NDArray[np.int64],
        domain_labels: Optional[npt.NDArray[np.int64]],
        n_total: int,
    ) -> npt.NDArray[np.int64]:
        if domain_labels is None:
            print("[EmbeddingCluster] No domain labels, assigning all to cluster 0")
            return np.zeros(n_total, dtype=np.int64)

        final_labels = np.full(n_total, -1, dtype=np.int64)
        final_labels[sample_indices] = sample_labels

        unique_domains = np.unique(domain_labels[domain_labels >= 0])
        K = int(sample_labels.max()) + 1

        for d in unique_domains:
            domain_mask = domain_labels == d
            domain_indices = np.where(domain_mask)[0]
            sampled_in_domain = domain_labels[sample_indices] == d

            if sampled_in_domain.sum() == 0:
                final_labels[domain_indices] = 0
                continue

            sampled_clusters = sample_labels[sampled_in_domain]
            valid_mask = sampled_clusters >= 0
            sampled_clusters = sampled_clusters[valid_mask]

            if len(sampled_clusters) == 0:
                final_labels[domain_indices] = 0
                continue

            unique_clusters, counts = np.unique(sampled_clusters, return_counts=True)

            unassigned = domain_indices[final_labels[domain_indices] == -1]
            n_unassigned = len(unassigned)
            if n_unassigned == 0:
                continue

            proportions = counts.astype(np.float64) / counts.sum()
            n_per_cluster = np.floor(proportions * n_unassigned).astype(int)
            remainder = n_unassigned - n_per_cluster.sum()
            order = np.argsort(-counts)
            for i in range(remainder):
                n_per_cluster[order[i % len(order)]] += 1

            rng = np.random.default_rng(42 + int(d))
            shuffled = unassigned.copy()
            rng.shuffle(shuffled)

            pos = 0
            for cid, n in zip(unique_clusters, n_per_cluster):
                final_labels[shuffled[pos:pos + n]] = int(cid)
                pos += n

        sampled_nan = np.zeros(n_total, dtype=bool)
        sampled_nan[sample_indices] = True
        nan_and_unassigned = (final_labels == -1) & (~sampled_nan)
        n_unassigned = int(nan_and_unassigned.sum())
        if n_unassigned > 0:
            print(f"[EmbeddingCluster] WARNING: {n_unassigned:,} docs unassigned, setting to cluster 0")
            final_labels[nan_and_unassigned] = 0

        n_excluded = int(((final_labels == -1) & sampled_nan).sum())
        if n_excluded > 0:
            print(f"[EmbeddingCluster] {n_excluded:,} sampled docs excluded (NaN/invalid embeddings)")

        n_clusters = len(np.unique(final_labels[final_labels >= 0]))
        print(f"[EmbeddingCluster] Assigned clusters to {n_total:,} docs "
              f"(sampled {len(sample_indices):,}, domain-mapped {n_total - len(sample_indices):,}), "
              f"{n_clusters} clusters used")
        return final_labels

    @staticmethod
    def _rebuild_cluster_info(
        final_labels: npt.NDArray[np.int64],
        old_cluster_info: list,
        token_counts: npt.NDArray[np.int64],
    ) -> list:
        from climbmix.core.types import ClusterInfo
        unique_ids = np.unique(final_labels[final_labels >= 0])
        clusters = []
        for cid in sorted(unique_ids):
            mask = final_labels == cid
            n_docs = int(mask.sum())
            n_tokens = int(token_counts[mask].sum())
            old = next((c for c in old_cluster_info if c.cluster_id == int(cid)), None)
            centroid = old.centroid if old else np.array([0.0])
            clusters.append(ClusterInfo(
                cluster_id=int(cid),
                centroid=centroid,
                num_docs=n_docs,
                num_tokens=n_tokens,
                label=f"C{int(cid)}",
            ))
        return clusters


class QualityClusterDiscovery:
    """
    Cluster by domain + quality scores — no text reading, no embedding.

    For each domain, sort docs by average quality score and divide into
    K_init // n_domains equal-sized sub-clusters. Then prune + merge
    as usual. This is instant (seconds) vs hours for embedding.
    """

    def __init__(self, config: Optional[ClusterDiscoveryConfig] = None):
        self._config = config or ClusterDiscoveryConfig()

    def discover(
        self,
        texts: Optional[list],
        cluster_labels: Optional[npt.NDArray[np.int64]],
        quality_scores: Optional[npt.NDArray[np.float64]],
        token_counts: Optional[npt.NDArray[np.int64]],
        metadata_manager: Optional[object],
        cache_dir: Optional[str] = None,
    ) -> Tuple[list, npt.NDArray[np.int64]]:
        from climbmix.core.cluster_merge import (
            compute_cluster_quality,
            prune_clusters,
            merge_clusters_by_distance,
            build_cluster_info,
        )

        config = self._config

        if cluster_labels is None:
            raise ValueError("quality_cluster requires cluster_labels (domain labels)")

        n_docs = len(cluster_labels)
        print(f"\n[QualityCluster] Clustering {n_docs:,} docs by domain + quality "
              f"(K_init={config.K_init}, K_enhanced={config.K_enhanced})")

        unique_domains = np.unique(cluster_labels[cluster_labels >= 0])
        n_domains = len(unique_domains)
        K_per_domain = max(1, config.K_init // n_domains)

        if quality_scores is not None and quality_scores.size > 0:
            avg_quality = quality_scores.mean(axis=1)
        else:
            avg_quality = np.zeros(n_docs, dtype=np.float64)

        labels = np.full(n_docs, -1, dtype=np.int64)
        centroids_list = []

        for d in unique_domains:
            domain_mask = cluster_labels == d
            domain_indices = np.where(domain_mask)[0]
            sorted_idx = domain_indices[np.argsort(avg_quality[domain_indices])]

            for k in range(K_per_domain):
                start = k * len(sorted_idx) // K_per_domain
                end = (k + 1) * len(sorted_idx) // K_per_domain
                group = sorted_idx[start:end]
                cluster_id = int(d) * K_per_domain + k
                labels[group] = cluster_id

                if quality_scores is not None and len(group) > 0:
                    centroid = quality_scores[group].mean(axis=0)
                else:
                    centroid = np.zeros(1)
                centroids_list.append(centroid)

        centroids = np.array(centroids_list, dtype=np.float32)
        n_clusters = len(np.unique(labels[labels >= 0]))
        print(f"[QualityCluster] Created {n_clusters} initial clusters "
              f"({n_domains} domains x {K_per_domain} quality tiers)")

        cluster_quality = compute_cluster_quality(
            labels, quality_scores, prune_threshold=config.prune_threshold,
        )

        pruned_labels, pruned_centroids, _ = prune_clusters(
            labels, centroids, cluster_quality, threshold=config.prune_threshold,
        )

        valid_mask = pruned_labels >= 0
        if valid_mask.sum() == 0:
            print("[QualityCluster] WARNING: All clusters pruned, using original labels")
            merged_labels = labels
            merged_centroids = centroids
        else:
            merged_labels, merged_centroids, _ = merge_clusters_by_distance(
                pruned_labels, pruned_centroids,
                merge_distance=config.merge_distance, target_K=config.K_enhanced,
            )

        if token_counts is None:
            token_counts = np.ones(n_docs, dtype=np.int64)

        cluster_info = build_cluster_info(merged_labels, merged_centroids, token_counts)

        return cluster_info, merged_labels


DISCOVERY_REGISTRY: Dict[str, ClusterDiscovery] = {
    "embedding_cluster": EmbeddingClusterDiscovery(),
    "quality_cluster": QualityClusterDiscovery(),
}


def get_discovery(method: str, config: Optional[ClusterDiscoveryConfig] = None) -> ClusterDiscovery:
    if method not in DISCOVERY_REGISTRY:
        raise ValueError(
            f"Unknown discovery method '{method}'. "
            f"Available: {list(DISCOVERY_REGISTRY.keys())}"
        )
    strategy = DISCOVERY_REGISTRY[method]
    if config is not None:
        strategy._config = config
    return strategy
