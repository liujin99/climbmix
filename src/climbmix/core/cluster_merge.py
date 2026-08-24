"""
CLIMB Step 2: Cluster pruning and merging.

Paper details (Section 2.1, "Cluster merging"):
  1. Cluster-level pruning: remove low-quality clusters based on quality scores
     (threshold=3.0), retaining K_pruned clusters
  2. Merge clusters by centroid Euclidean distance (threshold=1.5)
     into K_enhanced < K_pruned < K_init clusters

Quality labels (configurable via config/quality_columns.yaml):
  STEM: stem_relevance, knowledge_value, notation_fidelity,
        rigor_coherence, noise_level (all 1-5, higher = better)
  FineWeb: qs_dclm, qs_fineweb_edu_approx, qs_english, ...
  Nemotron: qs_quality, qs_educational, qs_informational, qs_advertisement

If no quality labels are found, pruning is skipped and clusters are
merged directly to K_enhanced.

This produces the final cluster set D = {D_1, ..., D_K_enhanced}
that defines the data mixture search space.
"""

import time
import numpy as np
import numpy.typing as npt
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist

from climbmix.core.types import ClusterInfo


def compute_cluster_quality(
    cluster_labels: npt.NDArray[np.int64],
    quality_scores: Optional[npt.NDArray[np.float64]] = None,
    quality_columns: Optional[List[str]] = None,
    prune_threshold: float = 3.0,
) -> Dict[int, float]:
    """
    Compute per-cluster quality score for pruning.

    Quality scores are 1-5 discrete, higher = better (all dimensions
    including noise_level). Prune clusters with average quality < threshold.

    If quality_scores is None or all-zero, skip pruning (assign 5.0 to all).
    This handles data without quality labels gracefully.

    Args:
        cluster_labels: Per-document cluster labels.
        quality_scores: Per-document quality scores (num_docs, N).
                        If None or all-zero, no pruning is done.
        quality_columns: Names of quality criteria.
        prune_threshold: Quality threshold for cluster pruning.

    Returns:
        Dict mapping cluster_id -> average quality score.
    """
    cluster_quality: Dict[int, float] = {}

    if quality_scores is None or np.all(quality_scores == 0):
        unique_clusters = np.unique(cluster_labels)
        for c in unique_clusters:
            cluster_quality[int(c)] = 5.0
        if quality_scores is not None and np.all(quality_scores == 0):
            print("[ClusterQuality] All-zero scores detected, skipping pruning")
        return cluster_quality

    unique_clusters = np.unique(cluster_labels)
    for c in unique_clusters:
        mask = cluster_labels == c
        avg_quality = float(quality_scores[mask].mean())
        cluster_quality[int(c)] = avg_quality

    return cluster_quality


def prune_clusters(
    cluster_labels: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float32],
    cluster_quality: Dict[int, float],
    threshold: float = 3.0,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], Dict[int, int]]:
    """
    Prune low-quality clusters based on quality threshold.

    Args:
        cluster_labels: Per-document cluster labels (K_init clusters).
        centroids: Cluster centroids (K_init, dim).
        cluster_quality: Per-cluster quality scores.
        threshold: Minimum quality threshold.

    Returns:
        Tuple of (pruned_labels, pruned_centroids, old_to_new_id_map).
    """
    unique_clusters = np.unique(cluster_labels)
    kept_clusters = []
    for c in unique_clusters:
        if cluster_quality.get(int(c), 0.0) >= threshold:
            kept_clusters.append(int(c))

    n_pruned = len(unique_clusters) - len(kept_clusters)
    print(f"[Prune] Pruned {n_pruned}/{len(unique_clusters)} clusters "
          f"(threshold={threshold}), keeping {len(kept_clusters)}")

    old_to_new: Dict[int, int] = {}
    for new_id, old_id in enumerate(sorted(kept_clusters)):
        old_to_new[old_id] = new_id

    pruned_labels = np.full(len(cluster_labels), -1, dtype=np.int64)
    for old_id, new_id in old_to_new.items():
        mask = cluster_labels == old_id
        pruned_labels[mask] = new_id

    kept_indices = sorted(old_to_new.keys())
    pruned_centroids = centroids[kept_indices].copy()

    n_removed = int((pruned_labels == -1).sum())
    print(f"[Prune] Removed {n_removed} documents from pruned clusters")

    return pruned_labels, pruned_centroids, old_to_new


def merge_clusters_by_distance(
    cluster_labels: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.float32],
    merge_distance: float = 1.5,
    target_K: Optional[int] = None,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], Dict[int, int]]:
    """
    Merge similar clusters based on centroid Euclidean distance.

    Paper: merge clusters with centroid distance < threshold (1.5),
    iteratively merging the closest pair until target K is reached.

    Args:
        cluster_labels: Per-document cluster labels (after pruning).
        centroids: Cluster centroids.
        merge_distance: Maximum distance for merging.
        target_K: Target number of clusters after merging. If None,
                   merges all pairs within distance threshold.

    Returns:
        Tuple of (merged_labels, merged_centroids, merge_map).
    """
    K = len(np.unique(cluster_labels[cluster_labels >= 0]))
    if K == 0:
        return cluster_labels, centroids, {}

    print(f"[Merge] Starting with K={K} clusters, target_K={target_K}, "
          f"merge_distance={merge_distance}")

    unique_ids = sorted(np.unique(cluster_labels[cluster_labels >= 0]).tolist())
    current_centroids = centroids.copy()
    current_to_final: Dict[int, int] = {uid: uid for uid in unique_ids}
    cluster_groups: Dict[int, List[int]] = {uid: [uid] for uid in unique_ids}

    iteration = 0
    while True:
        active_ids = list(cluster_groups.keys())
        K_current = len(active_ids)

        if target_K is not None and K_current <= target_K:
            break

        active_centroids = current_centroids[active_ids]
        dist_matrix = cdist(active_centroids, active_centroids, metric='euclidean')
        np.fill_diagonal(dist_matrix, np.inf)

        min_idx = np.argmin(dist_matrix)
        i, j = divmod(min_idx, len(active_ids))
        min_dist = dist_matrix[i, j]

        if min_dist > merge_distance and target_K is None:
            break

        id_i = active_ids[i]
        id_j = active_ids[j]

        docs_i = int(np.sum(cluster_labels == id_i))
        docs_j = int(np.sum(cluster_labels == id_j))
        new_centroid = (current_centroids[id_i] * docs_i + current_centroids[id_j] * docs_j) / (docs_i + docs_j)

        merged_id = min(id_i, id_j)
        absorbed_id = max(id_i, id_j)

        cluster_groups[merged_id] = cluster_groups.pop(id_i) + cluster_groups.pop(id_j)

        current_centroids[merged_id] = new_centroid
        for old_id in cluster_groups[merged_id]:
            current_to_final[old_id] = merged_id

        iteration += 1
        if iteration % 50 == 0:
            print(f"[Merge] Iteration {iteration}: K={K_current - 1}, "
                  f"merged {id_i}+{id_j} (dist={min_dist:.3f})")

        if target_K is not None and K_current - 1 <= target_K:
            break

    final_ids = sorted(cluster_groups.keys())
    final_to_consecutive: Dict[int, int] = {}
    for new_id, final_id in enumerate(final_ids):
        final_to_consecutive[final_id] = new_id

    merged_labels = np.full(len(cluster_labels), -1, dtype=np.int64)
    for old_id, final_id in current_to_final.items():
        mask = cluster_labels == old_id
        new_id = final_to_consecutive.get(final_id, -1)
        merged_labels[mask] = new_id

    merged_centroid_list = [current_centroids[final_id] for final_id in final_ids]
    merged_centroids = np.array(merged_centroid_list, dtype=np.float32)

    print(f"[Merge] Final K={len(final_ids)} clusters from K_init={K}")

    return merged_labels, merged_centroids, {old: final_to_consecutive[final] for old, final in current_to_final.items()}


def build_cluster_info(
    merged_labels: npt.NDArray[np.int64],
    merged_centroids: npt.NDArray[np.float32],
    token_counts: Optional[npt.NDArray[np.int64]] = None,
) -> List[ClusterInfo]:
    """
    Build ClusterInfo objects from merged cluster results.
    """
    if token_counts is None:
        token_counts = np.ones(len(merged_labels), dtype=np.int64)

    unique_ids = np.unique(merged_labels[merged_labels >= 0])
    clusters: List[ClusterInfo] = []

    for cid in sorted(unique_ids):
        mask = merged_labels == cid
        n_docs = int(mask.sum())
        n_tokens = int(token_counts[mask].sum())
        clusters.append(ClusterInfo(
            cluster_id=int(cid),
            centroid=merged_centroids[int(cid)].astype(np.float64),
            num_docs=n_docs,
            num_tokens=n_tokens,
            label=f"C{int(cid)}",
        ))

    print(f"[ClusterInfo] Built {len(clusters)} clusters, "
          f"total docs={sum(c.num_docs for c in clusters)}, "
          f"total tokens={sum(c.num_tokens for c in clusters)}")

    return clusters


def preprocess_pipeline(
    texts: Optional[List[str]] = None,
    quality_scores: Optional[npt.NDArray[np.float64]] = None,
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    embedding_model: str = "NovaSearch/stella_en_400M_v5",
    K_init: int = 1000,
    K_enhanced: int = 21,
    prune_threshold: float = 3.0,
    merge_distance: float = 1.5,
    embedding_cache: Optional[str] = None,
    cluster_cache: Optional[str] = None,
    device: str = "cpu",
    metadata_manager: Optional[object] = None,
) -> Tuple[List[ClusterInfo], npt.NDArray[np.int64]]:
    """
    Full CLIMB preprocessing pipeline: embed → cluster → prune → merge → build info.

    Args:
        texts: Raw document texts (for subsampled mode).
        metadata_manager: If provided (and texts is None), stream-embeds
            all texts shard-by-shard without loading everything into memory.
        quality_scores: Per-document quality scores for pruning.
        token_counts: Per-document token counts.
        embedding_model: Sentence-transformer model name.
        K_init: Initial number of clusters.
        K_enhanced: Target number of clusters after merging.
        prune_threshold: Quality threshold for cluster pruning.
        merge_distance: Centroid distance threshold for merging.
        embedding_cache: Cache path for embeddings.
        cluster_cache: Cache path for cluster results.
        device: Device for embedding.

    Returns:
        Tuple of (cluster_info_list, final_cluster_labels).
    """
    from climbmix.core.embedding_cluster import embed_documents, embed_texts_streaming, cluster_embeddings

    print("\n" + "=" * 70)
    print("  CLIMB Preprocessing Pipeline")
    print("=" * 70)

    t0 = time.time()

    if texts is not None:
        embeddings = embed_documents(
            texts, model_name=embedding_model,
            cache_path=embedding_cache, device=device,
        )
    elif metadata_manager is not None:
        print("[Preprocess] Streaming mode: embedding shard-by-shard (no full text load)")
        embeddings = embed_texts_streaming(
            metadata_manager, model_name=embedding_model,
            cache_path=embedding_cache, device=device,
        )
    else:
        raise ValueError("Either texts or metadata_manager must be provided")

    cluster_labels, centroids = cluster_embeddings(
        embeddings, K_init=K_init, cache_path=cluster_cache,
    )

    cluster_quality = compute_cluster_quality(cluster_labels, quality_scores, prune_threshold=prune_threshold)

    pruned_labels, pruned_centroids, _ = prune_clusters(
        cluster_labels, centroids, cluster_quality, threshold=prune_threshold,
    )

    valid_mask = pruned_labels >= 0
    if valid_mask.sum() == 0:
        print("[Preprocess] WARNING: All clusters pruned, using original labels")
        merged_labels = cluster_labels
        merged_centroids = centroids
    else:
        merged_labels, merged_centroids, _ = merge_clusters_by_distance(
            pruned_labels, pruned_centroids,
            merge_distance=merge_distance, target_K=K_enhanced,
        )

    valid_mask = merged_labels >= 0
    if token_counts is None:
        if metadata_manager is not None:
            token_counts_full = metadata_manager.estimate_token_counts()
        elif texts is not None:
            token_counts_full = np.array([max(1, len(t) // 4) for t in texts], dtype=np.int64)
        else:
            token_counts_full = np.ones(len(merged_labels), dtype=np.int64)
    else:
        token_counts_full = token_counts

    cluster_info = build_cluster_info(merged_labels, merged_centroids, token_counts_full)

    elapsed = time.time() - t0
    print(f"\n[Preprocess] Complete in {elapsed:.1f}s")
    print(f"  K_init={K_init} → K_enhanced={len(cluster_info)}")
    print(f"  Total docs: {sum(c.num_docs for c in cluster_info)}")
    print(f"  Total tokens: {sum(c.num_tokens for c in cluster_info)}")

    return cluster_info, merged_labels
