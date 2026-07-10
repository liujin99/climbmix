"""
Data selection based on mixture weights.

CLIMB selects data proportionally from each cluster based on
the mixture weights α = (α₁, ..., α_K). This is simpler than
QuaDMix's quality-based sigmoid sampling.

For cluster k with weight α_k:
  - Sample α_k * total_target_tokens tokens from cluster k
  - Within each cluster, sample documents uniformly
    (no quality-based selection within clusters)
"""

import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple

from climbmix.core.types import MixtureWeights, ClusterInfo


def select_data_by_mixture(
    cluster_labels: npt.NDArray[np.int64],
    mixture_weights: MixtureWeights,
    token_counts: Optional[npt.NDArray[np.int64]] = None,
    target_tokens: int = 0,
    seed: int = 42,
) -> Tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """
    Select documents from clusters based on mixture weights.

    Args:
        cluster_labels: Per-document cluster labels.
        mixture_weights: Mixture weights α for each cluster.
        token_counts: Per-document token counts.
        target_tokens: Total target tokens. If 0, uses all available.
        seed: Random seed.

    Returns:
        Tuple of (selected_indices, per_document_sampling_probability).
    """
    rng = np.random.default_rng(seed)
    K = mixture_weights.num_clusters
    weights = mixture_weights.weights

    if token_counts is None:
        token_counts = np.ones(len(cluster_labels), dtype=np.int64)

    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    cluster_indices: dict = {}
    cluster_tokens_total: dict = {}

    for c in unique_clusters:
        mask = cluster_labels == c
        cluster_indices[int(c)] = np.where(mask)[0]
        cluster_tokens_total[int(c)] = int(token_counts[mask].sum())

    if target_tokens == 0:
        target_tokens = int(token_counts.sum())

    selected_all = []
    sampling_probs = np.zeros(len(cluster_labels), dtype=np.float64)

    for c in unique_clusters:
        c_int = int(c)
        if c_int >= K:
            continue
        indices = cluster_indices[c_int]
        if len(indices) == 0:
            continue

        alpha_k = weights[c_int]
        cluster_tokens = cluster_tokens_total[c_int]
        cluster_target_tokens = int(alpha_k * target_tokens)

        if cluster_target_tokens <= 0:
            continue

        if cluster_tokens <= cluster_target_tokens:
            selected_all.extend(indices.tolist())
            sampling_probs[indices] = 1.0
        else:
            c_token_counts = token_counts[indices]
            cum_tokens = np.cumsum(c_token_counts)
            cutoff = np.searchsorted(cum_tokens, cluster_target_tokens, side='left')
            cutoff = min(cutoff + 1, len(indices))
            selected_all.extend(indices[:cutoff].tolist())
            sampling_probs[indices[:cutoff]] = alpha_k * target_tokens / cluster_tokens

    if len(selected_all) == 0:
        selected_all = rng.choice(len(cluster_labels), size=min(100, len(cluster_labels)), replace=False).tolist()

    selected_indices = np.array(selected_all, dtype=np.int64)
    return selected_indices, sampling_probs


def compute_mixture_dataset_stats(
    cluster_labels: npt.NDArray[np.int64],
    selected_indices: npt.NDArray[np.int64],
    cluster_info: List[ClusterInfo],
    mixture_weights: MixtureWeights,
) -> dict:
    """
    Compute statistics about the selected dataset.
    """
    K = mixture_weights.num_clusters
    weights = mixture_weights.weights

    orig_dist = np.bincount(cluster_labels[cluster_labels >= 0], minlength=K)
    sel_dist = np.bincount(
        cluster_labels[selected_indices][cluster_labels[selected_indices] >= 0],
        minlength=K,
    )

    stats = {
        "num_original_docs": len(cluster_labels),
        "num_selected_docs": len(selected_indices),
        "original_distribution": orig_dist.tolist(),
        "selected_distribution": sel_dist.tolist(),
        "mixture_weights": weights.tolist(),
        "cluster_labels": [c.label for c in cluster_info],
    }

    print("\n  Mixture selection summary:")
    print(f"    Original docs: {len(cluster_labels):,}")
    print(f"    Selected docs: {len(selected_indices):,}")
    for i, c in enumerate(cluster_info):
        if i < K:
            ratio = sel_dist[i] / max(1, orig_dist[i])
            print(f"    [{i}] {c.label:>5s}: {orig_dist[i]:>7,} → {sel_dist[i]:>7,} "
                  f"(α={weights[i]:.4f}, ratio={ratio:.2f})")

    return stats
