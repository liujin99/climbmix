"""
Cluster discovery strategies.

Provides a registry to select between different cluster discovery
methods via config:
  - "embedding_cluster": embed + K-means + prune + merge
"""

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
    ) -> Tuple[list, npt.NDArray[np.int64]]:
        from climbmix.core.cluster_merge import preprocess_pipeline

        config = self._config

        if texts is not None:
            loaded_texts = texts
        elif metadata_manager is not None:
            print("[EmbeddingCluster] Loading texts from metadata_manager (full dataset)")
            all_indices = np.arange(metadata_manager.num_docs)
            loaded_texts = metadata_manager.read_texts(all_indices)
        else:
            raise ValueError("EmbeddingCluster discovery requires texts or metadata_manager")

        if token_counts is None:
            from climbmix.utils.token_estimate import estimate_tokens_from_text
            token_counts = np.array(
                [estimate_tokens_from_text(t) for t in loaded_texts], dtype=np.int64
            )

        cluster_info, final_labels = preprocess_pipeline(
            loaded_texts,
            quality_scores=quality_scores,
            token_counts=token_counts,
            embedding_model=config.embedding_model,
            K_init=config.K_init,
            K_enhanced=config.K_enhanced,
            prune_threshold=config.prune_threshold,
            merge_distance=config.merge_distance,
            device="cpu",
        )

        return cluster_info, final_labels


DISCOVERY_REGISTRY: Dict[str, ClusterDiscovery] = {
    "embedding_cluster": EmbeddingClusterDiscovery(),
}


def get_discovery(method: str, config: Optional[ClusterDiscoveryConfig] = None) -> ClusterDiscovery:
    if method not in DISCOVERY_REGISTRY:
        raise ValueError(
            f"Unknown discovery method '{method}'. "
            f"Available: {list(DISCOVERY_REGISTRY.keys())}"
        )
    strategy = DISCOVERY_REGISTRY[method]
    if method == "embedding_cluster" and config is not None:
        strategy._config = config
    return strategy
