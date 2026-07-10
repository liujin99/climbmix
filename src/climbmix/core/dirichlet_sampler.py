"""
CLIMB mixture weight sampling using Dirichlet distribution.

Paper (Section 3.1):
  "We initialize a Dirichlet distribution based on each cluster's
   token count and sample configurations."

  Each mixture configuration α = (α₁, ..., α_K) is sampled from
  Dir(α₀ * p₁, α₀ * p₂, ..., α₀ * p_K) where p_i is the
  proportional token count of cluster i, and α₀ controls
  the concentration around the proportions.
"""

import numpy as np
import numpy.typing as npt
from typing import List, Optional

from climbmix.core.types import MixtureConfig, MixtureWeights, CLIMBConfig


class DirichletSampler:
    """
    Samples mixture configurations from Dirichlet distribution.

    In CLIMB, the Dirichlet concentration is set proportional to
    each cluster's token count, encouraging sampling around
    natural cluster proportions while allowing exploration.
    """

    def __init__(
        self,
        num_clusters: int,
        cluster_token_counts: npt.NDArray[np.int64],
        config: Optional[CLIMBConfig] = None,
        seed: int = 42,
    ):
        self.num_clusters = num_clusters
        self.cluster_token_counts = cluster_token_counts
        self.config = config or CLIMBConfig()
        self._rng = np.random.default_rng(seed)

        total = cluster_token_counts.sum()
        if total < 1:
            self._proportions = np.ones(num_clusters, dtype=np.float64) / num_clusters
        else:
            self._proportions = cluster_token_counts.astype(np.float64) / total

        self._concentration = self.config.get_dirichlet_concentration(cluster_token_counts)

    def sample_one(self) -> MixtureConfig:
        """Sample a single mixture configuration."""
        alpha = self._concentration * self._proportions
        weights = self._rng.dirichlet(alpha)
        mw = MixtureWeights(weights=weights.astype(np.float64))
        return MixtureConfig(mixture_weights=mw)

    def sample_batch(self, n: int) -> List[MixtureConfig]:
        """Sample n mixture configurations."""
        alpha = self._concentration * self._proportions
        all_weights = self._rng.dirichlet(alpha, size=n)
        configs = []
        for i in range(n):
            mw = MixtureWeights(weights=all_weights[i].astype(np.float64))
            configs.append(MixtureConfig(mixture_weights=mw, config_id=i))
        return configs

    def sample_from_top_n(
        self,
        top_n_configs: List[MixtureConfig],
        m: int,
        jitter_scale: float = 0.1,
    ) -> List[MixtureConfig]:
        """
        Sample M configurations near the top-N candidates.

        Used in iterative bootstrapping: add small jitter to
        promising configs to explore nearby region.
        """
        if m <= len(top_n_configs):
            indices = self._rng.choice(len(top_n_configs), size=m, replace=False)
            selected = [top_n_configs[i] for i in indices]
        else:
            selected = top_n_configs.copy()
            remaining = m - len(top_n_configs)
            extra = self.sample_batch(remaining)
            selected.extend(extra)

        jittered = []
        for config in selected:
            jitter = self._rng.normal(0, jitter_scale, size=self.num_clusters)
            new_weights = config.mixture_weights.weights + jitter
            new_weights = np.maximum(new_weights, 0.01)
            mw = MixtureWeights(weights=new_weights)
            jittered.append(MixtureConfig(mixture_weights=mw.normalize()))

        return jittered
