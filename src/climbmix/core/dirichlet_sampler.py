"""
CLIMB mixture weight sampling using Dirichlet distribution.

Paper (Section 3.1):
  "We initialize a Dirichlet distribution based on each cluster's
   token count and sample configurations."

  Each mixture configuration alpha = (alpha_1, ..., alpha_K) is sampled from
  Dir(alpha_0 * p_1, alpha_0 * p_2, ..., alpha_0 * p_K) where p_i is the
  proportional token count of cluster i, and alpha_0 controls
  the concentration around the proportions.

For iterative bootstrapping (Section 3.2):
  Iteration 1: sample from Dirichlet centered on natural proportions
  Iteration 2+: verbatim M-of-N draw from the predictor-ranked top-N
  (paper §2.2, handled in iterative_bootstrapper.py — no perturbation).
  sample_from_top_n (Dirichlet exploration around given bases) remains
  for the FINAL selection refinement around the predictor's argmax
  (see _search_full_design_space).
"""

import numpy as np
import numpy.typing as npt
from typing import List, Optional

from climbmix.core.types import MixtureConfig, MixtureWeights, CLIMBConfig


class DirichletSampler:

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
        alpha = self._concentration * self._proportions
        weights = self._rng.dirichlet(alpha)
        mw = MixtureWeights(weights=weights.astype(np.float64))
        return MixtureConfig(mixture_weights=mw)

    def sample_batch(self, n: int) -> List[MixtureConfig]:
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
        exploration_concentration: float = 5.0,
    ) -> List[MixtureConfig]:
        """
        Sample M configurations by Dirichlet exploration around top-N candidates.

        Paper: "randomly sample M new configurations from the top N ranked
        configurations." We implement this as Dirichlet sampling centered
        around each top-N config, which naturally stays on the simplex
        without needing clip-and-renormalize post-processing.

        For each selected top-N config with weights w, we sample from
        Dir(concentration * w) to generate nearby exploration configs.
        Higher concentration = closer to the original config.
        """
        if m <= len(top_n_configs):
            indices = self._rng.choice(len(top_n_configs), size=m, replace=False)
            base_configs = [top_n_configs[i] for i in indices]
        else:
            indices = self._rng.choice(len(top_n_configs), size=len(top_n_configs), replace=False)
            base_configs = [top_n_configs[i] for i in indices]

        new_configs = []
        for base in base_configs:
            alpha = exploration_concentration * base.mixture_weights.weights
            alpha = np.maximum(alpha, 0.01)
            new_weights = self._rng.dirichlet(alpha)
            mw = MixtureWeights(weights=new_weights.astype(np.float64))
            new_configs.append(MixtureConfig(mixture_weights=mw))

        if m > len(top_n_configs):
            remaining = m - len(top_n_configs)
            extras = self.sample_batch(remaining)
            new_configs.extend(extras)

        return new_configs
