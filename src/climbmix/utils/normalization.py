"""
Normalization utilities (adapted from quadmix).
"""

import numpy as np
import numpy.typing as npt
from scipy import stats as sp_stats
from typing import Callable


def zscore_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    mean = np.mean(scores)
    std = np.std(scores)
    if std < 1e-10:
        return np.zeros_like(scores)
    return (scores - mean) / std


def minmax_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    s_min, s_max = np.min(scores), np.max(scores)
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def rank_normalize(scores: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    n = len(scores)
    if n == 0:
        return scores
    ranks = sp_stats.rankdata(scores, method='average')
    return (ranks - 1).astype(np.float64) / n


NORMALIZATION_REGISTRY: dict[str, Callable] = {
    "zscore": zscore_normalize,
    "minmax": minmax_normalize,
    "rank": rank_normalize,
}


def get_normalizer(name: str = "rank") -> Callable:
    if name not in NORMALIZATION_REGISTRY:
        raise ValueError(f"Unknown normalizer '{name}'. Available: {list(NORMALIZATION_REGISTRY.keys())}")
    return NORMALIZATION_REGISTRY[name]
