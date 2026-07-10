"""
Strategy protocol definitions for the CLIMB framework.

Each protocol defines the interface for a swappable component.
Implementations are registered in their respective modules
and selected via CLIMBConfig.
"""

from typing import Dict, List, Optional, Tuple, Protocol, runtime_checkable
import numpy as np
import numpy.typing as npt

from climbmix.core.types import (
    ClusterInfo,
    MixtureConfig,
    MixtureWeights,
)


@runtime_checkable
class ClusterDiscovery(Protocol):
    def discover(
        self,
        texts: Optional[List[str]],
        cluster_labels: Optional[npt.NDArray[np.int64]],
        quality_scores: Optional[npt.NDArray[np.float64]],
        token_counts: Optional[npt.NDArray[np.int64]],
        metadata_manager: Optional[object],
    ) -> Tuple[List[ClusterInfo], npt.NDArray[np.int64]]:
        ...


@runtime_checkable
class QualityFilter(Protocol):
    def filter(
        self,
        cluster_labels: npt.NDArray[np.int64],
        quality_scores: npt.NDArray[np.float64],
        config: "QualityFilterConfig",
    ) -> Tuple[npt.NDArray[np.int64], Dict[int, float]]:
        ...


@runtime_checkable
class MixtureSampler(Protocol):
    def sample_one(self) -> MixtureConfig: ...
    def sample_batch(self, n: int) -> List[MixtureConfig]: ...
    def sample_from_top_n(self, top_configs: List[MixtureConfig], m: int) -> List[MixtureConfig]: ...


@runtime_checkable
class Predictor(Protocol):
    def fit(
        self,
        configs: List[MixtureConfig],
        losses: npt.NDArray[np.float64],
        val_configs: Optional[List[MixtureConfig]] = None,
        val_losses: Optional[npt.NDArray[np.float64]] = None,
    ) -> "Predictor": ...
    def predict(self, configs: List[MixtureConfig]) -> npt.NDArray[np.float64]: ...
    def predict_and_rank(self, configs: List[MixtureConfig]) -> List[int]: ...
    def score(self, configs: List[MixtureConfig], losses: npt.NDArray[np.float64]) -> float: ...
