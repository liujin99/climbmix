"""
Core type definitions for the CLIMB framework.

Data types: ClusterInfo, MixtureWeights, MixtureConfig, ProxyResult, IterationResult
Layered config: ClusterDiscoveryConfig, QualityFilterConfig, SearchConfig,
                ProxyConfig, PredictorConfig, DeviceConfig, CLIMBConfig
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import numpy.typing as npt


@dataclass
class ClusterInfo:
    cluster_id: int
    centroid: npt.NDArray[np.float64]
    num_docs: int
    num_tokens: int = 0
    label: str = ""
    quality_score: float = 0.0

    @property
    def token_fraction(self) -> float:
        if self.num_tokens == 0:
            return 0.0
        return self.num_tokens / max(1, self.num_docs)


@dataclass
class MixtureWeights:
    weights: npt.NDArray[np.float64]

    @property
    def num_clusters(self) -> int:
        return len(self.weights)

    def normalize(self) -> "MixtureWeights":
        total = self.weights.sum()
        if total < 1e-10:
            return MixtureWeights(weights=np.ones(len(self.weights)) / len(self.weights))
        return MixtureWeights(weights=self.weights / total)

    def to_dict(self, cluster_labels: Optional[List[str]] = None) -> Dict[str, float]:
        labels = cluster_labels or [f"C{i}" for i in range(len(self.weights))]
        return {labels[i]: float(self.weights[i]) for i in range(len(self.weights))}

    @classmethod
    def from_dict(cls, d: Dict[str, float], cluster_labels: Optional[List[str]] = None) -> "MixtureWeights":
        if cluster_labels is None:
            cluster_labels = sorted(d.keys())
        weights = np.array([d[label] for label in cluster_labels], dtype=np.float64)
        return cls(weights=weights)


@dataclass
class MixtureConfig:
    mixture_weights: MixtureWeights
    config_id: int = 0

    def flatten(self) -> npt.NDArray[np.float64]:
        return self.mixture_weights.weights.copy()

    @classmethod
    def from_flattened(cls, array: npt.NDArray[np.float64], config_id: int = 0) -> "MixtureConfig":
        mw = MixtureWeights(weights=array)
        return cls(mixture_weights=mw.normalize(), config_id=config_id)

    @property
    def num_clusters(self) -> int:
        return self.mixture_weights.num_clusters


@dataclass
class ProxyResult:
    mixture_config: MixtureConfig
    validation_loss: float
    validation_accuracy: float = 0.0
    per_task_accuracies: Optional[Dict[str, float]] = None
    per_task_losses: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if self.validation_accuracy > 0:
            return self.validation_accuracy
        return -self.validation_loss


@dataclass
class IterationResult:
    iteration: int
    n_configs: int
    n_trained: int
    predictor: Optional[Any] = None
    predictor_r2: Optional[float] = None
    best_config: Optional[MixtureConfig] = None
    best_score: Optional[float] = None
    best_loss: Optional[float] = None
    all_configs: List[MixtureConfig] = field(default_factory=list)
    all_scores: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    all_losses: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    selected_for_training: List[int] = field(default_factory=list)


# ── Layered configuration ──────────────────────────────────────

@dataclass
class ClusterDiscoveryConfig:
    method: str = "fdc_labels"
    K_init: int = 1000
    K_enhanced: int = 21
    embedding_model: str = "NovaSearch/stella_en_400M_v5"
    embedding_truncate_len: int = 512
    prune_threshold: float = 3.0
    merge_distance: float = 1.5
    fdc_num_domains: int = 22

    VALID_METHODS = ("fdc_labels", "embedding_cluster")


@dataclass
class QualityFilterConfig:
    method: str = "none"
    doc_english_min: float = 0.3
    doc_composite_min: float = 0.5
    cluster_avg_threshold: float = 3.0
    quality_columns: List[str] = field(default_factory=lambda: [
        "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
        "qs_eai_general_math", "qs_eai_open_web_math",
    ])

    VALID_METHODS = ("none", "doc_level", "cluster_level", "doc_and_cluster")


@dataclass
class SearchConfig:
    num_iterations: int = 3
    configs_per_iter: List[int] = field(default_factory=lambda: [64, 32, 16])
    dirichlet_alpha: Optional[float] = None
    predict_top_n_ratio: float = 0.5
    sample_from_top_m: int = 32

    @property
    def total_configs(self) -> int:
        return sum(self.configs_per_iter)


@dataclass
class ProxyConfig:
    model_size: str = "62M"
    training_steps: int = 1000
    training_tokens: int = 40_000_000_000
    batch_tokens: int = 2_000_000
    micro_batch_size: int = 8
    learning_rate: float = 5e-5
    decay_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    block_size: int = 2048
    lr_schedule: str = "wsd"
    warmup_fraction: float = 0.04
    stable_fraction: float = 0.76
    decay_fraction: float = 0.20
    phase1_checkpoint_path: Optional[str] = None
    validation_metric: str = "accuracy"

    VALID_SIZES = ("1M", "5M", "20M", "62M", "132M", "350M")
    VALID_LR_SCHEDULES = ("wsd", "cosine", "linear")
    VALID_VALIDATION_METRICS = ("accuracy", "loss")

    SIZE_PARAMS = {
        "1M": {"batch_tokens": 500_000, "learning_rate": 1e-3, "decay_learning_rate": 1e-4, "training_tokens": 1_000_000_000},
        "5M": {"batch_tokens": 1_000_000, "learning_rate": 5e-4, "decay_learning_rate": 5e-5, "training_tokens": 5_000_000_000},
        "20M": {"batch_tokens": 1_000_000, "learning_rate": 2e-4, "decay_learning_rate": 2e-5, "training_tokens": 10_000_000_000},
        "62M": {"batch_tokens": 1_000_000, "learning_rate": 1e-4, "decay_learning_rate": 1e-5, "training_tokens": 20_000_000_000},
        "132M": {"batch_tokens": 2_000_000, "learning_rate": 8e-5, "decay_learning_rate": 8e-6, "training_tokens": 30_000_000_000},
        "350M": {"batch_tokens": 2_000_000, "learning_rate": 5e-5, "decay_learning_rate": 1e-5, "training_tokens": 40_000_000_000},
    }

    def apply_size_defaults(self) -> "ProxyConfig":
        if self.model_size in self.SIZE_PARAMS:
            p = self.SIZE_PARAMS[self.model_size]
            self.batch_tokens = p["batch_tokens"]
            self.learning_rate = p["learning_rate"]
            self.decay_learning_rate = p["decay_learning_rate"]
            self.training_tokens = p["training_tokens"]
        return self


@dataclass
class PredictorConfig:
    method: str = "lightgbm"
    l1_reg: float = 1.0
    l2_reg: float = 1.0
    max_depth: int = 4
    min_samples_leaf: int = 5
    n_estimators: int = 500
    early_stopping_rounds: int = 20
    learning_rate: float = 0.02

    VALID_METHODS = ("lightgbm",)


@dataclass
class DeviceConfig:
    device_type: str = "cpu"
    npu_device_id: int = 0
    npu_devices: int = 8


@dataclass
class CLIMBConfig:
    discovery: ClusterDiscoveryConfig = field(default_factory=ClusterDiscoveryConfig)
    filtering: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    val_tasks: List[str] = field(default_factory=lambda: ["piqa", "arc_e", "hellaswag"])
    data_dir: str = "./data"
    output_dir: str = "./climbmix_output"

    @property
    def metric_direction(self) -> str:
        if self.proxy.validation_metric == "accuracy":
            return "maximize"
        return "minimize"

    def get_dirichlet_concentration(self, cluster_token_counts: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        if self.search.dirichlet_alpha is not None:
            return np.full(len(cluster_token_counts), self.search.dirichlet_alpha, dtype=np.float64)
        total = cluster_token_counts.sum()
        if total < 1:
            return np.ones(len(cluster_token_counts), dtype=np.float64)
        return cluster_token_counts.astype(np.float64) / total * len(cluster_token_counts)
