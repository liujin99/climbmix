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
    depth: int = 10
    num_iterations: Optional[int] = None
    ratio: Optional[float] = None
    phase1_checkpoint_path: Optional[str] = None
    validation_metric: str = "accuracy"
    lr_scale: float = 1.0
    warmup: float = 0.0
    warmdown: float = 0.9

    DEPTH_INFO = {
        4:  {"scaling_M": 3.2,   "total_M": 8.2,    "n_embd": 256,  "n_head": 2},
        6:  {"scaling_M": 17.6,  "total_M": 41.5,   "n_embd": 384,  "n_head": 3},
        8:  {"scaling_M": 40.4,  "total_M": 93.8,   "n_embd": 512,  "n_head": 4},
        10: {"scaling_M": 70.2,  "total_M": 196.0,  "n_embd": 640,  "n_head": 5},
        12: {"scaling_M": 110.1, "total_M": 286.3,  "n_embd": 768,  "n_head": 6},
        14: {"scaling_M": 164.2, "total_M": 399.1,  "n_embd": 896,  "n_head": 7},
        16: {"scaling_M": 234.9, "total_M": 536.9,  "n_embd": 1024, "n_head": 8},
        18: {"scaling_M": 324.4, "total_M": 701.9,  "n_embd": 1152, "n_head": 9},
        20: {"scaling_M": 435.2, "total_M": 896.5,  "n_embd": 1280, "n_head": 10},
        22: {"scaling_M": 569.5, "total_M": 1123.2, "n_embd": 1408, "n_head": 11},
        24: {"scaling_M": 729.8, "total_M": 1384.1, "n_embd": 1536, "n_head": 12},
    }

    @property
    def scaling_params(self) -> int:
        info = self.DEPTH_INFO.get(self.depth)
        if info is None:
            raise ValueError(f"Unsupported depth={self.depth}. Valid: {sorted(self.DEPTH_INFO.keys())}")
        return int(info["scaling_M"] * 1_000_000)

    @property
    def total_params(self) -> int:
        info = self.DEPTH_INFO.get(self.depth)
        if info is None:
            raise ValueError(f"Unsupported depth={self.depth}. Valid: {sorted(self.DEPTH_INFO.keys())}")
        return int(info["total_M"] * 1_000_000)

    @property
    def model_tag(self) -> str:
        return f"d{self.depth}"

    @property
    def scaling_M(self) -> float:
        info = self.DEPTH_INFO.get(self.depth)
        if info is None:
            raise ValueError(f"Unsupported depth={self.depth}. Valid: {sorted(self.DEPTH_INFO.keys())}")
        return info["scaling_M"]

    @property
    def total_M(self) -> float:
        info = self.DEPTH_INFO.get(self.depth)
        if info is None:
            raise ValueError(f"Unsupported depth={self.depth}. Valid: {sorted(self.DEPTH_INFO.keys())}")
        return info["total_M"]

    @property
    def training_iterations(self) -> int:
        if self.num_iterations is not None:
            return self.num_iterations
        if self.ratio is not None:
            return max(1, int(self.ratio * self.scaling_params / 500_000))
        return 500


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
class TargetConfig:
    depth: int = 24
    num_iterations: Optional[int] = None
    ratio: Optional[float] = None
    phase1_checkpoint_path: Optional[str] = None
    lr_scale: float = 1.0
    warmup: float = 0.0
    warmdown: float = 0.9

    @property
    def model_tag(self) -> str:
        return f"d{self.depth}"

    @property
    def training_iterations(self) -> int:
        if self.num_iterations is not None:
            return self.num_iterations
        if self.ratio is not None:
            return max(1, int(self.ratio * ProxyConfig.DEPTH_INFO[self.depth]["scaling_M"] * 1_000_000 / 500_000))
        return 1000


@dataclass
class DeviceConfig:
    device_type: str = "cpu"
    npu_device_id: int = 0
    npu_devices: int = 8


HIGH_SIGNAL_TASKS = [
    "piqa", "arc_easy", "lambada_openai",
    "commonsense_qa", "squad", "coqa",
]


@dataclass
class CLIMBConfig:
    discovery: ClusterDiscoveryConfig = field(default_factory=ClusterDiscoveryConfig)
    filtering: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    val_tasks: List[str] = field(default_factory=lambda: HIGH_SIGNAL_TASKS.copy())
    data_dir: str = "./data"
    output_dir: str = "./climbmix_output"
    nanochat_dir: str = "/home/liujin99/nanochat-npu"

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
