"""
Core type definitions for the CLIMB framework.

Data types: ClusterInfo, MixtureWeights, MixtureConfig, ProxyResult, IterationResult
Layered config: ClusterDiscoveryConfig, QualityFilterConfig, SearchConfig,
                ProxyConfig, PredictorConfig, DeviceConfig, CLIMBConfig
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import glob
import json
import os
import numpy as np
import numpy.typing as npt


# ── Auto-detect model info from checkpoint meta_*.json ─────────

DEFAULT_NANOCHAT_BASE_DIR = "/home/ma-user/work/nanochat_model_dir"


def _approx_scaling_params(model_config: dict) -> Optional[int]:
    """Approximate scaling params from model config dimensions.

    Ported from quadmix/nanochat_mid_compare/get_model_info.py.
    """
    n_embd = model_config.get("n_embd")
    n_layer = model_config.get("n_layer", 24)
    n_head = model_config.get("n_head")
    vocab_size = model_config.get("vocab_size", 32768)
    head_dim = model_config.get("head_dim", 128)
    aspect_ratio = model_config.get("aspect_ratio", 64)

    if n_embd is None:
        base_dim = n_layer * aspect_ratio
        n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
        if n_head is None:
            n_head = n_embd // head_dim
    else:
        if n_head is None:
            n_head = n_embd // head_dim

    n_kv_head = model_config.get("n_kv_head", n_head)
    pad_vocab_size_to = model_config.get("pad_vocab_size_to", 64)
    padded_vocab_size = ((vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to

    ve_gate_channels = model_config.get("ve_gate_channels", 12)
    has_ve_count = 0
    for layer_idx in range(n_layer):
        if layer_idx % 2 == (n_layer - 1) % 2:
            has_ve_count += 1

    transformer_matrices = 0
    for _ in range(n_layer):
        transformer_matrices += (
            n_embd * n_head * head_dim +
            n_embd * n_kv_head * head_dim +
            n_embd * n_kv_head * head_dim +
            n_embd * n_embd +
            n_embd * 4 * n_embd +
            4 * n_embd * n_embd
        )
    transformer_matrices += has_ve_count * ve_gate_channels * n_kv_head

    lm_head = padded_vocab_size * n_embd
    return transformer_matrices + lm_head


def auto_detect_depth_info(checkpoint_dir: Optional[str], depth: int) -> Optional[dict]:
    """Auto-detect model info from checkpoint meta_*.json.

    Three-tier fallback:
      1. meta_*.json -> GPTConfig -> num_scaling_params() (exact, needs nanochat)
      2. meta_*.json -> _approx_scaling_params() (approximate, no nanochat needed)
      3. DEPTH_INFO table lookup (fallback)

    Returns dict with keys: scaling_M, total_M, n_embd, n_head, n_layer
    or None if all methods fail.
    """
    if checkpoint_dir:
        meta_files = sorted(glob.glob(os.path.join(checkpoint_dir, "meta_*.json")))
        if meta_files:
            try:
                with open(meta_files[-1]) as f:
                    meta = json.load(f)
                model_config = meta.get("model_config", {})
                total_batch_size = meta.get("total_batch_size")

                n_layer = model_config.get("n_layer", depth)
                n_embd = model_config.get("n_embd")
                n_head = model_config.get("n_head")

                if n_embd is None:
                    aspect_ratio = model_config.get("aspect_ratio", 64)
                    head_dim = model_config.get("head_dim", 128)
                    base_dim = n_layer * aspect_ratio
                    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
                if n_head is None:
                    head_dim = model_config.get("head_dim", 128)
                    n_head = n_embd // head_dim

                # Tier 1: exact via nanochat GPTConfig
                try:
                    import torch
                    from nanochat.gpt import GPT, GPTConfig
                    config = GPTConfig(**model_config)
                    with torch.device("meta"):
                        model = GPT(config)
                    params_counts = model.num_scaling_params()
                    num_scaling = params_counts["transformer_matrices"] + params_counts["lm_head"]
                    total = sum(params_counts.values())
                    return {
                        "scaling_M": num_scaling / 1_000_000,
                        "total_M": total / 1_000_000,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                    }
                except Exception:
                    pass

                # Tier 2: approximate formula
                num_scaling = _approx_scaling_params(model_config)
                if num_scaling is not None:
                    return {
                        "scaling_M": num_scaling / 1_000_000,
                        "total_M": num_scaling / 1_000_000,
                        "n_embd": n_embd,
                        "n_head": n_head,
                        "n_layer": n_layer,
                    }
            except Exception:
                pass

    # Tier 3: DEPTH_INFO fallback
    return _DEPTH_INFO.get(depth)


_DEPTH_INFO = {
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
    validation_nll: float = 0.0
    per_task_accuracies: Optional[Dict[str, float]] = None
    per_task_losses: Optional[Dict[str, float]] = None
    per_task_nlls: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        return self.validation_accuracy


@dataclass
class IterationResult:
    iteration: int
    n_configs: int
    n_trained: int
    predictor: Optional[Any] = None
    predictor_r2: Optional[float] = None
    predictor_spearman: Optional[float] = None
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
    method: str = "embedding_cluster"
    K_init: int = 1000
    # Cluster-count band: K_final = clamp(natural_K(merge_distance), K_enhanced, K_max).
    # K_enhanced is the FLOOR — a permissive safety bound (default 3) against
    # degenerate collapse on coarse pools, so the pool's natural structure is
    # followed down to very few clusters; set it to the paper's fixed
    # K_enhanced (e.g. 10 or 20) for paper-faithful semantics. K_max is the
    # CAP (search-budget bound). Inside the band the distance guard refuses
    # to merge semantically distinct clusters; beyond K_max closest-pair
    # forced merges keep heterogeneous pools within budget (see cluster_merge.py).
    K_enhanced: int = 3
    K_max: int = 15
    embedding_model: str = "NovaSearch/stella_en_400M_v5"
    embedding_truncate_len: int = 512
    embedding_device: str = "cpu"
    embedding_sample_size: int = 0
    prune_threshold: float = 3.0
    # Merge legality threshold (tau) on unit-normalized embeddings:
    # d^2 = 2(1 - cos), so 0.9 ~ cosine similarity 0.6. NOT from the paper
    # (paper merges to fixed K_enhanced regardless of distance) — deliberate
    # deviation to prevent forced merges of semantically distinct clusters.
    merge_distance: float = 0.9

    VALID_METHODS = ("embedding_cluster", "quality_cluster")


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
    configs_per_iter: List[int] = field(default_factory=lambda: [15, 8, 4])
    dirichlet_alpha: Optional[float] = None
    predict_top_n_ratio: float = 0.5
    sample_from_top_m: int = 32
    w_floor: float = 0.0

    @property
    def total_configs(self) -> int:
        return sum(self.configs_per_iter)


@dataclass
class ProxyConfig:
    depth: int = 20
    num_iterations: Optional[int] = None
    ratio: Optional[float] = None
    phase1_checkpoint_path: Optional[str] = None
    validation_metric: str = "accuracy"
    lr_scale: float = 1.0
    warmup: float = 0.0
    warmdown: float = 0.9
    target_tokens: int = 0

    DEPTH_INFO = _DEPTH_INFO

    def _get_info(self) -> Optional[dict]:
        """Try auto_detect from checkpoint, fallback to DEPTH_INFO table."""
        info = auto_detect_depth_info(self.phase1_checkpoint_path, self.depth)
        if info is not None:
            return info
        return self.DEPTH_INFO.get(self.depth)

    @property
    def scaling_params(self) -> int:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return int(info["scaling_M"] * 1_000_000)

    @property
    def total_params(self) -> int:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return int(info["total_M"] * 1_000_000)

    @property
    def model_tag(self) -> str:
        return f"d{self.depth}"

    @property
    def scaling_M(self) -> float:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return info["scaling_M"]

    @property
    def total_M(self) -> float:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return info["total_M"]

    @property
    def training_iterations(self) -> int:
        if self.num_iterations is not None:
            return self.num_iterations
        if self.ratio is not None:
            return max(1, int(self.ratio * self.scaling_params / 500_000))
        return 1000


@dataclass
class PredictorConfig:
    method: str = "lightgbm"
    l1_reg: float = 1.0
    l2_reg: float = 1.0
    max_depth: int = 3
    min_samples_leaf: int = 3
    n_estimators: int = 500
    early_stopping_rounds: int = 20
    learning_rate: float = 0.02
    auto_adjust: bool = True

    VALID_METHODS = ("lightgbm",)

    def get_adjusted_params(self, n_samples: int, n_features: int) -> dict:
        """
        Auto-adjust max_depth and min_samples_leaf based on N and feature count.

        Formula:
          max_depth = min(4, max(2, int(log2(N / n_features)) + 2))
          min_samples_leaf = max(3, min(5, N // 20))

        Reference:
          N=27,  k=10 → max_depth=3, min_samples_leaf=3
          N=35,  k=10 → max_depth=3, min_samples_leaf=3
          N=112, k=21 → max_depth=4, min_samples_leaf=5 (matches paper)
        """
        if not self.auto_adjust:
            return {"max_depth": self.max_depth, "min_samples_leaf": self.min_samples_leaf}

        import math
        ratio = max(n_samples / max(n_features, 1), 2.0)
        adj_max_depth = min(4, max(2, int(math.log2(ratio)) + 2))
        adj_min_samples = max(3, min(5, n_samples // 20))
        return {"max_depth": adj_max_depth, "min_samples_leaf": adj_min_samples}


@dataclass
class TargetConfig:
    depth: int = 28
    num_iterations: Optional[int] = None
    ratio: Optional[float] = None
    phase1_checkpoint_path: Optional[str] = None
    lr_scale: float = 1.0
    warmup: float = 0.0
    warmdown: float = 0.9
    target_tokens: int = 0

    DEPTH_INFO = _DEPTH_INFO

    def _get_info(self) -> Optional[dict]:
        """Try auto_detect from checkpoint, fallback to DEPTH_INFO table."""
        info = auto_detect_depth_info(self.phase1_checkpoint_path, self.depth)
        if info is not None:
            return info
        return self.DEPTH_INFO.get(self.depth)

    @property
    def model_tag(self) -> str:
        return f"d{self.depth}"

    @property
    def scaling_params(self) -> int:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return int(info["scaling_M"] * 1_000_000)

    @property
    def total_params(self) -> int:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return int(info["total_M"] * 1_000_000)

    @property
    def scaling_M(self) -> float:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return info["scaling_M"]

    @property
    def total_M(self) -> float:
        info = self._get_info()
        if info is None:
            raise ValueError(f"Cannot determine model info for depth={self.depth}. "
                             f"Set phase1_checkpoint_path or use a supported depth.")
        return info["total_M"]

    @property
    def training_iterations(self) -> int:
        if self.num_iterations is not None:
            return self.num_iterations
        if self.ratio is not None:
            return max(1, int(self.ratio * self.scaling_params / 500_000))
        return 1000


@dataclass
class DeviceConfig:
    device_type: str = "npu"
    npu_device_id: int = 0
    npu_devices: int = 8


STEM_BENCHMARK_LABELS = [
    "arc_easy", "arc_challenge", "mmlu_stem",
    "gpqa_diamond", "gsm8k_cot", "math_cot_500",
]

BENCHMARK_SIZES = {
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "mmlu_stem": 3545,
    "gpqa_diamond": 198,
    "gsm8k_cot": 1319,
    "math_cot_500": 500,
}


@dataclass
class CLIMBConfig:
    discovery: ClusterDiscoveryConfig = field(default_factory=ClusterDiscoveryConfig)
    filtering: QualityFilterConfig = field(default_factory=QualityFilterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    val_tasks: List[str] = field(default_factory=lambda: STEM_BENCHMARK_LABELS.copy())
    data_dir: str = "./data"
    output_dir: str = "./climbmix_output"
    nanochat_dir: str = "/home/liujin99/nanochat-npu"
    nanochat_base_dir: str = DEFAULT_NANOCHAT_BASE_DIR
    general_data_dir: str = ""
    stem_ratio: float = 0.7
    eval_benchmarks: str = "stem"
    # Subsample cap for base_eval (--max-per-task). base_eval shuffles every
    # task with a FIXED seed (random.Random(1337)) before truncating, so all
    # experiments score the SAME subset and stay comparable. -1 = full eval
    # sets (production); small caps (e.g. 100) keep proxy evals cheap
    # (speedrun). Part of the run fingerprint: changing it changes the eval
    # protocol, so old results must not be mixed with new ones.
    eval_max_per_task: int = -1
    quality_config_path: str = ""
    schema_path: str = ""
    # Stable cache location for (expensive) pool-level artifacts: embeddings
    # and K-means labels/centroids, keyed by content hash of the data pool.
    # Lives OUTSIDE the fingerprinted output dir so K_enhanced/merge_distance
    # changes (which archive the output dir) reuse the embeddings instead of
    # re-embedding the whole pool. Empty = legacy behavior (cache alongside
    # the other cluster caches in the output dir).
    embedding_cache_dir: str = ""
    npu_per_exp: int = 0
    # Experiment name (like nanochat's model-tag): scopes proxy model tags
    # (climbmix_{name}_{id}) so parallel runs with different names never
    # overwrite each other's checkpoints / eval CSVs.
    experiment_name: str = "main"

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
