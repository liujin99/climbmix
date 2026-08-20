# CLIMBmix — Nemotron-CLIMB Unofficial Reproduction

> **DISCLAIMER**: This is an **unofficial** reproduction of the Nemotron-CLIMB paper.
> It is NOT affiliated with, endorsed by, or connected to NVIDIA or the original authors.
> This project is for **personal research purposes only** and should NOT be used for
> commercial applications. The original paper and official data can be found at:
> [https://research.nvidia.com/labs/lpr/climb/](https://research.nvidia.com/labs/lpr/climb/)

> **Paper**: Shizhe Diao et al. (NVIDIA, NeurIPS 2025)
>
> [arXiv:2504.13161](https://arxiv.org/abs/2504.13161)

Automated framework that discovers, evaluates, and refines data mixtures
for language model pre-training through embedding-driven clustering and
iterative bootstrapping, using **nanochat-npu** (8×910B Ascend NPU) as the
training backend via **method A** (subprocess calls).

## Algorithm Pipeline

```
STEM Data Pool (100B parquet, 116M docs, 1000 shards)
  ↓
Embedding Cluster (stella_en_400M_v5 → K-means K=21 → merge small clusters)
  ↓
Iterative Bootstrapping Search:
  Iteration 1: Dirichlet sample 64 configs → d14 proxy train+eval → fit predictor
  Iteration 2: Predictor-guided 32 configs → d14 proxy train+eval → update predictor
  Iteration 3: Predictor-guided 16 configs → d14 proxy train+eval → final predictor
  ↓
Each proxy experiment: 70% STEM (by cluster weights) + 30% ClimbMix general
  (adaptive 3-50 shards, reverse download from shard 6542 → avoids pretrain overlap)
  ↓
Predictor ranks candidates → optimal mixture α*
  ↓
Target training: d28 mid-train with α* + 30% ClimbMix (same mixing)
  ↓
STEM benchmark eval (arc_easy, arc_challenge, mmlu_stem, gpqa_diamond, gsm8k_cot, math_cot_500)
  ↓
Output: report + sampled_dataset.parquet + target_result.json
```

## Key Design Choices

- **method A**: ProxyRunner/TargetRunner call nanochat `mid_train.py` + `base_eval.py` as subprocesses
- **d14 proxy** (164.2M scaling, 500 iterations) → **d28 target** (auto-detected from `meta_*.json`)
- **70% STEM + 30% ClimbMix**: adaptive shard count (`calc_climbmix_count`, clamped [3, 50]), not full 400B
- **Reverse-order download**: shards from MAX_SHARD (6542) backwards, avoids overlap with pretrain (shards 0-999)
- **Stream-based mixing**: `stream_texts_uniform` + `endless_generator`, memory-efficient
- **STEM benchmark**: `--eval-benchmarks=stem` → CSV "STEM" row parsed as `stem_metric`
- **Annealing semantics**: lr_scale=1.0, warmup=0.0, warmdown=0.9 (CLIMB = annealing, not re-warmup)
- **NPU support**: `device_type=npu`, embedding tries `torch_npu` first, fallback to CPU (192 threads)
- **Self-contained**: `get_model_info.py` + `mix_general_data.py` in `scripts/`, no external repo dependency

## nanochat Model Sizes

| depth | scaling(M) | total(M) | VE占比 | CLIMB对标 |
|-------|-----------|---------|--------|----------|
| 14 | 164.2 | 399.1 | 51.5% | ≈132M proxy |
| 24 | 729.8 | 1384.1 | 43.6% | 56% of CLIMB 1.3B |
| 28 | auto-detect | auto-detect | — | target (from meta_*.json) |

VE (Value Embeddings) 占 ~50% 参数但不参与核心计算。对标 CLIMB 时看 **scaling_params**。
d28 参数从 checkpoint `meta_*.json` 自动读取（三层 fallback: GPTConfig → 公式估算 → DEPTH_INFO 表）。

## Project Structure

```
climbmix/
├── docs/
│   └── proxy_and_model_analysis.md     # 分析文档
├── runs/                                # Shell scripts
│   ├── search_d14.sh                    # Main entry: d14 search + d28 target (NPU)
│   ├── midtrain_validate.sh             # CLIMB vs random validation (NPU)
│   └── train_base_model.sh             # Generate base checkpoint (NPU)
├── scripts/
│   ├── run_climb.py                     # CLI entry point
│   ├── mix_general_data.py             # Adaptive shard download + stream mixing
│   ├── get_model_info.py               # Auto-detect scaling params from meta_*.json
│   └── prepare_random_baseline.py       # Random baseline data prep
└── src/climbmix/
    ├── core/
    │   ├── types.py                     # Config + auto_detect_depth_info + DEPTH_INFO
    │   ├── iterative_bootstrapper.py    # Search engine
    │   ├── dirichlet_sampler.py         # Dirichlet exploration
    │   ├── predictor.py                 # LightGBM predictor
    │   ├── discovery.py                 # EmbeddingClusterDiscovery only
    │   ├── embedding_cluster.py         # Embed + K-means (NPU/CPU dual mode)
    │   ├── cluster_merge.py             # Prune + merge
    │   ├── quality_filter.py            # Quality filtering
    │   └── protocols.py
    ├── pipeline/
    │   ├── climb_pipeline.py            # 7-stage pipeline (Stage 0-6)
    │   ├── proxy_runner.py              # d14 proxy: train + eval + mix
    │   ├── target_runner.py             # d28 target: train + eval + mix
    │   └── report_generator.py          # Markdown + matplotlib
    ├── data/
    │   ├── metadata_manager.py          # ShardMetadataManager (parquet)
    │   └── column_schema.py             # Column name mapping
    ├── sampling/
    │   └── data_selector.py             # Mixture-weighted doc sampling
    └── utils/
        ├── token_estimate.py
        └── perf_timer.py
```

## Quick Start

```bash
# Step 1: Generate base checkpoints (NPU)
DEPTH=14  bash runs/train_base_model.sh   # d14, ~4h
DEPTH=28  bash runs/train_base_model.sh   # d28

# Step 2: Run full pipeline — d14 proxy search + d28 target (NPU)
bash runs/search_d14.sh

# Step 3: Validate CLIMB vs random (NPU)
CLIMBMIX_RESULT=result/stage1_xxx bash runs/midtrain_validate.sh
```

Each script auto-checks dependencies, NPU availability, disk space, and exits with instructions if anything is missing.

## CLI Options

```bash
python scripts/run_climb.py --help

# Key options:
--proxy-depth 14          # nanochat model depth (14=164M scaling)
--target-depth 28         # Target model depth (auto-detected from meta_*.json)
--proxy-num-iterations 500  # Fixed training steps (not ratio-based)
--proxy-lr-scale 1.0      # Annealing LR scale (1.0 = continue from base)
--proxy-warmup 0.0        # No re-warmup (CLIMB annealing)
--proxy-warmdown 0.9      # 90% warmdown for annealing
--device-type npu          # NPU (default) or cpu
--nanochat-base-dir /path  # Checkpoint storage (default: /home/ma-user/work/nanochat_model_dir)
--general-data-dir /path   # ClimbMix shard cache dir
--stem-ratio 0.7           # 70% STEM + 30% ClimbMix (default)
--eval-benchmarks stem     # STEM benchmark subset for eval
--skip-target              # Skip d28 target training
--dry-run                 # Skip training (CPU only, logic check)
```

## Dependencies

- **nanochat-npu** (external): training backend, must be at configured path
- **Python**: numpy, lightgbm, scikit-learn, scipy, pandas, pyarrow, torch, matplotlib, tqdm
- **sentence-transformers**: for embedding (stella_en_400M_v5); NPU inference may require torch_npu, fallback to CPU
- **faiss-cpu**: for K-means clustering
- **torch_npu**: optional, for Ascend NPU support

## License

Apache 2.0
