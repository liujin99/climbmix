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
Embedding Cluster (stella_en_400M_v5 → FAISS K-means K_init=1000 →
   prune (threshold 3.0) + band merge (τ=0.9) → K ∈ [3, 15]; per-run
   merge_profile.json + printed tuning advice)
  ↓
Iterative Bootstrapping Search:
  Iteration 1: Dirichlet sample 20 configs → d20 proxy train+eval → fit predictor
  Iteration 2: Predictor-guided 10 configs → d20 proxy train+eval → update predictor
  Iteration 3: Predictor-guided  5 configs → d20 proxy train+eval → final predictor
  (8 experiments in parallel, 1 NPU each; token-capped data selection, default 200M/exp)
  ↓
Each proxy experiment: 70% STEM (by cluster weights) + 30% ClimbMix general
  (adaptive 3-50 shards, reverse download from shard 6542 → avoids pretrain overlap)
  ↓
Predictor ranks candidates → optimal mixture α*
  ↓
Target training: d28 mid-train with α* + 30% ClimbMix (same mixing)
  ↓
STEM benchmark eval (arc_easy, arc_challenge, mmlu_stem, gpqa_diamond, gsm8k_cot, math_cot_500)
  + random-baseline comparison (equal cluster weights 1/K, paper App. C.1;
    same token cap and same shortfall policy as the CLIMB arm, seed 42)
  ↓
Output: report + sampled_dataset.parquet + target_result.json
```

## Key Design Choices

- **method A**: ProxyRunner/TargetRunner call nanochat `mid_train.py` + `base_eval.py` as subprocesses
- **d20 proxy** (435.2M scaling, 1000 iterations) → **d28 target** (auto-detected from `meta_*.json`)
- **8 parallel experiments**: `--npu-per-exp 1` runs 8 proxy experiments concurrently on 8 NPUs (set 0 = sequential, all NPUs per experiment)
- **Token caps**: `--proxy-target-tokens 200M` / `--target-tokens 1B` cap data selection (0 = all available — never leave 0 on the full 100B-token pool; suffix syntax `2B/10M/500K` supported)
- **70% STEM + 30% ClimbMix**: adaptive shard count (`calc_climbmix_count`, clamped [3, 50]), not full 400B
- **Reverse-order download**: shards from MAX_SHARD (6542) backwards, avoids overlap with pretrain (shards 0-999)
- **Stream-based mixing**: `stream_texts_uniform` + `endless_generator`, memory-efficient
- **Val split convention**: last `shard_*.parquet` is validation (held out from train, DDP row-group safe)
- **STEM benchmark**: `--eval-benchmarks=stem` → CSV "STEM" row parsed as `stem_metric`
- **Annealing semantics**: lr_scale=1.0, warmup=0.0, warmdown=0.9 (CLIMB = annealing, not re-warmup)
- **NPU support**: `device_type=npu`, embedding tries `torch_npu` first, fallback to CPU (192 threads)
- **Self-contained**: `get_model_info.py` + `mix_general_data.py` in `scripts/`, no external repo dependency

## nanochat Model Sizes

| depth | scaling(M) | total(M) | VE占比 | CLIMB对标 |
|-------|-----------|---------|--------|----------|
| 20 | 435.2 | 896.5 | 51.5% | proxy (production) |
| 24 | 729.8 | 1384.1 | 43.6% | 56% of CLIMB 1.3B |
| 28 | auto-detect | auto-detect | — | target (from meta_*.json) |

VE (Value Embeddings) 占 ~50% 参数但不参与核心计算。对标 CLIMB 时看 **scaling_params**。
d28 参数从 checkpoint `meta_*.json` 自动读取（三层 fallback: GPTConfig → 公式估算 → DEPTH_INFO 表）。

## Project Structure

```
climbmix/
├── docs/
│   ├── paper_deviations.md             # 与论文 (arXiv:2504.13161) 的逐项偏差 + 一致性审计
│   ├── scoring_metric_design.md        # SNR 评分设计 + proxy/target 训练量对比
│   ├── proxy_and_model_analysis.md     # 分析文档
│   ├── embedding_performance.md        # 嵌入性能
│   └── nan_investigation.md            # stella NaN 修复调查
├── runs/                                # Shell scripts
│   ├── run_climbmix.sh                  # Main entry: full pipeline d20 search + d28 target (NPU)
│   ├── speedrun_climbmix.sh             # End-to-end validation (minimal data + steps)
│   ├── smoke_test.sh                    # Logic smoke test (CPU)
│   └── train_base_model.sh             # Generate base checkpoint (NPU)
├── scripts/
│   ├── run_climb.py                     # CLI entry point
│   ├── mix_general_data.py             # Adaptive shard download + stream mixing
│   ├── prepare_shards.py               # parquet → nanochat shards (val = last shard)
│   ├── get_model_info.py               # Auto-detect scaling params from meta_*.json
│   └── prepare_random_baseline.py       # Random baseline data prep
└── src/climbmix/
    ├── core/
    │   ├── types.py                     # Config + auto_detect_depth_info + DEPTH_INFO
    │   ├── iterative_bootstrapper.py    # Search engine
    │   ├── dirichlet_sampler.py         # Dirichlet exploration
    │   ├── predictor.py                 # LightGBM predictor
    │   ├── discovery.py                 # EmbeddingClusterDiscovery only
    │   ├── embedding_cluster.py         # Embed + FAISS K-means (NPU/CPU dual mode)
    │   ├── cluster_merge.py             # Prune + merge
    │   ├── quality_filter.py            # Quality filtering
    │   └── protocols.py
    ├── pipeline/
    │   ├── climb_pipeline.py            # 7-stage pipeline (Stage 0-6)
    │   ├── proxy_runner.py              # d20 proxy: train + eval + mix (parallel)
    │   ├── target_runner.py             # d28 target: train + eval + mix
    │   └── report_generator.py          # Markdown + matplotlib
    ├── data/
    │   ├── metadata_manager.py          # ShardMetadataManager (parquet)
    │   └── column_schema.py             # Column name mapping
    ├── sampling/
    │   └── data_selector.py             # Mixture-weighted doc sampling (seeded permutation)
    └── utils/
        ├── token_estimate.py            # chars→tokens heuristic + "2B"/"10M" parser
        ├── io_utils.py                  # Atomic write helpers (savez/json/parquet)
        ├── fingerprint.py               # Experiment fingerprint (code + params → reset)
        └── perf_timer.py
```

## Quick Start

```bash
# Step 1: Generate base checkpoints (NPU)
DEPTH=20  bash runs/train_base_model.sh   # d20 proxy checkpoint
DEPTH=28  bash runs/train_base_model.sh   # d28 target checkpoint

# Step 2: End-to-end validation first (minimal data, ~minutes)
bash runs/speedrun_climbmix.sh

# Step 3: Run full pipeline — d20 proxy search + d28 target + report (NPU)
bash runs/run_climbmix.sh
```

Each script auto-checks dependencies, NPU availability, disk space, and exits with instructions if anything is missing.

## Crash Resume (断点续跑)

Both runners are resumable: **re-run the same command after an interruption.**

- **Stage-scoped fingerprint auto-reset**: on start, each script compares TWO
  fingerprints against `result/$EXP_NAME/.fingerprint_search` and
  `.fingerprint_target` (each = stage-relevant repo sources + semantic params).
  A SEARCH mismatch (search semantics changed) archives the whole dir as
  `result/${EXP_NAME}_stale_<ts>`; a TARGET mismatch archives only target
  products (Steps 4-8 rerun; search results are kept). Legacy single-
  `.fingerprint` dirs are adopted unverified with
  `MIGRATE_LEGACY_FINGERPRINT=1` (one-time migration), else archived.
  `num_npu` is deliberately NOT fingerprinted (parallel shape only — the NPU
  pool may shrink/grow mid-campaign). `runs/*.sh` edits alone (comments/echo)
  do NOT reset; param-array knobs do. Not covered: nanochat-npu edits, data
  files with unchanged names/counts.
- **Granularity** (finest loss on crash):
  | Stage | Resume unit | Loss on crash |
  |---|---|---|
  | metadata scan | step | rescan |
  | embedding | **shard ledger** (`embedding_progress_w*.json` + memmap) | ≤ N_workers in-flight shards |
  | clustering | step (cache) | re-cluster |
  | search iteration | iteration (`search_state.json`, atomic) | ≈0 |
  | search experiment | **experiment** (`exp_XXXX/meta.json`, rc=0/0 + weight match) | only interrupted exp re-runs |
  | search done, pre-selection | predictor refit + full-design-space search (paper-faithful) | ≈0 |
  | shard/mix/sampled writes | atomic (tmp+rename, `.done` markers) | redo step |
  | target train (climb/random independent) | whole run (partial checkpoints cleared first) | 1 training run |
  | eval / report | `.done` marker / idempotent | minutes / 0 |
- **Not resumable**: inside a single nanochat training run (1000 steps) —
  interrupted trainings restart from step 0 by design.
- **Experiment isolation**: `EXP_NAME=myexp bash runs/run_climbmix.sh` scopes
  the output dir (`result/myexp`), proxy tags (`climbmix_myexp_*`) and target
  tags (`d28_climb_myexp`) so parallel/sequential experiments never overwrite
  each other. Valid chars: `[A-Za-z0-9_-]`.
- **Force fresh run**: change `EXP_NAME` or `rm -rf result/$EXP_NAME`.
- **HF download endpoint**: `runs/*.sh` default `HF_ENDPOINT` to
  `https://hf-mirror.com` (override: `HF_ENDPOINT=https://huggingface.co bash runs/...`).
  The corporate proxy selectively refuses Python-issued CONNECT tunnels to
  huggingface.co (90+ consecutive 503s over 80 min) while allowing both curl to
  huggingface.co and Python to hf-mirror.com — the mirror serves identical
  bytes, so downloads, Range resume and parquet validation work unchanged. One
  variable covers ClimbMix shards (`dataset.py`) and `eval_stem.zip`
  (`base_eval.py`); both read it at import time, so it must be set before launch.

## CLI Options

```bash
python scripts/run_climb.py --help

# Key options:
--proxy-depth 20          # nanochat model depth (20=435M scaling, production default)
--target-depth 28         # Target model depth (auto-detected from meta_*.json)
--proxy-num-iterations 1000  # Fixed training steps (not ratio-based)
--proxy-lr-scale 1.0      # Annealing LR scale (1.0 = continue from base)
--proxy-warmup 0.0        # No re-warmup (CLIMB annealing)
--proxy-warmdown 0.9      # 90% warmdown for annealing
--proxy-target-tokens 200M # Per-experiment data cap (0 = all; accepts 2B/10M/500K/1.5B)
--target-tokens 1B        # Cap for final target data selection (0 = all)
--K-init 1000             # Initial K-means clusters before prune+merge
--K-enhanced 3            # Cluster-count floor (safety bound; set to paper's K for fixed-K semantics)
--K-max 15                # Cluster-count cap; K_final = clamp(natural_K(0.9), 3, 15)
--configs-per-iter 20,10,5  # Search: 20 random + 10+5 predictor-guided
--npu-per-exp 1           # NPUs per proxy experiment (0=all sequential; 1=8 parallel)
--device-type npu          # NPU (default) or cpu
--nanochat-base-dir /path  # Checkpoint storage (default: /home/ma-user/work/nanochat_model_dir)
--general-data-dir /path   # ClimbMix shard cache dir
--stem-ratio 0.7           # 70% STEM + 30% ClimbMix (default)
--eval-benchmarks stem     # STEM benchmark subset for eval
--exp-name main            # Experiment name: scopes proxy/target tags + dirs
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
