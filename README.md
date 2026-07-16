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
iterative bootstrapping, using **nanochat-npu** (8×910B3 Ascend NPU) as the
training backend via **method A** (subprocess calls).

## Algorithm Pipeline

```
Raw Data → FDC domain labels (22 domains) or embedding clustering
  ↓
Iterative Bootstrapping Search:
  Iteration 1: Dirichlet sample 8 configs → mid-train proxy → evaluate → fit predictor
  Iteration 2: Predictor-guided 4 configs → mid-train → evaluate → update predictor
  Iteration 3: Predictor-guided 2 configs → mid-train → evaluate → final predictor
  ↓
Predictor ranks candidates → optimal mixture α*
  ↓
Target validation: mid-train d24 with α* vs baseline
  ↓
Output: report + distribution chart + sampled_dataset.parquet
```

## Staged Experiment Design

| Stage | Proxy | Target | Purpose | Time |
|-------|-------|--------|---------|------|
| 0 (dry-run) | — | — | CPU logic verification | ~5min |
| 1 | d10 (70M scaling) | d24 (730M scaling) | Quick: proxy→target transferability | ~5h |
| 2 | d14/d18 | d24 | Formal search after Stage 1 gate | ~8-26h |

**Stage 1 decision gate**: d24 accuracy with optimal mixture > baseline → proceed to Stage 2.

## Key Design Choices

- **method A**: ProxyRunner calls nanochat `mid_train.py` + `base_eval.py` as subprocesses
- **Annealing semantics**: lr_scale=1.0, warmup=0.0, warmdown=0.9 (CLIMB = annealing, not re-warmup)
- **Fixed training**: `--num-iterations` (500 steps proxy, 1000 steps target), not ratio-based
- **Continual pre-training**: all stages anneal from base checkpoint, never from-scratch
- **Metric**: high-signal 6-task subset (piqa, arc_easy, lambada, commonsense_qa, squad, coqa)
- **Same metric for search AND validation** — no distortion from different benchmark sets

## nanochat Model Sizes

| depth | scaling(M) | total(M) | VE占比 | CLIMB对标 |
|-------|-----------|---------|--------|----------|
| 10 | 70.2 | 196.0 | 53.5% | ≈62M proxy |
| 14 | 164.2 | 399.1 | 51.5% | ≈132M |
| 18 | 324.4 | 701.9 | 48.4% | ≈350M proxy |
| 24 | 729.8 | 1384.1 | 43.6% | target (56% of CLIMB 1.3B) |

VE (Value Embeddings) 占 ~50% 参数但不参与核心计算。对标 CLIMB 时看 **scaling_params**。

## Project Structure

```
climbmix/
├── docs/
│   └── proxy_and_model_analysis.md     # 分析文档
├── runs/                                # Shell scripts (staged)
│   ├── dryrun.sh                        # CPU dry-run (~5min)
│   ├── search_d10.sh                    # d10→d24 search (~5h, NPU)
│   ├── search_d14.sh                    # d14/d18→d24 search (~8-26h, NPU)
│   ├── midtrain_validate.sh             # CLIMB vs random validation (NPU)
│   └── train_base_model.sh             # Generate base checkpoint (NPU)
├── scripts/
│   ├── run_climb.py                     # CLI entry point
│   └── prepare_random_baseline.py       # Random baseline data prep
└── src/climbmix/
    ├── core/
    │   ├── types.py                     # Config (depth-based + annealing)
    │   ├── iterative_bootstrapper.py    # Search engine
    │   ├── dirichlet_sampler.py         # Dirichlet exploration
    │   ├── predictor.py                 # LightGBM predictor
    │   ├── discovery.py                 # Cluster discovery strategies
    │   ├── embedding_cluster.py         # Embed + FAISS K-means
    │   ├── cluster_merge.py             # Prune + merge
    │   ├── quality_filter.py            # Quality filtering
    │   └── protocols.py
    ├── pipeline/
    │   ├── climb_pipeline.py            # Main pipeline (injects cluster data)
    │   ├── proxy_runner.py              # method A: subprocess nanochat
    │   └── report_generator.py          # Markdown + matplotlib
    ├── data/
    │   ├── metadata_manager.py          # ShardMetadataManager (parquet)
    │   └── column_schema.py             # Column name mapping
    ├── sampling/
    │   └── data_selector.py             # Mixture-weighted doc sampling
    └── utils/
        ├── token_estimate.py
        ├── normalization.py
        └── perf_timer.py
```

## Quick Start

```bash
# Step 0: CPU dry-run (no NPU needed)
bash runs/dryrun.sh

# Step 1: Generate base checkpoints (NPU)
DEPTH=10  bash runs/train_base_model.sh   # d10, ~0.6h
DEPTH=24  bash runs/train_base_model.sh   # d24, ~29h

# Step 2: Search for optimal mixture (NPU)
bash runs/search_d10.sh             # d10→d24 (~5h)
PROXY_DEPTH=14 bash runs/search_d14.sh  # d14→d24 (~8h)

# Step 3: Validate CLIMB vs random (NPU)
CLIMBMIX_RESULT=result/stage1_xxx bash runs/midtrain_validate.sh
```

Each script auto-checks dependencies and exits with instructions if anything is missing.

## CLI Options

```bash
python scripts/run_climb.py --help

# Key options:
--proxy-depth 10          # nanochat model depth (10=70M scaling, 24=730M)
--proxy-num-iterations 500 # Fixed training steps (not ratio-based)
--proxy-lr-scale 1.0      # Annealing LR scale (1.0 = continue from base)
--proxy-warmup 0.0        # No re-warmup (CLIMB annealing)
--proxy-warmdown 0.9      # 90% warmdown for annealing
--target-depth 24         # Target model depth
--nanochat-dir /path      # nanochat-npu directory
--val-tasks piqa,arc_easy,lambada_openai,commonsense_qa,squad,coqa
--dry-run                 # Skip training (random scores, CPU only)
```

## Dependencies

- **nanochat-npu** (external): training backend, must be at configured path
- **Python**: numpy, lightgbm, scikit-learn, pandas, pyarrow, torch, matplotlib

## License

Apache 2.0
