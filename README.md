# CLIMBmix — Nemotron-CLIMB Reproduction

> **Disclaimer**: This repository is a personal clean-room implementation of the Nemotron-CLIMB paper,
> not an official NVIDIA release. For educational and research purposes only.

> **Paper**: Shizhe Diao et al. (NVIDIA, NeurIPS 2025)
>
> [arXiv:2504.13161](https://arxiv.org/abs/2504.13161)

Automated framework that discovers, evaluates, and refines data mixtures
for language model pre-training through embedding-driven clustering and
iterative bootstrapping.

## Algorithm Pipeline

```
Raw Data → Text Embedding (stella_en_400M_v5)
  ↓
FAISS K-means (K_init=1000)
  ↓
Cluster Pruning (fasttext quality, threshold=3.0 → K_pruned=240)
  ↓
Cluster Merging (centroid distance<1.5 → K_enhanced≈21)
  ↓
Iteration 1: Dirichlet sample 64 configs → train proxy → fit LightGBM predictor
Iteration 2: Predictor-guided sample 32 configs → train → update predictor
Iteration 3: Predictor-guided sample 16 configs → train → final predictor
  ↓
Search: predictor ranks 10K candidates → optimal α*
  ↓
Final selection: proportional sampling by α* per cluster
  ↓
Output: sampled_dataset.parquet + optimal_mixture_weights.json
```

## Key Differences from QuaDMix

| Aspect | QuaDMix | CLIMB |
|--------|---------|-------|
| Domain discovery | fastText predefined labels (22) | Embedding + K-means → merge (≈21) |
| Data selection | Quality ranking + sigmoid sampling | Mixture weights + proportional sampling |
| Search strategy | Single-shot (sample → regress → search) | Iterative bootstrapping (3 iterations) |
| Predictor | LightGBM (single fit) | LightGBM (iterative update) |
| Proxy model | 1M tinyllama | 350M (main) / 62M (ablation) |
| Parameter space | (N+4)×M per domain | K mixture weights per cluster |

## Project Structure

```
climbmix/
├── src/climbmix/                # Python package (pip install -e .)
│   ├── core/
│   │   ├── types.py                # Core types (MixtureWeights, ClusterInfo, etc.)
│   │   ├── embedding_cluster.py    # Step 1: Embed + FAISS K-means
│   │   ├── cluster_merge.py        # Step 2: Prune + merge clusters
│   │   ├── dirichlet_sampler.py    # Dirichlet mixture weight sampling
│   │   ├── iterative_bootstrapper.py # Step 3: Iterative search engine
│   │   └── proxy_model.py          # Proxy model (1M/62M/350M/1B variants)
│   ├── pipeline/
│   │   ├── climb_pipeline.py       # Main pipeline orchestrator
│   │   ├── proxy_runner.py         # Proxy training runner
│   │   └── loss_utils.py           # Chunked loss computation
│   ├── data/
│   │   └── metadata_manager.py     # Shard metadata manager
│   ├── sampling/
│   │   └── data_selector.py        # Mixture-based data selection
│   ├── npu/
│   │   └── device.py               # Device manager (CPU/CUDA/NPU)
│   └── utils/
│       ├── normalization.py         # Normalization utilities
│       └── perf_timer.py           # Performance timer
├── scripts/
│   ├── run_climb.py               # Main entry script
│   ├── create_test_data.py        # Test data generator
│   └── demo_run_cpu.sh            # Quick CPU demo (~1-2min)
├── data/                          # Data directory
├── temp/                          # Intermediate data
└── docs/                          # Documentation
```

## Quick Start

```bash
# Install
pip install -e .

# Create test data & run CPU demo
bash runs/demo_run_cpu.sh

# Custom run
python scripts/run_climb.py \
    --data-dir data/my_data.parquet \
    --output-dir result/my_run \
    --K-init 1000 \
    --K-enhanced 21 \
    --num-iterations 3 \
    --configs-per-iter "64,32,16" \
    --proxy-size 350M \
    --proxy-steps 5000 \
    --device-type cuda
```

## Reusable Components from Quadmix

The following components are adapted from the quadmix project:

- **Proxy model architecture**: Same RegMix-style tinyllama (SwiGLU+RMSNorm+RoPE)
- **Device manager**: CPU/CUDA/NPU abstraction
- **Chunked loss utilities**: Memory-efficient CE loss computation
- **Normalization utilities**: zscore, minmax, rank normalizers
- **Shard metadata manager**: Multi-shard parquet loading with on-demand text

## License

Apache 2.0
