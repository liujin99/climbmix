# CLIMBmix — Nemotron-CLIMB Unofficial Reproduction

> **DISCLAIMER**: This is an **unofficial** reproduction of the Nemotron-CLIMB paper.
> It is NOT affiliated with, endorsed by, or connected to NVIDIA or the original authors.
> This project is for **personal research purposes only** and should NOT be used for
> commercial applications. The original paper and official data can be found at:
> [https://research.nvidia.com/labs/lpr/climb/](https://research.nvidia.com/labs/lpr/climb/)

> **Paper**: *Nemotron-CLIMB: CLustering-based Iterative Data Mixture
> Bootstrapping for Language Model Pre-training* — Shizhe Diao et al.
> (NVIDIA, NeurIPS 2025)
>
> [arXiv:2504.13161](https://arxiv.org/abs/2504.13161) ·
> [Project page](https://research.nvidia.com/labs/lpr/climb/)

Automated framework that discovers, evaluates, and refines data mixtures
for language model pre-training through embedding-driven clustering and
iterative bootstrapping, using **nanochat-npu** as the training backend via
**method A** (subprocess calls) — a full 9-arm validation round consumes
**≈1,800 NPU-hours**.

The CLIMB premise is validated at target-model scale in our production
rounds: search-found mixtures beat uniform / natural / domain-ratio
baselines by **+0.014–0.031 STEM** on a d28 (~2.5B) model at a 3B-token
mid-training budget — see [Results](#results--round-reports).

## Results & Round Reports

Each production round gets one record file (`docs/experiment_prodN.md`):
a reader-friendly closeout report up front, developer details in the
appendix.

### prod4 (2026-09) — winner-validation round, 9 arms

All arms: d28 (~2.5B), 3B tokens, identical training recipe and eval
protocol; STEM = centered-accuracy mean over 6 tasks (4 MC + 2 generative
CoT); seed pairs where marked. Round cost: **≈1,800 NPU-hours**
(d20 search stage + 8 target arms + anchor/failed launches included).

| Arm | STEM | gsm8k_cot |
|---|---|---|
| **CLIMB-cfg72** (measured search config, rank #2) | **0.2142** | **0.3116** |
| **CLIMB-cfg25** (measured search config, rank #1) | 0.2066 | 0.2714 |
| CLIMB-winner (interpolated selection, seed 42/43) | 0.1972 / 0.1990 | 0.2578 / 0.2381 |
| uniform-cluster baseline (seed 42/43) | 0.1647 / 0.1833 | 0.1077 / 0.1228 |
| natural (pool-proportional) | 0.1705 | 0.1069 |
| domain-ratio (external manual ratio, math 60%) | 0.1804 | 0.1289 |
| base model (no mid-training, reference) | 0.1746 | 0.0273 |

Key findings:

- **CLIMB premise validated at target scale** — search-found mixtures beat
  every non-search baseline by +0.014–0.031 STEM; gains concentrate in
  generative math reasoning (gsm8k 2.4–2.9× the uniform family, 11× the
  base model).
- **Novel finding: the selection mechanism is the weak link** — the
  predictor's argmin extrapolation over the design space added nothing over
  *measured* configs (both measured points beat the never-measured
  interpolated winner; a soft winner's curse, presaged by the predictor's
  thin response in exactly the directions the winner extrapolated).
  Next-round priority: best-measured fallback + no-claim guard. The paper
  does not discuss this failure mode.
- **Domain-level ratios are not a substitute** — a hand-tuned 4-domain
  mixture (math 60%) lands inside the uniform band; the win comes from
  cluster-level structure, not "more math".
- **Rigor as a standing policy** — seed-pair replication (gaps reported as
  bands), remote-eval anchor reconciliation (4-decimal match), eval-protocol
  freeze within a round, quota-exact data cross-accounting.

Cross-round trajectory (CLIMB vs uniform, same-day budget-matched):
prod1 −0.004 → prod2 +0.010 → prod4 +0.016–0.032.

Full report: [docs/experiment_prod4.md](docs/experiment_prod4.md).

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
  (8 experiments in parallel, 1 NPU each; token-capped data selection, default 200M/exp;
   production rounds: history-injected [54, 36, 18] configs @ 400M single-pass)
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

## Model Sizes (nanochat backend)

| depth | scaling (M) | total (M) | VE share | role |
|-------|------------|-----------|----------|------|
| 20 | 435.2 | 896.5 | 46.8% | proxy (production) |
| 24 | 729.8 | 1,384.1 | 43.7% | 56% of CLIMB 1.3B (scaling) |
| 28 | 1,477 | 2,481 | 37.9% | target (production), ~1.1× CLIMB 1.3B (scaling) |

VE (Value Embeddings) sit on alternating layers and hold a large share of the
parameters without participating in the core matmul FLOPs — compare against
the paper in **scaling_params**. The d28 row is measured from the production
checkpoint (dtype audit: 182 fp32 transformer matrices + bf16 embeddings +
fp32 lm_head); depth is auto-detected from `meta_*.json` (three-level
fallback: GPTConfig → formula estimate → DEPTH_INFO table).

## Project Structure

```
climbmix/
├── docs/
│   ├── experiment_prod*.md            # Per-round experiment records (reader-facing report up front, dev details at the back)
│   ├── paper_deviations.md             # Itemized deviations from the paper (arXiv:2504.13161) + consistency audit
│   ├── scoring_metric_design.md        # SNR-weighted scoring design + proxy/target training-budget comparison
│   ├── proxy_and_model_analysis.md     # Proxy/target model-size analysis
│   ├── embedding_performance.md        # Embedding throughput notes
│   └── nan_investigation.md            # stella NaN fix investigation
├── runs/                                # Shell scripts
│   ├── run_experiment.sh               # Entry: run one full experiment (complete chain; from-scratch / crash-resume; pipeline engine included)
│   ├── run_extend_experiment.sh        # Entry: extension search (reuse a prior experiment's measured d20 points, incremental search)
│   ├── run_extend_traineval.sh         # Entry: extension train+eval (same mixture, vary token budget / base / params)
│   ├── run_extend_eval.sh              # Entry: extension eval (swap the benchmark suite, no training)
│   ├── lib/                            # Shared library + internal engines (arm_engine/stage_gate/...)
│   └── infra/                          # Outside the experiment lifecycle: one-time infrastructure + downstream export
│       ├── train_base_model.sh         # Generate base checkpoints (one-time, NPU)
│       ├── embed_wave.sh               # Full-pool embedding wave dispatch (per pool version)
│       ├── embed_merge.sh              # Merge embedding partials into the Step-1 cache
│       └── large_scale_sample.sh       # Large-scale sampling from a finished run's optimal mixture (no training)
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
# Step 1: Generate base checkpoints (one-time, NPU)
DEPTH=20  bash runs/infra/train_base_model.sh   # d20 proxy checkpoint
DEPTH=28  bash runs/infra/train_base_model.sh   # d28 target checkpoint

# Step 2: Production experiments — the entry-point family; parameters live in
#   the EDIT block at the top of each script
#   LAUNCH=0 <cmd> = dry-run (validate + print, no execution)
# ── Experiment level: run one full experiment ──
bash runs/run_experiment.sh          # Full chain: pool (embeddings reused if the
                                     #   pool is unchanged) → clustering → d20
                                     #   search → mixing → two arms → report;
                                     #   state-driven: from-scratch / crash-resume
                                     #   = re-run the same command
bash runs/run_extend_experiment.sh   # Extension search: reuse a prior experiment's
                                     #   measured d20 points → incremental search →
                                     #   full downstream chain (HISTORY_RUN=<old run>)
# ── Extension level: append to an existing experiment (not a mandatory next stop) ──
bash runs/run_extend_traineval.sh    # Extension train+eval: same optimal mixture,
                                     #   vary token budget / base_model / training
                                     #   params in a two-arm comparison
                                     #   train→eval→report (SCALE_TOKENS=20B NODES=8)
bash runs/run_extend_eval.sh         # Extension eval: re-evaluate existing ckpts on
                                     #   a different benchmark suite → report, no training
# One-time setup: put "obs_prod_base" (the production OBS root prefix) into
# ~/.config/climbmix/remote_ma.json (same file as the secret) — after that,
# every launch omits REMOTE_OBS_PREFIX (auto-appended /<run_name>).
```

Each script auto-checks dependencies, NPU availability, disk space, and exits with instructions if anything is missing.

## Crash Resume

Both runners are resumable: **re-run the same command after an interruption.**

- **Result-dir lifecycle** (`ls result/` is self-describing):
  - `result/${EXP_NAME}_current/` — ACTIVE run (in progress, crashed, or green)
  - `result/${EXP_NAME}_<ts>/` — COMPLETED: terminal `.done` markers all
    present; renamed automatically at the end of a green run
    (`mark_completed` in `runs/lib/stage_gate.sh`)
  - `result/${EXP_NAME}_stale_<scope>_<ts>/` — abandoned mid-run; scope =
    what changed: `search` (whole dir, search fingerprint), `target` (target
    products only), `legacy` (old single-fingerprint format), `orphan`
    (no fingerprints)
  - every archive carries `archive_meta.json`: reason, timestamps, old→new
    fingerprints, git HEAD, `was_complete`, moved items
  - re-running a COMPLETED experiment is idempotent: the newest
    `${EXP_NAME}_<ts>` whose search fingerprint matches is restored as
    `_current` (all `.done` markers skip, zero NPU work; a target mismatch
    then re-runs only Steps 4-8)
  - old server layouts (`result/$EXP_NAME`, `*_stale_<ts>`,
    `*_target_stale_<ts>`) migrate to the new names automatically on the
    next run (idempotent, one-time)
- **Stage-scoped fingerprint auto-reset**: on start, each script compares TWO
  fingerprints against `result/${EXP_NAME}_current/.fingerprint_search` and
  `.fingerprint_target` (each = stage-relevant repo sources + semantic params).
  A SEARCH mismatch (search semantics changed) archives the whole dir; a
  TARGET mismatch archives only target products (Steps 4-8 rerun; search
  results are kept). Legacy single-`.fingerprint` dirs are adopted unverified
  with `MIGRATE_LEGACY_FINGERPRINT=1` (one-time migration), else archived.
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
- **Experiment isolation**: `EXP_NAME=myexp bash runs/run_experiment.sh` scopes
  the output dir (`result/myexp`), proxy tags (`climbmix_myexp_*`) and target
  tags (`d28_climb_myexp`) so parallel/sequential experiments never overwrite
  each other. Valid chars: `[A-Za-z0-9_-]`.
- **Force fresh run**: change `EXP_NAME` or `rm -rf result/${EXP_NAME}_current`.
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
