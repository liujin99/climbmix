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

What the production rounds actually run. Every deliberate deviation from
the paper's defaults is itemized in
[docs/paper_deviations.md](docs/paper_deviations.md) (D1–D17).

```
STEM Data Pool (100B parquet, 116M docs, 1000 shards)
  ↓
Embedding Cluster (stella_en_400M_v5 → FAISS K-means K_init=1000 →
   prune (avg threshold 3.0 + per-column floor 2.0) → macro clusters:
   MERGE_STRATEGY=balanced, capacity-constrained partition to exactly
   K=15 — D14: distance merging chain-collapses on this pool's single
   continuous manifold; distance mode is kept only as a paper-faithful
   control. Per-run audit: balanced_profile.json)
  ↓
Iterative Bootstrapping Search — warm-started (docs/reuse_design.md):
   prod4 = 54 history-injected measured points (inherited from prod3,
   zero NPU cost) + [36, 18] fresh configs; each = d20 proxy train+eval
   with single-pass token-capped selection (400M/exp in prod4; code
   default 640M = 80% of the paper's ~800M)
  ↓
Each proxy experiment: 70% STEM (by cluster weights) + 30% ClimbMix general
   (adaptive 3-50 shards, reverse download from shard 6542 → avoids pretrain overlap)
  ↓
Final selection: the LightGBM argmin over the design space (4 concentration
   levels × 25K Dirichlet candidates + 5K refine near the argmin) must
   predict better than the best MEASURED config by more than a noise-floor
   margin to win the slot (no-claim guard) — else the best measured config
   is selected; the selection model refits on ALL measured points
   (early stopping only chooses the tree count). Top-k measured candidates
   are exported to topk_mixture_candidates.json for d28 arm promotion
   (paper_deviations.md D19)
  ↓
Target arms: d28 mid-train with α* + 30% ClimbMix (same mixing) vs
   uniform-cluster baseline (equal weights 1/K, paper App. C.1; same
   token cap and same shortfall policy, seed 42)
  ↓
STEM benchmark eval (arc_easy, arc_challenge, mmlu_stem, gpqa_diamond,
   gsm8k_cot, math_cot_500) → report + sampled_dataset.parquet
```

prod4's key negative result sits exactly at the final-selection step:
both *measured* top configs beat the never-measured extrapolated winner
(a soft winner's curse). The next round therefore ships a best-measured
fallback with a no-claim margin guard plus top-k measured-candidate arms
([paper_deviations.md D19](docs/paper_deviations.md)) — see
[Results](#results--round-reports).

## Key Design Choices

- **method A**: ProxyRunner/TargetRunner call nanochat `mid_train.py` + `base_eval.py` as subprocesses
- **d20 proxy** (435.2M scaling) → **d28 target** (~1.5B scaling, auto-detected from `meta_*.json`); training steps are **derived from token budgets** — steps = tokens / total_batch_size (read from ckpt meta); the old step knobs are gone (launch aborts if set)
- **8 parallel experiments**: `--npu-per-exp 1` runs 8 proxy experiments concurrently on 8 NPUs (set 0 = sequential, all NPUs per experiment); production fleets also mix in remote jobs (docs/remote_setup.md)
- **Token caps**: `--proxy-target-tokens 640M` / `--target-tokens 2B` cap data selection (0 = all available — never leave 0 on the full 100B-token pool; suffix syntax `2B/10M/500K` supported)
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
│   ├── algorithm_review.md             # Post-prod4 algorithm design review (clustering / predictor / selection)
│   ├── remote_setup.md                 # Remote-fleet setup + embedding wave/merge operations
│   ├── reuse_design.md                 # Cross-run reuse design (warm start, extension scripts)
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
│   ├── dispatch_target_arm.py           # Remote target-arm dispatch (three-layer fallback)
│   ├── inject_history.py                # Warm-start seed builder (history reuse)
│   ├── rescore_search.py                # Re-rank a finished search under a new scoring formula
│   ├── derive_target_steps.py           # steps = tokens / total_batch_size (single source of truth)
│   ├── gen_natural_weights.py           # Natural (pool-proportional) baseline weights
│   ├── mix_general_data.py             # Adaptive shard download + stream mixing
│   ├── prepare_shards.py               # parquet → nanochat shards (val = last shard)
│   ├── prepare_random_baseline.py       # Random / fixed-weight baseline data prep
│   ├── check_disk_budget.py             # Arm-launch disk preflight
│   ├── clean_derived_data.py            # Post-upload local cleanup (guarded, dry-run default)
│   ├── get_model_info.py               # Auto-detect scaling params from meta_*.json
│   └── diagnostics/                     # Test suites + ops probes
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
--target-depth 28         # Target model depth (~1.5B scaling, auto-detected from meta_*.json)
--proxy-target-tokens 640M # Per-experiment data budget; steps DERIVED = tokens/tbs
                           #   (610 steps @ tbs 1,048,576; 800M = paper-equivalent)
--proxy-lr-scale 1.0      # Annealing LR scale (1.0 = continue from base)
--proxy-warmup 0.0        # No re-warmup (CLIMB annealing)
--proxy-warmdown 0.9      # 90% warmdown for annealing
--target-tokens 2B        # Target data cap (1907 steps derived; prod1/2 ran 1B historically)
--K-init 1000             # Initial K-means clusters before prune+merge
--K-enhanced 15           # Macro-cluster count (balanced partition to exactly K; production)
--K-max 15                # Distance-mode cap only; balanced mode: K_max ≡ K_enhanced
--configs-per-iter 20,10,5  # Search: 20 random + 10+5 predictor-guided
                           #   (warm-start rounds: first slot = history points)
--npu-per-exp 1           # NPUs per proxy experiment (0=all sequential; 1=8 parallel)
--device-type npu         # NPU (default) or cpu
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

## Environment & Limitations

This repository is the **companion code of a research report**, not an
out-of-the-box product. Honest scope:

**Fully available here** — the complete pipeline code (embedding,
clustering, search, scoring, sampling, single-pass guards), the launcher
family, the diagnostics test suites, per-round experiment records with
every number, and the paper-deviation ledger
([docs/paper_deviations.md](docs/paper_deviations.md)).

**Not runnable as-is outside the original environment**, by design:

- **Ascend 910B NPUs + CANN** — training and eval go through nanochat-npu
  and torch_npu; there is no CUDA or CPU training path.
- **nanochat-npu** (external backend repo) must be present at the
  configured path with its base checkpoints.
- **Data pools are not redistributed** (data-governance policy): the
  100B STEM parquet pool, the ClimbMix general shards, and all model
  weights stay private. Mixtures are reproducible in *composition*
  (cluster weights + selection rules are fully specified), not in raw
  bytes.
- **The remote fleet** needs an OBS-compatible object store plus a
  private job-dispatch backend adapter (`REMOTE_BACKEND_MODULE`); a mock
  backend (`REMOTE_BACKEND=mock`) exists for local end-to-end
  simulation.
- The launch config (`~/.config/climbmix/remote_ma.json`) holds secrets
  and OBS prefixes and never enters git.

Reading order for reproduction purposes:
README → docs/experiment_prodN.md → docs/paper_deviations.md → source.

## License

Apache 2.0
