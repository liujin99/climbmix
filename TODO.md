# CLIMBmix — Progress & Remaining Work

## Completed

- **Stage-scoped fingerprints** (`runs/lib/stage_gate.sh` + `fingerprint.py --stage`): single global fingerprint meant a target-stage knob (MID_DEVICE_BATCH_SIZE) would archive multi-day search results. Now `.fingerprint_search` (Steps 1-3 products) and `.fingerprint_target` (Steps 4-8 products) are checked independently: search mismatch → archive all; target mismatch → archive only target products, search kept. File classification: search-only (proxy_runner/climb_pipeline/embedding_cluster/discovery/predictor/dirichlet/iterative_bootstrapper/cluster_merge/quality_filter/run_climb.py), target-only (target_runner/report_generator/prepare_shards/prepare_random_baseline/mix_general_data), everything else BOTH (conservative); diagnostics + get_model_info excluded. Params split by consuming stage; shared (stem_ratio, eval sets, dtype, data dirs) in both. `num_npu` removed from fingerprints (parallel shape only — pool may shrink/grow, remote executors later). Legacy single-`.fingerprint` dirs: `MIGRATE_LEGACY_FINGERPRINT=1` adopts unverified (one-time), else archives with hint. Guarded by `/tmp/opencode/test_stage_fingerprint.py` (44 checks) + updated `test_fingerprint_p1.py` (array parsing, NUM_NPU excluded-by-design)
- **d28 Step-6 OOM fix**: `MID_DEVICE_BATCH_SIZE` 8→4 in both runners — dbs=8 override + full AdamW optimizer state (ws=8 matches d28 base) hit 27.58/29.49 GiB in the FIRST forward (apply_rotary_emb, 66 MiB short; speedrun 2026-08-27). dbs=4 = the checkpoint's own inherited value; total batch 1,048,576 unchanged, both arms same value → scores comparable
- **d20 proxy → d28 target pipeline**: embedding cluster → Dirichlet search → LightGBM predictor → d28 target training
- **Paper-fidelity audit** (arXiv:2504.13161): stella_en_400M_v5 embeddings, FAISS spherical K-means (K_init=1000), prune threshold 3.0, Dirichlet init on cluster token counts, predictor-guided sampling, LightGBM features = mixture weights, WSD annealing semantics — all verified aligned
- **stella NaN fix on torch_npu**: `position_ids` buffer held uninitialized heap garbage → OOB RoPE → NaN; repaired via `_repair_stella_buffers` in `embedding_cluster.py`
- **embedding_cluster only**: fdc_labels removed entirely, single discovery strategy
- **STEM benchmark evaluation** (full benchmark set, no per-task cap): arc_easy, arc_challenge, mmlu_stem, gpqa_diamond, gsm8k_cot, math_cot_500
- **70% STEM + 30% ClimbMix mixing**: adaptive shard count (3-50 shards, calculated from STEM doc count), reverse-order download (shard 6542→6541→...) to avoid overlap with pretrain data (shards 0-999)
- **mix_general_data.py reuse**: proxy_runner and target_runner load scripts/mix_general_data.py via importlib, no code duplication
- **auto_detect_depth_info**: 3-tier fallback (GPTConfig from meta_*.json → _approx_scaling_params formula → DEPTH_INFO table)
- **NPU support**: device_type=npu, embedding_cluster tries torch_npu first with CPU fallback (192 threads)
- **Parallel proxy search**: --npu-per-exp 1 → 8 experiments concurrently on 8 NPUs
- **Token caps**: --proxy-target-tokens / --target-tokens (default 200M proxy / 1B target in production, 10M in speedrun; human-readable "2B/10M/500K" syntax) — caps per-experiment I/O and peak RAM (8 parallel read_texts)
- **Seeded data selection**: cluster docs randomly permuted (seed = exp_id + 42) before token-budget prefix cut
- **Val split hygiene**: last shard_*.parquet = held-out validation (train shards exclude val docs); DDP row-group sizing mirrors prepare_shards.py
- **Parallel-safe eval via private base dirs**: base_eval.py writes its CSV to `{NANOCHAT_BASE_DIR}/base_eval/mid_model_{step}.csv` — step-only, no model tag (confirmed on remote), so parallel evals used to overwrite each other and were serialized behind a global lock. Now each eval subprocess gets a private base dir (symlink farm: mid_checkpoints/tokenizer/eval_bundle/eval_stem → real shared data; private base_eval/ + report/) — no lock, evals fully parallel, CSV attribution unambiguous
- **Eval subsample cap (`--eval-max-per-task`)**: base_eval shuffles each task with fixed seed 1337 before truncating, so capped runs still score the same subset across experiments (comparable). Speedrun: 100/task; production: -1 (full sets). Recorded in meta.json and the run fingerprint
- **No duplicate in-training benchmark eval**: proxy/target mid_train now pass `--core-metric-every=-1` (the default fires all 28 benchmarks at last_step inside training — measured ~2h10m/exp on the speedrun, duplicating the external base_eval; val bpb stays on as the training signal)
- **mid_train resume marker (`.mid_train_ok`)**: weights-sha256 + model-tag + checkpoint presence; a crash/kill during eval resumes at eval only instead of retraining (speedrun 2026-08-27: ^C during iteration-2's first eval)
- **Subprocess heartbeat**: every 5min the runner prints stage elapsed + last log line (mid_train step / eval task progress) + log path at start — a 3h eval no longer looks like a hang
- **nanochat-npu integration**: all training via subprocess torchrun, --device-type npu
- **Self-contained scripts**: get_model_info.py and mix_general_data.py in climbmix/scripts/, no quadmix dependency
- **HF_ENDPOINT defaults to hf-mirror.com in runs/*.sh**: corporate proxy selectively rejects Python CONNECT tunnels to huggingface.co (curl + Python-to-mirror both fine); mirror serves identical bytes; covers ClimbMix shards + eval_stem.zip; override with the env var
- **Fingerprint gap closure**: all semantic knobs previously hardcoded in `runs/*.sh` (proxy/target lr-scale/warmup/warmdown, K_init, filter-method, prune-threshold, merge-distance, embedding-model, search num-iterations, mid/eval device-batch-size, core-metric-every, NANOCHAT_DTYPE) are now `${VAR:-default}` shell vars passed to BOTH the invocation and the fingerprint `--param` list — changing any of them archives the stale output dir instead of silently resuming. `/tmp/opencode/test_fingerprint_p1.py` guards this: it parses the scripts and fails if a semantic flag consumes a `$VAR` missing from the fingerprint
- **Cluster-count band (pool-adaptive K)**: `K_final = clamp(natural_K(τ), K_enhanced=10, K_max=15)`. Floor = min search dimensionality (paper's fixed K_enhanced semantics), cap = search-budget bound (35 configs), τ=0.9 merge legality on unit-normalized stella embeddings (cos ~0.6). Inside the band the distance guard NEVER force-merges semantically distinct clusters (the failure mode where distinct subtopics share one weight and the true optimum becomes inexpressible); beyond the cap, closest-pair forced merges (logged) keep heterogeneous pools within budget. The paper merges to a fixed K_enhanced (20 in the main experiment) regardless of distance — the band is a deliberate deviation. Diagnostics: every run writes `merge_profile.json` (full dendrogram cut profile, natural_K at τ∈{0.7..1.2}, elbow suggestion) — the audit trail for K decisions per data pool
- **Pool-keyed stable caches**: embeddings + K-means live in `cache/embeddings/<sha256(pool manifest, model, truncate_len)[:12]>/` (NOT inside the fingerprinted output dir) — changing K_enhanced/K_max/τ/prune re-runs only prune+merge (seconds) instead of re-embedding the pool (hours). flock serializes concurrent runs over the same pool; kmeans cache keyed by K_init. Run-level artifacts (cluster_cache.npz, merge_profile.json) stay in the output dir and archive normally
- **`--num-iterations` validation** (run_climb.py): defaults to `len(configs_per_iter)`; explicit value that doesn't match the per-iteration list length is a `parser.error` (previously: extra iterations silently skipped)
- **search_state.json records `n_clusters`**: on resume, if the cluster count changed (recomputed cluster cache, edited data, or a run_climb.py invocation bypassing the shell fingerprint), the stale state is discarded with a warning instead of mixing wrong-dimension weight vectors into the predictor; old-format states (no field) are checked via weights length, pending-only states via pending configs
- **Crash resume (A+B+C+D)**: fingerprint auto-reset (code/param change → archive stale output dir), shard-level embedding resume (memmap + per-worker progress ledgers), iteration-level search state (`search_state.json`, atomic, with pending-iteration configs), experiment-level reuse (`exp_XXXX/meta.json` rc=0/0 + weight match, globally unique exp ids), predictor refit on resume (search continues paper-faithfully incl. full-design-space selection), atomic writes + `.done` markers for shards/mix/sampled/cluster caches, per-target `.done` markers with partial-checkpoint cleanup. EXP_NAME isolates experiments (output dir + proxy/target tags)

## Resolved Design Decisions

- **d20 ≈ CLIMB proxy**: scaling_params=435.2M, 1000 iterations, 8 parallel experiments (1 NPU each)
- **d28 target**: auto-detected from meta_*.json checkpoint, not hardcoded in DEPTH_INFO
- **Cluster-count band [10, 15] with τ=0.9**: floor 10 = min search dimensionality (paper merges to a fixed K_enhanced — 20 in its main experiment — regardless of distance); cap 15 = search-budget bound for our 35-config budget (paper used 112); τ=0.9 (~cos 0.6) rejects merges of semantically distinct clusters inside the band. The 1.5/3.0 thresholds are NOT in the paper text (ported from the original implementation); see merge_profile.json diagnostics per pool
- **ClimbMix adaptive shards**: calc_climbmix_count(stem_docs, ratio) → ceil(stem_docs * 0.3/0.7 / 500000), clamped [3, 50] — not all 400B
- **Reverse-order download**: shards from MAX_SHARD (6542) backwards, avoids overlap with pretrain shards 0-999
- **Stream-based mixing**: stream_texts_uniform + endless_generator, memory-efficient (no full List[str] loading)
- **STEM eval metric**: --eval-benchmarks=stem → CSV "STEM" row parsed as stem_metric; SNR-weighted acc+NLL z-score (see docs/scoring_metric_design.md)
- **Continual pre-training**: all stages anneal from base checkpoint, lr_scale=1.0, warmup=0.0, warmdown=0.9

## Remaining

- Server: `git pull` + `MIGRATE_LEGACY_FINGERPRINT=1 bash runs/speedrun_climbmix.sh` — adopts today's completed search (7/7 exps, weights C0=0.9486/C2=0.0514), skips Steps 1-5 via .done markers, reruns Step 6 with dbs=4 (~35min) + Step 7 eval (~15min)
- Speedrun passes → build RemoteExecutor (OBS + ModelArts API: ExpExecutor abstraction, remote_worker.py, dispatch_remote.py, Mock-first JobAPI), validate with a byte-identical exp_0000 bundle on a remote node (Δstem_metric < 0.002), then launch production `bash runs/run_climbmix.sh`
- Confirm on remote: mid_train val-shard convention (last shard = val, minimum val size)
- Analyze results, write final report

## Known Limitations

- Resume is atomic per experiment/training run, NOT per training step: an interrupted nanochat training restarts from step 0 (partial target checkpoints are deliberately cleared before retrain)
- Fingerprint does not detect: edits inside nanochat-npu, or data files whose names/row-counts are unchanged (content swaps)
- Predictor refit on resume assumes LightGBM determinism (same accumulated data → same model); verified identical weights in local integration tests
- quality_cluster discovery breaks when K_init > number of domains (centroids shape error) — unused in production (embedding_cluster)
- NPU embedding compatibility relies on the stella buffer repair; re-verify NaN=0 after any sentence-transformers upgrade
- ClimbMix shard count is adaptive (3-50), not full 400B dataset — only enough to match 30% of STEM doc count
- d28 DEPTH_INFO requires meta_*.json checkpoint for auto-detection; without checkpoint, raises ValueError
- STEM data fact density D unknown — cannot determine if subdisciplines cross phase transition threshold
- Intentional deviations from paper documented in docs/scoring_metric_design.md (SNR score, 70/30 anti-forgetting mix, chars/4 token estimation)
