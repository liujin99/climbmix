# CLIMBmix — Progress & Remaining Work

## Completed

- **d20 proxy → d28 target pipeline**: embedding cluster → Dirichlet search → LightGBM predictor → d28 target training
- **Paper-fidelity audit** (arXiv:2504.13161): stella_en_400M_v5 embeddings, FAISS spherical K-means (K_init=1000), prune threshold 3.0, merge distance 1.5, Dirichlet init on cluster token counts, predictor-guided sampling, LightGBM features = mixture weights, WSD annealing semantics — all verified aligned
- **stella NaN fix on torch_npu**: `position_ids` buffer held uninitialized heap garbage → OOB RoPE → NaN; repaired via `_repair_stella_buffers` in `embedding_cluster.py`
- **embedding_cluster only**: fdc_labels removed entirely, single discovery strategy
- **STEM benchmark evaluation** (full benchmark set, no per-task cap): arc_easy, arc_challenge, mmlu_stem, gpqa_diamond, gsm8k_cot, math_cot_500
- **70% STEM + 30% ClimbMix mixing**: adaptive shard count (3-50 shards, calculated from STEM doc count), reverse-order download (shard 6542→6541→...) to avoid overlap with pretrain data (shards 0-999)
- **mix_general_data.py reuse**: proxy_runner and target_runner load scripts/mix_general_data.py via importlib, no code duplication
- **auto_detect_depth_info**: 3-tier fallback (GPTConfig from meta_*.json → _approx_scaling_params formula → DEPTH_INFO table)
- **NPU support**: device_type=npu, embedding_cluster tries torch_npu first with CPU fallback (192 threads)
- **Parallel proxy search**: --npu-per-exp 1 → 8 experiments concurrently on 8 NPUs
- **Token caps**: --proxy-target-tokens / --target-tokens (default 2B in production, 10M in speedrun; human-readable "2B/10M/500K" syntax) — cuts per-experiment I/O ~145× in speedrun
- **Seeded data selection**: cluster docs randomly permuted (seed = exp_id + 42) before token-budget prefix cut
- **Val split hygiene**: last shard_*.parquet = held-out validation (train shards exclude val docs); DDP row-group sizing mirrors prepare_shards.py
- **Eval CSV location**: 3-tier strategy (filename contains model_tag → mtime within eval window → global newest); failed evals skip parsing instead of reading stale CSVs
- **nanochat-npu integration**: all training via subprocess torchrun, --device-type npu
- **Self-contained scripts**: get_model_info.py and mix_general_data.py in climbmix/scripts/, no quadmix dependency

## Resolved Design Decisions

- **d20 ≈ CLIMB proxy**: scaling_params=435.2M, 1000 iterations, 8 parallel experiments (1 NPU each)
- **d28 target**: auto-detected from meta_*.json checkpoint, not hardcoded in DEPTH_INFO
- **K_enhanced=10 as lower bound**: cluster merging stops when closest pair > merge_distance (1.5) even if target K not reached, per paper §2.1
- **ClimbMix adaptive shards**: calc_climbmix_count(stem_docs, ratio) → ceil(stem_docs * 0.3/0.7 / 500000), clamped [3, 50] — not all 400B
- **Reverse-order download**: shards from MAX_SHARD (6542) backwards, avoids overlap with pretrain shards 0-999
- **Stream-based mixing**: stream_texts_uniform + endless_generator, memory-efficient (no full List[str] loading)
- **STEM eval metric**: --eval-benchmarks=stem → CSV "STEM" row parsed as stem_metric; SNR-weighted acc+NLL z-score (see docs/scoring_metric_design.md)
- **Continual pre-training**: all stages anneal from base checkpoint, lr_scale=1.0, warmup=0.0, warmdown=0.9

## Remaining

- Confirm on remote: how nanochat base_eval.py names its CSV files (tag vs timestamp) — 3-tier fallback handles both, but tier-1 confirmation removes ambiguity under parallel experiments
- Confirm on remote: mid_train val-shard convention (last shard = val, minimum val size)
- Pull latest code on remote and run `bash runs/speedrun_climbmix.sh` (watch: NaN=0, K=100 clustering, 50-step train+eval)
- Speedrun passes → run production `bash runs/run_climbmix.sh` (d20 search, 2B token caps)
- Analyze results, write final report

## Known Limitations

- NPU embedding compatibility relies on the stella buffer repair; re-verify NaN=0 after any sentence-transformers upgrade
- ClimbMix shard count is adaptive (3-50), not full 400B dataset — only enough to match 30% of STEM doc count
- d28 DEPTH_INFO requires meta_*.json checkpoint for auto-detection; without checkpoint, raises ValueError
- STEM data fact density D unknown — cannot determine if subdisciplines cross phase transition threshold
- Intentional deviations from paper documented in docs/scoring_metric_design.md (SNR score, 70/30 anti-forgetting mix, chars/4 token estimation)
