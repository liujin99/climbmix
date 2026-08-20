# CLIMBmix — Progress & Remaining Work

## Completed

- **d14 proxy → d28 target pipeline**: embedding cluster → Dirichlet search → LightGBM predictor → d28 target training
- **embedding_cluster only**: fdc_labels removed entirely, single discovery strategy
- **STEM benchmark evaluation**: arc_easy, arc_challenge, mmlu_stem, gpqa_diamond, gsm8k_cot, math_cot_500
- **70% STEM + 30% ClimbMix mixing**: adaptive shard count (3-50 shards, calculated from STEM doc count), reverse-order download (shard 6542→6541→...) to avoid overlap with pretrain data (shards 0-999)
- **mix_general_data.py reuse**: proxy_runner and target_runner load scripts/mix_general_data.py via importlib, no code duplication
- **auto_detect_depth_info**: 3-tier fallback (GPTConfig from meta_*.json → _approx_scaling_params formula → DEPTH_INFO table)
- **NPU support**: device_type=npu, embedding_cluster tries torch_npu first with CPU fallback (192 threads)
- **TargetRunner (Stage 6)**: d28 training + STEM eval with optimal mixture weights
- **nanochat-npu integration**: all training via subprocess torchrun, --device-type npu
- **Self-contained scripts**: get_model_info.py and mix_general_data.py copied to climbmix/scripts/, no quadmix dependency

## Resolved Design Decisions

- **d14 ≈ CLIMB 164M proxy**: scaling_params=164.2M, 500 iterations, sufficient signal for search
- **d28 target**: auto-detected from meta_*.json checkpoint, not hardcoded in DEPTH_INFO
- **ClimbMix adaptive shards**: calc_climbmix_count(stem_docs, ratio) → ceil(stem_docs * 0.3/0.7 / 500000), clamped [3, 50] — not all 400B
- **Reverse-order download**: shards from MAX_SHARD (6542) backwards, avoids overlap with pretrain shards 0-999
- **Stream-based mixing**: stream_texts_uniform + endless_generator, memory-efficient (no full List[str] loading)
- **STEM eval metric**: --eval-benchmarks=stem → CSV "STEM" row parsed as stem_metric
- **Continual pre-training**: all stages anneal from base checkpoint, lr_scale=1.0, warmup=0.0, warmdown=0.9

## Remaining

- Sync code to remote NPU machine (8×910B Ascend)
- Run search_d14.sh pre-flight checks (deps, nanochat, STEM data, NPU, checkpoints, disk)
- Test NPU embedding: sentence_transformers + stella_en_400M_v5 on Ascend 910B (fallback: 192 vCPU CPU)
- Test d28 meta_*.json auto-read from remote checkpoint
- Pre-download ClimbMix shards (adaptive 3-50, reverse order)
- Run embedding cluster (NPU or CPU fallback)
- Run full Stage 1: d14 proxy search → d28 target training
- Analyze results, write final report

## Known Limitations

- NPU embedding compatibility untested: sentence_transformers may not support Ascend NPU inference, CPU fallback uses 192 vCPUs
- ClimbMix shard count is adaptive (3-50), not full 400B dataset — only enough to match 30% of STEM doc count
- d28 DEPTH_INFO requires meta_*.json checkpoint for auto-detection; without checkpoint, raises ValueError
- STEM data fact density D unknown — cannot determine if subdisciplines cross phase transition threshold
