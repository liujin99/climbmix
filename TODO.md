# CLIMBmix — Progress & Remaining Work

## Completed

- **All 8 paper alignment fixes** (commit 3d10abd)
- **Staged experiment design**: d10→d24 (Stage 1) → d14/d18→d24 (Stage 2)
- **method A ProxyRunner**: subprocess calls nanochat mid_train.py + base_eval.py
- **Mixture data preparation**: per-experiment parquet dir with weighted sampling
- **Annealing config**: lr_scale=1.0, warmup=0.0, warmdown=0.9
- **nanochat depth-based sizing**: DEPTH_INFO table, scaling_params vs total_params
- **High-signal task subset**: 6 tasks (centered>0.1) for search metric
- **Unique model-tag per experiment**: symlink base checkpoint + copy mid checkpoint
- **nanochat dependency validation**: ProxyRunner._validate_nanochat() at init
- **Auto report generation**: Markdown + matplotlib per experiment run

## Resolved Design Decisions

- **Continual pre-training**: all stages anneal from base, never from-scratch
- **Same metric for search AND validation**: no distortion
- **Fixed training steps**: --num-iterations, not ratio × scaling_params
- **scaling_params for CLIMB alignment**: VE is embedding lookup, not core compute
- **d10 ≈ CLIMB 62M proxy**: ablation shows even 62M works (+0.74 vs RegMix)
- **Stage 1 = d10→d24**: validates proxy→target transferability, not just "can it run"

## Remaining

- Run Stage 0 dry-run on CPU
- Obtain d10 + d24 base checkpoints (phase-1 pretraining on NPU)
- Download more Essential-Web shards (currently only 2 of 3291)
- Run Stage 1 on NPU (d10 search + d24 validation)
- Analyze results, write final report

## Known Limitations

- d24 scaling=730M is 56% of CLIMB 1.3B — report must note this honestly
- d10 CORE=0.1001 marginal signal — using high-signal subset mitigates
- Only 2 Essential-Web shards available — need more for real training
