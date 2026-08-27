#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: STEM 数据混合优化 — 单脚本全流程 (d20 → d28)
#
#  用法:   bash runs/run_climbmix.sh
#  实验:   EXP_NAME=myexp bash runs/run_climbmix.sh   (输出 result/myexp)
#
#  断点续跑 (直接重跑同一命令即可):
#    - 阶段指纹匹配 → 自动续跑: 聚类/搜索状态/已完成实验/target 训练/eval 全部复用
#    - search 指纹变(搜索语义代码或参数变更) → 旧目录归档 result/${EXP_NAME}_stale_<ts> 后全新开始
#    - target 指纹变(仅 target 语义变更) → 只归档 target 产物, 搜索结果保留, Steps 4-8 重跑
#  恢复粒度: 步骤级(.done) / 迭代级(search_state.json) / 实验级(exp_*/meta.json)
#            / embedding 分片级(进度账本) / 训练内部不支持(整次重跑)
#  旧版单一 .fingerprint 目录: MIGRATE_LEGACY_FINGERPRINT=1 采纳(不校验)。
#  num_npu 不进指纹(并行形状可变, 见 runs/lib/stage_gate.sh)。
#  注意: nanochat-npu 侧代码变更、同名数据文件内容变化不在指纹检测范围内
# ═══════════════════════════════════════════════════════════════════════
# Source CANN env BEFORE set -euo pipefail (set_env.sh may have commands
# that fail under strict mode, causing incomplete env setup)
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

# ── Configuration ──
CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-main}"
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

PROXY_DEPTH="${PROXY_DEPTH:-20}"
TARGET_DEPTH="${TARGET_DEPTH:-28}"
PROXY_NUM_ITERATIONS="${PROXY_NUM_ITERATIONS:-1000}"
TARGET_STEPS="${TARGET_STEPS:-1000}"
# Token caps for data selection (full pool ≈ 100B tokens / 116M docs — NOT capped means
# every proxy exp would select the whole pool; that's the default 0, so always set these).
# Proxy: 200M tokens/exp (~80-230K docs, ~1-2GB). 35 exps × 8 parallel: peak RAM ~18GB
# (read_texts) and ~35-80GB disk total. Proxy trains 1000 iters × ~0.5-1M tokens ≈ 0.5-1B
# tokens, so 200M data cycles 2.5-5x — same data:train ratio as the speedrun (10M/52M).
# 2B/exp instead would peak 8×~23GB RAM (OOM risk) and churn 280-800GB of parquet.
# Target: 1B tokens ≈ d28 anneal budget (1000 iters × ~1M) ≈ 1 epoch, no repetition.
PROXY_TARGET_TOKENS="${PROXY_TARGET_TOKENS:-200M}"
TARGET_TOKENS="${TARGET_TOKENS:-1B}"
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10,5}"
SEARCH_NUM_ITERATIONS="${SEARCH_NUM_ITERATIONS:-3}"
K_ENHANCED="${K_ENHANCED:-10}"
K_CLUSTER_MAX="${K_CLUSTER_MAX:-15}"
K_INIT="${K_INIT:-1000}"
FILTER_METHOD="${FILTER_METHOD:-none}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-3.0}"
MERGE_DISTANCE="${MERGE_DISTANCE:-0.9}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
# Stable pool-keyed cache for embeddings + K-means (survives fingerprint
# resets; K/merge knob changes reuse embeddings instead of re-embedding)
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$CLIMBMIX_DIR/cache/embeddings}"
DISCOVERY_METHOD="${DISCOVERY_METHOD:-embedding_cluster}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-npu}"
EMBEDDING_SAMPLE_SIZE="${EMBEDDING_SAMPLE_SIZE:-0}"
# Proxy/target training dynamics (semantic: change → fingerprint → fresh run)
PROXY_LR_SCALE="${PROXY_LR_SCALE:-1.0}"
PROXY_WARMUP="${PROXY_WARMUP:-0.0}"
PROXY_WARMDOWN="${PROXY_WARMDOWN:-0.9}"
TARGET_LR_SCALE="${TARGET_LR_SCALE:-1.0}"
TARGET_WARMUP="${TARGET_WARMUP:-0.0}"
TARGET_WARMDOWN="${TARGET_WARMDOWN:-0.9}"
# d28 Step-6 OOM evidence (speedrun 2026-08-27): dbs=8 override + full AdamW
# optimizer state (Step 6 loads it by design, ws=8 matches the d28 base) →
# 27.58 GiB allocated / 29.49 GiB HBM, 66 MiB short in the FIRST forward
# (apply_rotary_emb). dbs=4 (the d28 checkpoint's own inherited value) halves
# activations → same memory envelope as the d20 proxy runs (~23.7 GiB peak).
# dbs only re-slices micro-batches (total batch 1,048,576 unchanged), and both
# arms (climb/random) use the same value → scores stay comparable.
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-4}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-32}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
STEM_RATIO="${STEM_RATIO:-0.7}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"
# Eval subsample cap per task: -1 = FULL eval sets (production default).
# base_eval shuffles each task with a fixed seed (1337) before truncating,
# so any cap still yields comparable scores across experiments.
EVAL_MAX_PER_TASK="${EVAL_MAX_PER_TASK:--1}"
NUM_NPU="${NUM_NPU:-8}"
NPU_PER_EXP="${NPU_PER_EXP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/result/$EXP_NAME}"

# ── HF download endpoint ──
# The corporate proxy (proxy.modelarts.com) selectively rejects Python's bare
# CONNECT tunnels to huggingface.co (observed: 90+ consecutive 503s across two
# independent runs / 80 min, while curl to the same host AND Python to
# hf-mirror.com both succeeded). hf-mirror.com serves the same bytes (Range
# resume verified, 206). Covers ClimbMix shards + eval_stem.zip (nanochat
# reads HF_ENDPOINT at import time in dataset.py AND base_eval.py).
# Override to use the origin: HF_ENDPOINT=https://huggingface.co bash runs/...
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── NPU Environment (deliberately minimal — matches proven train_base_model.sh) ──
# See runs/speedrun_climbmix.sh: the allocator block that used to live here
# (memory_pool:True + PYTORCH_NPU_ALLOC_MAX_SIZE=60G + friends) filled device 0
# outside torch_npu's allocator accounting → kernel-load OOMs (aclnnMean 207001
# / EL0004) in every proxy exp and Step 6 on 2026-08-26. Do not re-add
# allocator overrides unless a specific need is proven on this hardware.
export OMP_NUM_THREADS=1 WANDB_MODE=offline NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
mkdir -p "$NANOCHAT_BASE_DIR"
export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200 HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_NPU - 1)))
export RANK_SIZE=$NUM_NPU MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export PYTHONUNBUFFERED=1
export NANOCHAT_DTYPE="$NANOCHAT_DTYPE" PYTHONWARNINGS="ignore::UserWarning:torch_npu"

# ── Stage-scoped fingerprints (code + semantic params → auto-reset on change) ──
# search guards Steps 1-3 products (embedding/cluster/search_state/exp_*/
# sampled_dataset); target guards Steps 4-8 products (shards, mixes, .done_*).
# num_npu deliberately NOT fingerprinted — parallel shape only, see
# runs/lib/stage_gate.sh. Params are split by which stage consumes them;
# shared params (data mix, eval sets, dtype, data dirs) enter BOTH stages.
FP_SEARCH_PARAMS=(
    "proxy_depth=$PROXY_DEPTH"
    "proxy_num_iterations=$PROXY_NUM_ITERATIONS"
    "proxy_target_tokens=$PROXY_TARGET_TOKENS"
    "configs_per_iter=$CONFIGS_PER_ITER"
    "search_num_iterations=$SEARCH_NUM_ITERATIONS"
    "K_enhanced=$K_ENHANCED"
    "K_cluster_max=$K_CLUSTER_MAX"
    "K_init=$K_INIT"
    "filter_method=$FILTER_METHOD"
    "prune_threshold=$PRUNE_THRESHOLD"
    "merge_distance=$MERGE_DISTANCE"
    "embedding_model=$EMBEDDING_MODEL"
    "discovery_method=$DISCOVERY_METHOD"
    "embedding_device=$EMBEDDING_DEVICE"
    "embedding_sample_size=$EMBEDDING_SAMPLE_SIZE"
    "proxy_lr_scale=$PROXY_LR_SCALE"
    "proxy_warmup=$PROXY_WARMUP"
    "proxy_warmdown=$PROXY_WARMDOWN"
    "npu_per_exp=$NPU_PER_EXP"
    "stem_ratio=$STEM_RATIO"
    "eval_benchmarks=$EVAL_BENCHMARKS"
    "eval_max_per_task=$EVAL_MAX_PER_TASK"
    "nanochat_dtype=$NANOCHAT_DTYPE"
    "data_dir=$DATA_DIR"
    "general_data_dir=$GENERAL_DATA_DIR"
)
FP_TARGET_PARAMS=(
    "target_depth=$TARGET_DEPTH"
    "target_steps=$TARGET_STEPS"
    "target_tokens=$TARGET_TOKENS"
    "target_lr_scale=$TARGET_LR_SCALE"
    "target_warmup=$TARGET_WARMUP"
    "target_warmdown=$TARGET_WARMDOWN"
    "mid_device_batch_size=$MID_DEVICE_BATCH_SIZE"
    "eval_device_batch_size=$EVAL_DEVICE_BATCH_SIZE"
    "core_metric_every=$CORE_METRIC_EVERY"
    "stem_ratio=$STEM_RATIO"
    "eval_benchmarks=$EVAL_BENCHMARKS"
    "eval_max_per_task=$EVAL_MAX_PER_TASK"
    "nanochat_dtype=$NANOCHAT_DTYPE"
    "data_dir=$DATA_DIR"
    "general_data_dir=$GENERAL_DATA_DIR"
)

source "$CLIMBMIX_DIR/runs/lib/stage_gate.sh"
run_stage_gate

# ── Pre-flight ──
echo -e "\n════════════════════════════════════════════════════════════"
echo "  ClimbMix: d${PROXY_DEPTH} proxy → d${TARGET_DEPTH} target  |  $OUTPUT_DIR"
echo "  NPU: ${NUM_NPU}x910B4, npu_per_exp=${NPU_PER_EXP} ($((NUM_NPU / NPU_PER_EXP)) parallel)"
echo "════════════════════════════════════════════════════════════"

python3 -c "import torch_npu; import torch; assert torch.npu.is_available(), 'NPU not available'" || { echo "✗ NPU not available"; exit 1; }
[ -d "$NANOCHAT_DIR" ] || { echo "✗ nanochat-npu not found at $NANOCHAT_DIR"; exit 1; }
for d in "$PROXY_DEPTH" "$TARGET_DEPTH"; do
    ckpt="$NANOCHAT_BASE_DIR/base_checkpoints/d${d}"
    [ -d "$ckpt" ] && ls "$ckpt"/model_*.pt >/dev/null 2>&1 || { echo "✗ d${d} checkpoint not found"; exit 1; }
    echo "✓ d${d} checkpoint"
done

( cd "$NANOCHAT_DIR" && python3 -c "from scripts.base_eval import prepare_eval_data; prepare_eval_data('stem')" 2>/dev/null ) || true

# ═══════════════════════════════════════════════════════════════════════
#  Step 1-3: Embedding + Proxy Search + Data Selection
#  (embedding 分片级续跑 + 聚类缓存 + search_state 迭代级续跑 +
#   exp_*/meta.json 实验级复用 — 均自动)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$OUTPUT_DIR/sampled_dataset.parquet" ]; then
    echo -e "\n===== Step 1-3: Proxy Search — already complete (sampled_dataset.parquet), skip =====\n"
else
    echo -e "\n===== Step 1-3: Proxy Search (d${PROXY_DEPTH}) =====\n"

    python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
        --data-dir "$DATA_DIR" \
        --nanochat-dir "$NANOCHAT_DIR" \
        --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
        --general-data-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" \
        --eval-benchmarks "$EVAL_BENCHMARKS" \
        --eval-max-per-task "$EVAL_MAX_PER_TASK" \
        --proxy-depth "$PROXY_DEPTH" \
        --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
        --proxy-target-tokens "$PROXY_TARGET_TOKENS" \
        --proxy-lr-scale "$PROXY_LR_SCALE" --proxy-warmup "$PROXY_WARMUP" --proxy-warmdown "$PROXY_WARMDOWN" \
        --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
        --target-depth "$TARGET_DEPTH" \
        --target-tokens "$TARGET_TOKENS" \
        --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
        --K-enhanced "$K_ENHANCED" \
        --K-max "$K_CLUSTER_MAX" \
        --K-init "$K_INIT" \
        --filter-method "$FILTER_METHOD" \
        --prune-threshold "$PRUNE_THRESHOLD" \
        --merge-distance "$MERGE_DISTANCE" \
        --embedding-model "$EMBEDDING_MODEL" \
        --num-iterations "$SEARCH_NUM_ITERATIONS" \
        --discovery-method "$DISCOVERY_METHOD" \
        --embedding-device "$EMBEDDING_DEVICE" \
        --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
        --configs-per-iter "$CONFIGS_PER_ITER" \
        --device-type npu --npu-devices "$NUM_NPU" --npu-per-exp "$NPU_PER_EXP" \
        --output-dir "$OUTPUT_DIR" \
        --exp-name "$EXP_NAME" \
        --cluster-cache-dir "$OUTPUT_DIR" \
        --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
        --resume-search \
        --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
        --skip-target
fi

[ ! -f "$OUTPUT_DIR/sampled_dataset.parquet" ] && { echo "✗ No sampled_dataset.parquet"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Prepare Target Data (shards + random baseline)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 4: Prepare Target Data =====\n"

CLIMB_SHARDS="$OUTPUT_DIR/climb_shards"
RANDOM_SHARDS="$OUTPUT_DIR/random_shards"

python3 "$CLIMBMIX_DIR/scripts/prepare_shards.py" \
    --input "$OUTPUT_DIR/sampled_dataset.parquet" \
    --output-dir "$CLIMB_SHARDS" --num-npu "$NUM_NPU"

if [ -f "$RANDOM_SHARDS/.done" ]; then
    echo "  Random baseline: already complete (.done), skip"
else
    DOC_COUNT=$(python3 -c "import pyarrow.parquet as pq; print(pq.ParquetFile('$OUTPUT_DIR/sampled_dataset.parquet').metadata.num_rows)")
    python3 "$CLIMBMIX_DIR/scripts/prepare_random_baseline.py" \
        --data-dir "$DATA_DIR" --output-dir "$RANDOM_SHARDS" \
        --num-docs "$DOC_COUNT" --seed 42 --num-npu "$NUM_NPU"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Mix STEM + General Data (anti-forgetting)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 5: Mix STEM + General Data (ratio=$STEM_RATIO) =====\n"

mix_one() {
    local stem_dir="$1" out_dir="$2" label="$3"
    [ -d "$stem_dir" ] || return 0
    [ -f "$out_dir/.done" ] && { echo "  $label: already mixed (.done), skip"; return; }
    NANOCHAT_REPO="$NANOCHAT_DIR" python3 "$CLIMBMIX_DIR/scripts/mix_general_data.py" \
        --stem-dir "$stem_dir" --output-dir "$out_dir" \
        --climbmix-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" --num-workers "$NUM_NPU" --num-npu "$NUM_NPU" \
        || { echo "✗ Mix failed for $label"; exit 1; }
}

mix_one "$CLIMB_SHARDS" "$OUTPUT_DIR/climb_mixed" "CLIMB"
mix_one "$RANDOM_SHARDS" "$OUTPUT_DIR/random_mixed" "Random"
CLIMB_DATA="$OUTPUT_DIR/climb_mixed"
RANDOM_DATA="$OUTPUT_DIR/random_mixed"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6: Target Training (d28 mid_train) — climb/random 独立 .done 标记
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6: Target Training (d${TARGET_DEPTH}) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_climb_${EXP_NAME}"
RANDOM_TAG="d${TARGET_DEPTH}_random_${EXP_NAME}"

run_mid_train() {
    local data_dir="$1" tag="$2" name="$3"
    local link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
    # Clean a stale/broken symlink from a previous crashed attempt BEFORE the
    # `[ -e ] || ln -s`: a broken link fails `[ -e ]` yet still blocks ln -s
    # (EEXIST), which kills the script under set -e.
    if [ -L "$link_dir" ] && [ ! -e "$link_dir" ]; then rm -f "$link_dir"; fi
    [ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"
    # Clear partial checkpoints from a crashed attempt (nanochat may otherwise
    # try to auto-resume from inconsistent state; whole-run atomicity instead)
    rm -rf "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag"
    ( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$TARGET_STEPS" \
        --lr-scale="$TARGET_LR_SCALE" --warmup-ratio="$TARGET_WARMUP" --warmdown-ratio="$TARGET_WARMDOWN" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --device-batch-size="$MID_DEVICE_BATCH_SIZE" \
        --run="${name}_mid" --model-tag="$tag" \
        --data-dir="$data_dir" 2>&1 | tee "$OUTPUT_DIR/mid_train_${name}.log" )
    # NOT `[ -L ] && rm` as the last statement: when link_dir is absent or not
    # a symlink the function would return 1, and under set -e the script dies
    # AFTER successful training with .done unwritten → retrain on every resume.
    if [ -L "$link_dir" ]; then rm -f "$link_dir"; fi
}

if [ -f "$OUTPUT_DIR/.done_mid_train_climb" ]; then
    echo "  mid_train climb: already done, skip"
else
    run_mid_train "$CLIMB_DATA" "$CLIMB_TAG" "climb"
    touch "$OUTPUT_DIR/.done_mid_train_climb"
fi

if [ -f "$OUTPUT_DIR/.done_mid_train_random" ]; then
    echo "  mid_train random: already done, skip"
else
    run_mid_train "$RANDOM_DATA" "$RANDOM_TAG" "random"
    touch "$OUTPUT_DIR/.done_mid_train_random"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 7: Evaluation
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 7: Evaluation =====\n"

run_eval() {
    local tag="$1" name="$2"
    ( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --max-per-task="$EVAL_MAX_PER_TASK" \
        --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
        --model-tag="$tag" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_${name}.log" )
}

if [ -f "$OUTPUT_DIR/.done_eval_climb" ]; then
    echo "  eval climb: already done, skip"
else
    run_eval "$CLIMB_TAG" "climb"
    touch "$OUTPUT_DIR/.done_eval_climb"
fi

if [ -f "$OUTPUT_DIR/.done_eval_random" ]; then
    echo "  eval random: already done, skip"
else
    run_eval "$RANDOM_TAG" "random"
    touch "$OUTPUT_DIR/.done_eval_random"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 8: Report (幂等, 总是重新生成)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 8: Report =====\n"

python3 "$CLIMBMIX_DIR/src/climbmix/pipeline/report_generator.py" \
    --result-dir "$OUTPUT_DIR" \
    --climb-train-log "$OUTPUT_DIR/mid_train_climb.log" \
    --random-train-log "$OUTPUT_DIR/mid_train_random.log" \
    --climb-eval-log "$OUTPUT_DIR/eval_climb.log" \
    --random-eval-log "$OUTPUT_DIR/eval_random.log" \
    --base-model-tag "d${TARGET_DEPTH}" \
    --climb-model-tag "$CLIMB_TAG" \
    --random-model-tag "$RANDOM_TAG"

echo -e "\n════════════════════════════════════════════════════════════"
echo "  Done! → $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"
