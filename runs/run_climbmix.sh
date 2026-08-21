#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: STEM 数据混合优化 — 单脚本全流程 (d20 → d28)
#  用法:  bash runs/run_climbmix.sh
#  跳过步骤: 注释掉对应的行
#  恢复搜索/跳过聚类: 自动 (检测 search_state.json / cluster_cache.npz)
# ═══════════════════════════════════════════════════════════════════════
# Source CANN env BEFORE set -euo pipefail (set_env.sh may have commands
# that fail under strict mode, causing incomplete env setup)
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

# ── Configuration ──
CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

PROXY_DEPTH="${PROXY_DEPTH:-20}"
TARGET_DEPTH="${TARGET_DEPTH:-28}"
PROXY_NUM_ITERATIONS="${PROXY_NUM_ITERATIONS:-1000}"
TARGET_STEPS="${TARGET_STEPS:-1000}"
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10,5}"
K_ENHANCED="${K_ENHANCED:-10}"
STEM_RATIO="${STEM_RATIO:-0.7}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"
NUM_NPU="${NUM_NPU:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/result/run_$(date +%Y%m%d_%H%M%S)}"
TS=$(date +%Y%m%d_%H%M%S)

# ── NPU Environment ──
export OMP_NUM_THREADS=1 WANDB_MODE=offline NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
mkdir -p "$NANOCHAT_BASE_DIR"
export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200 HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
export PYTORCH_ALLOC_CONF=expandable_segments:True ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_NPU - 1)))
export RANK_SIZE=$NUM_NPU MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200 ASCEND_DISABLE_MEM_SWAP=1 ASCEND_LAUNCH_BLOCKING=0
export NPU_DISABLE_RECORD=1 PYTHONUNBUFFERED=1 ASCEND_COMPILE_OPT_LEVEL=O3
export TORCH_NPU_LAZY_COMPILE=1 PYTHONPRELOAD=torch_npu
export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G ASCEND_ENABLE_CACHE=1 ASCEND_FUSION_ENABLE=1
export NANOCHAT_DTYPE=bfloat16 PYTHONWARNINGS="ignore::UserWarning:torch_npu"

# ── Pre-flight ──
echo -e "\n════════════════════════════════════════════════════════════"
echo "  ClimbMix: d${PROXY_DEPTH} proxy → d${TARGET_DEPTH} target  |  $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

python3 -c "import torch_npu; import torch; assert torch.npu.is_available(), 'NPU not available'" || { echo "✗ NPU not available"; exit 1; }
[ -d "$NANOCHAT_DIR" ] || { echo "✗ nanochat-npu not found at $NANOCHAT_DIR"; exit 1; }
for d in "$PROXY_DEPTH" "$TARGET_DEPTH"; do
    ckpt="$NANOCHAT_BASE_DIR/base_checkpoints/d${d}"
    [ -d "$ckpt" ] && ls "$ckpt"/model_*.pt >/dev/null 2>&1 || { echo "✗ d${d} checkpoint not found"; exit 1; }
    echo "✓ d${d} checkpoint"
done

mkdir -p "$OUTPUT_DIR"
( cd "$NANOCHAT_DIR" && python3 -c "from scripts.base_eval import prepare_eval_data; prepare_eval_data('stem')" 2>/dev/null ) || true

# ═══════════════════════════════════════════════════════════════════════
#  Step 1-3: Embedding + Proxy Search + Data Selection
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 1-3: Proxy Search (d${PROXY_DEPTH}) =====\n"

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --nanochat-dir "$NANOCHAT_DIR" \
    --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
    --general-data-dir "$GENERAL_DATA_DIR" \
    --stem-ratio "$STEM_RATIO" \
    --eval-benchmarks "$EVAL_BENCHMARKS" \
    --proxy-depth "$PROXY_DEPTH" \
    --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
    --proxy-lr-scale 1.0 --proxy-warmup 0.0 --proxy-warmdown 0.9 \
    --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
    --target-depth "$TARGET_DEPTH" \
    --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
    --K-enhanced "$K_ENHANCED" \
    --configs-per-iter "$CONFIGS_PER_ITER" \
    --device-type npu --npu-devices "$NUM_NPU" \
    --output-dir "$OUTPUT_DIR" \
    --cluster-cache-dir "$OUTPUT_DIR" \
    --resume-search \
    --quality-config-path "$CLIMBMIX_DIR/config/quality_columns.yaml" \
    --skip-target

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

DOC_COUNT=$(python3 -c "import pyarrow.parquet as pq; print(len(pq.read_table('$OUTPUT_DIR/sampled_dataset.parquet', columns=['text'])['text']))")
python3 "$CLIMBMIX_DIR/scripts/prepare_random_baseline.py" \
    --data-dir "$DATA_DIR" --output-dir "$RANDOM_SHARDS" \
    --num-docs "$DOC_COUNT" --seed 42 --num-npu "$NUM_NPU"

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Mix STEM + General Data (anti-forgetting)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 5: Mix STEM + General Data (ratio=$STEM_RATIO) =====\n"

mix_one() {
    local stem_dir="$1" out_dir="$2" label="$3"
    [ -d "$stem_dir" ] || return 0
    [ "$(ls "$out_dir"/shard_*.parquet 2>/dev/null | wc -l)" -gt 0 ] && { echo "  $label: already mixed, skip"; return; }
    python3 "$CLIMBMIX_DIR/scripts/mix_general_data.py" \
        --stem-dir "$stem_dir" --output-dir "$out_dir" \
        --climbmix-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" --num-workers "$NUM_NPU" \
        || { echo "✗ Mix failed for $label"; exit 1; }
}

mix_one "$CLIMB_SHARDS" "$OUTPUT_DIR/climb_mixed" "CLIMB"
mix_one "$RANDOM_SHARDS" "$OUTPUT_DIR/random_mixed" "Random"
CLIMB_DATA="$OUTPUT_DIR/climb_mixed"
RANDOM_DATA="$OUTPUT_DIR/random_mixed"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6: Target Training (d28 mid_train)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6: Target Training (d${TARGET_DEPTH}) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_climb_${TS}"
RANDOM_TAG="d${TARGET_DEPTH}_random_${TS}"

run_mid_train() {
    local data_dir="$1" tag="$2" name="$3"
    local link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
    [ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"
    ( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$TARGET_STEPS" \
        --lr-scale=1.0 --warmup-ratio=0.0 --warmdown-ratio=0.9 \
        --device-batch-size=8 \
        --run="${name}_mid" --model-tag="$tag" \
        --eval-benchmarks="$EVAL_BENCHMARKS" \
        --data-dir="$data_dir" 2>&1 | tee "$OUTPUT_DIR/mid_train_${name}.log" )
    [ -L "$link_dir" ] && rm "$link_dir"
}

run_mid_train "$CLIMB_DATA" "$CLIMB_TAG" "climb"
run_mid_train "$RANDOM_DATA" "$RANDOM_TAG" "random"

# ═══════════════════════════════════════════════════════════════════════
#  Step 7: Evaluation
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 7: Evaluation =====\n"

run_eval() {
    local tag="$1" name="$2"
    ( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --device-batch-size=32 \
        --model-tag="$tag" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_${name}.log" )
}

run_eval "$CLIMB_TAG" "climb"
run_eval "$RANDOM_TAG" "random"

# ═══════════════════════════════════════════════════════════════════════
#  Step 8: Report
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
