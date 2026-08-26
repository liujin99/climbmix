#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix Speedrun — 全流程端到端验证 (最小数据 + 最少步数)
#
#  目的: 验证代码正确性 (embedding → cluster → search → mid_train → eval)
#  不关注结果质量, 只确认所有步骤跑通无报错
#
#  设置:
#    - 5 个 parquet 文件 (~580K docs)
#    - configs=2,3,2 (7 个), proxy_steps=50, target_steps=50
#    - 8 NPU 做 embedding, 1 NPU/experiment 做 proxy search
#    - 70% STEM + 30% ClimbMix general data (含 proxy 实验内混合 + Step 5 混合;
#      首次运行会下载 3 个 ClimbMix 分片到 GENERAL_DATA_DIR, 之后复用缓存;
#      该缓存也供 full run 复用)。不做 random baseline。
#
#  用法:  bash runs/speedrun_climbmix.sh
# ═══════════════════════════════════════════════════════════════════════
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

# ── Configuration ──
CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
SPEED_DATA="/tmp/speedrun_data"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

PROXY_DEPTH=20
TARGET_DEPTH=28
PROXY_NUM_ITERATIONS=50
TARGET_STEPS=50
CONFIGS_PER_ITER="2,3,2"
K_ENHANCED=10
K_INIT=100
NUM_NPU=8
NPU_PER_EXP=1
EMBEDDING_SAMPLE_SIZE=2000
EVAL_BENCHMARKS="stem"
STEM_RATIO="${STEM_RATIO:-0.7}"
PROXY_TARGET_TOKENS=10M
TARGET_TOKENS=10M

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$CLIMBMIX_DIR/result/speedrun_${TS}"

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
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G ASCEND_ENABLE_CACHE=1
export NANOCHAT_DTYPE=bfloat16 PYTHONWARNINGS="ignore::UserWarning:torch_npu"

# ── Pre-flight ──
echo -e "\n════════════════════════════════════════════════════════════"
echo "  ClimbMix Speedrun: d${PROXY_DEPTH} → d${TARGET_DEPTH}  |  $OUTPUT_DIR"
echo "  ${NUM_NPU}x910B4, ${NPU_PER_EXP} NPU/exp ($((NUM_NPU / NPU_PER_EXP)) parallel)"
echo "  Data: 5 parquet files | Configs: ${CONFIGS_PER_ITER} | Steps: ${PROXY_NUM_ITERATIONS}"
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
#  Step 0: Prepare small data (5 parquet files)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 0: Prepare small data (5 parquet files) =====\n"

if [ ! -d "$SPEED_DATA" ] || [ "$(ls "$SPEED_DATA"/*.parquet 2>/dev/null | wc -l)" -lt 5 ]; then
    mkdir -p "$SPEED_DATA"
    rm -f "$SPEED_DATA"/*.parquet "$SPEED_DATA"/*.npz 2>/dev/null
    ls "$DATA_DIR"/part-*.parquet 2>/dev/null | head -5 | xargs -I{} cp {} "$SPEED_DATA/"
fi
echo "Speedrun data files:"
ls -lh "$SPEED_DATA"/*.parquet

# ═══════════════════════════════════════════════════════════════════════
#  Step 1-3: Embedding + Proxy Search + Data Selection
#  Tests: ProxyRunner._build_mid_train_cmd + _build_eval_cmd (fixed with --)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 1-3: Proxy Search (d${PROXY_DEPTH}, ${CONFIGS_PER_ITER} configs, ${PROXY_NUM_ITERATIONS} steps) =====\n"

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$SPEED_DATA" \
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
    --K-init "$K_INIT" \
    --discovery-method embedding_cluster \
    --embedding-device npu \
    --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
    --configs-per-iter "$CONFIGS_PER_ITER" \
    --device-type npu --npu-devices "$NUM_NPU" --npu-per-exp "$NPU_PER_EXP" \
    --output-dir "$OUTPUT_DIR" \
    --cluster-cache-dir "$OUTPUT_DIR" \
    --resume-search \
    --proxy-target-tokens "$PROXY_TARGET_TOKENS" \
    --target-tokens "$TARGET_TOKENS" \
    --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
    --skip-target

[ ! -f "$OUTPUT_DIR/sampled_dataset.parquet" ] && { echo "✗ No sampled_dataset.parquet — Step 1-3 FAILED"; exit 1; }
echo -e "\n✓ Step 1-3 complete: sampled_dataset.parquet found"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Prepare Target Data (shards only, skip random baseline)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 4: Prepare Target Data =====\n"

CLIMB_SHARDS="$OUTPUT_DIR/climb_shards"

python3 "$CLIMBMIX_DIR/scripts/prepare_shards.py" \
    --input "$OUTPUT_DIR/sampled_dataset.parquet" \
    --output-dir "$CLIMB_SHARDS" --num-npu "$NUM_NPU"

echo "✓ Shards prepared: $(ls "$CLIMB_SHARDS"/shard_*.parquet 2>/dev/null | wc -l) files"

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Mix STEM + General Data (anti-forgetting, ratio=$STEM_RATIO)
#  Tests: mix_general_data.py CLI (download cache + stream mixing + val 透传)
#  输出 shard 的 row group 按 NUM_NPU 自适应 (DDP 安全)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 5: Mix STEM + General Data (ratio=$STEM_RATIO) =====\n"

CLIMB_MIXED="$OUTPUT_DIR/climb_mixed"
NANOCHAT_REPO="$NANOCHAT_DIR" python3 "$CLIMBMIX_DIR/scripts/mix_general_data.py" \
    --stem-dir "$CLIMB_SHARDS" --output-dir "$CLIMB_MIXED" \
    --climbmix-dir "$GENERAL_DATA_DIR" \
    --stem-ratio "$STEM_RATIO" --num-workers "$NUM_NPU" --num-npu "$NUM_NPU" \
    || { echo "✗ Mix failed"; exit 1; }

CLIMB_DATA="$CLIMB_MIXED"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6: Target Training (d28 mid_train, ${TARGET_STEPS} steps)
#  Tests: shell torchrun mid_train command (already has --)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6: Target Training (d${TARGET_DEPTH}, ${TARGET_STEPS} steps) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_speedrun_${TS}"
link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$CLIMB_TAG"
[ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"

( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
    --num-iterations="$TARGET_STEPS" \
    --lr-scale=1.0 --warmup-ratio=0.0 --warmdown-ratio=0.9 \
    --device-batch-size=8 \
    --run="speedrun_climb" --model-tag="$CLIMB_TAG" \
    --eval-benchmarks="$EVAL_BENCHMARKS" \
    --data-dir="$CLIMB_DATA" 2>&1 | tee "$OUTPUT_DIR/mid_train_climb.log" ) || {
    echo "✗ Target mid_train FAILED"
    [ -L "$link_dir" ] && rm "$link_dir"
    exit 1
}
[ -L "$link_dir" ] && rm "$link_dir"
echo "✓ Target training complete"

# ═══════════════════════════════════════════════════════════════════════
#  Step 7: Evaluation
#  Tests: shell torchrun base_eval command (already has --)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 7: Evaluation =====\n"

( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
    --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
    --device-batch-size=32 \
    --model-tag="$CLIMB_TAG" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_climb.log" ) || {
    echo "✗ Eval FAILED"
    exit 1
}
echo "✓ Evaluation complete"

# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n════════════════════════════════════════════════════════════"
echo "  Speedrun Complete! All steps passed ✓"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  Verified code paths:"
echo "    ✓ Embedding (fallback path, 0% NaN)"
echo "    ✓ Clustering (FAISS K-means + merge)"
echo "    ✓ Proxy search (ProxyRunner: mix + mid_train + eval × 7 configs)"
echo "    ✓ Data selection (mixture weights → sampled_dataset)"
echo "    ✓ Shard preparation (prepare_shards.py)"
echo "    ✓ General data mixing (mix_general_data.py, stem_ratio=$STEM_RATIO)"
echo "    ✓ Target training (d${TARGET_DEPTH} mid_train, ${TARGET_STEPS} steps)"
echo "    ✓ Target evaluation (base_eval)"
echo ""
echo "  Output files:"
ls -lh "$OUTPUT_DIR"/*.parquet "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.log 2>/dev/null || echo "    (check output dir)"
echo "════════════════════════════════════════════════════════════"
