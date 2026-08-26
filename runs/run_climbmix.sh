#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: STEM 数据混合优化 — 单脚本全流程 (d20 → d28)
#
#  用法:   bash runs/run_climbmix.sh
#  实验:   EXP_NAME=myexp bash runs/run_climbmix.sh   (输出 result/myexp)
#
#  断点续跑 (直接重跑同一命令即可):
#    - 指纹匹配  → 自动续跑: 聚类/搜索状态/已完成实验/target 训练/eval 全部复用
#    - 指纹不匹配(代码或参数变更) → 旧目录归档为 result/${EXP_NAME}_stale_<ts> 后全新开始
#    - 强制全新:  换 EXP_NAME 或 rm -rf result/$EXP_NAME
#  恢复粒度: 步骤级(.done) / 迭代级(search_state.json) / 实验级(exp_*/meta.json)
#            / embedding 分片级(进度账本) / 训练内部不支持(整次重跑)
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
K_ENHANCED="${K_ENHANCED:-10}"
DISCOVERY_METHOD="${DISCOVERY_METHOD:-embedding_cluster}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-npu}"
EMBEDDING_SAMPLE_SIZE="${EMBEDDING_SAMPLE_SIZE:-0}"
STEM_RATIO="${STEM_RATIO:-0.7}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"
NUM_NPU="${NUM_NPU:-8}"
NPU_PER_EXP="${NPU_PER_EXP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/result/$EXP_NAME}"

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

# ── Fingerprint: code + semantic params → auto-reset on change ──
FINGERPRINT=$(python3 -m climbmix.utils.fingerprint --base-dir "$CLIMBMIX_DIR" \
    --param "proxy_depth=$PROXY_DEPTH" \
    --param "target_depth=$TARGET_DEPTH" \
    --param "proxy_num_iterations=$PROXY_NUM_ITERATIONS" \
    --param "target_steps=$TARGET_STEPS" \
    --param "proxy_target_tokens=$PROXY_TARGET_TOKENS" \
    --param "target_tokens=$TARGET_TOKENS" \
    --param "configs_per_iter=$CONFIGS_PER_ITER" \
    --param "K_enhanced=$K_ENHANCED" \
    --param "discovery_method=$DISCOVERY_METHOD" \
    --param "embedding_device=$EMBEDDING_DEVICE" \
    --param "embedding_sample_size=$EMBEDDING_SAMPLE_SIZE" \
    --param "stem_ratio=$STEM_RATIO" \
    --param "eval_benchmarks=$EVAL_BENCHMARKS" \
    --param "num_npu=$NUM_NPU" \
    --param "npu_per_exp=$NPU_PER_EXP" \
    --param "data_dir=$DATA_DIR" \
    --param "general_data_dir=$GENERAL_DATA_DIR")

mkdir -p "$CLIMBMIX_DIR/result"
if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    if [ -f "$OUTPUT_DIR/.fingerprint" ] && [ "$(cat "$OUTPUT_DIR/.fingerprint")" = "$FINGERPRINT" ]; then
        echo "  RESUME: $OUTPUT_DIR (fingerprint ${FINGERPRINT} matches)"
    else
        STALE="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_$(date +%Y%m%d_%H%M%S)"
        echo "  Fingerprint changed (code or params) — archiving old output:"
        echo "    $OUTPUT_DIR -> $STALE"
        mv "$OUTPUT_DIR" "$STALE"
    fi
fi
mkdir -p "$OUTPUT_DIR"
echo "$FINGERPRINT" > "$OUTPUT_DIR/.fingerprint"

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
        --proxy-depth "$PROXY_DEPTH" \
        --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
        --proxy-target-tokens "$PROXY_TARGET_TOKENS" \
        --proxy-lr-scale 1.0 --proxy-warmup 0.0 --proxy-warmdown 0.9 \
        --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
        --target-depth "$TARGET_DEPTH" \
        --target-tokens "$TARGET_TOKENS" \
        --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
        --K-enhanced "$K_ENHANCED" \
        --discovery-method "$DISCOVERY_METHOD" \
        --embedding-device "$EMBEDDING_DEVICE" \
        --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
        --configs-per-iter "$CONFIGS_PER_ITER" \
        --device-type npu --npu-devices "$NUM_NPU" --npu-per-exp "$NPU_PER_EXP" \
        --output-dir "$OUTPUT_DIR" \
        --exp-name "$EXP_NAME" \
        --cluster-cache-dir "$OUTPUT_DIR" \
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
    [ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"
    # Clear partial checkpoints from a crashed attempt (nanochat may otherwise
    # try to auto-resume from inconsistent state; whole-run atomicity instead)
    rm -rf "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag"
    ( cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$TARGET_STEPS" \
        --lr-scale=1.0 --warmup-ratio=0.0 --warmdown-ratio=0.9 \
        --device-batch-size=8 \
        --run="${name}_mid" --model-tag="$tag" \
        --eval-benchmarks="$EVAL_BENCHMARKS" \
        --data-dir="$data_dir" 2>&1 | tee "$OUTPUT_DIR/mid_train_${name}.log" )
    [ -L "$link_dir" ] && rm "$link_dir"
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
        --device-batch-size=32 \
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
