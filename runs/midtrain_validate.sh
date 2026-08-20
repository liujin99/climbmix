#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# CLIMB mid-training validation
# ──────────────────────────────────────────────────────────────
# Compare CLIMB optimal mixture vs random baseline on d28 target.
#
# Usage:
#   CLIMBMIX_RESULT=/path/to/stage_result \
#   bash runs/midtrain_validate.sh
#
# CLIMBMIX_RESULT should point to a completed Stage 1/2 output dir
# containing: sampled_dataset.parquet + optimal_mixture_weights.json
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLIMBMIX_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Config ──
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
NANOCHAT_REPO="${NANOCHAT_REPO:-$NANOCHAT_DIR}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"
STEM_RATIO="${STEM_RATIO:-0.7}"
MIX_GENERAL_DATA="${MIX_GENERAL_DATA:-1}"

BASE_MODEL_TAG="${BASE_MODEL_TAG:-d28}"
NUM_NPU="${NUM_NPU:-8}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"

TARGET_PARAM_DATA_RATIO="${TARGET_PARAM_DATA_RATIO:-0.5}"
NUM_SCALING_PARAMS="${NUM_SCALING_PARAMS:-}"

CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
EVAL_EVERY="${EVAL_EVERY:--1}"

RESULT_DIR="${RESULT_DIR:-$SCRIPT_DIR/../result/validate_${TIMESTAMP}}"

# ── Auto-detect CLIMBMIX_RESULT ──
if [ -z "${CLIMBMIX_RESULT:-}" ]; then
    LATEST=$(ls -dt "$CLIMBMIX_DIR"/result/stage* 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "$LATEST/sampled_dataset.parquet" ]; then
        CLIMBMIX_RESULT="$LATEST"
    fi
fi

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Checks ──
echo "=== CLIMB mid-training validation ==="
echo ""

for pkg in numpy pandas pyarrow torch; do
    python3 -c "import $pkg" 2>/dev/null || die "Missing: $pkg — pip install $pkg"
done; ok "Python deps"

if [ ! -d "$NANOCHAT_DIR" ]; then
    die "nanochat-npu not found at $NANOCHAT_DIR"
fi
ok "nanochat-npu found"

if [ -z "${CLIMBMIX_RESULT:-}" ] || [ ! -f "$CLIMBMIX_RESULT/sampled_dataset.parquet" ]; then
    die "CLIMBMIX_RESULT not set or sampled_dataset.parquet not found\n  Run Stage 1 first, or set CLIMBMIX_RESULT=<path>"
fi
ok "CLIMBMIX sampled data: $CLIMBMIX_RESULT/sampled_dataset.parquet"

BASE_CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$BASE_MODEL_TAG"
if [ ! -d "$BASE_CKPT_DIR" ]; then
    die "$BASE_MODEL_TAG base checkpoint not found at $BASE_CKPT_DIR"
fi
ok "$BASE_MODEL_TAG base checkpoint"

python3 -c "import torch_npu; assert torch.npu.is_available()" 2>/dev/null || die "NPU not available"
ok "NPU"

# Auto-detect NUM_SCALING_PARAMS from checkpoint meta JSON
if [ -z "$NUM_SCALING_PARAMS" ]; then
    echo "  Auto-detecting NUM_SCALING_PARAMS from checkpoint..."
    GET_MODEL_INFO="$SCRIPT_DIR/../scripts/get_model_info.py"
    if [ ! -f "$GET_MODEL_INFO" ]; then
        warn "get_model_info.py not found at $GET_MODEL_INFO"
    fi
    if [ -f "$GET_MODEL_INFO" ]; then
        MODEL_INFO=$(python3 "$GET_MODEL_INFO" --ckpt-dir "$BASE_CKPT_DIR" --nanochat-repo "$NANOCHAT_DIR")
        NUM_SCALING_PARAMS=$(echo "$MODEL_INFO" | grep NUM_SCALING_PARAMS | cut -d= -f2)
        CKPT_TOTAL_BATCH_SIZE=$(echo "$MODEL_INFO" | grep TOTAL_BATCH_SIZE | cut -d= -f2)
    fi
    if [ -z "$NUM_SCALING_PARAMS" ]; then
        NUM_SCALING_PARAMS=730000000
        warn "Could not auto-detect, using default NUM_SCALING_PARAMS=$NUM_SCALING_PARAMS"
    fi
    echo "    NUM_SCALING_PARAMS=$NUM_SCALING_PARAMS"
fi

# ── Model tags ──
CLIMB_MODEL_TAG="${BASE_MODEL_TAG}_climb_${TIMESTAMP}"
RANDOM_MODEL_TAG="${BASE_MODEL_TAG}_random_${TIMESTAMP}"

echo ""
echo "  CLIMB data:       $CLIMBMIX_RESULT/sampled_dataset.parquet"
echo "  Base checkpoint:  $BASE_CKPT_DIR"
echo "  CLIMB model tag:  $CLIMB_MODEL_TAG"
echo "  Random model tag: $RANDOM_MODEL_TAG"
echo "  Output:           $RESULT_DIR"
echo ""

mkdir -p "$RESULT_DIR"

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT (from run_stem_experiment.sh)
# ══════════════════════════════════════════════════════════════

export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
mkdir -p "$NANOCHAT_BASE_DIR"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3

ASCEND_DEVICE_LIST=$(seq -s, 0 $((NUM_NPU - 1)))
export ASCEND_VISIBLE_DEVICES="$ASCEND_DEVICE_LIST"
export RANK_SIZE=$NUM_NPU
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export ASCEND_DISABLE_MEM_SWAP=1
export ASCEND_LAUNCH_BLOCKING=0
export NPU_DISABLE_RECORD=1
export PYTHONUNBUFFERED=1
export ASCEND_COMPILE_OPT_LEVEL=O3
export TORCH_NPU_LAZY_COMPILE=1
export PYTHONPRELOAD=torch_npu
export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
export PYTORCH_NPU_ALLOC_MAX_SIZE=60G
export ASCEND_ENABLE_CACHE=1
export ASCEND_CACHE_POLICY=2
export ASCEND_FUSION_ENABLE=1
export NANOCHAT_DTYPE=bfloat16
export PYTHONWARNINGS="ignore::UserWarning:torch_npu"

# ══════════════════════════════════════════════════════════════
#  STEP 1: DATA PREPARATION
# ══════════════════════════════════════════════════════════════

echo ""
echo "Step 1: Prepare training data..."

CLIMB_DATA="$RESULT_DIR/climb_data"
RANDOM_DATA="$RESULT_DIR/random_data"

prepare_shards() {
    local src_parquet="$1"
    local out_dir="$2"
    local label="$3"

    if [ -d "$out_dir" ] && [ "$(ls "$out_dir"/shard_*.parquet 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "$label: shards already exist, skipping"
        return 0
    fi

    mkdir -p "$out_dir"

    python3 -c "
import sys, os, pyarrow.parquet as pq, pyarrow as pa, math

src = '$src_parquet'
out = '$out_dir'
npu = int('$NUM_NPU')
shard_size = 10000

table = pq.read_table(src, columns=['text'])
texts = table['text'].to_pylist()

n = len(texts)
n_shards = max(1, math.ceil(n / shard_size))
rg_size = max(1, shard_size // (npu * 2))

for i in range(n_shards):
    start = i * shard_size
    end = min(start + shard_size, n)
    shard_texts = texts[start:end]
    shard_table = pa.table({'text': shard_texts})
    pq.write_table(shard_table, os.path.join(out, f'shard_{i:05d}.parquet'), row_group_size=rg_size)

# nanochat expects last shard = val split (dummy)
dummy = pa.table({'text': ['dummy']})
pq.write_table(dummy, os.path.join(out, f'shard_{n_shards:05d}.parquet'), row_group_size=1)

print(f'  {n} docs -> {n_shards} train shards + 1 dummy val -> {out}')
"
    ok "$label: shards prepared"
}

# CLIMB mixture data
prepare_shards "$CLIMBMIX_RESULT/sampled_dataset.parquet" "$CLIMB_DATA" "CLIMB"

# Random baseline: sample same number of docs from preprocessed shards
CLIMB_DOC_COUNT=$(python3 -c "
import pyarrow.parquet as pq
t = pq.read_table('$CLIMBMIX_RESULT/sampled_dataset.parquet', columns=['text'])
print(len(t['text'].to_pylist()))
")

if [ ! -d "$RANDOM_DATA" ] || [ "$(ls "$RANDOM_DATA"/shard_*.parquet 2>/dev/null | wc -l)" -eq 0 ]; then
    echo "  Preparing random baseline ($CLIMB_DOC_COUNT docs)..."
    python3 "$CLIMBMIX_DIR/scripts/prepare_random_baseline.py" \
        --data-dir "$DATA_DIR" \
        --output-dir "$RANDOM_DATA" \
        --num-docs "$CLIMB_DOC_COUNT" \
        --seed 42 \
        --num-npu "$NUM_NPU"
    ok "Random baseline prepared"
else
    ok "Random baseline: shards already exist"
fi

# ══════════════════════════════════════════════════════════════
#  STEP 1.5: MIX STEM WITH CLIMBMIX (anti-forgetting)
# ══════════════════════════════════════════════════════════════

CLIMBMIX_SHARDS_DIR="$GENERAL_DATA_DIR"

if [ "$MIX_GENERAL_DATA" = "1" ]; then
    echo ""
    echo "Step 1.5: Mix STEM + ClimbMix (anti-forgetting)..."
    echo "  STEM ratio: $STEM_RATIO"

    MIX_SCRIPT="$SCRIPT_DIR/../scripts/mix_general_data.py"
    if [ ! -f "$MIX_SCRIPT" ]; then
        echo "  WARNING: mix_general_data.py not found at $MIX_SCRIPT, skipping mix"
        MIX_GENERAL_DATA=0
    fi

    mix_one() {
        local STEM_DIR="$1"
        local MIXED_DIR="$2"
        local LABEL="$3"

        if [ ! -d "$STEM_DIR" ]; then
            echo "  Skipping $LABEL: $STEM_DIR not found"
            return 0
        fi

        echo "  Mixing $LABEL..."
        python3 "$MIX_SCRIPT" \
            --stem-dir "$STEM_DIR" \
            --output-dir "$MIXED_DIR" \
            --climbmix-dir "$CLIMBMIX_SHARDS_DIR" \
            --stem-ratio "$STEM_RATIO" \
            --num-workers "$NUM_NPU" \
            || { echo "  ERROR: mixing failed for $LABEL"; return 1; }
    }

    mix_one "$CLIMB_DATA" "$RESULT_DIR/climb_mixed" "CLIMB"
    mix_one "$RANDOM_DATA" "$RESULT_DIR/random_mixed" "Random"

    CLIMB_DATA="$RESULT_DIR/climb_mixed"
    RANDOM_DATA="$RESULT_DIR/random_mixed"
    echo ""
fi

# Count tokens
CLIMB_TOKENS=$(python3 -c "
import os
total = 0
import pyarrow.parquet as pq
for f in sorted(os.listdir('$CLIMB_DATA')):
    if f.startswith('shard_') and not f.endswith(f'shard_{int(\"$NUM_NPU\")}+1.parquet'):
        t = pq.read_table(os.path.join('$CLIMB_DATA', f), columns=['text'])
        total += sum(len(doc) // 4 for doc in t['text'].to_pylist())
print(total)
" 2>/dev/null || echo "0")

RANDOM_TOKENS=$(python3 -c "
import os
total = 0
import pyarrow.parquet as pq
for f in sorted(os.listdir('$RANDOM_DATA')):
    if f.startswith('shard_') and not f.endswith(f'shard_{int(\"$NUM_NPU\")}+1.parquet'):
        t = pq.read_table(os.path.join('$RANDOM_DATA', f), columns=['text'])
        total += sum(len(doc) // 4 for doc in t['text'].to_pylist())
print(total)
" 2>/dev/null || echo "0")

echo "  CLIMB tokens:  $CLIMB_TOKENS (estimated)"
echo "  Random tokens: $RANDOM_TOKENS (estimated)"

# ══════════════════════════════════════════════════════════════
#  STEP 2: MID-TRAINING
# ══════════════════════════════════════════════════════════════

run_mid_train() {
    local data_path="$1"
    local model_tag="$2"
    local run_name="$3"
    local log_file="$4"
    local dataset_tokens="$5"

    CKPT_META_JSON=$(ls "$BASE_CKPT_DIR"/meta_*.json 2>/dev/null | sort | tail -1)
    if [ -n "$CKPT_META_JSON" ]; then
        TOTAL_BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CKPT_META_JSON'))['total_batch_size'])")
    else
        TOTAL_BATCH_SIZE=524288
    fi

    TARGET_TOKENS=$(python3 -c "print(int($TARGET_PARAM_DATA_RATIO * $NUM_SCALING_PARAMS))")
    ACTUAL_TOKENS=$(python3 -c "print(min($TARGET_TOKENS, $dataset_tokens))")
    NUM_ITERATIONS=$(python3 -c "print(max(1, int($ACTUAL_TOKENS / $TOTAL_BATCH_SIZE)))")

    echo ""
    echo "  Mid-training: $run_name"
    echo "    Data:       $data_path"
    echo "    Base:       $BASE_MODEL_TAG → mid as $model_tag"
    echo "    Steps:      $NUM_ITERATIONS"
    echo "    Tokens:     $ACTUAL_TOKENS"
    echo "    Log:        $log_file"

    LINK_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$model_tag"
    if [ ! -e "$LINK_DIR" ]; then
        ln -s "$BASE_CKPT_DIR" "$LINK_DIR"
    fi

    cd "$NANOCHAT_DIR"
    torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$NUM_ITERATIONS" \
        --lr-scale=1.0 \
        --warmup-ratio=0.0 \
        --warmdown-ratio=0.9 \
        --device-batch-size="$DEVICE_BATCH_SIZE" \
        --total-batch-size="$TOTAL_BATCH_SIZE" \
        --run="$run_name" \
        --model-tag="$model_tag" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --eval-every="$EVAL_EVERY" \
        --eval-benchmarks="$EVAL_BENCHMARKS" \
        --data-dir="$data_path" \
        2>&1 | tee "$log_file"

    cd "$CLIMBMIX_DIR"
    if [ -L "$LINK_DIR" ]; then
        rm "$LINK_DIR"
    fi
}

echo ""
echo "Step 2: Mid-training..."

CLIMB_LOG="$RESULT_DIR/mid_train_climb.log"
run_mid_train "$CLIMB_DATA" "$CLIMB_MODEL_TAG" "climb_mid" "$CLIMB_LOG" "$CLIMB_TOKENS"

RANDOM_LOG="$RESULT_DIR/mid_train_random.log"
run_mid_train "$RANDOM_DATA" "$RANDOM_MODEL_TAG" "random_mid" "$RANDOM_LOG" "$RANDOM_TOKENS"

# ══════════════════════════════════════════════════════════════
#  STEP 3: EVALUATION
# ══════════════════════════════════════════════════════════════

run_eval() {
    local model_tag="$1"
    local model_type="$2"
    local log_file="$3"

    echo "  Evaluating: $model_tag ($model_type)"

    cd "$NANOCHAT_DIR"
    torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core \
        --eval-benchmarks="$EVAL_BENCHMARKS" \
        --device-batch-size="$EVAL_BATCH_SIZE" \
        --model-tag="$model_tag" \
        --model-type="$model_type" \
        2>&1 | tee "$log_file"

    cd "$CLIMBMIX_DIR"
}

echo ""
echo "Step 3: Evaluation..."

CLIMB_EVAL_LOG="$RESULT_DIR/eval_climb.log"
run_eval "$CLIMB_MODEL_TAG" "mid" "$CLIMB_EVAL_LOG"

RANDOM_EVAL_LOG="$RESULT_DIR/eval_random.log"
run_eval "$RANDOM_MODEL_TAG" "mid" "$RANDOM_EVAL_LOG"

# ══════════════════════════════════════════════════════════════
#  STEP 4: REPORT
# ══════════════════════════════════════════════════════════════

echo ""
echo "Step 4: Generate report..."

python3 "$SCRIPT_DIR/../src/climbmix/pipeline/report_generator.py" \
    --result-dir "$RESULT_DIR" \
    --climb-train-log "$CLIMB_LOG" \
    --random-train-log "$RANDOM_LOG" \
    --climb-eval-log "$CLIMB_EVAL_LOG" \
    --random-eval-log "$RANDOM_EVAL_LOG" \
    --base-model-tag "$BASE_MODEL_TAG" \
    --climb-model-tag "$CLIMB_MODEL_TAG" \
    --random-model-tag "$RANDOM_MODEL_TAG"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Validation Complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Output:   $RESULT_DIR"
echo "  Report:   $RESULT_DIR/validation_report.md"
echo ""
