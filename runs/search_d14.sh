#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# ClimbMix Stage 1: d14 proxy → d28 target — STEM data mixture optimization
# ──────────────────────────────────────────────────────────────
# Pipeline:
#   1. Embedding cluster discovery (stella_400M → K-means → 21 domains)
#   2. d14 proxy search (14 configs × 3 iterations)
#      Each experiment: 70% STEM (by cluster weights) + 30% ClimbMix (adaptive shards)
#   3. d28 target training with optimal mixture + STEM evaluation
#
# Needs: 8×910B NPU + d14/d28 base checkpoints + STEM data
# Time:  ~12-20h (embedding ~2-4h + proxy ~8h + target ~4h)
#
# Usage:
#   bash runs/search_d14.sh
#   DATA_DIR=/path/to/stem_data bash runs/search_d14.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

# STEM data (main data pool)
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"

# Nanochat
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
NANOCHAT_REPO="${NANOCHAT_REPO:-$NANOCHAT_DIR}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"

# General data (ClimbMix, adaptive 3-50 shards, reverse download, cached)
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

# Model config
PROXY_DEPTH="${PROXY_DEPTH:-14}"
TARGET_DEPTH="${TARGET_DEPTH:-28}"
PROXY_NUM_ITERATIONS="${PROXY_NUM_ITERATIONS:-500}"
TARGET_NUM_ITERATIONS="${TARGET_NUM_ITERATIONS:-1000}"
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-8,4,2}"
NUM_ITERATIONS="${NUM_ITERATIONS:-3}"

# Data mixing
STEM_RATIO="${STEM_RATIO:-0.7}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"

# Output
OUTPUT_DIR="${OUTPUT_DIR:-${CLIMBMIX_DIR}/result/stage1_d${PROXY_DEPTH}_$(date +%Y%m%d_%H%M%S)}"

# NPU
NUM_NPU="${NUM_NPU:-8}"

# ══════════════════════════════════════════════════════════════
#  COLORS & UTILS
# ══════════════════════════════════════════════════════════════

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ══════════════════════════════════════════════════════════════
#  NPU ENVIRONMENT SETUP (from run_stem_experiment.sh)
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
#  PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ClimbMix Stage 1: d${PROXY_DEPTH} → d${TARGET_DEPTH}"
echo "  STEM data mixture optimization with embedding clustering"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  STEM data:        $DATA_DIR"
echo "  General data:     $GENERAL_DATA_DIR (ClimbMix, adaptive shards)"
echo "  Nanochat dir:     $NANOCHAT_DIR"
echo "  Base dir:         $NANOCHAT_BASE_DIR"
echo "  Proxy:            d${PROXY_DEPTH} (${PROXY_NUM_ITERATIONS} iterations)"
echo "  Target:           d${TARGET_DEPTH} (${TARGET_NUM_ITERATIONS} iterations)"
echo "  Data mix:         ${STEM_RATIO}% STEM + $(python3 -c "print(100-$STEM_RATIO)")% general"
echo "  Eval benchmarks:  $EVAL_BENCHMARKS"
echo "  Search:           $NUM_ITERATIONS iterations, configs=$CONFIGS_PER_ITER"
echo "  Output:           $OUTPUT_DIR"
echo ""

# Python deps
echo "=== Pre-flight checks ==="
for pkg in numpy lightgbm sklearn scipy pandas pyarrow torch matplotlib; do
    python3 -c "import $pkg" 2>/dev/null || die "Missing: $pkg — pip install $pkg"
done; ok "Python deps"

# nanochat-npu
if [ ! -d "$NANOCHAT_DIR" ]; then
    die "nanochat-npu not found at $NANOCHAT_DIR"
fi
ok "nanochat-npu found"

# STEM data
if [ ! -d "$DATA_DIR" ]; then
    die "STEM data directory not found: $DATA_DIR"
fi
ls "$DATA_DIR"/*.parquet >/dev/null 2>&1 || die "No parquet files in $DATA_DIR"
ok "STEM data found"

# NPU
python3 -c "import torch_npu; assert torch.npu.is_available()" 2>/dev/null || die "NPU not available — requires 8×910B NPU"
ok "NPU available ($NUM_NPU devices)"

# Checkpoints
need_ckpt() {
    local depth=$1
    local ckpt_dir="$NANOCHAT_BASE_DIR/base_checkpoints/d${depth}"
    if [ -d "$ckpt_dir" ] && [ "$(ls "$ckpt_dir"/model_*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "d${depth} checkpoint"
    else
        die "d${depth} checkpoint NOT found at $ckpt_dir"
    fi
}

need_ckpt "$PROXY_DEPTH"
need_ckpt "$TARGET_DEPTH"

# Auto-detect scaling params from checkpoint meta JSON
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GET_MODEL_INFO="$SCRIPT_DIR/../scripts/get_model_info.py"
if [ ! -f "$GET_MODEL_INFO" ]; then
    die "get_model_info.py not found at $GET_MODEL_INFO"
fi

if [ -f "$GET_MODEL_INFO" ]; then
    echo ""
    echo "  Auto-detecting model info from checkpoints..."
    for depth in "$PROXY_DEPTH" "$TARGET_DEPTH"; do
        CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/d${depth}"
        if [ -d "$CKPT_DIR" ]; then
            MODEL_INFO=$(python3 "$GET_MODEL_INFO" --ckpt-dir "$CKPT_DIR" --nanochat-repo "$NANOCHAT_DIR" 2>/dev/null || echo "")
            if [ -n "$MODEL_INFO" ]; then
                echo "    d${depth}: $MODEL_INFO"
            fi
        fi
    done
fi

# Disk space check
echo ""
echo "  Disk space check..."
MIN_REQUIRED_GB=80
AVAILABLE_KB=$(df -P "$NANOCHAT_BASE_DIR" | awk 'NR==2{print $4}')
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
echo "    Available: ${AVAILABLE_GB}GB at $NANOCHAT_BASE_DIR"
echo "    Minimum needed: ${MIN_REQUIRED_GB}GB"
if [ "$AVAILABLE_GB" -lt "$MIN_REQUIRED_GB" ]; then
    warn "Low disk space (${AVAILABLE_GB}GB < ${MIN_REQUIRED_GB}GB)"
fi

# Eval data
echo ""
echo "  Downloading eval data (stem)..."
cd "$NANOCHAT_DIR"
python3 -c "from scripts.base_eval import prepare_eval_data; prepare_eval_data('stem')" 2>/dev/null || warn "Eval data download skipped"
cd "$CLIMBMIX_DIR"

echo ""
ok "All pre-flight checks passed"

# ══════════════════════════════════════════════════════════════
#  RUN CLIMBMIX PIPELINE
# ══════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Starting ClimbMix pipeline..."
echo "════════════════════════════════════════════════════════════"
echo ""

mkdir -p "$OUTPUT_DIR"

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --nanochat-dir "$NANOCHAT_DIR" \
    --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
    --general-data-dir "$GENERAL_DATA_DIR" \
    --stem-ratio "$STEM_RATIO" \
    --eval-benchmarks "$EVAL_BENCHMARKS" \
    --proxy-depth "$PROXY_DEPTH" \
    --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
    --proxy-lr-scale 1.0 \
    --proxy-warmup 0.0 \
    --proxy-warmdown 0.9 \
    --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
    --target-depth "$TARGET_DEPTH" \
    --target-num-iterations "$TARGET_NUM_ITERATIONS" \
    --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
    --discovery-method embedding_cluster \
    --filter-method none \
    --num-iterations "$NUM_ITERATIONS" \
    --configs-per-iter "$CONFIGS_PER_ITER" \
    --device-type npu \
    --npu-devices "$NUM_NPU" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo ""
echo "════════════════════════════════════════════════════════════"
ok "Done → $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"
