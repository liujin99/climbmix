#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase 1: Base checkpoint pretraining (CLIMB-aligned)
# ──────────────────────────────────────────────────────────────
# Produces a base checkpoint with CONSTANT LR (no warmdown).
# CLIMB annealing (LR decay) happens ONLY in mid_train.
# This matches the 2024-2025 industry trend and the CLIMB paper.
# Run this BEFORE Stage 1/2 to generate required checkpoints.
#
# Time: d10 ~0.6h, d14 ~4h, d18 ~15h, d24 ~29h
#
# Usage:
#   DEPTH=10  bash runs/train_base_model.sh
#   DEPTH=24  bash runs/train_base_model.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
NANOCHAT_BRANCH="dev-data-mix"
DEPTH="${DEPTH:-10}"
RATIO="${RATIO:-9.5}"
MODEL_TAG="${MODEL_TAG:-d${DEPTH}}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Check ──
echo "=== Phase 1: d${DEPTH} base pretraining ==="
if [ ! -d "$NANOCHAT_DIR" ]; then
    die "nanochat-npu not found — git clone -b $NANOCHAT_BRANCH https://github.com/liujin99/nanochat-npu.git $NANOCHAT_DIR"
fi
branch=$(cd "$NANOCHAT_DIR" && git rev-parse --abbrev-ref HEAD)
[ "$branch" = "$NANOCHAT_BRANCH" ] || die "nanochat-npu on '$branch', need '$NANOCHAT_BRANCH'"
ok "nanochat-npu ($NANOCHAT_BRANCH)"

python3 -c "import torch_npu; assert torch.npu.is_available()" 2>/dev/null || die "NPU not available — needs 8×910B3"
ok "NPU"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

echo ""
echo "Training d${DEPTH} (ratio=$RATIO, tag=$MODEL_TAG)..."
echo "  LR schedule: warmup + constant (no warmdown)"
echo "  CLIMB annealing will happen in mid_train, not here"
echo "  Estimated time: see header comments"

cd "$NANOCHAT_DIR"
torchrun --standalone --nproc_per_node=8 -m scripts.base_train \
    -- --depth="$DEPTH" \
    --target-param-data-ratio="$RATIO" \
    --device-batch-size=8 \
    --warmdown-ratio=0.0 \
    --run=dummy \
    --model-tag="$MODEL_TAG"

echo ""
echo "Evaluating d${DEPTH} base model..."
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval \
    -- --device-batch-size=32 \
    --model-tag="$MODEL_TAG" \
    --model-type=base \
    --eval=core

ok "Phase 1 done → ~/.cache/nanochat/base_checkpoints/$MODEL_TAG"
