#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 3: Full validation — verify optimal mixture on target model
# ──────────────────────────────────────────────────────────────
# Purpose: Train the target model (depth=24, ~1.38B) with the
#          optimal mixture found in Stage 2, then evaluate on
#          benchmarks. Compare against:
#   - Random mixture baseline
#   - Proportional-to-token-count mixture
#   - (optional) Other data mixing methods (DoReMi, RegMix)
#
# Prerequisites:
#   - Stage 2 completed (optimal mixture weights available)
#   - Phase-1 checkpoint: depth=24 pretrained on ~20B tokens
#     (nanochat default: target-param-data-ratio=9.5 → 1.38B×9.5≈13B)
#
# Time estimate (8×910B3):
#   Phase-1 (13B tokens, depth=24): ~13.8h (nanochat measured)
#   Target mid-training (40B tokens): ~40h (extrapolated)
#   Per comparison: same as mid-training
#   3 comparisons = ~120h
#
# Usage:
#   OPTIMAL_WEIGHTS=/path/to/stage2/optimal_weights.json \
#   bash runs/stage3_validate.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/stage3_validate_$(date +%Y%m%d_%H%M%S)}"
PHASE1_CKPT="${PHASE1_CKPT:-}"
OPTIMAL_WEIGHTS="${OPTIMAL_WEIGHTS:-}"

# ── Validation config ──
DEPTH=24                    # ~1.38B params (nanochat standard)
K_ENHANCED=10               # same as Stage 2
MID_TRAIN_TOKENS=40000000000   # 40B tokens (paper standard)
BATCH_TOKENS=2000000            # 2M batch (paper standard)
VALIDATION_METRIC="accuracy"

echo "╔══ Stage 3: Full Validation ═══╗"
echo ""
echo "  Purpose:    Verify optimal mixture on target model"
echo "  Model:      nanochat depth=24 (~1.38B params)"
echo "  Cluster:    FDC labels, K_enhanced=$K_ENHANCED"
echo "  Mid-train:  40B tokens, accuracy-based evaluation"
echo "  Phase-1:    ${PHASE1_CKPT:-required!}"
echo "  Weights:    ${OPTIMAL_WEIGHTS:-required! from Stage 2}"
echo "  Comparisons: optimal vs random vs proportional"
echo "  Device:     8×910B3 NPU"
echo "  Time:       ~120h (3 comparisons × 40h each)"
echo ""
echo "╚════════════════════════════════╝"

if [ -z "$OPTIMAL_WEIGHTS" ]; then
    echo "ERROR: OPTIMAL_WEIGHTS not set. Run Stage 2 first."
    exit 1
fi
if [ -z "$PHASE1_CKPT" ]; then
    echo "ERROR: PHASE1_CKPT not set. Run phase-1 pretraining first."
    exit 1
fi

EXTRA_ARGS="--phase1-checkpoint-path $PHASE1_CKPT --optimal-weights-path $OPTIMAL_WEIGHTS"

torchrun --standalone --nproc_per_node=8 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --discovery-method fdc_labels \
    --K-enhanced "$K_ENHANCED" \
    --filter-method none \
    --proxy-model-depth "$DEPTH" \
    --proxy-training-tokens "$MID_TRAIN_TOKENS" \
    --proxy-batch-tokens "$BATCH_TOKENS" \
    --validation-metric "$VALIDATION_METRIC" \
    --device-type npu \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS \
    "$@"
