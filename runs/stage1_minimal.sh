#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 1: Minimal validation — verify proxy training works
# ──────────────────────────────────────────────────────────────
# Purpose: End-to-end with real proxy training on nanochat.
#          Tiny model (depth=5, ~59M) + few clusters + few configs.
#          Just verify the pipeline produces meaningful results.
#
# Key question: Does LightGBM predictor learn anything?
#   - Predictor R² should be > 0 (even negative is ok for 6 samples)
#   - Spearman rank correlation should be > 0.3
#   - If not, there's a bug or the signal is too weak at this scale
#
# Prerequisites:
#   - Phase-1 checkpoint: depth=5 model pretrained on ~1B tokens
#     (if not available, uses random init — validation_metric should
#      be "loss" since model won't have enough accuracy signal)
#   - NPU cluster available (8×910B3)
#
# Time estimate:
#   Phase-1 (1B tokens, depth=5): ~10 min
#   Proxy search (6×0.5B tokens): ~15 min
#   Total: ~25 min + phase-1
#
# Usage:
#   bash runs/stage1_minimal.sh
#   PHASE1_CKPT=/path/to/d5_phase1 bash runs/stage1_minimal.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/stage1_minimal_$(date +%Y%m%d_%H%M%S)}"
PHASE1_CKPT="${PHASE1_CKPT:-}"

# ── Minimal config ──
DEPTH=5               # ~59M params
K_ENHANCED=5          # 5 clusters (reduced from 21)
NUM_ITERATIONS=2      # 2 iterations (not 3)
CONFIGS_PER_ITER="4,2"  # 6 total configs
TRAINING_TOKENS=500000000   # 0.5B tokens per proxy run
BATCH_TOKENS=250000         # 0.25M batch
VALIDATION_METRIC="loss"    # depth=5 too small for lm-eval accuracy

echo "╔══ Stage 1: Minimal Validation ═══╗"
echo ""
echo "  Purpose:    Verify proxy training + predictor works"
echo "  Model:      nanochat depth=5 (~59M params)"
echo "  Cluster:    FDC labels, K_enhanced=5"
echo "  Filter:     none"
echo "  Proxy:      0.5B tokens/config, loss-based validation"
echo "  Search:     2 iterations, 4/2 = 6 configs"
echo "  Phase-1:    ${PHASE1_CKPT:-random init (no checkpoint)}"
echo "  Device:     8×910B3 NPU"
echo "  Time:       ~25 min (+ phase-1 if needed)"
echo ""
echo "╚════════════════════════════════════╝"

EXTRA_ARGS=""
if [ -n "$PHASE1_CKPT" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --phase1-checkpoint-path $PHASE1_CKPT"
fi

torchrun --standalone --nproc_per_node=8 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --discovery-method fdc_labels \
    --K-enhanced "$K_ENHANCED" \
    --filter-method none \
    --proxy-model-depth "$DEPTH" \
    --proxy-training-tokens "$TRAINING_TOKENS" \
    --proxy-batch-tokens "$BATCH_TOKENS" \
    --validation-metric "$VALIDATION_METRIC" \
    --num-iterations "$NUM_ITERATIONS" \
    --configs-per-iter "$CONFIGS_PER_ITER" \
    --device-type npu \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS \
    "$@"
