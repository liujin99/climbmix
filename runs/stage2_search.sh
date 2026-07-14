#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 2: Scaled search — first real mixture optimization
# ──────────────────────────────────────────────────────────────
# Purpose: Real data mixture search with sufficient configs
#          and clusters. This is the first experiment that
#          could produce a meaningful optimal mixture.
#
# Key questions:
#   1. Does predictor Spearman > 0.8? (ranking reliability)
#   2. Does the optimal mixture differ from proportional sampling?
#   3. Does predictor generalize across iterations?
#
# Prerequisites:
#   - Stage 1 passed (proxy training works, predictor learns)
#   - Phase-1 checkpoint: depth=10 pretrained on ~4B tokens
#   - More Essential-Web shards downloaded
#
# Time estimate (8×910B3):
#   Phase-1 (4B tokens, depth=10): ~30-40 min
#   Proxy search (28×5B tokens):   ~28h
#   Total: ~28.5h + phase-1
#
# Usage:
#   bash runs/stage2_search.sh
#   PHASE1_CKPT=/path/to/d10_phase1 bash runs/stage2_search.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/stage2_search_$(date +%Y%m%d_%H%M%S)}"
PHASE1_CKPT="${PHASE1_CKPT:-}"

# ── Search config ──
DEPTH=10                 # ~196M params
K_ENHANCED=10            # 10 clusters (reduced from 21 for 28 configs)
NUM_ITERATIONS=3         # 3 iterations (paper standard)
CONFIGS_PER_ITER="16,8,4"  # 28 total configs
TRAINING_TOKENS=5000000000   # 5B tokens per proxy run
BATCH_TOKENS=500000          # 0.5M batch
VALIDATION_METRIC="accuracy"  # depth=10 should be enough for lm-eval

echo "╔══ Stage 2: Scaled Search ═══╗"
echo ""
echo "  Purpose:    First real mixture optimization"
echo "  Model:      nanochat depth=10 (~196M params)"
echo "  Cluster:    FDC labels, K_enhanced=10"
echo "  Filter:     none"
echo "  Proxy:      5B tokens/config, accuracy-based validation"
echo "  Search:     3 iterations, 16/8/4 = 28 configs"
echo "  Phase-1:    ${PHASE1_CKPT:-required! run phase1_d10.sh first}"
echo "  Device:     8×910B3 NPU"
echo "  Time:       ~28h"
echo ""
echo "╚══════════════════════════════╝"

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
