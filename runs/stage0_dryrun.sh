#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 0: Pipeline dry-run (no real training)
# ──────────────────────────────────────────────────────────────
# Purpose: Verify the entire pipeline runs end-to-end with
#          random scores (no GPU/NPU needed).
# Expected output: Markdown report + matplotlib chart
# Time: ~5 min on CPU
#
# Usage:
#   bash runs/stage0_dryrun.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/stage0_dryrun_$(date +%Y%m%d_%H%M%S)"

echo "╔══ Stage 0: Dry-Run ═══╗"
echo ""
echo "  Purpose:  Verify pipeline logic without training"
echo "  Cluster:  FDC labels (22 domains)"
echo "  Filter:   none"
echo "  Proxy:    DRY-RUN (random scores)"
echo "  Search:   3 iterations, 8/4/2 configs"
echo "  Device:   CPU"
echo "  Time:     ~5 min"
echo ""
echo "╚════════════════════════╝"

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --discovery-method fdc_labels \
    --filter-method none \
    --num-iterations 3 \
    --configs-per-iter "8,4,2" \
    --dry-run \
    --device-type cpu \
    --output-dir "$OUTPUT_DIR" \
    "$@"
