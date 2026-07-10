#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Baseline: FDC 22-domain labels as clusters
# ──────────────────────────────────────────────────────────────
# Uses predefined FDC taxonomy (domain 0-21) as clusters,
# applies quality filtering, then runs iterative search.
#
# Usage:
#   bash runs/baseline_fdc.sh
#   DATA_DIR=/other/path bash runs/baseline_fdc.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

# ── Paths ──
DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/baseline_fdc_$(date +%Y%m%d_%H%M%S)}"

# ── Cluster discovery ──
DISCOVERY_METHOD="fdc_labels"

# ── Quality filtering ──
FILTER_METHOD="doc_and_cluster"
DOC_ENGLISH_MIN=0.3
DOC_COMPOSITE_MIN=0.5
CLUSTER_AVG_THRESHOLD=3.0

# ── Proxy ──
PROXY_SIZE="62M"
PROXY_STEPS=1000

# ── Search ──
NUM_ITERATIONS=3
CONFIGS_PER_ITER="64,32,16"

# ── Device ──
DEVICE_TYPE="cpu"

# ── Validate data ──
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Run: bash runs/preprocess.sh first"
    exit 1
fi

PREPROCESSED=$(ls "$DATA_DIR"/preprocessed_*.parquet 2>/dev/null | wc -l || echo 0)
if [ "$PREPROCESSED" -eq 0 ]; then
    echo "ERROR: No preprocessed_*.parquet files in $DATA_DIR"
    echo "Run: bash runs/preprocess.sh first"
    exit 1
fi

echo "╔══ FDC Baseline ═══╗"
echo ""
echo "  Cluster:   FDC 22-domain labels"
echo "  Filter:    $FILTER_METHOD (english≥$DOC_ENGLISH_MIN, composite≥$DOC_COMPOSITE_MIN, cluster_avg≥$CLUSTER_AVG_THRESHOLD)"
echo "  Proxy:     $PROXY_SIZE, $PROXY_STEPS steps"
echo "  Search:    $NUM_ITERATIONS iterations, $CONFIGS_PER_ITER"
echo "  Device:    $DEVICE_TYPE"
echo "  Data:      $DATA_DIR ($PREPROCESSED shards)"
echo "  Output:    $OUTPUT_DIR"
echo ""
echo "╚════════════════════╝"
echo ""

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --discovery-method "$DISCOVERY_METHOD" \
    --filter-method "$FILTER_METHOD" \
    --doc-english-min "$DOC_ENGLISH_MIN" \
    --doc-composite-min "$DOC_COMPOSITE_MIN" \
    --cluster-avg-threshold "$CLUSTER_AVG_THRESHOLD" \
    --proxy-size "$PROXY_SIZE" \
    --proxy-steps "$PROXY_STEPS" \
    --num-iterations "$NUM_ITERATIONS" \
    --configs-per-iter "$CONFIGS_PER_ITER" \
    --device-type "$DEVICE_TYPE" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
