#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# CLIMB full: embedding cluster → prune → merge → iterative search
# ──────────────────────────────────────────────────────────────
# Embeds documents with stella_en_400M_v5, runs K-means,
# prunes low-quality clusters, merges nearby clusters,
# then runs iterative bootstrapping search with proxy models.
#
# Default uses 62M proxy for faster iteration.
# Switch to 350M by changing PROXY_SIZE when ready.
#
# Usage:
#   bash runs/climb.sh
#   DATA_DIR=/other/path PROXY_SIZE=350M bash runs/climb.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

# ── Paths ──
DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/liujin99/climbmix/result/climb_${PROXY_SIZE:-62M}_$(date +%Y%m%d_%H%M%S)}"

# ── Cluster discovery ──
DISCOVERY_METHOD="embedding_cluster"
K_INIT=1000
K_ENHANCED=21
EMBEDDING_MODEL="NovaSearch/stella_en_400M_v5"
PRUNE_THRESHOLD=3.0
MERGE_DISTANCE=1.5

# ── Quality filtering ──
FILTER_METHOD="doc_and_cluster"
DOC_ENGLISH_MIN=0.3
DOC_COMPOSITE_MIN=0.5
CLUSTER_AVG_THRESHOLD=3.0

# ── Proxy ──
PROXY_SIZE="${PROXY_SIZE:-62M}"
PROXY_STEPS="${PROXY_STEPS:-1000}"

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

echo "╔══ CLIMB Full ═══╗"
echo ""
echo "  Cluster:   embedding($EMBEDDING_MODEL) → K-means(K=$K_INIT) → prune($PRUNE_THRESHOLD) → merge($MERGE_DISTANCE) → K_enhanced=$K_ENHANCED"
echo "  Filter:    $FILTER_METHOD (english≥$DOC_ENGLISH_MIN, composite≥$DOC_COMPOSITE_MIN, cluster_avg≥$CLUSTER_AVG_THRESHOLD)"
echo "  Proxy:     $PROXY_SIZE, $PROXY_STEPS steps"
echo "  Search:    $NUM_ITERATIONS iterations, $CONFIGS_PER_ITER"
echo "  Device:    $DEVICE_TYPE"
echo "  Data:      $DATA_DIR ($PREPROCESSED shards)"
echo "  Output:    $OUTPUT_DIR"
echo ""
echo "╚══════════════════╝"
echo ""

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --discovery-method "$DISCOVERY_METHOD" \
    --K-init "$K_INIT" \
    --K-enhanced "$K_ENHANCED" \
    --embedding-model "$EMBEDDING_MODEL" \
    --prune-threshold "$PRUNE_THRESHOLD" \
    --merge-distance "$MERGE_DISTANCE" \
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
