#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Preprocess Essential-Web data: download + extract FDC + quality + filter
# ──────────────────────────────────────────────────────────────
# Downloads raw Essential-Web shards from HuggingFace,
# extracts FDC domain labels and quality scores,
# applies document-level quality filtering,
# saves preprocessed parquet files to data directory.
#
# Usage:
#   bash runs/preprocess.sh
#   NUM_SHARDS=100 bash runs/preprocess.sh
#   HF_ENDPOINT=https://hf-mirror.com bash runs/preprocess.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

RAW_DATA_DIR="${RAW_DATA_DIR:-/home/liujin99/data/essential-web-v1}"
PREPROCESSED_DIR="${PREPROCESSED_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
NUM_SHARDS="${NUM_SHARDS:-50}"

echo "╔══ Essential-Web Preprocessing ═══╗"
echo ""
echo "  Shards to download:  $NUM_SHARDS"
echo "  Raw data dir:        $RAW_DATA_DIR"
echo "  Preprocessed dir:    $PREPROCESSED_DIR"
echo "  HF endpoint:         ${HF_ENDPOINT:-https://huggingface.co}"
echo ""
echo "╚══════════════════════════════════════╝"
echo ""

# ── Step 1: Download raw data ──────────────────────────
if [ ! -d "$RAW_DATA_DIR" ]; then
    mkdir -p "$RAW_DATA_DIR"
fi

EXISTING_RAW=$(ls "$RAW_DATA_DIR"/train-*.parquet 2>/dev/null | wc -l || echo 0)
if [ "$EXISTING_RAW" -lt "$NUM_SHARDS" ]; then
    echo "Downloading Essential-Web raw data..."
    python3 "$CLIMBMIX_DIR/scripts/preprocess/download_essential_web.py" \
        --num-files "$NUM_SHARDS" \
        --output-dir "$RAW_DATA_DIR" \
        --workers 16
else
    echo "Raw data already exists ($EXISTING_RAW shards), skipping download"
fi

# ── Step 2: Preprocess ──────────────────────────────────
EXISTING_PP=$(ls "$PREPROCESSED_DIR"/preprocessed_*.parquet 2>/dev/null | wc -l || echo 0)
if [ "$EXISTING_PP" -lt 1 ]; then
    echo "Preprocessing shards..."
    python3 "$CLIMBMIX_DIR/scripts/preprocess/preprocess_essential_web.py" \
        --input-dir "$RAW_DATA_DIR" \
        --output-dir "$PREPROCESSED_DIR" \
        --workers 64
else
    echo "Preprocessed data already exists ($EXISTING_PP shards), skipping"
fi

echo ""
echo "═══════════════════════════════════════════"
echo "  Preprocessing complete!"
echo "  Data: $PREPROCESSED_DIR/"
echo "═══════════════════════════════════════════"
