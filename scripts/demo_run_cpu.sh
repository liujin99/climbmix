#!/bin/bash
# CLIMB quick CPU demo (~1-2min)
# Uses tiny 1M proxy model, 20 steps, 3 iterations with 4/2/1 configs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create small test dataset if not exists
DATA_PATH="${PROJECT_DIR}/data/test_data.parquet"
if [ ! -f "$DATA_PATH" ]; then
    echo "[Demo] Creating small test dataset..."
    python "${PROJECT_DIR}/scripts/create_test_data.py" \
        --output "$DATA_PATH" --num-docs 500 --num-clusters 10
fi

echo "[Demo] Running CLIMB quick demo (CPU, 1M proxy, 20 steps)"
echo "[Demo] Data: $DATA_PATH"

python "${PROJECT_DIR}/scripts/run_climb.py" \
    --data-path "$DATA_PATH" \
    --output-dir "${PROJECT_DIR}/temp/demo_cpu" \
    --K-init 10 \
    --K-enhanced 5 \
    --num-iterations 3 \
    --configs-per-iter "4,2,1" \
    --proxy-model-size 1M \
    --proxy-training-steps 20 \
    --device cpu \
    --dry-run \
    --skip-preprocess

echo "[Demo] Complete! Results in ${PROJECT_DIR}/temp/demo_cpu/"
