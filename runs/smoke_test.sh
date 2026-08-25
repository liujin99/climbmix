#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix Smoke Test — 小规模验证 pipeline 端到端
#  Phase 1: dry-run (embedding + clustering + output)
#  Phase 2: minimal proxy search (2 configs × 50 steps)
# ═══════════════════════════════════════════════════════════════════════
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
SMOKE_DATA="/tmp/smoke_data"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"

# ── NPU env ──
export OMP_NUM_THREADS=1
export ASCEND_VISIBLE_DEVICES=0
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTHONUNBUFFERED=1

# ── Step 0: Prepare small data (2 parquet files) ──
echo -e "\n════════════════════════════════════════════════════════════"
echo "  Step 0: Prepare small data (2 parquet files)"
echo "════════════════════════════════════════════════════════════"

if [ ! -d "$SMOKE_DATA" ] || [ "$(ls "$SMOKE_DATA"/*.parquet 2>/dev/null | wc -l)" -lt 2 ]; then
    mkdir -p "$SMOKE_DATA"
    cp "$DATA_DIR"/part-0000{0,1}.parquet "$SMOKE_DATA/" 2>/dev/null || {
        echo "Cannot copy parquet files from $DATA_DIR"
        echo "Trying alternative: first 2 files matching part-*.parquet"
        ls "$DATA_DIR"/part-*.parquet 2>/dev/null | head -2 | xargs -I{} cp {} "$SMOKE_DATA/"
    }
fi
echo "Smoke data files:"
ls -lh "$SMOKE_DATA"/*.parquet

# ═══════════════════════════════════════════════════════════════════════
#  Phase 1: Dry-run (embedding + clustering + output, no proxy training)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n════════════════════════════════════════════════════════════"
echo "  Phase 1: Dry-run (embedding + clustering + output)"
echo "════════════════════════════════════════════════════════════"

PHASE1_OUT="$CLIMBMIX_DIR/smoke_out_phase1"
rm -rf "$PHASE1_OUT"
mkdir -p "$PHASE1_OUT"

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$SMOKE_DATA" \
    --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
    --discovery-method embedding_cluster \
    --embedding-sample-size 2000 \
    --embedding-device npu \
    --dry-run \
    --num-iterations 1 \
    --configs-per-iter 2 \
    --device-type npu --npu-devices 1 --npu-per-exp 1 \
    --output-dir "$PHASE1_OUT" \
    --cluster-cache-dir "$PHASE1_OUT" \
    --skip-target

echo -e "\n── Phase 1 Results ──"
echo "Output files:"
ls -lh "$PHASE1_OUT"/*.parquet "$PHASE1_OUT"/*.json 2>/dev/null || echo "  (no output files found — ERROR)"

if [ -f "$PHASE1_OUT/sampled_dataset.parquet" ]; then
    echo "sampled_dataset.parquet exists ✓"
    python3 -c "import pyarrow.parquet as pq; t=pq.read_table('$PHASE1_OUT/sampled_dataset.parquet'); print(f'  Rows: {len(t)}, Cols: {t.column_names}')"
else
    echo "sampled_dataset.parquet MISSING ✗"
    echo "Phase 1 FAILED, skipping Phase 2"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════
#  Phase 2: Minimal proxy search (2 configs × 50 steps)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n════════════════════════════════════════════════════════════"
echo "  Phase 2: Minimal proxy search (2 configs × 50 steps)"
echo "════════════════════════════════════════════════════════════"

# Pre-flight checks
[ -d "$NANOCHAT_DIR" ] || { echo "✗ nanochat-npu not found at $NANOCHAT_DIR — skip Phase 2"; exit 0; }
ckpt="$NANOCHAT_BASE_DIR/base_checkpoints/d20"
[ -d "$ckpt" ] && ls "$ckpt"/model_*.pt >/dev/null 2>&1 || { echo "✗ d20 checkpoint not found — skip Phase 2"; exit 0; }
echo "✓ nanochat + d20 checkpoint found"

PHASE2_OUT="$CLIMBMIX_DIR/smoke_out_phase2"
rm -rf "$PHASE2_OUT"
mkdir -p "$PHASE2_OUT"

# Reuse cluster cache from Phase 1 (skip re-embedding)
cp "$PHASE1_OUT"/cluster_cache.npz "$PHASE2_OUT/" 2>/dev/null || true
cp "$PHASE1_OUT"/cluster_info_cache.json "$PHASE2_OUT/" 2>/dev/null || true

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$SMOKE_DATA" \
    --nanochat-dir "$NANOCHAT_DIR" \
    --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
    --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
    --discovery-method embedding_cluster \
    --embedding-sample-size 2000 \
    --embedding-device npu \
    --num-iterations 1 \
    --configs-per-iter 2 \
    --proxy-num-iterations 50 \
    --proxy-lr-scale 1.0 --proxy-warmup 0.0 --proxy-warmdown 0.9 \
    --phase1-checkpoint-path "$ckpt" \
    --device-type npu --npu-devices 1 --npu-per-exp 1 \
    --output-dir "$PHASE2_OUT" \
    --cluster-cache-dir "$PHASE2_OUT" \
    --resume-search \
    --skip-target

echo -e "\n── Phase 2 Results ──"
echo "Output files:"
ls -lh "$PHASE2_OUT"/*.parquet "$PHASE2_OUT"/*.json 2>/dev/null || echo "  (no output files)"

if [ -f "$PHASE2_OUT/sampled_dataset.parquet" ]; then
    echo "sampled_dataset.parquet exists ✓"
    python3 -c "import pyarrow.parquet as pq; t=pq.read_table('$PHASE2_OUT/sampled_dataset.parquet'); print(f'  Rows: {len(t)}, Cols: {t.column_names}')"
fi

echo -e "\n════════════════════════════════════════════════════════════"
echo "  Smoke test complete!"
echo "  Phase 1 (dry-run): $PHASE1_OUT"
echo "  Phase 2 (proxy):   $PHASE2_OUT"
echo "════════════════════════════════════════════════════════════"
