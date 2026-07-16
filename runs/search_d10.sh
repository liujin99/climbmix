#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 1: d10 proxy → d24 target — quick validation
# ──────────────────────────────────────────────────────────────
# Purpose: Verify proxy→target transferability
#   - d10 proxy searches 14 mixture configs (3 iterations)
#   - d24 target validates best mixture vs baseline
#   - If d24 improves → proceed to Stage 2
#
# Needs: NPU + d10 base checkpoint + d24 base checkpoint
# Time:  ~5h (d10 search ~3h + d24 validation ~2h)
#
# Usage: bash runs/1_validate.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
NANOCHAT_BRANCH="dev-data-mix"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIMBMIX_DIR}/result/stage1_$(date +%Y%m%d_%H%M%S)}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

need_ckpt() {
    local depth=$1
    local ckpt_dir="${NANOCHAT_BASE_DIR}/base_checkpoints/d${depth}"
    if [ -d "$ckpt_dir" ] && [ "$(ls "$ckpt_dir"/model_*.pt 2>/dev/null | wc -l)" -gt 0 ]; then
        ok "d${depth} checkpoint"; return 0
    fi
    warn "d${depth} checkpoint NOT found at $ckpt_dir"
    echo "  Generate on 8×NPU:"
    echo "    cd $NANOCHAT_DIR && source /usr/local/Ascend/ascend-toolkit/set_env.sh"
    echo "    torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=$depth --target-param-data-ratio=9.5 --device-batch-size=8 --model-tag=d${depth}"
    echo "    torchrun --standalone --nproc_per_node=8 -m scripts.base_eval  -- --model-tag=d${depth} --model-type=base --eval=core"
    echo "  Or copy from a shared checkpoint store."
    return 1
}

# ── Check ──
echo "=== Stage 1: d10→d24 (~5h, NPU) ==="
for pkg in numpy lightgbm sklearn scipy pandas pyarrow torch matplotlib; do
    python3 -c "import $pkg" 2>/dev/null || die "Missing: $pkg — pip install $pkg"
done; ok "Python deps"

if [ ! -d "$NANOCHAT_DIR" ]; then
    die "nanochat-npu not found — git clone -b $NANOCHAT_BRANCH https://github.com/liujin99/nanochat-npu.git $NANOCHAT_DIR"
fi
branch=$(cd "$NANOCHAT_DIR" && git rev-parse --abbrev-ref HEAD)
[ "$branch" = "$NANOCHAT_BRANCH" ] || die "nanochat-npu on '$branch', need '$NANOCHAT_BRANCH'"
ok "nanochat-npu ($NANOCHAT_BRANCH)"

[ -d "$DATA_DIR" ] && ls "$DATA_DIR"/preprocessed_*.parquet >/dev/null 2>&1 || die "No data in $DATA_DIR"
ok "Data"

python3 -c "import torch_npu; assert torch.npu.is_available()" 2>/dev/null || die "NPU not available — this stage requires 8×910B3 NPU"
ok "NPU"

echo ""
echo "=== Required checkpoints ==="
need_ckpt 10 || die "Need d10 checkpoint (see instructions above)"
need_ckpt 24 || die "Need d24 checkpoint (see instructions above)"

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
echo ""
echo "All checks passed. Starting experiment..."

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --nanochat-dir "$NANOCHAT_DIR" \
    --proxy-depth 10 \
    --proxy-num-iterations 500 \
    --proxy-lr-scale 1.0 \
    --proxy-warmup 0.0 \
    --proxy-warmdown 0.9 \
    --phase1-checkpoint-path "${NANOCHAT_BASE_DIR}/base_checkpoints/d10" \
    --target-depth 24 \
    --target-num-iterations 1000 \
    --target-phase1-checkpoint-path "${NANOCHAT_BASE_DIR}/base_checkpoints/d24" \
    --discovery-method fdc_labels \
    --filter-method none \
    --num-iterations 3 \
    --configs-per-iter "8,4,2" \
    --device-type npu \
    --npu-devices 8 \
    --output-dir "$OUTPUT_DIR" \
    "$@"

ok "Done → $OUTPUT_DIR"
