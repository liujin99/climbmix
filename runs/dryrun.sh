#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Stage 0: CPU dry-run — verify pipeline logic without training
# ──────────────────────────────────────────────────────────────
# Time: ~5min, no NPU/checkpoint needed
#
# Usage: bash runs/dryrun.sh
# ──────────────────────────────────────────────────────────────

set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
NANOCHAT_BRANCH="dev-data-mix"
OUTPUT_DIR="${OUTPUT_DIR:-${CLIMBMIX_DIR}/result/stage0_$(date +%Y%m%d_%H%M%S)}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
die()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── Check ──
echo "=== Stage 0: CPU dry-run (~5min) ==="
for pkg in numpy lightgbm sklearn scipy pandas pyarrow torch matplotlib; do
    python3 -c "import $pkg" 2>/dev/null || die "Missing: $pkg — pip install $pkg"
done; ok "Python deps"

if [ ! -d "$NANOCHAT_DIR" ]; then
    die "nanochat-npu not found at $NANOCHAT_DIR — git clone -b $NANOCHAT_BRANCH https://github.com/liujin99/nanochat-npu.git $NANOCHAT_DIR"
fi
branch=$(cd "$NANOCHAT_DIR" && git rev-parse --abbrev-ref HEAD)
[ "$branch" = "$NANOCHAT_BRANCH" ] || die "nanochat-npu on '$branch', need '$NANOCHAT_BRANCH' — cd $NANOCHAT_DIR && git checkout $NANOCHAT_BRANCH"
ok "nanochat-npu ($NANOCHAT_BRANCH)"

[ -d "$DATA_DIR" ] && ls "$DATA_DIR"/preprocessed_*.parquet >/dev/null 2>&1 || die "No data in $DATA_DIR — set DATA_DIR=<path>"
ok "Data"

echo ""
echo "All checks passed. Starting dry-run..."

python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
    --data-dir "$DATA_DIR" \
    --nanochat-dir "$NANOCHAT_DIR" \
    --proxy-depth 10 \
    --target-depth 24 \
    --discovery-method fdc_labels \
    --filter-method none \
    --num-iterations 3 \
    --configs-per-iter "8,4,2" \
    --dry-run \
    --device-type cpu \
    --output-dir "$OUTPUT_DIR" \
    "$@"

ok "Done → $OUTPUT_DIR"
