#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  Diagnose which env var(s) in speedrun cause 100% NaN in embedding.
#
#  Strategy: binary search + individual testing
#    Phase 1: baseline (smoke test env)  → expect 0% NaN
#    Phase 2: full speedrun env         → expect 100% NaN
#    Phase 3: add vars in groups         → narrow down
#    Phase 4: test individual vars       → pinpoint culprit
#
#  Each test loads model + encodes 200 docs (~30s). Total ~8-10 min.
# ═══════════════════════════════════════════════════════════════════════
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"
export DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"

# ── Clear NPU compilation cache (if exists) ──
echo "=== Clearing NPU compilation cache ==="
rm -rf ~/.cache/torch/npu/ ~/.cache/ascend/ kernel_meta/ 2>/dev/null || true
echo "Done."
echo ""

# ── Test runner ──
test_env() {
    local label="$1"
    shift
    # Build env: minimal baseline + extra vars
    local env_str="OMP_NUM_THREADS=1 ASCEND_GLOBAL_LOG_LEVEL=3 PYTHONUNBUFFERED=1 ASCEND_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True"
    if [ $# -gt 0 ]; then
        env_str="$env_str $*"
    fi
    echo -n "  [TEST] ${label}: "
    local logfile="/tmp/diag_env_$$.log"
    env $env_str python3 "$CLIMBMIX_DIR/scripts/diagnostics/diagnose_env_test.py" 2>&1 | tee "$logfile" | grep "^RESULT:" || {
        echo "ERROR (no RESULT line, showing last 20 lines of output)"
        tail -20 "$logfile"
    }
    rm -f "$logfile"
}

echo "════════════════════════════════════════════════════════════"
echo "  Phase 1: Baseline (smoke test env) — expect 0% NaN"
echo "════════════════════════════════════════════════════════════"
test_env "baseline"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 2: Full speedrun env (minus HCCL/NCCL comm vars)"
echo "════════════════════════════════════════════════════════════"
test_env "full-speedrun" \
    ASCEND_COMPILE_OPT_LEVEL=O3 \
    ASCEND_ENABLE_CACHE=1 \
    ASCEND_FUSION_ENABLE=0 \
    TORCH_NPU_LAZY_COMPILE=1 \
    ASCEND_DISABLE_MEM_SWAP=1 \
    NPU_DISABLE_RECORD=1 \
    PYTORCH_NPU_ALLOC_MAX_SIZE=60G \
    'TORCH_NPU_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,memory_pool:True' \
    PYTHONPRELOAD=torch_npu \
    NANOCHAT_DTYPE=bfloat16 \
    ASCEND_LAUNCH_BLOCKING=0

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 3: Group A — compiler/cache vars"
echo "════════════════════════════════════════════════════════════"
test_env "group-A-compiler" \
    ASCEND_COMPILE_OPT_LEVEL=O3 \
    ASCEND_ENABLE_CACHE=1 \
    TORCH_NPU_LAZY_COMPILE=1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 3: Group B — memory/runtime vars"
echo "════════════════════════════════════════════════════════════"
test_env "group-B-memory" \
    ASCEND_DISABLE_MEM_SWAP=1 \
    NPU_DISABLE_RECORD=1 \
    PYTORCH_NPU_ALLOC_MAX_SIZE=60G \
    'TORCH_NPU_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,memory_pool:True' \
    PYTHONPRELOAD=torch_npu \
    NANOCHAT_DTYPE=bfloat16 \
    ASCEND_LAUNCH_BLOCKING=0

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 4: Individual vars (only if Phase 2 or 3 found NaN)"
echo "════════════════════════════════════════════════════════════"
test_env "only-O3" ASCEND_COMPILE_OPT_LEVEL=O3
test_env "only-CACHE" ASCEND_ENABLE_CACHE=1
test_env "only-LAZY" TORCH_NPU_LAZY_COMPILE=1
test_env "only-MEMSWAP" ASCEND_DISABLE_MEM_SWAP=1
test_env "only-MAXALLOC" PYTORCH_NPU_ALLOC_MAX_SIZE=60G
test_env "only-ALLOCCONF" 'TORCH_NPU_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256,memory_pool:True'
test_env "only-PRELOAD" PYTHONPRELOAD=torch_npu
test_env "only-DTYPE" NANOCHAT_DTYPE=bfloat16
test_env "only-FUSION0" ASCEND_FUSION_ENABLE=0
test_env "only-LAUNCHBLOCK" ASCEND_LAUNCH_BLOCKING=0
test_env "only-DISABLE-RECORD" NPU_DISABLE_RECORD=1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Phase 5: O3 + CACHE combined (most likely combo)"
echo "════════════════════════════════════════════════════════════"
test_env "O3+CACHE" ASCEND_COMPILE_OPT_LEVEL=O3 ASCEND_ENABLE_CACHE=1

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Diagnosis complete. Review results above."
echo "════════════════════════════════════════════════════════════"
