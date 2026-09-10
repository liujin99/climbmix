#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  从评测阶段开始: 只评测, 不训练
#  (对照 quadmix run_eval_only.sh)
#
#  当前用途: base 锚点评测 (远端 d28 base 模型) — 检验平台/评测管线
#  健康度; run 后补发 (锚点失败/漏跑时)。
#  (待 mmlu 上游修复后的 d20 re-eval 也挂这里, docs/reuse_design.md §4.3)
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_eval_only.sh
#        后台: nohup ./runs/run_eval_only.sh > eval_only.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod2_k15bal_20260909_200323}"   # run 目录 (归档目录也可)
RETRY_FAILED="${RETRY_FAILED:-1}"  # 1 = 之前失败过也重试 (锚点几乎总是补发)
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

[ -f "$RUN_DIR/launch_env.json" ] || { echo "✗ $RUN_DIR/launch_env.json 不存在 — 非 run 目录"; exit 1; }

EXTRA=()
if [ "$RETRY_FAILED" = "1" ]; then
    EXTRA+=(--retry-failed)
fi

exec python3 scripts/dispatch_target_arm.py \
    --arm base_eval_check --output-dir "$RUN_DIR" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
