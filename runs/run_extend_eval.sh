#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  扩展评测: 对既有 ckpt 换评测基准集评测 → 报告 (不训练)
#
#  当前用途: base 锚点评测 (远端 d28 base 模型) — 检验平台/评测管线
#  健康度; run 后补发 (锚点失败/漏跑时)。
#  (待 mmlu 上游修复后的 d20 re-eval 也挂这里, docs/reuse_design.md §4.3)
#
#  用法: 编辑下方 EDIT 块 → bash runs/run_extend_eval.sh
#        后台: nohup bash runs/run_extend_eval.sh > extend_eval.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod2_k15bal_20260909_200323}"   # run 目录 (归档目录也可)
RETRY_FAILED="${RETRY_FAILED:-1}"  # 1 = 之前失败过也重试 (锚点几乎总是补发)
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

# 深整理布局 (2026-10-10): 归档 run 的 launch_env 在 detail/ — 双布局判存
if [ ! -f "$RUN_DIR/launch_env.json" ] && [ ! -f "$RUN_DIR/detail/launch_env.json" ]; then
    echo "✗ $RUN_DIR/launch_env.json 不存在 — 非 run 目录"; exit 1
fi

EXTRA=()
if [ "$RETRY_FAILED" = "1" ]; then
    EXTRA+=(--retry-failed)
fi

python3 scripts/dispatch_target_arm.py \
    --arm base_eval_check --output-dir "$RUN_DIR" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

# ── 落地后的发布形态维护 (2026-10-10): 锚点 CSV 落在根层 —
# ① 决策图刷新 (有搜索状态才画; 判定块留档 logs/report_charts.log)
# ② 已归档的 run 重新深整理, 根层回到 9 项发布形态;
#    活跃 run 跳过 (未收官, 状态文件必须原位)。
# 锚点臂无 dispatch 跳过检查 (总是重发), 整理不影响幂等语义。──
mkdir -p "$RUN_DIR/logs"
if [ -f "$RUN_DIR/search_state.json" ] || [ -f "$RUN_DIR/detail/search_state.json" ]; then
    if python3 scripts/diagnostics/report_charts.py "$RUN_DIR" \
            > "$RUN_DIR/logs/report_charts.log" 2>&1; then
        tail -n 40 "$RUN_DIR/logs/report_charts.log"
    else
        echo "  (report_charts 未完成 — 详见 $RUN_DIR/logs/report_charts.log)"
    fi
fi
if [ -f "$RUN_DIR/archive_meta.json" ] || [ -f "$RUN_DIR/detail/archive_meta.json" ]; then
    python3 scripts/diagnostics/tidy_result_dir.py "$RUN_DIR" --deep --apply \
        || echo "  (tidy 未完成 — 手动: tidy_result_dir.py $RUN_DIR --deep --apply)"
else
    echo "  (run 未归档 — 跳过深整理)"
fi
