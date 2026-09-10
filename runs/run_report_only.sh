#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  从报告阶段开始: 对一个已完成/已归档的 run 出全套事后分析
#  (对照 quadmix 的报告步; docs/reuse_design.md §4.1)
#
#  两份产物 (输入缺哪个就跳过哪个, 都缺才报错):
#    1. 分数重算 sidecar: <run>/search_state.json.rescored.json
#       — 用当前评分公式重算历史点分数 (评分修 bug/改设计后的影响评估)
#    2. CP4 臂对比报告: 任意两臂 stem_metric 对比 + 显著性 + 符号检验
#       — 默认 climb vs random; 自定义臂 (run_arm_only.sh 产出) 直接换名
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_report_only.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod2_k15bal_20260909_200323}"   # run 目录
ARM_A="${ARM_A:-climb}"         # 臂 A (对比对象; 自定义臂如 fixratio_v1)
ARM_B="${ARM_B:-random}"        # 臂 B (参照)
DO_RESCORE="${DO_RESCORE:-1}"   # 1 = 先重算分数 sidecar (便宜, 无副作用)
TOP="${TOP:-10}"                # rescore 打印前 N 名
JSON_OUT="${JSON_OUT:-}"        # CP4 机读 verdict 输出路径 (空 = 不写)
# ─── CP4 对照设定 (通常用默认) ─────────────────────────────────────
BASE_EXPECTED="${BASE_EXPECTED:-0.1738}"   # base 锚点 stem_metric
SE="${SE:-0.006}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

STATE="$RUN_DIR/search_state.json"
CSV_A="$RUN_DIR/eval_${ARM_A}.csv"
CSV_B="$RUN_DIR/eval_${ARM_B}.csv"

HAVE_RESCORE=0
HAVE_CP4=0
if [ -f "$STATE" ]; then
    HAVE_RESCORE=1
fi
if [ -f "$CSV_A" ] && [ -f "$CSV_B" ]; then
    HAVE_CP4=1
fi

if [ "$HAVE_RESCORE" -eq 0 ] && [ "$HAVE_CP4" -eq 0 ]; then
    echo "✗ ${RUN_DIR} 既无 search_state.json 也无 eval_${ARM_A}/${ARM_B}.csv — 没有可报告的东西"
    exit 1
fi

# ── 1. 分数重算 (raw → 当前公式; sidecar, 原始 state 不动) ──
if [ "$DO_RESCORE" = "1" ] && [ "$HAVE_RESCORE" -eq 1 ]; then
    echo "═══ rescore ═══"
    python3 scripts/rescore_search.py --state "$STATE" --top "$TOP" || true
fi

# ── 2. CP4 臂对比 ──
if [ "$HAVE_CP4" -eq 1 ]; then
    echo
    echo "═══ CP4: ${ARM_A} vs ${ARM_B} ═══"
    EXTRA=()
    if [ -n "$JSON_OUT" ]; then
        EXTRA+=(--json "$JSON_OUT")
    fi
    python3 scripts/diagnostics/cp4_report.py "$RUN_DIR" \
        --arm-a "$ARM_A" --arm-b "$ARM_B" \
        --base-expected "$BASE_EXPECTED" --se "$SE" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
elif [ "$HAVE_RESCORE" -eq 1 ]; then
    echo "(跳过 CP4: 缺 ${CSV_A} 或 ${CSV_B})"
fi
