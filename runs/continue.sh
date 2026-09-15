#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  continue.sh — 唯一执行入口
#
#  语义: <stage> 是"重入点"——从该阶段开始, 把它及其后所有步骤跑完
#        (不是只执行该子步骤)。
#
#    scratch  全新实验: 池 → 聚类 → d20 搜索 → 混料 → 双臂训练评测 → 报告
#             (同 EXP_NAME 重跑 = 断点续跑, 三级断点不从头来)
#    search   从搜索步续: 注入历史 run 已测点做增量 d20 → 混料 → 双臂 → 报告
#    mix      从混料步续 (验证轮): 新预算/seed 重采样重混双臂 → 训练评测 → 报告
#    arm      从臂步续: (重)发一个臂 (自定义配比/赢家重训/重发) → 评测 → 报告
#    eval     从评测步续: 评测既有 ckpt / base 锚点 → 报告
#
#  用法:  bash runs/continue.sh <stage>
#         (阶段旋钮一律走 env, 见 runs/stages/<stage>.sh 头注的 EDIT 块)
#  干跑:  LAUNCH=0 bash runs/continue.sh <stage>
#  实现:  runs/stages/<stage>.sh (arm 同时是 mix 的逐臂引擎)
#  引擎:  runs/run_climbmix.sh = scratch 阶段背后的主管道 (高级直接路径)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail
RUNS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

menu() { sed -n '2,20p' "$RUNS_DIR/continue.sh" | sed 's/^# \{0,1\}//'; }

STAGE="${1:-}"
case "$STAGE" in
    scratch|search|mix|arm|eval)
        shift
        [ "$#" -eq 0 ] || { echo "✗ 阶段旋钮一律走 env (如 SCALE_TOKENS=20B NODES=8), 不收位置参数: $*"; exit 1; }
        exec bash "$RUNS_DIR/stages/$STAGE.sh"
        ;;
    -h|--help)
        menu; exit 0
        ;;
    "")
        menu; exit 1
        ;;
    *)
        echo "✗ 未知阶段: '$STAGE' — 可选: scratch | search | mix | arm | eval"
        exit 1
        ;;
esac
