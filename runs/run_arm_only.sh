#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  从臂阶段开始: 训练/重发一个 d28 目标臂 (跳过搜索)
#  (docs/reuse_design.md §4.4; 对照 quadmix run_stem_random_only.sh)
#
#  三种用法 (由 ARM_NAME / WEIGHTS 组合决定):
#    a) 自定义配比臂:  ARM_NAME=fixratio_v1 WEIGHTS="0.2,0.3,..."
#    b) 赢家重训:      ARM_NAME=winner_v2 WEIGHTS=result/<run>/optimal_mixture_weights.json
#                      + 取消注释训练参数覆盖 (TARGET_STEPS 等)
#    c) 已有臂重发:    ARM_NAME=random (WEIGHTS 留空 — 数据已混合,
#                      失败重试场景; 前次 SUCCEEDED 的臂会被 .done 跳过)
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_arm_only.sh
#  对比: run_report_only.sh ARM_A=<臂名> ARM_B=climb
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod3_current}"        # 主 run 目录
ARM_NAME="${ARM_NAME:-fixratio_v1}"               # 臂名 [A-Za-z0-9_-]+ (random/climb/自定义)
WEIGHTS="${WEIGHTS:-}"                            # 空 = 不选点不混合, 直接重发已有臂
TARGET_TOKENS="${TARGET_TOKENS:-2B}"              # (选点时) 与对比臂相同的 token 预算!
STEM_RATIO="${STEM_RATIO:-0.7}"
RETRY_FAILED="${RETRY_FAILED:-1}"                 # 1 = 前次 dispatch 失败也重试
WAIT_CACHE_MIN="${WAIT_CACHE_MIN:-0}"             # run 还在搜索中时: 等池缓存的分钟数
LAUNCH="${LAUNCH:-1}"                             # 0=干跑 (只准备数据+打印发射命令)
# ─── 训练参数覆盖 (可选; 取消注释即生效, 其余用 run 的 launch_env) ──
# TARGET_STEPS="3000"
# MID_DEVICE_BATCH_SIZE="1"
# TARGET_LR_SCALE="1.0"
# TARGET_WARMUP="0.0"
# TARGET_WARMDOWN="0.9"
# TARGET_ARM_NODES="4"
# ─── 本机路径 (与主 run 相同; 服务器默认通常不用改) ─────────────────
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"
NUM_NPU="${NUM_NPU:-8}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

echo "═══ arm only: ${ARM_NAME} @ ${RUN_DIR} ═══"

# ── 校验 ──
[[ "$ARM_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "✗ ARM_NAME 必须匹配 [A-Za-z0-9_-]+ (进文件名/tag/OBS 路径)"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "✗ RUN_DIR 不存在: ${RUN_DIR}"; exit 1; }

MIXED="$RUN_DIR/${ARM_NAME}_mixed"

# ── 选点 + 混合 (仅当 WEIGHTS 给出; 每步 .done 幂等) ──
if [ -n "$WEIGHTS" ]; then
    echo "  weights: ${WEIGHTS}"

    CACHE="$RUN_DIR/cluster_cache.npz"
    DEADLINE=$(( $(date +%s) + WAIT_CACHE_MIN * 60 ))
    while [ ! -f "$CACHE" ]; do
        if [ "$(date +%s)" -ge "$DEADLINE" ]; then
            echo "✗ ${CACHE} 等待超时 (${WAIT_CACHE_MIN}min) — 主 run 还没到聚类完成?"
            exit 1
        fi
        echo "  等待池缓存 (剩余 $(( (DEADLINE - $(date +%s)) / 60 ))min)..."; sleep 30
    done

    # 池数据目录: 优先 run 的 launch_env 快照, 其次 env
    DATA_DIR="$(python3 - "$RUN_DIR" <<'PY'
import json, os, sys
le = os.path.join(sys.argv[1], "launch_env.json")
if os.path.exists(le):
    print(json.load(open(le)).get("DATA_DIR", ""))
else:
    print(os.environ.get("DATA_DIR", ""))
PY
)"
    [ -n "$DATA_DIR" ] || { echo "✗ DATA_DIR 无法解析 (${RUN_DIR} 无 launch_env.json 且 env 未设) — 在 EDIT 块 export DATA_DIR"; exit 1; }
    echo "  pool: ${DATA_DIR}"

    SHARDS="$RUN_DIR/${ARM_NAME}_shards"
    if [ -f "$SHARDS/.done" ]; then
        echo "  选点已完成 (.done): ${SHARDS}"
    else
        python3 scripts/prepare_random_baseline.py \
            --data-dir "$DATA_DIR" \
            --output-dir "$SHARDS" \
            --cluster-cache "$CACHE" \
            --schema config/schema_stem.yaml \
            --target-tokens "$TARGET_TOKENS" \
            --seed 42 --num-npu "$NUM_NPU" \
            --weights "$WEIGHTS"
    fi

    if [ -f "$MIXED/.done" ]; then
        echo "  混合已完成 (.done): ${MIXED}"
    else
        NANOCHAT_REPO="$NANOCHAT_DIR" python3 scripts/mix_general_data.py \
            --stem-dir "$SHARDS" --output-dir "$MIXED" \
            --climbmix-dir "$GENERAL_DATA_DIR" \
            --stem-ratio "$STEM_RATIO" \
            --num-workers "$NUM_NPU" --num-npu "$NUM_NPU"
    fi
else
    echo "  WEIGHTS 为空 — 直接重发已有臂 (要混合数据已在: ${MIXED})"
    [ -f "$MIXED/.done" ] || echo "  ⚠ ${MIXED}/.done 不存在 — 若该臂从未混合过, dispatch 会因缺数据报错"
fi

# ── 训练参数覆盖透传 (CLI env 优先于 run 的 launch_env.json) ──
for v in TARGET_STEPS MID_DEVICE_BATCH_SIZE TARGET_LR_SCALE \
         TARGET_WARMUP TARGET_WARMDOWN TARGET_ARM_NODES \
         EVAL_BENCHMARKS EVAL_MAX_PER_TASK; do
    if [ -n "${!v:-}" ]; then
        export "$v"
    fi
done

EXTRA=()
if [ "$RETRY_FAILED" = "1" ]; then
    EXTRA+=(--retry-failed)
fi

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 将执行:"
    echo "  python3 scripts/dispatch_target_arm.py --arm ${ARM_NAME} --output-dir ${RUN_DIR}$( [ "$RETRY_FAILED" = "1" ] && echo ' --retry-failed' )"
    exit 0
fi

exec python3 scripts/dispatch_target_arm.py \
    --arm "$ARM_NAME" --output-dir "$RUN_DIR" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
