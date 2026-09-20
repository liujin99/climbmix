#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  臂引擎 (内部, 非用户入口): 训练/(重)发一个 d28 目标臂
#  由 runs/run_extend_traineval.sh 逐臂调用 (选样→混合→single-pass 守卫→
#  远端 dispatch→报告); 应急手动直调亦可 (一次性自定义臂/失败臂补发)。
#  (docs/reuse_design.md §4.4)
#
#  三种用法 (由 ARM_NAME / WEIGHTS 组合决定):
#    a) 自定义配比臂:  ARM_NAME=fixratio_v1 WEIGHTS="0.2,0.3,..."
#    b) 赢家重训:      ARM_NAME=winner_v2 WEIGHTS=result/<run>/optimal_mixture_weights.json
#                      + 取消注释训练参数覆盖 (TARGET_TOKENS 等)
#    c) 已有臂重发:    ARM_NAME=random (WEIGHTS 留空 — 数据已混合,
#                      失败重试场景; 前次 SUCCEEDED 的臂会被 .done 跳过)
#
#  常规路径: 对比轮用 runs/run_extend_traineval.sh 的 ARMS 臂表, 不直调本引擎
#  训完自动出 CP4 全景报告 (run 内所有臂, 对照默认 random; 臂没落地自动跳过)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod3_current}"        # 主 run 目录
ARM_NAME="${ARM_NAME:-fixratio_v1}"               # 臂名 [A-Za-z0-9_-]+ (random/climb/自定义)
WEIGHTS="${WEIGHTS:-}"                            # 空 = 不选点不混合, 直接重发已有臂
LABEL_SOURCE="${LABEL_SOURCE:-cluster}"           # cluster = 搜索簇(cluster_cache.npz);
                                                   # domain = 四域(schema domain_names, 配 WEIGHTS 域名键)
TARGET_TOKENS="${TARGET_TOKENS:-2B}"              # token 预算: 选点大小 + 步数派生的唯一真源
                                                  # (与 run 快照不同时, 下方自动重派生 TARGET_STEPS)
STEM_RATIO="${STEM_RATIO:-0.7}"
RETRY_FAILED="${RETRY_FAILED:-1}"                 # 1 = 前次 dispatch 失败也重试
WAIT_CACHE_MIN="${WAIT_CACHE_MIN:-0}"             # run 还在搜索中时: 等池缓存的分钟数
SEED="${SEED:-42}"                                # 选样种子 (复刻/换 seed 重跑时改它)
LAUNCH="${LAUNCH:-1}"                             # 0=干跑 (只准备数据+打印发射命令)
# ─── 训练参数覆盖 (可选; 取消注释即生效, 其余用 run 的 launch_env) ──
# TARGET_TOKENS="3B"   # 更大预算: 重选点重混合, 步数自动派生 (TARGET_STEPS 非旋钮)
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

# TARGET_STEPS 非旋钮 (与 run_experiment.sh 同规则): 步数来自 run 的
# launch_env.json; TARGET_TOKENS 覆盖 run 预算时由下方推导块重派生。
if [ -n "${TARGET_STEPS:-}" ]; then
    echo "✗ TARGET_STEPS=${TARGET_STEPS} is no longer a knob — set the budget instead (TARGET_TOKENS=3B re-derives steps)."
    exit 1
fi

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CLIMBMIX_DIR"
source "$CLIMBMIX_DIR/runs/lib/auto_report.sh"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

echo "═══ arm only: ${ARM_NAME} @ ${RUN_DIR} ═══"

# ── 校验 ──
[[ "$ARM_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "✗ ARM_NAME 必须匹配 [A-Za-z0-9_-]+ (进文件名/tag/OBS 路径)"; exit 1; }
[ -d "$RUN_DIR" ] || { echo "✗ RUN_DIR 不存在: ${RUN_DIR}"; exit 1; }

# ── HF 镜像防裸壳 (战后清单 #2): mix 拉取 ClimbMix 分片需要 HF_ENDPOINT;
#    优先级 shell env > run 的 launch_env 快照 > hf-mirror 默认 ──
if [ -z "${HF_ENDPOINT:-}" ]; then
    HF_EP_LE="$(python3 - "$RUN_DIR" <<'PY'
import json, os, sys
try:
    print(json.load(open(os.path.join(sys.argv[1], "launch_env.json"))).get("HF_ENDPOINT", ""))
except (FileNotFoundError, ValueError):
    print("")
PY
)"
    if [ -n "$HF_EP_LE" ]; then
        export HF_ENDPOINT="$HF_EP_LE"
    else
        export HF_ENDPOINT="https://hf-mirror.com"
    fi
fi

MIXED="$RUN_DIR/${ARM_NAME}_mixed"

# ── 选点 + 混合 (仅当 WEIGHTS 给出; 每步 .done 幂等) ──
if [ -n "$WEIGHTS" ]; then
    echo "  weights: ${WEIGHTS}"
    echo "  label source: ${LABEL_SOURCE}"

    # ── 磁盘预算 preflight (战后清单 #8; ARM_DISK_CHECK=0 关闭) ──
    # prod4 教训: 3×6B 臂全链本地落地 ~100G 打穿 /work, winner 混到 95% 阵亡。
    if [ "${ARM_DISK_CHECK:-1}" = "1" ]; then
        python3 scripts/check_disk_budget.py \
            --run-dir "$RUN_DIR" --arm "$ARM_NAME" \
            --target-tokens "$TARGET_TOKENS" \
            ${ARM_DISK_HEADROOM_X:+--headroom-x "$ARM_DISK_HEADROOM_X"} \
            ${ARM_DISK_MIN_FREE_GB:+--min-free-gb "$ARM_DISK_MIN_FREE_GB"} \
            || { echo "✗ 磁盘预算不足 — 清理 (scripts/clean_derived_data.py) 或 ARM_DISK_CHECK=0 自担风险"; exit 1; }
    fi

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
            --label-source "$LABEL_SOURCE" \
            --target-tokens "$TARGET_TOKENS" \
            --seed "$SEED" --num-npu "$NUM_NPU" \
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

# ── 步数来源: run 的 launch_env.json; TARGET_TOKENS 覆盖 run 预算时重派生 ──
LE_INFO="$(python3 - "$RUN_DIR" <<'PY'
import json, os, sys
try:
    with open(os.path.join(sys.argv[1], "launch_env.json")) as f:
        env = json.load(f)
except FileNotFoundError:
    env = {}
print(f"{(env.get('TARGET_TOKENS') or '').strip()}\t{(env.get('TARGET_DEPTH') or '').strip() or '28'}")
PY
)"
LE_TOKENS="${LE_INFO%%$'\t'*}"
TARGET_DEPTH="${LE_INFO##*$'\t'}"
if ! python3 - "$TARGET_TOKENS" "$LE_TOKENS" <<'PY'
import sys
sys.path.insert(0, "src")
from climbmix.utils.token_estimate import parse_token_count
sys.exit(0 if sys.argv[2] and
         parse_token_count(sys.argv[1]) == parse_token_count(sys.argv[2]) else 1)
PY
then
    TARGET_STEPS="$(python3 scripts/derive_target_steps.py \
        --target-tokens "$TARGET_TOKENS" \
        --ckpt-dir "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}")" \
        || { echo "✗ TARGET_STEPS derivation failed"; exit 1; }
    export TARGET_STEPS
    echo "  TARGET_STEPS derived from TARGET_TOKENS=$TARGET_TOKENS (run 快照: ${LE_TOKENS:-无}) -> $TARGET_STEPS steps"
    [ -n "$WEIGHTS" ] || echo "  ⚠ WEIGHTS 为空 — 未按新预算重选数据; 若池子不足, dispatch 守卫会拦截"
fi

# ── 训练参数覆盖透传 (CLI env 优先于 run 的 launch_env.json) ──
for v in MID_DEVICE_BATCH_SIZE TARGET_LR_SCALE \
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
# DISPATCH_EXTRA: 透传给 dispatch 的额外参数 (空格分隔, 如 "--job-timeout-h 22")
if [ -n "${DISPATCH_EXTRA:-}" ]; then
    read -r -a _DE <<< "$DISPATCH_EXTRA"
    EXTRA+=("${_DE[@]}")
fi

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 将执行:"
    echo "  python3 scripts/dispatch_target_arm.py --arm ${ARM_NAME} --output-dir ${RUN_DIR}$( [ "$RETRY_FAILED" = "1" ] && echo ' --retry-failed' )"
    echo "  训完自动出全景对比报告 (run 内所有臂, 对照默认 random)"
    exit 0
fi

# 非 exec — dispatch 结束后要出自动全景报告 (dispatch 失败时 set -e 直接退出, 不报告)
python3 scripts/dispatch_target_arm.py \
    --arm "$ARM_NAME" --output-dir "$RUN_DIR" \
    "${EXTRA[@]+"${EXTRA[@]}"}"

# ── 成功后本地清理 (战后清单 #8; CLEANUP_LOCAL=0 关闭) ──
# OBS 已有内容键化副本 (mixture_data_k*); 本地 {arm}_shards/{arm}_mixed
# 每臂 ~31G (3B), 不清就重演 prod4 磁盘打穿。守卫在脚本内 (未成功不删)。
if [ "${CLEANUP_LOCAL:-1}" = "1" ]; then
    python3 scripts/clean_derived_data.py \
        --run-dir "$RUN_DIR" --arms "$ARM_NAME" --apply \
        || echo "  ⚠ 本地清理未执行 (不影响臂结果; 手动补: python3 scripts/clean_derived_data.py --run-dir ${RUN_DIR} --arms ${ARM_NAME} --apply)"
fi

if [ "${SKIP_AUTO_REPORT:-0}" = "1" ]; then
    echo "  (SKIP_AUTO_REPORT=1 — 报告由调用方统一渲染)"
else
    auto_cp4_report "$RUN_DIR"
fi
