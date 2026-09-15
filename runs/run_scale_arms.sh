#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  run_scale_arms.sh — 同一配比策略, 不同 token 预算的成对对比臂 (尺度阶梯)
#
#  场景 (CP4 核心): "winner 配比 vs 均匀基线, 训 X tokens" 的可重复实验。
#  主 run (3B) 之后逐档上量:
#    RUN_DIR=result/prod3_current SCALE_TOKENS=20B NODES=8 \
#      nohup bash runs/run_scale_arms.sh > ~/work/tmp/scale_20b.log 2>&1 &
#  再上一档: 同命令改 SCALE_TOKENS=35B (.done 幂等, 已完成的臂自动跳过)。
#
#  做什么: 校验 → 后台预取 general 分片 (与选样重叠, 跨进程锁保护) →
#  并行调 runs/run_arm_only.sh 两遍 (winner_<tok> + random_<tok>:
#  选样 → 混合 → single-pass 守卫 → 远端 dispatch) → 等两臂落地 →
#  统一渲染 CP4 全景报告 (run 内所有臂)。
#  依赖: RUN_DIR 里已有 optimal_mixture_weights.json + cluster_cache.npz
#  (即主 run 搜索已完成)。臂数据/锁/audit 按臂名独立命名空间, 不碰主流程。
#
#  NODES 必须 2 的幂 (1/2/4/8/16): d28 优化器按 world_size=8×NODES 切分
#  参数, 而 d28 全部张量维度是 2^k — ws 含因子 3 (如 12/15 节点) 时
#  optim.py:499 断言必炸 (2026-09-09 probe C 实证 ws=24 即此死法)。
#  尺度自由度在 token 轴 (连续), 节点轴只有 2 的幂 — 实验变量本来就是前者。
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod3_current}"   # 主 run 目录 (权重/池缓存来源)
SCALE_TOKENS="${SCALE_TOKENS:-20B}"          # token 预算 (10B/20B/35B...)
NODES="${NODES:-8}"                          # 每臂节点数 — 必须 2 的幂 (见上)
ARMS="${ARMS:-winner random}"                # v1 支持 winner|random 成对
SEED="${SEED:-42}"                           # 选样种子 (换 seed 复刻时改它)
PREFETCH="${PREFETCH:-1}"                    # 1=选样期间后台预取 general 分片
LAUNCH="${LAUNCH:-1}"                        # 0=干跑 (只打印计划)
# ─── 本机路径 (与主 run 相同; 服务器默认通常不用改) ─────────────────
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"
SCALE_LOG_DIR="${SCALE_LOG_DIR:-$HOME/work/tmp}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
source "$CLIMBMIX_DIR/runs/lib/auto_report.sh"

# ── 校验 ──
[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 1 && (NODES & (NODES-1)) == 0 )) \
    || { echo "✗ NODES=$NODES 非法 — 必须 2 的幂 (1/2/4/8/16): d28 优化器分片约束 (optim.py:499)"; exit 1; }
WFILE="$RUN_DIR/optimal_mixture_weights.json"
[ -f "$WFILE" ] || { echo "✗ $WFILE 不存在 — 需要主 run 的最终选点 (搜索完成后产出)"; exit 1; }
[ -f "$RUN_DIR/cluster_cache.npz" ] || { echo "✗ $RUN_DIR/cluster_cache.npz 不存在"; exit 1; }
for a in $ARMS; do
    case "$a" in winner|random) : ;; *) echo "✗ ARMS v1 只支持 winner|random, got '$a'"; exit 1 ;; esac
done

# ── 计划 (步数/超时按节点数估算; 分片数按 85K docs × 2940 chars 实测) ──
PLAN="$(python3 - "$SCALE_TOKENS" "$NODES" "$WFILE" <<'PY'
import json, math, sys
sys.path.insert(0, "src")
from climbmix.utils.token_estimate import parse_token_count
tokens = parse_token_count(sys.argv[1]); nodes = int(sys.argv[2])
w = json.load(open(sys.argv[3]))
K = len(w)
uniform = ",".join(["1"] * K)
steps = int(tokens / 1_048_576)                    # 估算 (真值由 run_arm_only 派生)
dt = 145.6 / (nodes * 8) / 0.75                    # 锚点: 18.2s@ws8, 6.1s@ws32
timeout_h = math.ceil(steps * dt / 3600) + 4       # 训练 + 评测 + 引导余量
gen_tok = tokens * 3 / 7                           # stem_ratio 0.7 的 general 侧
needed = math.ceil(gen_tok / 62.5e6)               # 85K docs × 2940 chars / 4 per shard
cap = needed + 32                                  # CLIMBMIX_MAX_SHARDS (留余量)
prefetch = needed + 16
disk_gb = math.ceil(tokens / 1e9 * 36)             # 3B 实测 ~52GB/臂 外推两臂
tag = "".join(c for c in sys.argv[1].lower() if c.isalnum())
print(f"{tokens}\t{steps}\t{timeout_h}\t{needed}\t{cap}\t{prefetch}\t{disk_gb}\t{tag}\t{uniform}")
PY
)" || exit 1
IFS=$'\t' read -r TOK_BYTES STEPS_EST TIMEOUT_H NEEDED CAP PREFETCH_N DISK_GB TOKTAG UNIFORM <<< "$PLAN"

FREE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d G)
[ "$FREE_GB" -ge $((DISK_GB + 50)) ] \
    || { echo "✗ 磁盘空闲 ${FREE_GB}G < 预算 ${DISK_GB}G + 50G (两臂混料+选样)"; exit 1; }

echo "═══ scale arms: ${ARMS} @ ${SCALE_TOKENS} (=${TOK_BYTES} tokens) ═══"
echo "  run:     ${RUN_DIR}"
echo "  nodes:   ${NODES}/臂 (ws=$((NODES*8))) → est ${STEPS_EST} 步, job timeout ${TIMEOUT_H}h"
echo "  general: ~${NEEDED} 分片 (cap=${CAP}, 预取 ${PREFETCH_N})"
echo "  disk:    est ${DISK_GB}G, free ${FREE_GB}G"
echo "  臂名:    $(for a in $ARMS; do printf '%s_%s ' "$a" "$TOKTAG"; done)"

[ "$LAUNCH" = "1" ] || { echo; echo "[dry-run] LAUNCH=0 — 只打印计划"; exit 0; }

mkdir -p "$SCALE_LOG_DIR"

# ── 后台预取 general 分片 (与选样重叠; .download.lock 跨进程保护) ──
if [ "$PREFETCH" = "1" ]; then
    nohup python3 - "$GENERAL_DATA_DIR" "$PREFETCH_N" \
        > "$SCALE_LOG_DIR/scale_${TOKTAG}_prefetch.log" 2>&1 <<'PY' &
import sys
sys.path.insert(0, "scripts")
from mix_general_data import download_climbmix
download_climbmix(sys.argv[1], int(sys.argv[2]), num_workers=16)
print("[prefetch] done")
PY
    echo "  prefetch pid=$!"
fi

# ── 并行发射两臂 (复用 run_arm_only.sh: 选样→混合→守卫→dispatch) ──
declare -a PIDS=()
for a in $ARMS; do
    NAME="${a}_${TOKTAG}"
    if [ "$a" = "winner" ]; then WVAL="$WFILE"; else WVAL="$UNIFORM"; fi
    env RUN_DIR="$RUN_DIR" ARM_NAME="$NAME" WEIGHTS="$WVAL" \
        TARGET_TOKENS="$SCALE_TOKENS" TARGET_ARM_NODES="$NODES" \
        CLIMBMIX_MAX_SHARDS="$CAP" SEED="$SEED" \
        SKIP_AUTO_REPORT=1 DISPATCH_EXTRA="--job-timeout-h $TIMEOUT_H" \
        bash runs/run_arm_only.sh > "$SCALE_LOG_DIR/scale_${NAME}.log" 2>&1 &
    PIDS+=($!)
    echo "  ${NAME} pid=$! (log: $SCALE_LOG_DIR/scale_${NAME}.log)"
done

FAILED=0
for i in "${!PIDS[@]}"; do
    a=$(echo $ARMS | cut -d' ' -f$((i+1)))
    if wait "${PIDS[$i]}"; then
        echo "  ${a}_${TOKTAG} ✓"
    else
        echo "  ✗ ${a}_${TOKTAG} 失败 — 看 $SCALE_LOG_DIR/scale_${a}_${TOKTAG}.log"
        FAILED=1
    fi
done

# ── 统一渲染 CP4 全景报告 (run 内所有臂, 含主 run 的 3B 臂) ──
auto_cp4_report "$RUN_DIR"
echo "═══ done (FAILED=$FAILED) — 报告: ${RUN_DIR}/report.md ═══"
exit $FAILED
