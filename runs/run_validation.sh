#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  run_validation.sh — continue from 训练验证 (两阶段目录模式, quadmix 同款)
#
#  算法阶段 (run_search.sh: 搜索 + 首次验证) 产出 weights / cluster cache;
#  本脚本从那里 "continue": 建立独立验证目录 result/<VAL_NAME>_current,
#  按配置的臂 + token 预算跑一轮新的训练验证对比, 自带 CP4 报告。
#
#  用法 (服务器: git pull 后一条命令):
#    SRC_RUN_DIR=result/prod3_current SCALE_TOKENS=20B NODES=8 \
#      nohup bash runs/run_validation.sh > ~/work/tmp/val20b.log 2>&1 &
#  再上一档 (独立新目录, .done 幂等):
#    SRC_RUN_DIR=result/prod3_current SCALE_TOKENS=35B NODES=8 ...同上
#  臂配置 (可任意组合):
#    ARMS="winner random"                       # 默认: 搜索最优 vs 均匀基线
#    ARMS="winner random fix1=0.1,0.2,...,15个" # 自定义配比臂 (逗号列表或权重文件路径)
#
#  目录协议 (与算法阶段隔离):
#    SRC_RUN_DIR (只读)  →  result/<VAL_NAME>_current (本实验独立目录)
#    复制: weights / cluster_cache(+info) / launch_env / search_state
#    重写: remote_config 的 obs_prefix → .../<VAL_NAME> (per-run OBS 隔离)
#  ⚠ VAL 目录只归本脚本管 — 不要拿它的名字跑 run_search.sh
#    (无指纹目录会被 stage gate 当 orphan 归档)。
#
#  分层: 本脚本是编排层; 每臂由 runs/run_arm_only.sh (单臂引擎:
#  选样→混合→single-pass 守卫→远端 dispatch) 执行。
#  NODES 必须 2 的幂 (1/2/4/8/16): d28 优化器按 ws=8×NODES 切分全部
#  2^k 张量维度, ws 含因子 3 时 optim.py:499 断言必炸 (probe C 实证)。
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
SRC_RUN_DIR="${SRC_RUN_DIR:-result/prod3_current}"   # 算法阶段目录 (只读)
VAL_NAME="${VAL_NAME:-}"           # 空=自动派生 <src名>_val<tokens>, 如 prod3_val20b
SCALE_TOKENS="${SCALE_TOKENS:-20B}"          # token 预算 (10B/20B/35B...)
NODES="${NODES:-8}"                          # 每臂节点数 — 必须 2 的幂
ARMS="${ARMS:-winner random}"                # 臂表: winner | random | 名字=权重
SEED="${SEED:-42}"                           # 选样种子 (换 seed 复刻时改它)
PREFETCH="${PREFETCH:-1}"                    # 1=选样期间后台预取 general 分片
LAUNCH="${LAUNCH:-1}"                        # 0=干跑 (只打印计划)
# ─── 本机路径 (与主 run 相同; 服务器默认通常不用改) ─────────────────
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"
VAL_LOG_DIR="${VAL_LOG_DIR:-$HOME/work/tmp}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
source "$CLIMBMIX_DIR/runs/lib/auto_report.sh"

# ── 校验 ──
[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 1 && (NODES & (NODES-1)) == 0 )) \
    || { echo "✗ NODES=$NODES 非法 — 必须 2 的幂 (1/2/4/8/16): d28 优化器分片约束 (optim.py:499)"; exit 1; }
WFILE="$SRC_RUN_DIR/optimal_mixture_weights.json"
[ -f "$WFILE" ] || { echo "✗ $WFILE 不存在 — 需要算法阶段的最终选点"; exit 1; }
[ -f "$SRC_RUN_DIR/cluster_cache.npz" ] || { echo "✗ $SRC_RUN_DIR/cluster_cache.npz 不存在"; exit 1; }
[ -f "$SRC_RUN_DIR/launch_env.json" ]  || { echo "✗ $SRC_RUN_DIR/launch_env.json 不存在"; exit 1; }
[ -f "$SRC_RUN_DIR/remote_config.json" ] || { echo "✗ $SRC_RUN_DIR/remote_config.json 不存在"; exit 1; }
for spec in $ARMS; do
    case "$spec" in
        winner|random) : ;;
        *=*) [[ "${spec%%=*}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "✗ 臂名非法: ${spec%%=*}"; exit 1; } ;;
        *) echo "✗ 臂配置 '$spec' 无法解析 (winner | random | 名字=权重)"; exit 1 ;;
    esac
done

# ── 计划 (python: 步数/超时按节点数估算; 分片按 85K docs × 2940 chars 实测) ──
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

# VAL_NAME 自动派生: result/prod3_current → prod3_val20b
if [ -z "$VAL_NAME" ]; then
    SRC_BASE="$(basename "$SRC_RUN_DIR")"; SRC_BASE="${SRC_BASE%_current}"
    VAL_NAME="${SRC_BASE}_val${TOKTAG}"
fi
VAL_DIR="result/${VAL_NAME}_current"

FREE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d G)
[ "$FREE_GB" -ge $((DISK_GB + 50)) ] \
    || { echo "✗ 磁盘空闲 ${FREE_GB}G < 预算 ${DISK_GB}G + 50G (两臂混料+选样)"; exit 1; }

echo "═══ validation: ${ARMS} @ ${SCALE_TOKENS} (=${TOK_BYTES} tokens) ═══"
echo "  src:     ${SRC_RUN_DIR} (只读)"
echo "  val dir: ${VAL_DIR} (独立实验目录)"
echo "  nodes:   ${NODES}/臂 (ws=$((NODES*8))) → est ${STEPS_EST} 步, job timeout ${TIMEOUT_H}h"
echo "  general: ~${NEEDED} 分片 (cap=${CAP}, 预取 ${PREFETCH_N})"
echo "  disk:    est ${DISK_GB}G, free ${FREE_GB}G"

[ "$LAUNCH" = "1" ] || { echo; echo "[dry-run] LAUNCH=0 — 只打印计划"; exit 0; }

# ── 初始化 VAL 目录 (幂等; 从算法阶段复制数据源, OBS 前缀重写隔离) ──
mkdir -p "$VAL_DIR"
for f in optimal_mixture_weights.json cluster_cache.npz cluster_info_cache.json \
         launch_env.json search_state.json; do
    [ -f "$SRC_RUN_DIR/$f" ] && { [ -f "$VAL_DIR/$f" ] || cp "$SRC_RUN_DIR/$f" "$VAL_DIR/$f"; }
done
[ -f "$VAL_DIR/remote_config.json" ] || cp "$SRC_RUN_DIR/remote_config.json" "$VAL_DIR/remote_config.json"
python3 - "$VAL_DIR/remote_config.json" "$VAL_NAME" <<'PY'
import json, sys
p, name = sys.argv[1], sys.argv[2]
c = json.load(open(p))
pre = (c.get("obs_prefix") or "").rstrip("/")
if pre:
    c["obs_prefix"] = pre.rsplit("/", 1)[0] + "/" + name
json.dump(c, open(p, "w"), indent=2)
print("[setup] obs_prefix ->", c["obs_prefix"])
PY
cat > "$VAL_DIR/.validation_only" <<MSG
validation-only dir managed by runs/run_validation.sh
src: ${SRC_RUN_DIR}   budget: ${SCALE_TOKENS}   nodes: ${NODES}   seed: ${SEED}
不要用 ${VAL_NAME} 作为 EXP_NAME 跑 run_search.sh (无指纹目录会被 orphan 归档)
MSG
echo "  setup ✓ (${VAL_DIR})"

mkdir -p "$VAL_LOG_DIR"

# ── 后台预取 general 分片 (与选样重叠; .download.lock 跨进程保护) ──
if [ "$PREFETCH" = "1" ]; then
    nohup python3 - "$GENERAL_DATA_DIR" "$PREFETCH_N" \
        > "$VAL_LOG_DIR/val_${VAL_NAME}_prefetch.log" 2>&1 <<'PY' &
import sys
sys.path.insert(0, "scripts")
from mix_general_data import download_climbmix
download_climbmix(sys.argv[1], int(sys.argv[2]), num_workers=16)
print("[prefetch] done")
PY
    echo "  prefetch pid=$!"
fi

# ── 逐臂发射 (引擎: run_arm_only.sh = 选样→混合→守卫→dispatch) ──
declare -a PIDS=() NAMES=()
for spec in $ARMS; do
    case "$spec" in
        winner) NAME=winner; WVAL="$VAL_DIR/optimal_mixture_weights.json" ;;
        random) NAME=random; WVAL="$UNIFORM" ;;
        *)      NAME="${spec%%=*}"; WVAL="${spec#*=}" ;;
    esac
    env RUN_DIR="$VAL_DIR" ARM_NAME="$NAME" WEIGHTS="$WVAL" \
        TARGET_TOKENS="$SCALE_TOKENS" TARGET_ARM_NODES="$NODES" \
        CLIMBMIX_MAX_SHARDS="$CAP" SEED="$SEED" \
        SKIP_AUTO_REPORT=1 DISPATCH_EXTRA="--job-timeout-h $TIMEOUT_H" \
        bash runs/run_arm_only.sh > "$VAL_LOG_DIR/val_${VAL_NAME}_${NAME}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$NAME")
    echo "  ${NAME} pid=$! (log: $VAL_LOG_DIR/val_${VAL_NAME}_${NAME}.log)"
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ${NAMES[$i]} ✓"
    else
        echo "  ✗ ${NAMES[$i]} 失败 — 看 $VAL_LOG_DIR/val_${VAL_NAME}_${NAMES[$i]}.log"
        FAILED=1
    fi
done

# ── CP4 报告 (本验证目录内的臂, 对照 random) ──
auto_cp4_report "$VAL_DIR"
echo "═══ done (FAILED=$FAILED) — 报告: ${VAL_DIR}/report.md ═══"
exit $FAILED
