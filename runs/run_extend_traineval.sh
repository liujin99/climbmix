#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  扩展训练评测: 基于同一最优配比, 变 token 预算 / base_model / 训练参数的
#  双臂对比实验 (训练 → 评测 → 报告)
#
#  实验身份 = 一个最优配比策略的产出条件 (数据池 + 聚类 + 搜索配置,
#  即搜索指纹); 池/聚类/搜索变了才是新实验 (run_experiment /
#  run_extend_experiment)。同一配比的更多训练评测 (更大 token 预算 /
#  换 seed / 对比臂) 不是新实验 — 是本实验的 traineval 轮, 住进实验
#  目录的 traineval/ 子目录:
#
#    result/prod3_current/                 # 实验 prod3 (一个 _current)
#    ├── 搜索产物 + 首轮 3B 验证 (run_experiment 的 Step 1-8, 根目录历史约定)
#    └── traineval/
#        ├── val20b/                       # 本脚本产出的轮 (独立成套)
#        └── val35b_s7/
#
#  本脚本做什么: 校验 → 建 traineval/<轮名>/ (从实验根目录只读复制
#  weights/cluster_cache/launch_env/remote_config, OBS 前缀重写隔离) →
#  后台预取 general 分片 → 按臂表逐臂调 runs/lib/arm_engine.sh (选样→
#  混合→single-pass 守卫→远端 dispatch) → 落地后渲染本轮 CP4 报告。
#
#  入口家族 (后两个 extend_* = 对已有实验的扩展动作, 非必经下一站):
#    run_experiment / run_extend_experiment / run_extend_traineval(本) /
#    run_extend_eval
#
#  用法:
#    SRC_RUN_DIR=result/prod3_current SCALE_TOKENS=20B NODES=8 \
#      nohup bash runs/run_extend_traineval.sh > ~/work/tmp/traineval20b.log 2>&1 &
#  再上一档: SCALE_TOKENS=35B (轮名自动派生 val35b; .done 幂等可重入;
#  失败臂重跑同命令即重试)
#  臂表: ARMS="winner random" (默认) | 含自定义: "winner random fix1=0.1,..."
#
#  注意:
#  · 实验 run 绿色收官后目录会改名 (prod3_current → prod3_<ts>);
#    下一轮把 SRC_RUN_DIR 指向归档目录即可 (轮目录内容自洽, 不受影响)
#  · NODES 必须 2 的幂 (1/2/4/8/16): d28 优化器按 ws=8×NODES 切分全部
#    2^k 张量维度, ws 含因子 3 时 optim.py:499 断言必炸 (probe C 实证)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
SRC_RUN_DIR="${SRC_RUN_DIR:-result/prod3_current}"   # 实验目录 (只读来源)
ROUND_NAME="${ROUND_NAME:-}"         # 空=自动派生 val<tokens>[_s<seed>]
SCALE_TOKENS="${SCALE_TOKENS:-20B}"          # token 预算 (10B/20B/35B...)
NODES="${NODES:-8}"                          # 每臂节点数 — 必须 2 的幂
ARMS="${ARMS:-winner random}"                # 臂表: winner | random | 名字=权重
SEED="${SEED:-42}"                           # 选样种子 (≠42 时进轮名)
PREFETCH="${PREFETCH:-1}"                    # 1=选样期间后台预取 general 分片
LAUNCH="${LAUNCH:-1}"                        # 0=干跑 (只打印计划)
# ─── 本机路径 (与主 run 相同; 服务器默认通常不用改) ─────────────────
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"
TRAINEVAL_LOG_DIR="${TRAINEVAL_LOG_DIR:-$HOME/work/tmp}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
source "$CLIMBMIX_DIR/runs/lib/auto_report.sh"

# ── 校验 ──
[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 1 && (NODES & (NODES-1)) == 0 )) \
    || { echo "✗ NODES=$NODES 非法 — 必须 2 的幂 (1/2/4/8/16): d28 优化器分片约束 (optim.py:499)"; exit 1; }
WFILE="$SRC_RUN_DIR/optimal_mixture_weights.json"
[ -f "$WFILE" ] || { echo "✗ $WFILE 不存在 — 需要实验的最终选点 (搜索完成后产出; run 收官改名后把 SRC_RUN_DIR 指向归档目录)"; exit 1; }
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
steps = int(tokens / 1_048_576)                    # 估算 (真值由 arm_engine 派生)
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

# 轮名自动派生: val20b / val35b_s7
if [ -z "$ROUND_NAME" ]; then
    ROUND_NAME="val${TOKTAG}"
    [ "$SEED" != "42" ] && ROUND_NAME="${ROUND_NAME}_s${SEED}"
fi
ROUND_DIR="$SRC_RUN_DIR/traineval/$ROUND_NAME"

FREE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d G)
[ "$FREE_GB" -ge $((DISK_GB + 50)) ] \
    || { echo "✗ 磁盘空闲 ${FREE_GB}G < 预算 ${DISK_GB}G + 50G (两臂混料+选样)"; exit 1; }

echo "═══ traineval round: ${ARMS} @ ${SCALE_TOKENS} (=${TOK_BYTES} tokens) ═══"
echo "  src:     ${SRC_RUN_DIR} (只读)"
echo "  round:   ${ROUND_DIR} (实验内的独立验证轮)"
echo "  nodes:   ${NODES}/臂 (ws=$((NODES*8))) → est ${STEPS_EST} 步, job timeout ${TIMEOUT_H}h"
echo "  general: ~${NEEDED} 分片 (cap=${CAP}, 预取 ${PREFETCH_N})"
echo "  disk:    est ${DISK_GB}G, free ${FREE_GB}G"

[ "$LAUNCH" = "1" ] || { echo; echo "[dry-run] LAUNCH=0 — 只打印计划"; exit 0; }

# ── 初始化轮目录 (幂等; 从实验根目录复制数据源, OBS 前缀重写隔离) ──
mkdir -p "$ROUND_DIR"
for f in optimal_mixture_weights.json cluster_cache.npz cluster_info_cache.json \
         launch_env.json search_state.json; do
    [ -f "$SRC_RUN_DIR/$f" ] && { [ -f "$ROUND_DIR/$f" ] || cp "$SRC_RUN_DIR/$f" "$ROUND_DIR/$f"; }
done
[ -f "$ROUND_DIR/remote_config.json" ] || cp "$SRC_RUN_DIR/remote_config.json" "$ROUND_DIR/remote_config.json"
python3 - "$ROUND_DIR/remote_config.json" "$ROUND_NAME" <<'PY'
import json, sys
p, round_name = sys.argv[1], sys.argv[2]
c = json.load(open(p))
pre = (c.get("obs_prefix") or "").rstrip("/")
if pre:
    base, _, exp = pre.rpartition("/")
    c["obs_prefix"] = f"{base}/{exp}/traineval/{round_name}"
json.dump(c, open(p, "w"), indent=2)
print("[setup] obs_prefix ->", c["obs_prefix"])
PY
cat > "$ROUND_DIR/.traineval_round" <<MSG
traineval round managed by runs/run_extend_traineval.sh
src experiment: ${SRC_RUN_DIR}   budget: ${SCALE_TOKENS}   nodes: ${NODES}   seed: ${SEED}
不要把此目录当 run 目录使用 (无指纹, 非 stage-gate 管理)
MSG
echo "  setup ✓ (${ROUND_DIR})"

mkdir -p "$TRAINEVAL_LOG_DIR"

# ── 后台预取 general 分片 (与选样重叠; .download.lock 跨进程保护) ──
if [ "$PREFETCH" = "1" ]; then
    nohup python3 - "$GENERAL_DATA_DIR" "$PREFETCH_N" \
        > "$TRAINEVAL_LOG_DIR/traineval_${ROUND_NAME}_prefetch.log" 2>&1 <<'PY' &
import sys
sys.path.insert(0, "scripts")
from mix_general_data import download_climbmix
download_climbmix(sys.argv[1], int(sys.argv[2]), num_workers=16)
print("[prefetch] done")
PY
    echo "  prefetch pid=$!"
fi

# ── 逐臂发射 (引擎: runs/lib/arm_engine.sh = 选样→混合→守卫→dispatch) ──
declare -a PIDS=() NAMES=()
for spec in $ARMS; do
    case "$spec" in
        winner) NAME=winner; WVAL="$ROUND_DIR/optimal_mixture_weights.json" ;;
        random) NAME=random; WVAL="$UNIFORM" ;;
        *)      NAME="${spec%%=*}"; WVAL="${spec#*=}" ;;
    esac
    env RUN_DIR="$ROUND_DIR" ARM_NAME="$NAME" WEIGHTS="$WVAL" \
        TARGET_TOKENS="$SCALE_TOKENS" TARGET_ARM_NODES="$NODES" \
        CLIMBMIX_MAX_SHARDS="$CAP" SEED="$SEED" \
        SKIP_AUTO_REPORT=1 DISPATCH_EXTRA="--job-timeout-h $TIMEOUT_H" \
        bash runs/lib/arm_engine.sh > "$TRAINEVAL_LOG_DIR/traineval_${ROUND_NAME}_${NAME}.log" 2>&1 &
    PIDS+=($!); NAMES+=("$NAME")
    echo "  ${NAME} pid=$! (log: $TRAINEVAL_LOG_DIR/traineval_${ROUND_NAME}_${NAME}.log)"
done

FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ${NAMES[$i]} ✓"
    else
        echo "  ✗ ${NAMES[$i]} 失败 — 看 $TRAINEVAL_LOG_DIR/traineval_${ROUND_NAME}_${NAMES[$i]}.log"
        FAILED=1
    fi
done

# ── CP4 报告 (本轮目录内的臂, 对照 random) ──
auto_cp4_report "$ROUND_DIR"
echo "═══ done (FAILED=$FAILED) — 报告: ${ROUND_DIR}/report.md ═══"
exit $FAILED
