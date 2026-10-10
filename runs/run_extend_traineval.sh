#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  扩展训练评测: 基于同一最优配比, 变 token 预算 / base_model / 训练参数的
#  双臂对比实验 (训练 → 评测 → 报告)
#
#  实验身份 = 一个最优配比策略的产出条件 (数据池 + 聚类 + 搜索配置,
#  即搜索指纹); 池/聚类/搜索变了才是新实验 (run_experiment /
#  run_extend_experiment)。同一配比的更多训练评测 (更大 token 预算 /
#  换 seed / 对比臂) 不是搜索新实验 — 是独立成 run 的验证轮, 与源实验
#  平级并存 (2026-10-10 裁决: 开训练 = 新 run, 只重打分 = 原地):
#
#    result/prod3_<ts>/                  # 实验 prod3 (搜索 + 首轮验证, 封存只读)
#    result/prod3_<ts>_val20b/           # 本脚本产出 (state/ 布局, 自洽)
#    result/prod3_<ts>_val35b_s7/
#
#  本脚本做什么: 校验 → 建独立轮 run 目录 result/<源名>_<轮名>/ (从源
#  只读复制 state 文件进 state/, OBS 前缀重写隔离) → 后台预取 general
#  分片 → 按臂表逐臂调 runs/lib/arm_engine.sh (选样→混合→single-pass
#  守卫→远端 dispatch) → 落地后渲染本轮 CP4 报告。
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
#  · 轮目录名 = <源目录名>_<轮名>; 源收官改名 (prod3_current →
#    prod3_<ts>) 后重跑同命令会生成新轮目录、丢掉 .done 幂等 —
#    重跑/续跑请始终指向同一源目录
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

# ── 三代布局 (2026-10-10 统一布局): state/ → 根级 → detail/
#    (SRC_RUN_DIR 常指向已收官的归档目录, 三代形态都可能) ──
sfile() {
    if [ -f "$SRC_RUN_DIR/state/$1" ]; then printf '%s\n' "$SRC_RUN_DIR/state/$1"
    elif [ -f "$SRC_RUN_DIR/$1" ]; then printf '%s\n' "$SRC_RUN_DIR/$1"
    else printf '%s\n' "$SRC_RUN_DIR/detail/$1"; fi
}

# ── 校验 ──
[[ "$NODES" =~ ^[0-9]+$ ]] && (( NODES >= 1 && (NODES & (NODES-1)) == 0 )) \
    || { echo "✗ NODES=$NODES 非法 — 必须 2 的幂 (1/2/4/8/16): d28 优化器分片约束 (optim.py:499)"; exit 1; }
WFILE="$(sfile optimal_mixture_weights.json)"
[ -f "$WFILE" ] || { echo "✗ $SRC_RUN_DIR/optimal_mixture_weights.json 不存在 — 需要实验的最终选点 (搜索完成后产出; run 收官改名后把 SRC_RUN_DIR 指向归档目录)"; exit 1; }
[ -f "$(sfile cluster_cache.npz)" ] || { echo "✗ $SRC_RUN_DIR/cluster_cache.npz 不存在"; exit 1; }
[ -f "$(sfile launch_env.json)" ]  || { echo "✗ $SRC_RUN_DIR/launch_env.json 不存在"; exit 1; }
[ -f "$(sfile remote_config.json)" ] || { echo "✗ $SRC_RUN_DIR/remote_config.json 不存在"; exit 1; }
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
# 独立轮 run 目录 (2026-10-10 裁决): 与源实验平级, 名字携带来源、排序相邻
SRC_BASE="${SRC_RUN_DIR%/}"; SRC_BASE="${SRC_BASE##*/}"
[ -n "$SRC_BASE" ] || { echo "✗ SRC_RUN_DIR 无法解析出目录名: ${SRC_RUN_DIR}"; exit 1; }
ROUND_DIR="${ROUND_DIR:-result/${SRC_BASE}_${ROUND_NAME}}"

FREE_GB=$(df -BG . | awk 'NR==2{print $4}' | tr -d G)
[ "$FREE_GB" -ge $((DISK_GB + 50)) ] \
    || { echo "✗ 磁盘空闲 ${FREE_GB}G < 预算 ${DISK_GB}G + 50G (两臂混料+选样)"; exit 1; }

echo "═══ traineval round: ${ARMS} @ ${SCALE_TOKENS} (=${TOK_BYTES} tokens) ═══"
echo "  src:     ${SRC_RUN_DIR} (只读)"
echo "  round:   ${ROUND_DIR} (独立验证 run, 与源平级)"

# 防改名脚枪: 同轮名兄弟轮已存在 (多半是源改名后重跑) → 点名提示;
# 20B 级重训很贵, 不要静默开新轮
for _d in result/*_"$ROUND_NAME"; do
    if [ -f "$_d/.traineval_round" ] && [ "$_d" != "$ROUND_DIR" ]; then
        echo "  ⚠ 已存在同轮名目录: $_d"
        echo "    $(sed -n 2p "$_d/.traineval_round")"
        echo "    若为源改名后的重跑, 请改回原源/原轮目录 (保住 .done 幂等)"
    fi
done
echo "  nodes:   ${NODES}/臂 (ws=$((NODES*8))) → est ${STEPS_EST} 步, job timeout ${TIMEOUT_H}h"
echo "  general: ~${NEEDED} 分片 (cap=${CAP}, 预取 ${PREFETCH_N})"
echo "  disk:    est ${DISK_GB}G, free ${FREE_GB}G"

[ "$LAUNCH" = "1" ] || { echo; echo "[dry-run] LAUNCH=0 — 只打印计划"; exit 0; }

# ── 初始化轮 run 目录 (幂等; 从源只读复制 state 文件进 state/ — 轮是
#    正经 state/ 布局 run; OBS 前缀重写隔离; 源文件经 sfile 三代解析,
#    平铺/detail/ 归档皆可为源) ──
mkdir -p "$ROUND_DIR/state"
for f in optimal_mixture_weights.json cluster_cache.npz cluster_info_cache.json \
         macro_info.json launch_env.json search_state.json; do
    src="$(sfile "$f")"
    [ -f "$src" ] && { [ -f "$ROUND_DIR/state/$f" ] || cp "$src" "$ROUND_DIR/state/$f"; }
done
[ -f "$ROUND_DIR/state/remote_config.json" ] || cp "$(sfile remote_config.json)" "$ROUND_DIR/state/remote_config.json"
python3 - "$ROUND_DIR/state/remote_config.json" "$ROUND_NAME" <<'PY'
import json, sys
p, round_name = sys.argv[1], sys.argv[2]
c = json.load(open(p))
pre = (c.get("obs_prefix") or "").rstrip("/")
if pre:
    base, _, exp = pre.rpartition("/")
    c["obs_prefix"] = f"{base}/{exp}_{round_name}"
json.dump(c, open(p, "w"), indent=2)
print("[setup] obs_prefix ->", c.get("obs_prefix") or "(none)")
PY
cat > "$ROUND_DIR/.traineval_round" <<MSG
traineval round managed by runs/run_extend_traineval.sh
src experiment: ${SRC_RUN_DIR}   budget: ${SCALE_TOKENS}   nodes: ${NODES}   seed: ${SEED}
独立验证 run (state/ 布局, state 文件从源只读复制而来); 非 stage-gate
管理 — 无指纹, 报告由本脚本直接渲染, 不走 mark_completed 改名流
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
        winner) NAME=winner; WVAL="$ROUND_DIR/state/optimal_mixture_weights.json" ;;
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

# ── 决策图 (2026-10-10): B 联 winner vs random = 本轮主结果;
# A/D/E 为来源实验的搜索语境 (轮 state/ 复制了 search_state/cluster 信息);
# 无 climb 族臂 → C 自动跳过。判定块留档 logs/report_charts.log。
# 轮目录即发布形态 (report + 图表在根层, state 文件在 state/),
# .done 幂等标记在轮根层 (arm_engine/dispatch 的跳过检查只认根层)。──
mkdir -p "$ROUND_DIR/logs"
if python3 scripts/diagnostics/report_charts.py "$ROUND_DIR" \
        > "$ROUND_DIR/logs/report_charts.log" 2>&1; then
    tail -n 30 "$ROUND_DIR/logs/report_charts.log"
else
    echo "  (report_charts 未完成 — 详见 $ROUND_DIR/logs/report_charts.log)"
fi
echo "═══ done (FAILED=$FAILED) — 报告: ${ROUND_DIR}/report.md ═══"
exit $FAILED
