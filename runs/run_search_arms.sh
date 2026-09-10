#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  主实验: d20 搜索 + d28 两臂 (climb/random) + 报告
#  nanochat 式状态驱动 — 同一命令覆盖三种情形, 重跑即续:
#    · 从零:      新 EXP_NAME, HISTORY_RUN 留空
#    · 中断继续:  同 EXP_NAME 重跑本命令 (exp .done / search_state /
#                 指纹三级断点自动续, 不从头来)
#    · 热启动:    新 EXP_NAME + HISTORY_RUN=<旧run> — 注入旧 run 的历史点
#                 作为 iter1, 新点从 iter2 起 guided (docs/reuse_design.md)
#                 [仅当新 run 尚无 search_state.json 时注入一次, 之后同 resume]
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_search_arms.sh
#        后台: nohup ./runs/run_search_arms.sh > run.log 2>&1 &
#        干跑: LAUNCH=0 ./runs/run_search_arms.sh (校验+注入+打印, 不启动)
#  固定比例基线/赢家重训臂/已有臂重发: runs/run_arm_only.sh
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖, 文件值为默认) ──────────────────────
EXP_NAME="${EXP_NAME:-prod3}"          # run 名 (result/<名>_current, OBS 前缀)
HISTORY_RUN="${HISTORY_RUN:-}"         # 热启动源 run; 空 = 从零/继续
K_ENHANCED="${K_ENHANCED:-15}"         # 池聚类数 (热启动必须与源池一致, 会校验)
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-30,20,10}"   # 热启动: 第 1 槽 = 历史点数
ADAPTIVE_CONFIGS="${ADAPTIVE_CONFIGS:-1}"          # 自适应波预算 (生产开)
ADAPTIVE_COMPACT="${ADAPTIVE_COMPACT:-1}"          # 紧凑画像 (时间盒; 进指纹)
NPU_PER_EXP="${NPU_PER_EXP:-8}"         # 每 d20 实验卡数 k (全程固定)
REMOTE_MAX_JOBS="${REMOTE_MAX_JOBS:-10}"           # 远端在飞上限
REMOTE_LOCAL_PARALLEL="${REMOTE_LOCAL_PARALLEL:-1}"  # 本地卡加入舰队 (k=8 整槽)
REMOTE_OBS_PREFIX="${REMOTE_OBS_PREFIX:-}"         # 必填 (obs://bucket/prefix, 须含 ${EXP_NAME} 做 per-run 隔离)
TARGET_ARM_NODES="${TARGET_ARM_NODES:-1}"          # 1=单节点 ~10h/臂; 4=多节点 ~3.4h/臂 (2 的幂)
DISPATCH_RANDOM_ARM="${DISPATCH_RANDOM_ARM:-1}"    # 1=搜索期间并行预发 random 臂
LAUNCH="${LAUNCH:-1}"                   # 0=干跑
# ─── 平台身份 (不确定就保持默认/留空, 由后端配置文件解析) ───────────
# REMOTE_BACKEND_MODULE="climbmix_ma:create_backend"
# REMOTE_PLATFORM_CONFIG=""
# REMOTE_IMAGE=""
# REMOTE_FLAVOR=""
# REMOTE_POOL_NAME=""
# 子进程 (run_climbmix.sh / dispatch / inject) 只见已 export 的变量:
export EXP_NAME K_ENHANCED CONFIGS_PER_ITER ADAPTIVE_CONFIGS ADAPTIVE_COMPACT \
       NPU_PER_EXP REMOTE_MAX_JOBS REMOTE_LOCAL_PARALLEL REMOTE_OBS_PREFIX \
       TARGET_ARM_NODES \
       REMOTE_BACKEND_MODULE REMOTE_PLATFORM_CONFIG REMOTE_IMAGE \
       REMOTE_FLAVOR REMOTE_POOL_NAME
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
OUTPUT_DIR="$CLIMBMIX_DIR/result/${EXP_NAME}_current"
HISTORY_RUN="${HISTORY_RUN#"$CLIMBMIX_DIR"/}"   # repo 相对路径也可

echo "═══ search+arms: ${EXP_NAME} ═══"
echo "  K=${K_ENHANCED}  CONFIGS_PER_ITER=${CONFIGS_PER_ITER}  k=${NPU_PER_EXP}"
echo "  output: ${OUTPUT_DIR}"
if [ -n "$REMOTE_OBS_PREFIX" ]; then
    echo "  obs:    ${REMOTE_OBS_PREFIX}"
else
    echo "  obs:    (未设置 — LAUNCH=1 时必须提供 obs://bucket/prefix, 含 run 名 '${EXP_NAME}')"
fi

# ── 情形判定 (状态驱动, nanochat 式) ──
if [ -f "$OUTPUT_DIR/search_state.json" ]; then
    echo "  → 检测到已有 search_state — 续跑 (HISTORY_RUN 若设置了也不重注入)"
elif [ -n "$HISTORY_RUN" ]; then
    # 热启动: 校验 → 继承池缓存 → 注入 (docs/reuse_design.md §4.2/§8)
    echo "  → 热启动: 注入 ${HISTORY_RUN} 的历史点"
    [ -f "$HISTORY_RUN/search_state.json" ] || { echo "✗ ${HISTORY_RUN}/search_state.json 不存在"; exit 1; }
    [ -f "$HISTORY_RUN/cluster_cache.npz" ] || { echo "✗ ${HISTORY_RUN}/cluster_cache.npz 不存在 (池缓存来源)"; exit 1; }
    if [ -f "$HISTORY_RUN/balanced_profile.json" ]; then
        SRC_K=$(python3 -c "import json;print(json.load(open('${HISTORY_RUN}/balanced_profile.json'))['K_final'])")
        [ "$SRC_K" = "$K_ENHANCED" ] || { echo "✗ 源池 K_final=${SRC_K} != K_ENHANCED=${K_ENHANCED} — 聚类空间不同, 历史点不可复用 (docs/reuse_design.md §2)"; exit 1; }
        echo "    K 校验: 源池 K_final=${SRC_K} == K_ENHANCED ✓"
    fi
    mkdir -p "$OUTPUT_DIR"
    if [ ! -f "$OUTPUT_DIR/cluster_cache.npz" ]; then
        cp "$HISTORY_RUN/cluster_cache.npz" "$OUTPUT_DIR/"
        if [ -f "$HISTORY_RUN/balanced_profile.json" ]; then
            cp "$HISTORY_RUN/balanced_profile.json" "$OUTPUT_DIR/"
        fi
        echo "    池缓存已继承 (继承, 不重新聚类 — 重新聚类 = 历史点静默作废)"
    fi
    python3 scripts/inject_history.py \
        --source "$HISTORY_RUN/search_state.json" \
        --target-dir "$OUTPUT_DIR" \
        --pool "$OUTPUT_DIR/cluster_cache.npz"
    N_HIST=$(python3 -c "import json;print(json.load(open('${OUTPUT_DIR}/search_state.json'))['realized_configs_per_iter'][0])")
    SLOT1="${CONFIGS_PER_ITER%%,*}"
    if [ "$N_HIST" != "$SLOT1" ]; then
        echo "⚠ CONFIGS_PER_ITER 第 1 槽 = ${SLOT1}, 但历史点 = ${N_HIST}"
        echo "  语义: 第 1 槽被历史点整体替换 → 总预算 = ${N_HIST} + ${CONFIGS_PER_ITER#*,}"
        echo "  (要按发射线记账就改成 \"${N_HIST},${CONFIGS_PER_ITER#*,}\")"
        read -r -p "  回车继续, Ctrl-C 退出: " _ || true
    fi
    echo "    热启动就绪: iter1 = ${N_HIST} 历史点, 新点从 iter2 起 guided"
else
    echo "  → 从零开始 (HISTORY_RUN 为空)"
fi

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 将执行: bash runs/run_climbmix.sh (env 如上)"
    exit 0
fi

# per-run 隔离 (docs/reuse_design.md §8): 前缀不含 run 名则
# upload_dir_if_missing 会跨 run 复用旧臂数据 — 池变了就是错误复用。
case "$REMOTE_OBS_PREFIX" in
    "") echo "✗ LAUNCH=1 需要 REMOTE_OBS_PREFIX (obs://bucket/prefix)"; exit 1 ;;
    obs://*) : ;;
    *) echo "✗ REMOTE_OBS_PREFIX 必须以 obs:// 开头 (got: ${REMOTE_OBS_PREFIX})"; exit 1 ;;
esac
case "$REMOTE_OBS_PREFIX" in
    *"${EXP_NAME}"*) : ;;
    *) echo "✗ REMOTE_OBS_PREFIX 未含 run 名 '${EXP_NAME}' — per-run 隔离要求前缀以 /${EXP_NAME} 结尾"; exit 1 ;;
esac

# random 臂预发: dispatch 自带 cluster-cache 等待 + flock, 与主脚本 Step 4
# 的准备共享 .done 产物, 双方幂等。热启动/续跑时池缓存已在, 秒过等待。
if [ "$DISPATCH_RANDOM_ARM" = "1" ]; then
    mkdir -p "$OUTPUT_DIR"
    setsid nohup python3 scripts/dispatch_target_arm.py --arm random \
        --output-dir "$OUTPUT_DIR" \
        > "$OUTPUT_DIR/dispatch_random.log" 2>&1 &
    echo "  random 臂已预发 (log: ${OUTPUT_DIR}/dispatch_random.log)"
fi

exec bash runs/run_climbmix.sh
