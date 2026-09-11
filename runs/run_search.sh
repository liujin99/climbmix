#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  主实验: d20 搜索 + d28 两臂 (climb/random) + 报告
#  状态驱动 — 同一命令覆盖两种情形, 重跑即续:
#    · 从零:      新 EXP_NAME
#    · 中断继续:  同 EXP_NAME 重跑本命令 (exp .done / search_state /
#                 指纹三级断点自动续, 不从头来)
#  复用历史实验热启动 (inject 旧 run 结果为新 run 地基):
#    → runs/run_extend_search.sh
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_search.sh
#        后台: nohup ./runs/run_search.sh > run.log 2>&1 &
#        干跑: LAUNCH=0 ./runs/run_search.sh (校验+打印, 不启动)
#  其他阶段: run_arm_only.sh (臂, 训完自动全景报告) / run_eval_only.sh (评测)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖, 文件值为默认) ──────────────────────
EXP_NAME="${EXP_NAME:-prod3}"          # run 名 (result/<名>_current, OBS 前缀)
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10}"   # 本 run 轮次计划 (每轮新 d20 实验数)
K_ENHANCED="${K_ENHANCED:-15}"         # 池聚类数
ADAPTIVE_CONFIGS="${ADAPTIVE_CONFIGS:-1}"          # 自适应波预算 (生产开)
ADAPTIVE_COMPACT="${ADAPTIVE_COMPACT:-1}"          # 紧凑画像 (时间盒; 进指纹)
NPU_PER_EXP="${NPU_PER_EXP:-8}"         # 每 d20 实验卡数 k (全程固定)
REMOTE_MAX_JOBS="${REMOTE_MAX_JOBS:-10}"           # 远端在飞上限
REMOTE_LOCAL_PARALLEL="${REMOTE_LOCAL_PARALLEL:-1}"  # 本地卡加入舰队 (k=8 整槽)
REMOTE_OBS_PREFIX="${REMOTE_OBS_PREFIX:-}"         # 留空则自动读 climbmix-ma 配置 (见下)
TARGET_ARM_NODES="${TARGET_ARM_NODES:-1}"          # 1=单节点 ~10h/臂; 4=多节点 ~3.4h/臂 (2 的幂)
DISPATCH_RANDOM_ARM="${DISPATCH_RANDOM_ARM:-1}"    # 1=搜索期间并行预发 random 臂
LAUNCH="${LAUNCH:-1}"                   # 0=干跑
# ─── 平台身份 (不确定就保持默认/留空, 由后端配置文件解析) ───────────
# REMOTE_BACKEND_MODULE="climbmix_ma:create_backend"
# REMOTE_PLATFORM_CONFIG=""
# REMOTE_IMAGE=""
# REMOTE_FLAVOR=""
# REMOTE_POOL_NAME=""

# ─── OBS 前缀自动配置 (真实内网路径不进 public repo) ────────────────
# 优先级: env 显式 > climbmix-ma 配置的 obs_prod_base (自动拼 /<EXP_NAME>)
# 一次性: 把 "obs_prod_base": "obs://bucket/…/climbmix" 写进
#         ~/.config/climbmix/remote_ma.json (与 secret 同文件, 私有, 永不进 git)
if [ -z "$REMOTE_OBS_PREFIX" ]; then
    _OBS=$(python3 -c "
import os, sys
_d = os.path.join(os.getcwd(), 'climbmix-ma')
os.path.isdir(_d) and sys.path.insert(0, _d)
try:
    from climbmix_ma.modelarts_job_api import load_ma_config
    b = str(load_ma_config().get('obs_prod_base') or '').strip()
    if b.startswith('obs://'):
        print(b.rstrip('/') + '/' + os.environ['EXP_NAME'])
except Exception as e:
    sys.stderr.write(f'[obs] climbmix_ma 配置读取失败: {e}\n')
" || true)
    [ -n "$_OBS" ] && REMOTE_OBS_PREFIX="$_OBS"
fi

# 子进程 (run_climbmix.sh / dispatch) 只见已 export 的变量:
export EXP_NAME CONFIGS_PER_ITER K_ENHANCED ADAPTIVE_CONFIGS ADAPTIVE_COMPACT \
       NPU_PER_EXP REMOTE_MAX_JOBS REMOTE_LOCAL_PARALLEL REMOTE_OBS_PREFIX \
       TARGET_ARM_NODES \
       REMOTE_BACKEND_MODULE REMOTE_PLATFORM_CONFIG REMOTE_IMAGE \
       REMOTE_FLAVOR REMOTE_POOL_NAME
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
OUTPUT_DIR="$CLIMBMIX_DIR/result/${EXP_NAME}_current"

echo "═══ search+arms: ${EXP_NAME} ═══"
echo "  轮次计划: ${CONFIGS_PER_ITER}  K=${K_ENHANCED}  k=${NPU_PER_EXP}"
echo "  output: ${OUTPUT_DIR}"
if [ -n "$REMOTE_OBS_PREFIX" ]; then
    echo "  obs:    ${REMOTE_OBS_PREFIX}"
else
    echo "  obs:    (未解析 — LAUNCH=1 时必须提供; 配置 obs_prod_base 可免填, 见上方注释)"
fi

# ── 情形判定 (状态驱动) ──
if [ -f "$OUTPUT_DIR/search_state.json" ]; then
    echo "  → 检测到已有 search_state — 续跑 (三级断点, 不从头来)"
else
    echo "  → 从零开始"
    echo "    (要基于已有 d20 实验结果做增量实验? 用 runs/run_extend_search.sh)"
fi

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 将执行: bash runs/run_climbmix.sh (env 如上)"
    exit 0
fi

# per-run 隔离 (docs/reuse_design.md §8): 前缀不含 run 名则
# upload_dir_if_missing 会跨 run 复用旧臂数据 — 池变了就是错误复用。
case "$REMOTE_OBS_PREFIX" in
    "") echo "✗ LAUNCH=1 需要 REMOTE_OBS_PREFIX (obs://bucket/prefix, 末尾含 run 名)"; echo "  免填配置: 在 ~/.config/climbmix/remote_ma.json 加 \"obs_prod_base\": \"obs://bucket/…/climbmix\" (与 secret 同文件)"; exit 1 ;;
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
