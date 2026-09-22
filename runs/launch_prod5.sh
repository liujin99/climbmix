#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  runs/launch_prod5.sh — prod5 版本化发射器（2026-09-22）
#
#  原则（TODO #115 版本化规程，2026-09-22 用户重申）：发射 = bash 本脚本。
#  不拼裸命令、不贴 LLM 现生成的 env——所有值要么写死在本文件（git 版本
#  管理），要么从 SRC 轮的落盘工件机器重建，逐键对照由脚本完成。
#
#  用法：
#    bash runs/launch_prod5.sh              # 干跑（默认）：重建 + 对照表 + 引擎干跑门
#    LAUNCH=1 bash runs/launch_prod5.sh     # 真发射：preflight → 后台引擎 → 监控提示
#
#  重建来源（机器逐键对照）：
#    result/${SRC_ROUND}_current/launch_env.json     30 个 target/共享旋钮
#    result/${SRC_ROUND}_current/remote_config.json  fleet 旋钮（REMOTE_*）
#  本轮有意变更 / 无法重建的旋钮：本文件 EDIT 块（标注裁决出处）。
#
#  缓存继承：3 个聚类缓存文件 cp 进 result/${EXP_NAME}_current——stage-gate
#  的 cache-seed 豁免（3f99ef2）保它不被 orphan 归档。
#  注意：EMBEDDING_CACHE_DIR 刻意不设——run 级种子是主机制；池级 OBS 缓存
#  是人工兜底（内部值不入库，见服务器记录/TODO #124-ii），种子意外丢失时
#  手动 export 后重跑。
#
#  重发射注意：若 result/${EXP_NAME}_current 已有指纹且代码/参数变过，
#  引擎会把整个目录（含缓存种子）归档后空目录重来 → 那次发射会退化成
#  重聚类/重嵌入。此时先 rm 掉 _current 再跑本脚本（种子会重新 cp）。
# ═══════════════════════════════════════════════════════════════════════

# ─── EDIT：本轮参数（env 可覆盖，文件值为默认） ──────────────────────
EXP_NAME="${EXP_NAME:-prod5}"
SRC_ROUND="${SRC_ROUND:-prod4}"
LAUNCH="${LAUNCH:-0}"                     # 默认干跑；真发射显式 LAUNCH=1

# 本轮有意变更（E1/E2 裁决；其余一切 = prod4 同值，由工件重建）
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-64,32,16}"    # E2：112 新点（prod4: 54,36,18 含注入）
PROXY_TARGET_TOKENS="${PROXY_TARGET_TOKENS:-400M}"  # E1：同 prod4（launcher 默认 640M，必须显式）
TARGET_TOKENS="${TARGET_TOKENS:-6B}"                # E1：同 prod4 引擎值（实录 launch_env=6B）——
                                                     # Stage 5 终选 6B 口径 + 20B 可行性耦合同基；
                                                     # d28 臂预算 3B 是臂派发时的 env 覆盖（runbook
                                                     # 4.6），不在引擎层设（launcher 默认 2B，必须显式）
DISPATCH_RANDOM_ARM="${DISPATCH_RANDOM_ARM:-0}"     # 臂族 = 搜索收官后手动发射（基线臂已更名
                                                    # uniform；预发逻辑 special-case 'random' 名，
                                                    # launcher 默认 1，必须显式归零）
REMOTE_ENABLED="${REMOTE_ENABLED:-1}"

# 无法从 SRC 工件重建的搜索旋钮（SRC 未落盘；值 = prod4 记录在案）
SEARCH_NUM_ITERATIONS="${SEARCH_NUM_ITERATIONS:-3}"  # = len(64,32,16)
ADAPTIVE_CONFIGS="${ADAPTIVE_CONFIGS:-1}"             # prod4 实现数超计划（[54,38,19]）= 自适应开
ADAPTIVE_COMPACT="${ADAPTIVE_COMPACT:-1}"
# ─────────────────────────────────────────────────────────────────────

set -euo pipefail
CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
SRC_DIR="result/${SRC_ROUND}_current"
OUTPUT_DIR="result/${EXP_NAME}_current"
LOG_FILE="${EXP_NAME}.log"

# 防守：清除遗留旋钮（引擎守卫：设置即报错）
unset TARGET_STEPS PROXY_NUM_ITERATIONS 2>/dev/null || true

echo "═══ ${EXP_NAME} 版本化发射器（源轮 ${SRC_ROUND}）═══"
[ -f "$SRC_DIR/launch_env.json" ] || { echo "✗ 缺 $SRC_DIR/launch_env.json"; exit 1; }
[ -f "$SRC_DIR/remote_config.json" ] || { echo "✗ 缺 $SRC_DIR/remote_config.json（REMOTE_ENABLED=1 需要）"; exit 1; }

# ── 缓存种子（幂等；stage-gate cache-seed 豁免保它存活） ──
mkdir -p "$OUTPUT_DIR"
for f in cluster_cache.npz cluster_info_cache.json balanced_profile.json; do
    if [ ! -f "$OUTPUT_DIR/$f" ] && [ -f "$SRC_DIR/$f" ]; then
        cp "$SRC_DIR/$f" "$OUTPUT_DIR/"
        echo "  缓存种子: 继承 ${f}"
    elif [ -f "$OUTPUT_DIR/$f" ]; then
        echo "  缓存种子: 已在 ${f}"
    else
        echo "  ⚠ 源缺 ${f} —— Step 1 将重聚类（确认不是失误再继续）"
    fi
done
if [ -f "$OUTPUT_DIR/.fingerprint_search" ]; then
    echo "  ⚠ ${OUTPUT_DIR} 已有指纹（上次发射残留）——若其间代码/参数变过，"
    echo "    引擎会归档整个目录（含种子）后空目录重来；先 rm 再跑本脚本更稳。"
fi

# ── 环境重建 + 逐键对照（embedded python） ──
ENV_FILE="$(mktemp)"
trap 'rm -f "$ENV_FILE"' EXIT
python3 - "$SRC_DIR" "$ENV_FILE" "$EXP_NAME" "$CONFIGS_PER_ITER" \
         "$TARGET_TOKENS" "$PROXY_TARGET_TOKENS" "$DISPATCH_RANDOM_ARM" \
         "$SEARCH_NUM_ITERATIONS" "$ADAPTIVE_CONFIGS" "$ADAPTIVE_COMPACT" \
         "$REMOTE_ENABLED" <<'PYEOF'
import json, os, shlex, sys

(src_dir, env_file, exp_name, cpi, target_tok, proxy_tok, dispatch_arm,
 ssi, adapt, adaptc, remote_enabled) = sys.argv[1:12]
le = json.load(open(os.path.join(src_dir, "launch_env.json")))
rc = json.load(open(os.path.join(src_dir, "remote_config.json")))

# 1) launch_env.json → env。跳过：本轮覆盖（TARGET_TOKENS 由 EDIT 提供，
#    下面会对照）、引擎派生/自算（TARGET_STEPS / TARGET_LOAD_OPTIMIZER /
#    TARGET_BASE_CKPT / OUTPUT_DIR / CLIMBMIX_DIR / EXP_NAME）、空值
#    （空 = 用引擎默认，语义等价）。
SKIP = {"EXP_NAME", "OUTPUT_DIR", "CLIMBMIX_DIR", "TARGET_STEPS",
        "TARGET_LOAD_OPTIMIZER", "TARGET_BASE_CKPT", "TARGET_TOKENS"}
env, src_le = {}, {}
for k, v in le.items():
    if k in SKIP or v == "":
        continue
    env[k] = str(v)
    src_le[k] = str(v)

# 2) remote_config.json → REMOTE_*（单位回换算：秒 → 小时）
def hours(s):
    return str(int(s // 3600)) if s % 3600 == 0 else repr(s / 3600.0)

rc_map = [
    ("REMOTE_BACKEND",             rc["backend"]),
    ("REMOTE_BACKEND_MODULE",      rc["backend_module"]),
    ("REMOTE_PLATFORM_CONFIG",     rc.get("platform_config") or ""),
    ("REMOTE_IMAGE",               rc.get("image") or ""),
    ("REMOTE_FLAVOR",              rc.get("flavor") or ""),
    ("REMOTE_POOL_NAME",           rc.get("pool_name") or ""),
    ("REMOTE_NPU_PER_JOB",         str(rc["npu_per_job"])),
    ("REMOTE_MAX_JOBS",            str(rc["max_concurrent_jobs"])),
    ("REMOTE_SUBMIT_RETRY_H",      hours(rc["submit_retry_timeout_s"])),
    ("REMOTE_MAX_PREP",            str(rc["max_prep_parallel"])),
    ("REMOTE_LOCAL_PARALLEL",      "1" if rc.get("local_parallel") else "0"),
    ("REMOTE_STORAGE_KIND",        rc.get("storage_kind") or ""),
    ("REMOTE_STORAGE_ROOT",        rc.get("storage_root") or ""),
    ("REMOTE_JOB_TIMEOUT_H",       hours(rc["job_timeout_s"])),
    ("REMOTE_QUEUE_TIMEOUT_H",     hours(rc["queue_timeout_s"])),
    ("REMOTE_QUEUE_RETRY",         str(rc.get("queue_resubmit_attempts", 0))),
    ("REMOTE_PENDING_GRACE_MIN",   str(rc.get("pending_grace_min", 0))),
]
for k, v in rc_map:
    if v != "":
        env[k] = v
if rc.get("code_wheels"):
    env["REMOTE_CODE_WHEELS"] = ",".join(rc["code_wheels"])
if rc.get("asset_mounts"):
    env["REMOTE_ASSET_MOUNTS"] = json.dumps(rc["asset_mounts"])

# 3) OBS 前缀：SRC 末段换成本轮名（与 activate 的代码同步目标一致）
prefix = rc["obs_prefix"].rstrip("/")
env["REMOTE_OBS_PREFIX"] = prefix.rsplit("/", 1)[0] + "/" + exp_name

# 4) 机器对照：EDIT 的 TARGET_TOKENS 必须与 SRC 记录一致（E1 同规模裁决）
fatal = []
if str(le.get("TARGET_TOKENS", "")) != target_tok:
    fatal.append(f"TARGET_TOKENS: EDIT={target_tok} vs SRC 记录={le.get('TARGET_TOKENS')}"
                 " —— E1 裁决 = 同 prod4，不一致请先改 EDIT 或确认意图")
if int(rc["npu_per_job"]) < 1:
    fatal.append("REMOTE_NPU_PER_JOB < 1")

print("  ── 重建自 launch_env.json（与 SRC 逐键同值）──")
for k in sorted(src_le):
    print(f"    {k}={env[k]}")
print("  ── 重建自 remote_config.json（fleet 形态）──")
for k, _ in rc_map:
    if k in env:
        print(f"    {k}={env[k]}")
print(f"    REMOTE_OBS_PREFIX={env['REMOTE_OBS_PREFIX']}  （SRC 末段换名）")
print("  ── EDIT 提供（SRC 未落盘，无法机器对照；值 = 裁决/记录在案）──")
for k, v in (("CONFIGS_PER_ITER", cpi), ("PROXY_TARGET_TOKENS", proxy_tok),
             ("TARGET_TOKENS", target_tok), ("DISPATCH_RANDOM_ARM", dispatch_arm),
             ("SEARCH_NUM_ITERATIONS", ssi), ("ADAPTIVE_CONFIGS", adapt),
             ("ADAPTIVE_COMPACT", adaptc), ("REMOTE_ENABLED", remote_enabled)):
    print(f"    {k}={v}")
if fatal:
    print("✗ 对照失败：")
    for m in fatal:
        print(f"    {m}")
    sys.exit(1)

with open(env_file, "w") as f:
    for k, v in sorted(env.items()):
        f.write(f"export {k}={shlex.quote(v)}\n")
print(f"  重建完成：{len(env)} 个变量已写入发射环境")
PYEOF

# EDIT 值导出（在 python 重建之上）
export EXP_NAME CONFIGS_PER_ITER PROXY_TARGET_TOKENS TARGET_TOKENS \
       DISPATCH_RANDOM_ARM REMOTE_ENABLED SEARCH_NUM_ITERATIONS \
       ADAPTIVE_CONFIGS ADAPTIVE_COMPACT
source "$ENV_FILE"

echo
echo "  轮次计划: ${CONFIGS_PER_ITER}  代理预算: ${PROXY_TARGET_TOKENS}  臂预算: ${TARGET_TOKENS}"
echo "  obs: ${REMOTE_OBS_PREFIX}"

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[干跑] 引擎干跑门如下（校验守卫 + 解析后的计划）："
    LAUNCH=0 bash runs/run_experiment.sh
    echo
    echo "[干跑] 确认无误后真发射：LAUNCH=1 bash runs/launch_prod5.sh"
    exit 0
fi

# ── 真发射：preflight → 后台引擎 ──
echo
echo "  preflight..."
python3 scripts/diagnostics/preflight_launch.py --run-dir "$OUTPUT_DIR" --main-log "$LOG_FILE"

echo
echo "  发射引擎（后台，日志 ${LOG_FILE}）..."
setsid nohup bash runs/run_experiment.sh > "$LOG_FILE" 2>&1 &
echo "  监控："
echo "    tail -f ${LOG_FILE}"
echo "    bash scripts/diagnostics/prod2_watch.sh ${OUTPUT_DIR}"
echo "  停止（如需）：pkill -f run_experiment.sh"
