#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  smoke_round.sh — D19 + 新 eval 协议的端到端彩排（最小真数据轮）
#
#  背景（runs/ 重构时预登记，TODO 2026-09-20）: speedrun 删除后,"改码后
#  烧卡前预检"由下一轮改 P1 代码时的最小真数据轮承担。D19 改了终选机制 +
#  D17/D18 改了 eval 栈, 两者从未在真数据上端到端跑过; prod5 要烧 ~16h
#  搜索 + ~20h 臂 —— 这是投前最便宜的保险。
#
#  做什么（服务器, D1 激活之后; 全本地零远端提交, 无排队暴露）:
#    1. 聚类缓存继承（cluster_cache 双文件 + balanced_profile 从
#       SMOKE_CACHE_SRC 抄入 —— Steps 0-2 纯缓存命中, 簇与源轮逐位一致）
#    2. 发射 Step 1-3: 12 个 d20 实验（8+4, 凑够 10 触发 A2 全量重拟合）
#       × 50M token 代理预算 × 100 题/任务子采样评测 × NPU_PER_EXP=1
#       （8 卡 8 路并行, 2 波）; REMOTE_ENABLED=0 + DISPATCH_RANDOM_ARM=0
#    3. 看到搜索完成（run_climb "Done!" / 引擎进入 Step 4 横幅）即停
#       —— smoke 范围 = Step 1-3, 臂准备与派发不进彩排
#    4. 自动验证清单（舰队/终选模式/topk JSON/claim 渲染/权重和/落盘）
#    5. 清理: 验证通过后默认整目录删除（~30GB: 12 个 d20 ckpt + mixture
#       + parquet; 用户裁决 2026-09-21 不长期保存测试重件）; 验证报告落
#       result/${名}_verification.txt（轻量, 保留）
#
#  预计 ~30-45 min。前置: D1 激活完成（主树 = 新代码）、8 NPU 空闲、
#  磁盘余量 ≥50G。预计终选路径声明: tiny 预算下分数噪声主导, 可能走
#  no-signal 守卫路径而非 claim 比较路径 —— 两条都是真路径, 验证目标是
#  "管线跑通 + 落盘齐全", 不预设哪个路径命中（claim/A2 正常路径另有
#  test_final_selection_p1.py 真 LightGBM 单测 + predictor_audit.py 对
#  prod4 真舰队的重放覆盖）。
#
#  用法:  bash scripts/diagnostics/smoke_round.sh
#    SMOKE_NAME=smoke5                result 目录名（默认）
#    SMOKE_CACHE_SRC=result/prod4_current   聚类缓存继承源
#    SMOKE_KEEP=1                     保留产物排障（默认验证通过后删除）
#    SMOKE_LAUNCH=0                   干跑（env + 缓存检查, 不发射; 本地可测）
#    SMOKE_TIMEOUT_H=2                看门狗时限
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SMOKE_NAME="${SMOKE_NAME:-smoke5}"
SMOKE_CACHE_SRC="${SMOKE_CACHE_SRC:-result/prod4_current}"
SMOKE_KEEP="${SMOKE_KEEP:-0}"
SMOKE_LAUNCH="${SMOKE_LAUNCH:-1}"
SMOKE_TIMEOUT_H="${SMOKE_TIMEOUT_H:-2}"

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$CLIMBMIX_DIR"
OUTPUT_DIR="$CLIMBMIX_DIR/result/${SMOKE_NAME}_current"
ENGINE_LOG="$CLIMBMIX_DIR/result/${SMOKE_NAME}_engine.log"
VERIFY_LOG="$CLIMBMIX_DIR/result/${SMOKE_NAME}_verification.txt"

echo "═══ smoke 彩排: D19 + 新协议端到端（最小真数据轮）═══"
echo "  run:      ${SMOKE_NAME}  →  ${OUTPUT_DIR}"
echo "  计划:     12 实验 (8+4) × 50M tokens × 100 题/任务, 全本地 8 路"

# ── 前置检查 ────────────────────────────────────────────────────────
if pgrep -f "run_experiment.sh" >/dev/null 2>&1; then
    echo "✗ 已有 run_experiment.sh 在跑 —— 彩排会互相干扰, 先处理它"
    exit 1
fi
_avail=$(df -BG --output=avail "$CLIMBMIX_DIR/result" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)
echo "  磁盘:    ${_avail}G 可用（需 ≥50G）"
if [ "${_avail:-0}" -lt 50 ]; then
    echo "✗ 磁盘余量不足 50G —— 12 个 d20 实验的 ckpt+mixture 约 30G"
    exit 1
fi

if [ -f "$OUTPUT_DIR/search_state.json" ]; then
    echo "  → 检测到已有 search_state —— 续跑彩排（同命令同 env = 同指纹）"
elif [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
    echo "✗ ${OUTPUT_DIR} 非空且无 search_state —— 请移走或换 SMOKE_NAME"
    exit 1
fi

# ── 聚类缓存继承（全新跑 ≠ 重新聚类; 簇与源轮逐位一致）──────────────
echo "  缓存继承: 源 = ${SMOKE_CACHE_SRC}"
if [ "$SMOKE_LAUNCH" != "1" ]; then
    for f in cluster_cache.npz cluster_info_cache.json balanced_profile.json; do
        if [ -f "$SMOKE_CACHE_SRC/$f" ]; then
            echo "    [dry-run] 将复制 ${f}"
        else
            echo "    [dry-run] ⚠ 源缺 ${f}"
        fi
    done
else
    mkdir -p "$OUTPUT_DIR"
    for f in cluster_cache.npz cluster_info_cache.json balanced_profile.json; do
        if [ ! -f "$OUTPUT_DIR/$f" ] && [ -f "$SMOKE_CACHE_SRC/$f" ]; then
            cp "$SMOKE_CACHE_SRC/$f" "$OUTPUT_DIR/"
            echo "    继承 ${f}"
        elif [ -f "$OUTPUT_DIR/$f" ]; then
            echo "    已在 ${f}"
        else
            echo "    ⚠ 源缺 ${f} —— Step 0-2 将退化（嵌入缓存仍在, 但会重聚类）"
        fi
    done
fi

# ── 发射 env（全部 env 覆盖, 引擎 EDIT 块默认值不参与）──────────────
export EXP_NAME="$SMOKE_NAME"
export CONFIGS_PER_ITER="8,4"
export PROXY_TARGET_TOKENS="50M"
export EVAL_MAX_PER_TASK=100
export NPU_PER_EXP=1
export ADAPTIVE_CONFIGS=0
export ADAPTIVE_COMPACT=0
export REMOTE_ENABLED=0            # 零远端提交 —— 无排队暴露
export DISPATCH_RANDOM_ARM=0       # 搜索期不并行预发 random 臂（那是远端派发）
export TARGET_TOKENS=200M          # 看门狗若滞后进 Step 4, 其工作量也被钳小
export LAUNCH="$SMOKE_LAUNCH"

if [ "$SMOKE_LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] env 就绪; 引擎干跑门如下（核对 轮次计划 / k=1）:"
    bash runs/run_experiment.sh
    echo
    echo "[dry-run] 正式彩排: SMOKE_LAUNCH=1 bash scripts/diagnostics/smoke_round.sh"
    exit 0
fi

# ── 发射 + 看门狗（搜索完成即停; smoke 范围 = Step 1-3）──────────────
echo
echo "  发射引擎（后台, 日志 ${ENGINE_LOG}）..."
mkdir -p "$CLIMBMIX_DIR/result"
setsid bash runs/run_experiment.sh >"$ENGINE_LOG" 2>&1 &
ENGINE_PID=$!
ENGINE_PGID="$(ps -o pgid= -p "$ENGINE_PID" 2>/dev/null | tr -d '[:space:]' || true)"

_search_done=0
_timed_out=0
_deadline=$(( $(date +%s) + SMOKE_TIMEOUT_H * 3600 ))
echo "  看门狗: 每 5s 轮询, 超时 ${SMOKE_TIMEOUT_H}h; 完成标志 = 'Done! Results in' 或 Step 4 横幅"
while :; do
    if grep -q "Done! Results in\|===== Step 4" "$ENGINE_LOG" 2>/dev/null; then
        _search_done=1
        break
    fi
    if ! kill -0 "$ENGINE_PID" 2>/dev/null; then
        break
    fi
    if [ "$(date +%s)" -gt "$_deadline" ]; then
        _timed_out=1
        break
    fi
    sleep 5
done

if [ "$_timed_out" = "1" ]; then
    echo "  ⚠ 看门狗超时（${SMOKE_TIMEOUT_H}h）—— 终止引擎, 按现状验证"
fi
if kill -0 "$ENGINE_PID" 2>/dev/null; then
    echo "  搜索阶段结束 —— 终止引擎树（smoke 不进臂准备/派发）..."
    if [ -n "$ENGINE_PGID" ]; then
        kill -TERM -- "-$ENGINE_PGID" 2>/dev/null || kill -TERM "$ENGINE_PID" 2>/dev/null || true
    else
        kill -TERM "$ENGINE_PID" 2>/dev/null || true
    fi
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        kill -0 "$ENGINE_PID" 2>/dev/null || break
        sleep 5
    done
    kill -KILL -- "-$ENGINE_PGID" 2>/dev/null || kill -KILL "$ENGINE_PID" 2>/dev/null || true
    pkill -KILL -f "run_experiment.sh" 2>/dev/null || true
fi

# ── 验证清单 ────────────────────────────────────────────────────────
echo
echo "═══ 验证清单 ═══"
set +e
python3 - "$OUTPUT_DIR" <<'PYEOF' 2>&1 | tee "$VERIFY_LOG"
import json
import os
import sys

try:
    import numpy as np
except ImportError:
    np = None

out = sys.argv[1]


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


fails = []


def check(name, cond, extra=""):
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {extra}" if extra and not cond else ""))
    if not cond:
        fails.append(name)


def isfinite(v):
    return v is not None and np is not None and bool(np.isfinite(v))


# 1. 舰队完整性（12 点, 8+4, 全部有限分数）
st = load(os.path.join(out, "search_state.json"))
check("fleet: search_state.json 可读", st is not None)
if st:
    cfgs = st.get("accumulated_configs") or []
    scores = st.get("accumulated_scores") or []
    pb = st.get("accumulated_per_benchmark") or []
    check("fleet: 12 个配置", len(cfgs) == 12, f"{len(cfgs)}")
    check("fleet: realized = [8, 4]",
          st.get("realized_configs_per_iter") == [8, 4],
          str(st.get("realized_configs_per_iter")))
    n_fin = sum(1 for s in scores if isfinite(s))
    check("fleet: 12 个有限分数", n_fin == 12, f"{n_fin}")
    ok_pb = 0
    for d in pb:
        acc = (d or {}).get("acc") or {}
        vals = [v for v in acc.values() if v is not None and np.isfinite(v)]
        if len(vals) >= 5 and all(0.0 <= v <= 1.0 for v in vals):
            ok_pb += 1
    check("fleet: 12 点逐任务 acc 有限且在 [0,1]", ok_pb == 12, f"{ok_pb}")

# 2. 终选模式（search.log + report.md 双源一致）
mode = None
log_txt = ""
try:
    with open(os.path.join(out, "search.log"), errors="replace") as f:
        log_txt = f.read()
except Exception:
    pass
for line in log_txt.splitlines():
    if "Selection mode:" in line:
        mode = line.split("Selection mode:")[1].strip()
KNOWN = ("predictor_design_space_claimed", "best_measured_no_claim",
         "no_signal_best_measured", "no_predictor_best_measured")
check("log: Selection mode 落盘且为已知值", mode in KNOWN, str(mode))
md = ""
try:
    with open(os.path.join(out, "report.md"), errors="replace") as f:
        md = f.read()
except Exception:
    pass
check("report: Final Selection 节存在", "## Final Selection" in md)
check("report: Mode 行与 log 一致",
      mode is not None and f"Mode: `{mode}`" in md)
if mode in KNOWN[:2]:
    check("log: A2 全量重拟合已执行（n=12 ≥ 10 正常路径）",
          "Refit on full" in log_txt)
    check("log: claim 判决行存在",
          "NO-CLAIM" in log_txt or "CLAIMED the final slot" in log_txt)
else:
    print("[INFO] 守卫路径触发 —— claim/A2 正常路径由单测 + 审计重放覆盖"
          "（informational, 不判失败）")

# 3. top-k 导出（D19 A3）
topk = load(os.path.join(out, "topk_mixture_candidates.json"))
check("topk: topk_mixture_candidates.json 存在", topk is not None)
if topk and st:
    cands = topk.get("candidates") or []
    check("topk: 1-3 个候选", 1 <= len(cands) <= 3, f"{len(cands)}")
    score_by_id = {c.get("config_id"): s
                   for c, s in zip(st.get("accumulated_configs") or [],
                                   st.get("accumulated_scores") or [])}
    ok = True
    for c in cands:
        w = c.get("weights")
        if not isinstance(w, dict) or abs(sum(w.values()) - 1.0) > 0.01:
            ok = False
        cid = c.get("config_id")
        if cid not in score_by_id:
            ok = False
        elif abs((c.get("score") or 1e9) - score_by_id[cid]) > 1e-6:
            ok = False
    check("topk: 权重按簇标签键控且和≈1, id/score 与 state 对账", ok)
    sc = [c.get("score") or 0.0 for c in cands]
    check("topk: 分数降序", all(sc[i] >= sc[i + 1] - 1e-9
                                for i in range(len(sc) - 1)))

# 4. 终选权重 + Step 1-3 完成门
opt = load(os.path.join(out, "optimal_mixture_weights.json"))
check("optimal: optimal_mixture_weights.json 存在且和≈1",
      opt is not None and abs(sum(opt.values()) - 1.0) <= 0.01)
check("gate: sampled_dataset.parquet 存在（Step 1-3 完成门）",
      os.path.exists(os.path.join(out, "sampled_dataset.parquet")))

print()
if fails:
    print(f"SMOKE FAILED ({len(fails)}): {fails}")
    sys.exit(1)
print("SMOKE PASSED — D19 + 新协议端到端验证绿")
PYEOF
_verify_rc=${PIPESTATUS[0]}
set -e
echo

if [ "$_verify_rc" != "0" ]; then
    echo "✗ smoke 验证未过 —— 产物保留排障: ${OUTPUT_DIR}"
    echo "  引擎日志: ${ENGINE_LOG}（尾部 40 行如下）"
    tail -40 "$ENGINE_LOG" 2>/dev/null || true
    echo "  修复后同命令重跑 = search_state 续跑（已完成实验不重训）"
    exit 1
fi

if [ "$SMOKE_KEEP" = "1" ]; then
    echo "✓ SMOKE PASSED —— SMOKE_KEEP=1, 产物保留: ${OUTPUT_DIR}（排障后请手动删, ~30G）"
    exit 0
fi

echo "✓ SMOKE PASSED —— 清理测试重件（用户裁决: 不长期保存）..."
du -sh "$OUTPUT_DIR" 2>/dev/null || true
rm -rf "$OUTPUT_DIR"
echo "  已删 ${OUTPUT_DIR}; 保留: ${VERIFY_LOG} + ${ENGINE_LOG}（轻量）"
echo "═══ 彩排通过, prod5 可以发射（runbook 下一步）═══"
