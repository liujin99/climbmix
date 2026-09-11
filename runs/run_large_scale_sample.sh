#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  大规模采样 (解耦的生产出口): 拿已完成 run 的最优配比 α*, 只做
#  采样 + 混合, 输出最终数据集 — 不训练不评测 (大规模验证是下游
#  项目的事, 不在本仓库范围)。
#
#  流程定位:
#    ① 搜索最优配比 (含 1.5B 验证)   = run_climbmix.sh 全流程 (已有)
#    ② 大规模采样, 输出最终数据集    = 本脚本
#    ③ 大规模验证                    = 下游项目 (不在范围)
#
#  用法: 编辑下方 EDIT 块 → ./runs/run_large_scale_sample.sh
#    RUN_DIR=result/<run> TARGET_TOKENS=20B ./runs/run_large_scale_sample.sh
#  干跑 (校验 + 打印命令, 不执行): LAUNCH=0
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

# ─── EDIT HERE (env 可临时覆盖) ────────────────────────────────────
RUN_DIR="${RUN_DIR:-result/prod2_current}"   # 完成的 run: optimal_mixture_weights.json + cluster_cache.npz
TARGET_TOKENS="${TARGET_TOKENS:-20B}"        # 大规模 STEM 预算 (唯一真源; 池 = 预算/STEM_RATIO)
STEM_RATIO="${STEM_RATIO:-0.7}"
WEIGHTS="${WEIGHTS:-}"                       # 空 = $RUN_DIR/optimal_mixture_weights.json (搜索输出的 α*)
OUT_DIR="${OUT_DIR:-}"                       # 空 = result/<run名>_final
MAX_CLIMBMIX_SHARDS="${MAX_CLIMBMIX_SHARDS:-50}"  # 通用数据分片上限; 预算大时不足会大声警告 (配比失真)
LAUNCH="${LAUNCH:-1}"                        # 0=干跑
MEASURE="${MEASURE:-1}"                      # 1=实测产出 token 并写 manifest
# ─── 本机路径 (与主 run 相同; 服务器默认通常不用改) ────────────────
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
NUM_NPU="${NUM_NPU:-8}"
# ───────────────────────────────────────────────────────────────────

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"
RUN_DIR="${RUN_DIR#"$CLIMBMIX_DIR"/}"

echo "═══ large-scale sample: ${RUN_DIR} @ ${TARGET_TOKENS} ═══"

# ── 校验 ──
[ -d "$RUN_DIR" ] || { echo "✗ RUN_DIR 不存在: ${RUN_DIR}"; exit 1; }
[ -n "$OUT_DIR" ] || OUT_DIR="result/$(basename "$RUN_DIR")_final"
WEIGHTS="${WEIGHTS:-$RUN_DIR/optimal_mixture_weights.json}"
[ -f "$WEIGHTS" ] || { echo "✗ 配比文件不存在: ${WEIGHTS} (搜索未完成? 或用 WEIGHTS= 显式给出)"; exit 1; }
CACHE="$RUN_DIR/cluster_cache.npz"
[ -f "$CACHE" ] || { echo "✗ 池缓存不存在: ${CACHE} (嵌入/聚类是复用前提, 不重跑)"; exit 1; }
if [ "${TARGET_TOKENS}" = "0" ]; then
    echo "✗ TARGET_TOKENS=0 ('all available') 无意义 — 大规模采样要一个明确预算 (e.g. 20B)。"
    exit 1
fi

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
GENERAL_DATA_DIR="$(python3 - "$RUN_DIR" <<'PY'
import json, os, sys
le = os.path.join(sys.argv[1], "launch_env.json")
if os.path.exists(le):
    print(json.load(open(le)).get("GENERAL_DATA_DIR", ""))
else:
    print(os.environ.get("GENERAL_DATA_DIR", ""))
PY
)"
[ -n "$DATA_DIR" ] || { echo "✗ DATA_DIR 无法解析 (${RUN_DIR} 无 launch_env.json 且 env 未设)"; exit 1; }
[ -n "$GENERAL_DATA_DIR" ] || { echo "✗ GENERAL_DATA_DIR 无法解析 (同上)"; exit 1; }
echo "  pool: ${DATA_DIR}"
echo "  general: ${GENERAL_DATA_DIR}"
echo "  weights: ${WEIGHTS}"
echo "  out: ${OUT_DIR}"

SHARDS="$OUT_DIR/stem_shards"
MIXED="$OUT_DIR/mixed"

SELECT_CMD=(python3 scripts/prepare_random_baseline.py
    --data-dir "$DATA_DIR" --output-dir "$SHARDS"
    --cluster-cache "$CACHE" --schema config/schema_stem.yaml
    --target-tokens "$TARGET_TOKENS" --seed 42 --num-npu "$NUM_NPU"
    --weights "$WEIGHTS")
MIX_CMD=(env NANOCHAT_REPO="$NANOCHAT_DIR" python3 scripts/mix_general_data.py
    --stem-dir "$SHARDS" --output-dir "$MIXED"
    --climbmix-dir "$GENERAL_DATA_DIR" --stem-ratio "$STEM_RATIO"
    --num-workers "$NUM_NPU" --num-npu "$NUM_NPU"
    --max-climbmix-shards "$MAX_CLIMBMIX_SHARDS")

if [ "$LAUNCH" != "1" ]; then
    echo
    echo "[dry-run] 将执行:"
    echo "  ${SELECT_CMD[*]}"
    echo "  ${MIX_CMD[*]}"
    echo "  manifest: ${OUT_DIR}/manifest.json (MEASURE=${MEASURE})"
    exit 0
fi

mkdir -p "$OUT_DIR"

# ── ① 选点 (按 α* 等比放大; .done 幂等) ──
if [ -f "$SHARDS/.done" ]; then
    echo "  选点已完成 (.done): ${SHARDS}"
else
    "${SELECT_CMD[@]}" 2>&1 | tee "$OUT_DIR/select.log"
fi

# ── ② 混入通用语料 (.done 幂等) ──
if [ -f "$MIXED/.done" ]; then
    echo "  混合已完成 (.done): ${MIXED}"
else
    "${MIX_CMD[@]}" 2>&1 | tee "$OUT_DIR/mix.log"
fi

# ── ③ 实测 + manifest (把真相写进磁盘: 产了多少、配比是否达成) ──
MEASURE="$MEASURE" python3 - "$OUT_DIR" "$RUN_DIR" "$WEIGHTS" \
    "$TARGET_TOKENS" "$STEM_RATIO" "$MAX_CLIMBMIX_SHARDS" "$MIXED" <<'PY'
import datetime, json, os, re, subprocess, sys

out_dir, run_dir, weights, target_tokens, stem_ratio, cap, mixed = sys.argv[1:8]
env = dict(os.environ)
manifest = {
    "created": datetime.datetime.now().isoformat(timespec="seconds"),
    "git_head": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        cwd=out_dir).stdout.strip() or None,
    "source_run": run_dir,
    "weights_source": weights,
    "target_tokens": target_tokens,
    "stem_ratio": float(stem_ratio),
    "out_dir": out_dir,
    "mixed_dir": mixed,
    "logs": {"select": os.path.join(out_dir, "select.log"),
             "mix": os.path.join(out_dir, "mix.log")},
    "warnings": [],
}

try:
    from climbmix.utils.token_estimate import parse_token_count
    budget = parse_token_count(target_tokens)
    manifest["target_tokens_parsed"] = budget
    expected_pool = int(budget / float(stem_ratio))
except ValueError as e:
    sys.exit(f"✗ manifest: {e}")

# mix.log 真相: STEM 文档数 / 通用分片需求 vs 实用
mix_log = manifest["logs"]["mix"]
if os.path.exists(mix_log):
    text = open(mix_log).read()
    m = re.search(r"STEM: (\d+) train shards, ([\d,]+) docs", text)
    if m:
        stem_docs = int(m.group(2).replace(",", ""))
        manifest["stem_docs"] = stem_docs
        n = re.search(r"Need ~([\d,]+) ClimbMix docs -> (\d+) shards", text)
        if n:
            needed_shards = int(n.group(2))
            manifest["general_shards_needed"] = needed_shards
            u = re.search(r"already downloaded: (\d+) files", text)
            used = int(u.group(1)) if u else needed_shards
            manifest["general_shards_used"] = used
            if needed_shards >= int(cap):
                manifest["warnings"].append(
                    f"general shards hit the cap ({needed_shards} >= {cap}): "
                    f"STEM ratio drifts above {stem_ratio} — raise "
                    f"MAX_CLIMBMIX_SHARDS (check general-pool availability first)")

if env.get("MEASURE") == "1":
    from climbmix.sampling.single_pass import measure_train_tokens
    measured = measure_train_tokens(mixed)
    manifest["measured_tokens"] = measured
    manifest["d28_steps_cap"] = measured // 1_048_576
    if measured < int(expected_pool * 0.95):
        manifest["warnings"].append(
            f"measured pool {measured:,} < 95% of expected {expected_pool:,} "
            f"(budget/{stem_ratio}) — selection shortfall or mix cap; see select.log")

with open(os.path.join(out_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"  manifest: {os.path.join(out_dir, 'manifest.json')}")
for w in manifest["warnings"]:
    print(f"  ⚠ {w}")
if "measured_tokens" in manifest:
    print(f"  measured: {manifest['measured_tokens']:,} tokens "
          f"(d28 单遍上限 ≈ {manifest['d28_steps_cap']:,} 步)")
PY

echo
echo "═══ done: 最终数据集 ${MIXED} ═══"
