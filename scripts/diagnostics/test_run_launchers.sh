#!/usr/bin/env bash
# run-launcher verification (docs/reuse_design.md §8 one-command launchers):
#   runs/run_search_arms.sh (3-state: fresh / warm-start / resume)
#   runs/run_arm_only.sh, run_eval_only.sh, run_report_only.sh
# — syntax, dry-runs, state-driven dispatch, the K-consistency refusal,
# the slot-accounting warning, and the stage scripts' guards, all against
# synthetic run dirs (no NPU, no real pool; heavy steps skipped via .done).
#
# Run:  bash scripts/diagnostics/test_run_launchers.sh   (from repo root)
# Exit 0 = all checks pass.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

FAILED=0
check() {  # name, cond(0=ok), detail
    if [ "$2" -eq 0 ]; then
        echo "  [PASS] $1"
    else
        echo "  [FAIL] $1${3:+ — $3}"
        FAILED=1
    fi
}

TMP="$(mktemp -d /tmp/run_launchers_XXXXXX)"
trap 'rm -rf "$TMP"' EXIT

# ── synthetic history run (K=15, 6 points — inject only needs shape) ──
HIST="$TMP/hist_run"; mkdir -p "$HIST"
python3 - "$HIST" <<'PY'
import json, sys, os
import numpy as np
d = sys.argv[1]
rng = np.random.default_rng(42)
n, K = 6, 15
acc = [{"arc_easy": round(0.2 + 0.05 * rng.random(), 4),
        "arc_challenge": round(0.18 + 0.04 * rng.random(), 4)} for _ in range(n)]
nll = [{k: round(2.2 + 0.4 * rng.random(), 4) for k in a} for a in acc]
json.dump({
    "last_completed_iter": 2, "n_clusters": K,
    "accumulated_scores": [0.5] * n,
    "accumulated_configs": [
        {"weights": [round(float(x), 6) for x in rng.dirichlet(np.ones(K))],
         "config_id": i} for i in range(n)],
    "accumulated_per_benchmark": [{"acc": a, "nll": b} for a, b in zip(acc, nll)],
    "predictor_eval": [], "online_eval": [], "pruning_history": [],
    "pending": None, "realized_configs_per_iter": [4, 2], "last_c_eff": 6,
}, open(os.path.join(d, "search_state.json"), "w"))
np.savez(os.path.join(d, "cluster_cache.npz"),
         final_labels=np.repeat(np.arange(15), 3))
json.dump({"K_final": 15, "max_share": 0.08},
          open(os.path.join(d, "balanced_profile.json"), "w"))
PY

echo "── run_search_arms.sh (fresh / warm-start / resume) ──"
bash -n runs/run_search_arms.sh; check "bash -n" $?

# state 1: fresh (no state, no HISTORY_RUN)
LAUNCH=0 EXP_NAME=fresh_test ./runs/run_search_arms.sh > "$TMP/fresh.log" 2>&1
check "fresh dry-run exit 0" $? "$(tail -2 "$TMP/fresh.log")"
grep -q "从零开始 (HISTORY_RUN 为空)" "$TMP/fresh.log"
check "fresh path chosen" $?

# state 2: warm-start (no state, HISTORY_RUN) — inject runs for real
HISTORY_RUN="$HIST" EXP_NAME=ws_test LAUNCH=0 CONFIGS_PER_ITER="6,4" \
    ./runs/run_search_arms.sh > "$TMP/ws_ok.log" 2>&1
check "warm-start dry-run exit 0" $? "$(tail -3 "$TMP/ws_ok.log")"
[ -f result/ws_test_current/search_state.json ]
check "seed injected into result/ws_test_current" $?
python3 - <<'PY'
import json, sys
s = json.load(open("result/ws_test_current/search_state.json"))
hs = s.get("history_seed") or {}
ok = (s["last_completed_iter"] == 1
      and s["realized_configs_per_iter"] == [6]
      and len(s["accumulated_configs"]) == 6
      and hs.get("n_points") == 6 and hs.get("pool_k") == 15
      and hs.get("source_runs") == ["hist_run"])
sys.exit(0 if ok else 1)
PY
check "seed shape: iter1=6 points, provenance stamped" $?
[ -f result/ws_test_current/cluster_cache.npz ]
check "pool cache inherited (copied, not regenerated)" $?

# state 3: resume — same command with HISTORY_RUN still set must NOT re-inject
sleep 1  # injected_at has second granularity
HISTORY_RUN="$HIST" EXP_NAME=ws_test LAUNCH=0 CONFIGS_PER_ITER="6,4" \
    ./runs/run_search_arms.sh > "$TMP/ws_r2.log" 2>&1
check "re-run with same EXP_NAME exits 0" $?
grep -q "检测到已有 search_state — 续跑" "$TMP/ws_r2.log"
check "resume path chosen (no re-injection)" $?
rm -rf result/ws_test_current

# K mismatch refusal (profile says 14, K_ENHANCED default 15)
HIST_BAD="$TMP/hist_bad"; cp -r "$HIST" "$HIST_BAD"
python3 -c "import json; p='$HIST_BAD/balanced_profile.json'; d=json.load(open(p)); d['K_final']=14; json.dump(d, open(p,'w'))"
HISTORY_RUN="$HIST_BAD" EXP_NAME=ws_bad LAUNCH=0 ./runs/run_search_arms.sh \
    > "$TMP/ws_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "历史点不可复用" "$TMP/ws_bad.log"
check "K mismatch refused" $?
[ ! -e result/ws_bad_current/search_state.json ]
check "K mismatch: no seed written" $?

# missing source artifacts
HISTORY_RUN="$TMP/nope" EXP_NAME=ws_missing LAUNCH=0 ./runs/run_search_arms.sh \
    > "$TMP/ws_missing.log" 2>&1
[ $? -ne 0 ] && grep -q "search_state.json 不存在" "$TMP/ws_missing.log"
check "missing history run refused" $?

# slot accounting warning (history 6, slot1 4)
HISTORY_RUN="$HIST" EXP_NAME=ws_slot LAUNCH=0 CONFIGS_PER_ITER="4,3" \
    ./runs/run_search_arms.sh < /dev/null > "$TMP/ws_slot.log" 2>&1
check "slot-mismatch warning auto-continues (non-tty)" $?
grep -q "第 1 槽被历史点整体替换" "$TMP/ws_slot.log"
check "slot warning names the semantics" $?
rm -rf result/ws_slot_current

echo "── run_arm_only.sh ──"
bash -n runs/run_arm_only.sh; check "bash -n" $?

RUN="$TMP/arm_run"; mkdir -p "$RUN"
cp "$HIST/cluster_cache.npz" "$RUN/"
python3 -c "
import json
json.dump({'DATA_DIR': '$TMP/pool'}, open('$RUN/launch_env.json', 'w'))"
mkdir -p "$TMP/pool"

RUN_DIR="$RUN" ARM_NAME="bad name!" WEIGHTS="0.25,0.25,0.25" LAUNCH=0 \
    ./runs/run_arm_only.sh > "$TMP/arm_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "必须匹配" "$TMP/arm_bad.log"
check "unsafe arm name refused" $?

# custom-ratio path: steps skipped via .done, dispatch line printed
mkdir -p "$RUN/fixratio_shards" "$RUN/fixratio_mixed"
touch "$RUN/fixratio_shards/.done" "$RUN/fixratio_mixed/.done"
RUN_DIR="$RUN" ARM_NAME=fixratio WEIGHTS="0.25,0.25,0.25,0.25" LAUNCH=0 \
    ./runs/run_arm_only.sh > "$TMP/arm_ok.log" 2>&1
check "custom-ratio dry-run exit 0 (.done skips prep+mix)" $? "$(tail -3 "$TMP/arm_ok.log")"
grep -q "dispatch_target_arm.py --arm fixratio --output-dir" "$TMP/arm_ok.log"
check "dispatch command printed with --retry-failed default" $?
grep -q -- "--retry-failed" "$TMP/arm_ok.log"
check "retry-failed in the dispatch line" $?

# re-dispatch path: WEIGHTS empty = no prep, warns when mixed data missing
RUN_DIR="$RUN" ARM_NAME=fresharm WEIGHTS="" LAUNCH=0 \
    ./runs/run_arm_only.sh > "$TMP/arm_re.log" 2>&1
check "re-dispatch dry-run exit 0" $?
grep -q "直接重发已有臂" "$TMP/arm_re.log"
check "re-dispatch path chosen (WEIGHTS empty)" $?
grep -q "⚠" "$TMP/arm_re.log"
check "missing-mixed-data warning shown" $?

echo "── run_eval_only.sh ──"
bash -n runs/run_eval_only.sh; check "bash -n" $?
RUN_DIR="$TMP/nope" ./runs/run_eval_only.sh > "$TMP/anchor_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "launch_env.json 不存在" "$TMP/anchor_bad.log"
check "non-run dir refused" $?

echo "── run_report_only.sh ──"
bash -n runs/run_report_only.sh; check "bash -n" $?

# rescore-only run (state present, no eval CSVs)
cp -r "$HIST" "$TMP/rep_run"
RUN_DIR="$TMP/rep_run" ./runs/run_report_only.sh > "$TMP/rep1.log" 2>&1
check "rescore-only run exit 0" $? "$(tail -2 "$TMP/rep1.log")"
[ -f "$TMP/rep_run/search_state.json.rescored.json" ]
check "rescore sidecar written" $?
grep -q "跳过 CP4" "$TMP/rep1.log"
check "cp4 skip is explicit" $?
rm -f "$TMP/rep_run/search_state.json.rescored.json"

# nothing to report
RUN_DIR="$TMP/nope" ./runs/run_report_only.sh > "$TMP/rep_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "没有可报告的东西" "$TMP/rep_bad.log"
check "empty run dir refused" $?

# cp4 missing-arm guard: state present but wrong arm names -> cp4 skipped, not crash
RUN_DIR="$HIST" ARM_A=fixratio_v1 ARM_B=climb ./runs/run_report_only.sh \
    > "$TMP/rep2.log" 2>&1
check "missing cp4 arms -> skip + exit 0" $?
grep -q "跳过 CP4" "$TMP/rep2.log"
check "cp4 skip names the missing arms" $?

echo
if [ "$FAILED" -eq 0 ]; then
    echo "✓ all run-launcher checks passed"
else
    echo "✗ run-launcher checks FAILED"
fi
exit "$FAILED"
