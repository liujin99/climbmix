#!/usr/bin/env bash
# run-launcher verification (docs/reuse_design.md §8 one-command launchers):
#   runs/run_search.sh (3-state: fresh / warm-start / resume)
#   runs/run_arm_only.sh (auto panorama report), run_eval_only.sh
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
# pool/: 2 fake parquet shards → real pool key via pool_embedding_cache_key
# (the same function run_extend_search.sh uses), recorded in search.log.
HIST="$TMP/hist_run"; mkdir -p "$HIST" "$TMP/pool" "$TMP/general"
touch "$TMP/pool/a.parquet" "$TMP/pool/b.parquet"
python3 - "$HIST" "$TMP/pool" "$TMP/general" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from climbmix.utils.embed_cache import pool_embedding_cache_key
d, pool, general = sys.argv[1], sys.argv[2], sys.argv[3]
import numpy as np
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
# 池缓存的另一半 (climb_pipeline 命中条件 = npz + info json 同在)
json.dump({"clusters": [{"id": i, "n_docs": 3} for i in range(15)]},
          open(os.path.join(d, "cluster_info_cache.json"), "w"))
# 真实池 key (与发射时 search.log 记录同源)
shards = sorted(f for f in os.listdir(pool) if f.endswith(".parquet"))
pool_key = pool_embedding_cache_key(
    ((s, os.path.getsize(os.path.join(pool, s))) for s in shards),
    "NovaSearch/stella_en_400M_v5", 512, 0)
open(os.path.join(d, "search.log"), "w").write(
    "[Stage 1] Pool-level embedding/kmeans cache: /x/cache/embeddings/" + pool_key + "\n")
# 源 run 的发射参数 (不可变层 diff 的数据源; 语义键值 = 当前默认)
json.dump({"EVAL_BENCHMARKS": "stem", "EVAL_MAX_PER_TASK": "-1",
           "PROXY_DEPTH": "20", "STEM_RATIO": "0.7",
           "DATA_DIR": pool, "GENERAL_DATA_DIR": general},
          open(os.path.join(d, "launch_env.json"), "w"))
print(pool_key)
PY
POOL_KEY=$(grep -oE "[0-9a-f]{12}" "$HIST/search.log" | tail -1)
echo "(fixture pool key: ${POOL_KEY})"

echo "── run_search.sh (fresh / warm-start / resume) ──"
bash -n runs/run_search.sh; check "bash -n" $?

# state 1: fresh (no state, no HISTORY_RUN)
LAUNCH=0 EXP_NAME=fresh_test ./runs/run_search.sh > "$TMP/fresh.log" 2>&1
check "fresh dry-run exit 0" $? "$(tail -2 "$TMP/fresh.log")"
grep -q "→ 从零开始" "$TMP/fresh.log"
check "fresh path chosen" $?
grep -q "要基于已有 d20 实验结果做增量实验? 用 runs/run_extend_search.sh" "$TMP/fresh.log"
check "fresh path hints at warmstart entry" $?
echo "── run_extend_search.sh (reuse entry point) ──"
bash -n runs/run_extend_search.sh; check "bash -n" $?

# state 2: warm-start (no state, HISTORY_RUN) — inject runs for real.
# DATA_DIR env → new-side pool identity resolves to the fixture pool.
HISTORY_RUN="$HIST" EXP_NAME=ws_test LAUNCH=0 CONFIGS_PER_ITER="6,4" \
    DATA_DIR="$TMP/pool" \
    ./runs/run_extend_search.sh > "$TMP/ws_ok.log" 2>&1
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
[ -f result/ws_test_current/cluster_info_cache.json ]
check "cluster_info_cache.json inherited too (cache hit needs BOTH files)" $?
grep -q "✓ 池一致 (key=${POOL_KEY}, 2 个分片)" "$TMP/ws_ok.log"
check "pool identity: content-level key match" $?
grep -q "✓ 4 个测量层参数全部一致" "$TMP/ws_ok.log"
check "immutable-layer diff: source launch_env vs current defaults" $?

# slot semantics: plan "6,4" with history 6 → includes-history reading (no prepend)
grep -q "新实验: 第 1 轮 4 个 = 4 个新 d20 实验" "$TMP/ws_ok.log"
check "round breakdown in new-experiment terms" $?
grep -q "总实验数 = 10  \[底层 CONFIGS_PER_ITER=6,4\]" "$TMP/ws_ok.log"
check "totals for includes-history plan: 6 + 4 = 10" $?

# rounds-only plan "4,3" → history slot auto-prepended → underlying "6,4,3"
HISTORY_RUN="$HIST" EXP_NAME=ws_slot LAUNCH=0 CONFIGS_PER_ITER="4,3" \
    ./runs/run_extend_search.sh < /dev/null > "$TMP/ws_slot.log" 2>&1
check "rounds-only plan auto-normalized (no interaction)" $?
grep -q "新实验: 第 1 轮 4 个 + 第 2 轮 3 个 = 7 个新 d20 实验" "$TMP/ws_slot.log"
check "rounds-only plan: rounds listed in new-experiment terms" $?
grep -q "总实验数 = 13  \[底层 CONFIGS_PER_ITER=6,4,3\]" "$TMP/ws_slot.log"
check "history slot auto-prepended (6,4,3) + totals" $?
rm -rf result/ws_slot_current

# state 3: resume — same command with HISTORY_RUN still set must NOT re-inject
sleep 1  # injected_at has second granularity
HISTORY_RUN="$HIST" EXP_NAME=ws_test LAUNCH=0 CONFIGS_PER_ITER="6,4" \
    ./runs/run_extend_search.sh > "$TMP/ws_r2.log" 2>&1
check "re-run with same EXP_NAME exits 0" $?
grep -q "检测到已有 search_state — 续跑 (不重注入)" "$TMP/ws_r2.log"
check "resume path chosen (no re-injection)" $?
grep -q "底层 CONFIGS_PER_ITER=6,4" "$TMP/ws_r2.log"
check "resume re-prepends history slot (6,4)" $?

# slot1 == history count is the same includes-history reading — covered above;
# here: resume with plan written as new-exp rounds only ("4" → "6,4")
sleep 1
HISTORY_RUN="$HIST" EXP_NAME=ws_test LAUNCH=0 CONFIGS_PER_ITER="4" \
    ./runs/run_extend_search.sh > "$TMP/ws_r3.log" 2>&1
check "resume with rounds-only plan exits 0" $?
grep -q "底层 CONFIGS_PER_ITER=6,4" "$TMP/ws_r3.log"
check "resume normalizes rounds-only plan to 6,4" $?
rm -rf result/ws_test_current

# HISTORY_RUN missing → lists available history runs (scan target = repo result/)
mkdir -p result/ws_scan_fixture && cp "$HIST/search_state.json" "$HIST/search.log" "$HIST/launch_env.json" result/ws_scan_fixture/
EXP_NAME=ws_nohist LAUNCH=0 HISTORY_RUN= ./runs/run_extend_search.sh \
    > "$TMP/ws_list.log" 2>&1
[ $? -ne 0 ] && grep -q "HISTORY_RUN 必填" "$TMP/ws_list.log"
check "missing HISTORY_RUN refused with hint" $?
grep -q "result/ws_scan_fixture  (6 个 d20 实验, 池 K=15, 池 key=${POOL_KEY})  ✓ 可复用" "$TMP/ws_list.log"
check "history listing: counts + K + pool key + verdict" $?
rm -rf result/ws_scan_fixture

# history listing marks K-mismatched runs as unusable
HIST_LIST_BAD="$TMP/hist_list_bad"; mkdir -p "$HIST_LIST_BAD"
cp "$HIST/search_state.json" "$HIST_LIST_BAD/"
python3 -c "
import json
s = json.load(open('$HIST_LIST_BAD/search_state.json')); s['n_clusters'] = 14
json.dump(s, open('$HIST_LIST_BAD/search_state.json', 'w'))"
mkdir -p result/ws_scan_bad && cp "$HIST_LIST_BAD/search_state.json" result/ws_scan_bad/
EXP_NAME=ws_nohist2 LAUNCH=0 HISTORY_RUN= ./runs/run_extend_search.sh \
    > "$TMP/ws_list2.log" 2>&1
grep -q "result/ws_scan_bad  (6 个 d20 实验, 池 K=14, 池 ?)  ✗ K=14 != K_ENHANCED=15, 不可复用" "$TMP/ws_list2.log"
check "K-mismatched history run flagged unusable in listing" $?
rm -rf result/ws_scan_bad

# immutable-layer mismatch → loud warning + refusal
HIST_IMM="$TMP/hist_imm"; cp -r "$HIST" "$HIST_IMM"
python3 -c "
import json
p = '$HIST_IMM/launch_env.json'; d = json.load(open(p))
d['EVAL_MAX_PER_TASK'] = '100'   # 源 run 用了 cap, 当前默认 -1
json.dump(d, open(p, 'w'))"
HISTORY_RUN="$HIST_IMM" EXP_NAME=ws_imm LAUNCH=0 DATA_DIR="$TMP/pool" \
    ./runs/run_extend_search.sh \
    > "$TMP/ws_imm.log" 2>&1
[ $? -ne 0 ] && grep -q "EVAL_MAX_PER_TASK: 源=100" "$TMP/ws_imm.log"
check "immutable-layer mismatch refused with per-key detail" $?
grep -q "不可复用\|可比性存疑" "$TMP/ws_imm.log"
check "mismatch message explains the consequence" $?

# arm-layer mismatch → warn (NOT refuse): TARGET_* only affects the new
# run's d28 arms, never the historical d20 search-point scores (prod2
# launched with TARGET_TOKENS=1B before the 2B default landed)
HIST_ARM="$TMP/hist_arm"; cp -r "$HIST" "$HIST_ARM"
python3 -c "
import json
p = '$HIST_ARM/launch_env.json'; d = json.load(open(p))
d['TARGET_TOKENS'] = '1B'; d['TARGET_STEPS'] = '1000'
d['TARGET_BASE_CKPT'] = '/home/ma-user/work/nanochat_model_dir/base_checkpoints/d28'
d['TARGET_LR_SCALE'] = '1.0'; d['TARGET_WARMUP'] = '0.0'
d['TARGET_WARMDOWN'] = '0.9'; d['MID_DEVICE_BATCH_SIZE'] = '1'
json.dump(d, open(p, 'w'))"
HISTORY_RUN="$HIST_ARM" EXP_NAME=ws_arm LAUNCH=0 DATA_DIR="$TMP/pool" \
    ./runs/run_extend_search.sh \
    > "$TMP/ws_arm.log" 2>&1
check "arm-layer budget mismatch (1B vs 2B) passes with warning" $? "$(tail -3 "$TMP/ws_arm.log")"
grep -q "⚠ TARGET_TOKENS: 源=1B  当前=2B (臂层" "$TMP/ws_arm.log"
check "arm-layer warn names the key and both values" $?
grep -q "1 个臂层警告" "$TMP/ws_arm.log"
check "arm warns counted in the summary line" $?
if grep -q "TARGET_STEPS" "$TMP/ws_arm.log"; then FAILED=1; fi
check "derived TARGET_STEPS not compared (knob TARGET_TOKENS is)" $?
if grep -q "TARGET_BASE_CKPT" "$TMP/ws_arm.log"; then FAILED=1; fi
check "TARGET_BASE_CKPT synthesized from NANOCHAT_BASE_DIR+TARGET_DEPTH (no false positive)" $?
if grep -q "✗" "$TMP/ws_arm.log"; then FAILED=1; fi
check "no refusal lines in arm-layer-only mismatch" $?
rm -rf result/ws_arm_current
[ ! -e result/ws_imm_current/search_state.json ] || rm -rf result/ws_imm_current

# missing cluster_info_cache.json in source → loud warning
HIST_NOINFO="$TMP/hist_noinfo"; mkdir -p "$HIST_NOINFO"
cp "$HIST/search_state.json" "$HIST/cluster_cache.npz" "$HIST/balanced_profile.json" "$HIST_NOINFO/"
HISTORY_RUN="$HIST_NOINFO" EXP_NAME=ws_noinfo LAUNCH=0 \
    ./runs/run_extend_search.sh > "$TMP/ws_noinfo.log" 2>&1
grep -q "源 run 缺 cluster_info_cache.json" "$TMP/ws_noinfo.log"
check "incomplete pool cache in source: re-cluster risk warned" $?
rm -rf result/ws_noinfo_current

# pool identity: same K but DIFFERENT pool → refused (content-level key)
HIST_POOLBAD="$TMP/hist_poolbad"; mkdir -p "$HIST_POOLBAD" "$TMP/pool_other"
touch "$TMP/pool_other/x.parquet"   # different shard manifest → different key
cp -r "$HIST/." "$HIST_POOLBAD/"
HISTORY_RUN="$HIST_POOLBAD" EXP_NAME=ws_poolbad LAUNCH=0 DATA_DIR="$TMP/pool_other" \
    ./runs/run_extend_search.sh > "$TMP/ws_poolbad.log" 2>&1
[ $? -ne 0 ] && grep -q "池不一致: 源 key=" "$TMP/ws_poolbad.log"
check "same-K different-pool refused at content level" $?
grep -q "簇空间不同" "$TMP/ws_poolbad.log"
check "pool-mismatch message explains the consequence" $?
[ ! -e result/ws_poolbad_current/search_state.json ]
check "pool mismatch: no seed written" $?
rm -rf result/ws_poolbad_current

# pool identity: source has no search.log → path-level fallback
HIST_NOLOG="$TMP/hist_nolog"; mkdir -p "$HIST_NOLOG"
cp -r "$HIST/." "$HIST_NOLOG/"
rm "$HIST_NOLOG/search.log"
HISTORY_RUN="$HIST_NOLOG" EXP_NAME=ws_nolog LAUNCH=0 DATA_DIR="$TMP/pool" \
    ./runs/run_extend_search.sh > "$TMP/ws_nolog.log" 2>&1
check "no-search.log source: path-level fallback passes" $?
grep -q "仅路径级核验通过 (DATA_DIR=${TMP}/pool" "$TMP/ws_nolog.log"
check "fallback says path-level (no content key)" $?
rm -rf result/ws_nolog_current

# pool identity: no search.log AND paths differ → refused
HISTORY_RUN="$HIST_NOLOG" EXP_NAME=ws_nolog2 LAUNCH=0 DATA_DIR="$TMP/other" \
    ./runs/run_extend_search.sh > "$TMP/ws_nolog2.log" 2>&1
[ $? -ne 0 ] && grep -q "池不一致: 源 DATA_DIR=" "$TMP/ws_nolog2.log"
check "no-search.log + path mismatch refused" $?
rm -rf result/ws_nolog2_current

# stage_gate must NOT archive a warm-start seed dir as an orphan
mkdir -p result/sg_seed_current
cp "$HIST/search_state.json" result/sg_seed_current/
python3 -c "
import json
p = 'result/sg_seed_current/search_state.json'; d = json.load(open(p))
d['history_seed'] = {'n_points': 6, 'source_runs': ['hist_run']}
json.dump(d, open(p, 'w'))"
(
    CLIMBMIX_DIR="$PWD" OUTPUT_DIR="$PWD/result/sg_seed_current" EXP_NAME=sg_seed
    FP_SEARCH_PARAMS=("k=15"); FP_TARGET_PARAMS=("t=1")
    COMPLETION_MARKERS=(".done_eval_climb")
    source runs/lib/stage_gate.sh
    run_stage_gate
) > "$TMP/sg_seed.log" 2>&1
grep -q "Warm-start seed detected" "$TMP/sg_seed.log"
check "stage_gate keeps history_seed dir (not archived as orphan)" $?
[ -f result/sg_seed_current/search_state.json ]
check "seed state still in place after stage_gate" $?
[ -f result/sg_seed_current/.fingerprint_search ]
check "fresh fingerprints written into seed dir" $?
rm -rf result/sg_seed_current result/sg_seed_stale_* 2>/dev/null

# regression: a plain orphan dir (no seed) is still archived
mkdir -p result/sg_orphan_current && touch result/sg_orphan_current/junk
(
    CLIMBMIX_DIR="$PWD" OUTPUT_DIR="$PWD/result/sg_orphan_current" EXP_NAME=sg_orphan
    FP_SEARCH_PARAMS=("k=15"); FP_TARGET_PARAMS=("t=1")
    source runs/lib/stage_gate.sh
    run_stage_gate
) > "$TMP/sg_orphan.log" 2>&1
[ ! -e result/sg_orphan_current/junk ] && ls result/ | grep -q "sg_orphan_stale"
check "plain orphan dir still archived (regression)" $?
rm -rf result/sg_orphan_current result/sg_orphan_stale_* 2>/dev/null

# K mismatch refusal (profile says 14, K_ENHANCED default 15)
HIST_BAD="$TMP/hist_bad"; cp -r "$HIST" "$HIST_BAD"
python3 -c "import json; p='$HIST_BAD/balanced_profile.json'; d=json.load(open(p)); d['K_final']=14; json.dump(d, open(p,'w'))"
HISTORY_RUN="$HIST_BAD" EXP_NAME=ws_bad LAUNCH=0 ./runs/run_extend_search.sh \
    > "$TMP/ws_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "历史点不可复用" "$TMP/ws_bad.log"
check "K mismatch refused" $?
[ ! -e result/ws_bad_current/search_state.json ]
check "K mismatch: no seed written" $?

# missing source artifacts
HISTORY_RUN="$TMP/nope" EXP_NAME=ws_missing LAUNCH=0 ./runs/run_extend_search.sh \
    > "$TMP/ws_missing.log" 2>&1
[ $? -ne 0 ] && grep -q "search_state.json 不存在" "$TMP/ws_missing.log"
check "missing history run refused" $?

# slot1 == history count → read as includes-history form, no double-prepend
HISTORY_RUN="$HIST" EXP_NAME=ws_eq LAUNCH=0 CONFIGS_PER_ITER="6,4" \
    ./runs/run_extend_search.sh > "$TMP/ws_eq.log" 2>&1
check "slot1==history accepted (includes-history form)" $?
grep -q "总实验数 = 10  \[底层 CONFIGS_PER_ITER=6,4\]" "$TMP/ws_eq.log"
check "includes-history form not double-prepended" $?
rm -rf result/ws_eq_current

# run_search.sh itself has NO history concept: fresh path points at warmstart
grep -q "要基于已有 d20 实验结果做增量实验? 用 runs/run_extend_search.sh" "$TMP/fresh.log"
check "fresh path points reuse-intent at warmstart entry" $?

# REMOTE_OBS_PREFIX auto-derived from climbmix-ma config (obs_prod_base key)
mkdir -p "$TMP/fake_ma/climbmix_ma"
cat > "$TMP/fake_ma/climbmix_ma/__init__.py" <<'PY'
PY
cat > "$TMP/fake_ma/climbmix_ma/modelarts_job_api.py" <<'PY'
import json, os
def load_ma_config(path=None):
    p = os.environ.get("CLIMBMIX_MA_CONFIG")
    return json.load(open(p)) if p else {}
PY
echo '{"auth": {"x_auth_token": "fake"}, "obs_prod_base": "obs://bucket-test/user/climbmix"}' \
    > "$TMP/ma_config.json"
PYTHONPATH="$TMP/fake_ma" CLIMBMIX_MA_CONFIG="$TMP/ma_config.json" \
    EXP_NAME=ws_obs LAUNCH=0 ./runs/run_search.sh > "$TMP/ws_obs.log" 2>&1
check "obs_prod_base fallback dry-run exit 0" $?
grep -q "obs:    obs://bucket-test/user/climbmix/ws_obs" "$TMP/ws_obs.log"
check "obs prefix auto-derived as <obs_prod_base>/<EXP_NAME>" $?
rm -rf result/ws_obs_current

echo "── run_arm_only.sh ──"
bash -n runs/run_arm_only.sh; check "bash -n" $?

RUN="$TMP/arm_run"; mkdir -p "$RUN"
cp "$HIST/cluster_cache.npz" "$RUN/"
# TARGET_TOKENS 与壳默认 (2B) 一致 → 不触发步数重派生 (重派生需要 d28 ckpt)
python3 -c "
import json
json.dump({'DATA_DIR': '$TMP/pool', 'TARGET_TOKENS': '2B'}, open('$RUN/launch_env.json', 'w'))"
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

# auto-report tail: exec gone (tail reachable) + all-arms panorama preview
grep -q '^exec python3 scripts/dispatch_target_arm.py' runs/run_arm_only.sh
[ $? -ne 0 ]
check "no exec into dispatch (auto report reachable)" $?
grep -q "训完自动出全景对比报告" "$TMP/arm_ok.log"
check "dry-run previews auto panorama report" $?
[ ! -f runs/run_report_only.sh ]
check "run_report_only.sh removed (report = arm tail + cp4_report.py)" $?

# helper behavior (runs/lib/auto_report.sh): arms -> cp4 panorama; none -> skip
source runs/lib/auto_report.sh
TAIL="$TMP/tail_run"; mkdir -p "$TAIL"
printf 'arc_easy,0.25,0.2360,2.41\narc_challenge,0.21,0.1960,2.55\nSTEM,,0.2160,2.48\n' > "$TAIL/eval_climb.csv"
printf 'arc_easy,0.23,0.2160,2.47\narc_challenge,0.19,0.1760,2.63\nSTEM,,0.1960,2.55\n' > "$TAIL/eval_random.csv"
printf 'arc_easy,0.26,0.2460,2.38\narc_challenge,0.22,0.2060,2.51\nSTEM,,0.2260,2.44\n' > "$TAIL/eval_fixratio_v1.csv"
auto_cp4_report "$TAIL" > "$TMP/tail_ok.log" 2>&1
check "auto_cp4_report all-arms exit 0" $? "$(tail -2 "$TMP/tail_ok.log")"
grep -q "3 arms; ref = random" "$TMP/tail_ok.log"
check "panorama discovers all 3 arms (ref=random default)" $?
grep -q "ranking:" "$TMP/tail_ok.log"
check "ranking line present" $?
grep -q "fixratio_v1" "$TMP/tail_ok.log"
check "custom arm ranked in headline" $?
auto_cp4_report "$TMP/empty_dir" > "$TMP/tail_skip.log" 2>&1
check "no arm CSVs -> explicit skip, exit 0" $?
grep -q "跳过自动对比报告" "$TMP/tail_skip.log"
check "skip is explicit" $?

# cp4_report.py direct: --ref override + machine-readable json
python3 scripts/diagnostics/cp4_report.py "$TAIL" --ref climb --json "$TMP/cp4.json" > "$TMP/cp4_ref.log" 2>&1
check "cp4 --ref override exit 0" $? "$(tail -2 "$TMP/cp4_ref.log")"
grep -q "ref = climb" "$TMP/cp4_ref.log"
check "ref override applied" $?
python3 -c "
import json; v = json.load(open('$TMP/cp4.json'))
assert v['ref'] == 'climb' and set(v['arms']) == {'climb', 'random', 'fixratio_v1'}
assert v['ranking'][0] == 'fixratio_v1' and 'per_benchmark' in v"
check "json verdict: all arms + ranking + per_benchmark" $?

echo "── run_eval_only.sh ──"
bash -n runs/run_eval_only.sh; check "bash -n" $?
RUN_DIR="$TMP/nope" ./runs/run_eval_only.sh > "$TMP/anchor_bad.log" 2>&1
[ $? -ne 0 ] && grep -q "launch_env.json 不存在" "$TMP/anchor_bad.log"
check "non-run dir refused" $?

echo
if [ "$FAILED" -eq 0 ]; then
    echo "✓ all run-launcher checks passed"
else
    echo "✗ run-launcher checks FAILED"
fi
exit "$FAILED"
