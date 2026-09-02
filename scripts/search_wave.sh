#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  search_wave.sh — speedrun 配置网格的 k4 并发 wave dispatch (M5 搜索)
#
#  从 speedrun 归档 (SPEEDRUN_DIR/exps/*/meta.json) 读取全部配置的
#  weights, 逐个 dispatch 成隔离的 k4 远程实验 (exp 910+), 最多
#  MAX_PARALLEL 个 dispatch 进程并发 (每个进程 1 个 4 卡 job)。
#
#  eval 语义 (已定): 搜索阶段 cap 100 (--eval-max-per-task 100, 固定
#  seed-1337 子集 -> 所有实验同一把尺, 可比); 生产/验收阶段用全集
#  (不传该 flag)。四个 speedrun flag 全部由本脚本内置, 无粘贴输入:
#    --proxy-num-iterations 50 / --proxy-target-tokens 10M /
#    --eval-max-per-task 100 / --general-data-dir
#
#  并发安全:
#    - 每 exp 的 mixture_data 在独立 exp 目录准备+上传, 互不干扰
#    - 共享只读: speedrun 元数据缓存 / 远程配置
#    - worker assets 每次无条件重传 (两个小文件, OBS PUT 原子, 内容
#      相同); 本地 staging 为原子 rename, 并发安全
#    - ClimbMix 通用分片已下载过则跳过; 首个 dispatch 单独启动并等
#      其 job 提交后再放行其余 (错峰, 兜底新分片下载竞态)
#
#  用法 (服务器上, 阻塞 ~1.5-2h, 建议 tmux):
#    bash scripts/search_wave.sh dry-run    # 打印全部命令, 不执行
#    bash scripts/search_wave.sh run        # 错峰启动 + 并发池 + 汇总
#    bash scripts/search_wave.sh summary    # 只打印汇总 (可随时重跑)
#
#  环境变量 (默认服务器布局):
#    SRC_CFG=/tmp/remote_smoke.json         源 RemoteConfig (npu_per_job=1)
#    SPEEDRUN_DIR=result/speedrun_20260828_222129
#    START_EXP_ID=910   MAX_PARALLEL=6
#    OUT_DIR=/tmp/remote_validation_k4
#    FORCE_WAVE=1       覆盖已有结果的 exp id (默认拒绝, 防误清)
#    (DATA_DIR CLUSTER_CACHE_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR PHASE1_CKPT
#     GENERAL_DATA_DIR 同 k4_smoke self 模式; CLUSTER_CACHE_DIR 默认
#     取 SPEEDRUN_DIR)
#
#  本脚本不进指纹。日志: /tmp/wave_logs/run_<exp_id>.log
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_CFG="${SRC_CFG:-/tmp/remote_smoke.json}"
WAVE_CFG="/tmp/remote_wave_k4.json"
OUT_DIR="${OUT_DIR:-/tmp/remote_validation_k4}"
SPEEDRUN_DIR="${SPEEDRUN_DIR:-result/speedrun_20260828_222129}"
START_EXP_ID="${START_EXP_ID:-910}"
MAX_PARALLEL="${MAX_PARALLEL:-6}"
WAVE_NAME="${WAVE_NAME:-searchwave-k4}"

mode="${1:-dry-run}"
die() { echo "[wave] ERROR: $*" >&2; exit 1; }

[ -f "$SRC_CFG" ] || die "remote config not found: $SRC_CFG (set SRC_CFG=...)"
grep -q "parse_token_count" "$REPO/scripts/dispatch_remote.py" \
  || die "dispatch_remote.py predates 29295e8 (10M suffix rejected) — git pull first"

# ── collect configs + build commands ────────────────────────────────────
# The collector writes each run_<exp_id>.sh itself (no TAB parsing between
# processes — a malformed meta once dispatched a bare word as a command) and
# prints ONLY the exp ids, one per line, to stdout. The last stderr line is
# the manifest of archive paths in dispatch order.
CMD_DIR="${CMD_DIR:-/tmp/wave_cmds}"
mkdir -p "$CMD_DIR"
rm -f "$CMD_DIR"/run_*.sh
IDS_FILE="$(mktemp)"
if ! python3 - "$REPO" "$SRC_CFG" "$WAVE_CFG" "$SPEEDRUN_DIR" \
       "$START_EXP_ID" "$MAX_PARALLEL" "$OUT_DIR" "$WAVE_NAME" "$mode" \
       "$CMD_DIR" > "$IDS_FILE" <<'PY'
import glob, json, os, shlex, sys

repo, src_cfg, wave_cfg, speedrun_dir, start_id, max_par, out_dir, name, \
    mode, cmd_dir = sys.argv[1:11]
start_id, max_par = int(start_id), int(max_par)

cfg = json.load(open(src_cfg))
cfg["npu_per_job"] = 4
cfg["max_concurrent_jobs"] = max_par
json.dump(cfg, open(wave_cfg, "w"), indent=2)

# tolerate exps/*/meta.json and exp*/meta.json archive layouts
metas = []
for pat in (os.path.join(speedrun_dir, "exps", "*", "meta.json"),
            os.path.join(speedrun_dir, "exp*", "meta.json")):
    metas = sorted(glob.glob(os.path.join(repo, pat)))
    if metas:
        break
if not metas:
    print(f"[wave] ERROR: no meta.json under {repo}/{speedrun_dir} "
          f"(set SPEEDRUN_DIR=...)", file=sys.stderr)
    sys.exit(1)

exp_ids = [start_id + i for i in range(len(metas))]
# summary is read-only — existing results are its input, not a clash
if mode != "summary":
    clash = [i for i in exp_ids
             if os.path.isfile(os.path.join(out_dir, f"exp_{i:04d}",
                                            "meta.json"))]
    if clash and not os.environ.get("FORCE_WAVE"):
        print(f"[wave] ERROR: results already exist for exp "
              f"{', '.join(str(c) for c in clash)} under {out_dir} — bump "
              f"START_EXP_ID or set FORCE_WAVE=1 to wipe and re-run",
              file=sys.stderr)
        sys.exit(1)

env = lambda k, d: os.environ.get(k, d)
paths = []
for exp_id, meta_path in zip(exp_ids, metas):
    meta = json.load(open(meta_path))
    ref = meta.get("stem_metric")
    weights = meta.get("weights")
    if not isinstance(weights, list) or not weights or not all(
            isinstance(w, (int, float)) for w in weights):
        print(f"[wave] ERROR: {meta_path} has no usable weights list "
              f"(got {type(weights).__name__}); not an experiment meta — "
              f"clean the archive or point SPEEDRUN_DIR elsewhere",
              file=sys.stderr)
        sys.exit(1)
    parts = [
        "python3", "scripts/dispatch_remote.py",
        "--remote-config", wave_cfg,
        "--data-dir", env("DATA_DIR", "/tmp/speedrun_data"),
        "--schema", "config/schema_stem.yaml",
        "--cluster-cache-dir", env("CLUSTER_CACHE_DIR", speedrun_dir),
        "--nanochat-dir", env("NANOCHAT_DIR", "~/work/nanochat-npu"),
        "--nanochat-base-dir", env("NANOCHAT_BASE_DIR",
                                   "~/work/nanochat_model_dir"),
        "--phase1-checkpoint-path", env(
            "PHASE1_CKPT",
            "~/work/nanochat_model_dir/base_checkpoints/d20"),
        "--output-dir", out_dir,
        "--exp-id", str(exp_id),
        "--exp-name", name,
        "--weights", ",".join(repr(w) for w in weights),
        "--proxy-num-iterations", "50",
        "--proxy-target-tokens", "10M",
        "--eval-max-per-task", "100",
        "--general-data-dir", env(
            "GENERAL_DATA_DIR",
            "/home/ma-user/work/nanochat_model_dir/climbmix_shards"),
    ]
    # expanduser: shlex.join would quote bare ~/ paths and silently break
    # tilde expansion when the command file is re-executed via bash
    parts = [os.path.expanduser(a) for a in parts]
    script = ("#!/usr/bin/env bash\n"
              f"# from {meta_path}"
              f" (speedrun stem_metric={ref})\n"
              f"cd {shlex.quote(repo)}\n"
              f"{shlex.join(parts)}\n")
    with open(os.path.join(cmd_dir, f"run_{exp_id}.sh"), "w") as f:
        f.write(script)
    paths.append(meta_path)
    print(exp_id)
print(f"[wave] {len(metas)} configs, dispatch order:", file=sys.stderr)
for exp_id, p in zip(exp_ids, paths):
    print(f"[wave]   exp {exp_id} <- {p}", file=sys.stderr)
print(f"[wave] k4, eval cap 100", file=sys.stderr)
PY
then
  die "config collection failed (see messages above)"
fi

mapfile -t IDS < "$IDS_FILE"
rm -f "$IDS_FILE"
[ "${#IDS[@]}" -gt 0 ] || die "no configs collected"
for i in "${IDS[@]}"; do
  [[ "$i" =~ ^[0-9]+$ ]] || die "collector emitted a non-numeric exp id: '$i'"
done
FILES=()
for i in "${IDS[@]}"; do
  f="$CMD_DIR/run_${i}.sh"
  [ -f "$f" ] || die "collector did not write $f"
  FILES+=("$f")
done

LOG_DIR="${LOG_DIR:-/tmp/wave_logs}"
mkdir -p "$LOG_DIR"

LAST_ID="${IDS[$(( ${#IDS[@]} - 1 ))]}"
echo "[wave] ${#FILES[@]} configs, exp ${IDS[0]}..$LAST_ID, " \
     "k4 x MAX_PARALLEL=$MAX_PARALLEL, eval cap 100"

# ── dry-run ─────────────────────────────────────────────────────────────
if [ "$mode" = "dry-run" ]; then
  for f in "${FILES[@]}"; do
    echo "  --- $(basename "$f") ---"
    tail -n +4 "$f" | sed 's/^/  /'
  done
  echo "[wave] dry-run — nothing executed"
  exit 0
fi

# ── summary (also the run tail) ─────────────────────────────────────────
summary() {
  LOG_DIR_HINT="$LOG_DIR" python3 - "$REPO" "$SPEEDRUN_DIR" "$OUT_DIR" \
    "$START_EXP_ID" <<'PY'
import glob, json, os, re, sys

repo, speedrun_dir, out_dir, start_id = sys.argv[1:5]
start_id = int(start_id)
log_hint = os.environ.get("LOG_DIR_HINT", "/tmp/wave_logs")

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except OSError:
        return None

metas = []
for pat in (os.path.join(speedrun_dir, "exps", "*", "meta.json"),
            os.path.join(speedrun_dir, "exp*", "meta.json")):
    metas = sorted(glob.glob(os.path.join(repo, pat)))
    if metas:
        break
if not metas:
    print("no configs found", file=sys.stderr)
    sys.exit(1)

def gsm8k_time(exp_id):
    try:
        cur, t = None, {}
        with open(os.path.join(out_dir, f"exp_{exp_id:04d}", "eval.log"),
                  errors="replace") as f:
            for line in f:
                m = re.search(r"Evaluating: (\S+)", line)
                if m:
                    cur = m.group(1)
                    continue
                m = re.search(r"time: ([\d.]+)s", line)
                if m and cur:
                    t[cur] = float(m.group(1))
                    cur = None
        return t.get("gsm8k_cot")
    except OSError:
        return None

rows = []
for i, p in enumerate(metas):
    exp_id = start_id + i
    ref = (load(p) or {}).get("stem_metric")
    meta = load(os.path.join(out_dir, f"exp_{exp_id:04d}", "meta.json"))
    rows.append((exp_id, ref, meta, gsm8k_time(exp_id)))

done = sorted((r for r in rows if r[2]),
              key=lambda r: r[2].get("stem_metric", -1), reverse=True)
pending = [r for r in rows if not r[2]]
print("=" * 72)
print(f"search wave summary  [{len(done)}/{len(rows)} done]  "
      f"(search cap-100 eval; speedrun column = its own ruler, "
      f"offset expected)")
print("=" * 72)
print(f"{'exp':>5}  {'stem_metric':>11}  {'speedrun':>9}  {'gsm8k':>6}  status")
for exp_id, ref, meta, gk in done:
    m = meta.get("stem_metric")
    d = "" if (m is None or ref is None) else f"{m - ref:+.4f}"
    ref_s = "?" if ref is None else f"{ref:.4f}"
    rc = f"{meta.get('mid_train_rc')}/{meta.get('eval_rc')}"
    if rc != "0/0":
        status = f"rc {rc} FAILED"
    elif gk is None:
        status = "ok (eval.log not local)"
    elif gk >= 1200:
        status = f"CAP LOST? gsm8k {gk:.0f}s"
    else:
        status = f"ok {d}"
    gk_s = "?" if gk is None else f"{gk:.0f}s"
    m_s = f"{m:.4f}" if m is not None else "?"
    print(f"{exp_id:>5}  {m_s:>11}  {ref_s:>9}  {gk_s:>6}  {status}")
for exp_id, _, _, _ in pending:
    print(f"{exp_id:>5}  {'-':>11}  {'-':>9}  {'-':>6}  "
          f"pending/failed (log: {log_hint}/run_{exp_id}.log)")
print("=" * 72)
PY
}

if [ "$mode" = "summary" ]; then
  summary
  exit 0
fi
[ "$mode" = "run" ] || die "unknown mode: $mode (dry-run | run | summary)"

# ── run: staggered first dispatch, then a capped job pool ───────────────
first_log="$LOG_DIR/run_${IDS[0]}.log"
echo "[wave] first dispatch (exp ${IDS[0]}) starts alone — asset/shard stagger"
bash "${FILES[0]}" > "$first_log" 2>&1 &
stagger_ok=""
for _ in $(seq 1 90); do
  if grep -q "submitted job" "$first_log" 2>/dev/null; then stagger_ok=1; break; fi
  if grep -qE "Traceback|RuntimeError" "$first_log" 2>/dev/null; then
    echo "[wave] WARNING: first dispatch errored — pool continues anyway"
    break
  fi
  sleep 10
done
if [ -n "$stagger_ok" ]; then
  echo "[wave] stagger done — pool launches (max $((MAX_PARALLEL - 1)) more)"
else
  echo "[wave] stagger timed out after 15m — pool launches anyway"
fi
echo "[wave] progress: bash scripts/search_wave.sh summary  (anytime)"
echo "[wave] live:     grep -H 'stem_metric' $LOG_DIR/run_*.log | tail"

for f in "${FILES[@]:1}"; do
  # slot polling (bash 4.2-safe, no wait -n)
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    sleep 20
  done
  bash "$f" > "$LOG_DIR/$(basename "$f" .sh).log" 2>&1 &
done
wait || true

echo
summary

