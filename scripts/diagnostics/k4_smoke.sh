#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  k4_smoke.sh — k=4 远程性能 smoke (M5 闸门)
#
#  用与 M3 验收 (exp_0000, npu_per_job=1, 已采纳) 完全相同的 exp_0000
#  混合权重重新 dispatch 一次, 但 npu_per_job=4, 且落在隔离的 exp id
#  (900) — 避免 fresh-run 的陈旧产物清理覆盖 OBS exps/exp_0000 与
#  remote_validation/exp_0000 里的 M3 证据。
#
#  预期 (数据并行语义: total_batch_size 继承自 d20 预训练, 每个
#  optimizer step 的 token 数固定):
#    - mid_train.log 里 "Grad accum steps:" 128 -> 32 (k4 = M3/4)
#    - tok/sec ~4x M3; train ~54m -> ~11m; 整 job ~25m (cap 语义下;
#      900/901 的 64m 正是 flag 丢失 -> 全池训练 + 全量 eval)
#    - stem_metric 接近 M3 的 0.1622 (抽样噪声带内, 不要求相等;
#      900 vs 901 同配置复跑实测噪声 ~0.0003)
#  分片哈希与 M3 必然不同 (nproc 相关切分, 设计使然), 不比对哈希。
#
#  用法 (服务器上; 阻塞 ~25 分钟, 建议 tmux):
#    bash scripts/diagnostics/k4_smoke.sh self dry-run  # 推荐: 内部构建完整命令
#    bash scripts/diagnostics/k4_smoke.sh self run      # (无需粘贴/历史命令)
#    bash scripts/diagnostics/k4_smoke.sh dry-run '<cmd>'  # 显式命令 (缺任一
#    bash scripts/diagnostics/k4_smoke.sh run '<cmd>'      # 语义 flag 直接拒绝)
#    bash scripts/diagnostics/k4_smoke.sh analyze       # 随时可跑, 见下
#
#  self 模式: 权重读自 M3 meta.json, 四个 speedrun 语义 flag
#  (--proxy-num-iterations 50 / --proxy-target-tokens 10M /
#   --eval-max-per-task 100 / --general-data-dir) 全部内置, 服务器路径
#  走已知默认 — 免疫 shell 历史丢失与终端粘贴损坏 (exp_0901 事故:
#  实际执行的命令丢了两个 flag, 训练落在全池、eval 跑了全量)。
#
#  显式命令模式: 从 ~/.bash_history 恢复 M3 原命令 (过滤 --check-assets
#  与缺 --proxy-num-iterations 的版本), 或 run '<cmd>' 传入。四个语义
#  flag 缺一即 die。
#
#  analyze 语义自检 (随时可跑 — job 未完成也行, 分钟级判定, 不用等):
#    mixture_data 分片数  (dispatch 侧, 提交前就绪:  ~3 = cap / ~73 = 全池)
#    spec.json eval_cmd   (提交即写, FUSE 可读: 必须含 --max-per-task)
#    train token budget   (train 结束才打印: ~14-15M = cap / 52M = 全池)
#    gsm8k 每 rank 计数   (eval 开始后: 25 = cap / 330 = 全量; 25/rank
#                          无 80 步进进度行时, 回退用 gsm8k 任务耗时判定)
#    任一硬性信号显示 flag 丢失 -> exit 1 并提示立刻取消 job (提交时打印
#    过 console 链接), 不再白烧一小时。metric 差值 (band ±0.008,
#    seed 不同 -> ~0.003 抽样噪声) 仅在 meta.json 落地后给出。
#
#  环境变量:
#    SRC_CFG=/tmp/remote_smoke.json   源 RemoteConfig (npu_per_job=1)
#    EXP_ID=902 EXP_NAME=remoteval-k4c   目标 exp (默认 900)
#    (self 模式可覆盖: WEIGHTS DATA_DIR CLUSTER_CACHE_DIR NANOCHAT_DIR
#     NANOCHAT_BASE_DIR PHASE1_CKPT GENERAL_DATA_DIR — 默认服务器布局)
#    (REPO 自动取本脚本所在仓库根, 无需设置)
#
#  本脚本在 scripts/diagnostics/ 下, 不进指纹。
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_CFG="${SRC_CFG:-/tmp/remote_smoke.json}"
K4_CFG="/tmp/remote_smoke_k4.json"
OUT_DIR="/tmp/remote_validation_k4"
EXP_ID="${EXP_ID:-900}"
EXP_NAME="${EXP_NAME:-remoteval-k4}"
M3_DIR="$REPO/remote_validation/exp_0000"
K4_DIR="$OUT_DIR/exp_$(printf '%04d' "$EXP_ID")"

mode="${1:-run}"
explicit_cmd="${2:-}"

die() { echo "[k4] ERROR: $*" >&2; exit 1; }

# Canonical M3-semantics dispatch command, built from the repo layout.
# All four speedrun flags baked in; weights read from the M3 artifacts.
# Immune to lost shell history and terminal paste corruption.
build_self_cmd() {
    python3 - "$M3_DIR" <<'PY'
import json, os, shlex, sys
m3 = sys.argv[1]
weights = os.environ.get("WEIGHTS") or ",".join(
    repr(x) for x in json.load(open(os.path.join(m3, "meta.json")))["weights"])
env = lambda k, d: os.environ.get(k, d)
parts = [
    "python3", "scripts/dispatch_remote.py",
    "--remote-config", env("K4_CFG", "/tmp/remote_smoke_k4.json"),
    "--data-dir", env("DATA_DIR", "/tmp/speedrun_data"),
    "--schema", "config/schema_stem.yaml",
    "--cluster-cache-dir", env("CLUSTER_CACHE_DIR",
                               "result/speedrun_20260828_222129"),
    "--nanochat-dir", env("NANOCHAT_DIR", "~/work/nanochat-npu"),
    "--nanochat-base-dir", env("NANOCHAT_BASE_DIR", "~/work/nanochat_model_dir"),
    "--phase1-checkpoint-path", env("PHASE1_CKPT",
                                    "~/work/nanochat_model_dir/base_checkpoints/d20"),
    "--output-dir", env("OUT_DIR", "/tmp/remote_validation_k4"),
    "--exp-id", env("EXP_ID", "900"),
    "--exp-name", env("EXP_NAME", "remoteval-k4"),
    "--weights", weights,
    "--proxy-num-iterations", "50",
    "--proxy-target-tokens", "10M",
    "--eval-max-per-task", "100",
    "--general-data-dir", env("GENERAL_DATA_DIR",
                              "/home/ma-user/work/nanochat_model_dir/climbmix_shards"),
]
print(shlex.join(parts))
PY
}

[ -f "$SRC_CFG" ] || die "remote config not found: $SRC_CFG (set SRC_CFG=...)"
[ -f "$M3_DIR/meta.json" ] || die "M3 artifacts missing: $M3_DIR/meta.json"

# self mode: canonical command, no pasted input
if [ "$mode" = "self" ]; then
    mode="${2:-dry-run}"
    explicit_cmd="$(build_self_cmd)" || die "self command build failed"
fi

# ── 1. k4 remote config (npu_per_job 1 -> 4) ────────────────────────────
_GEO="$(python3 - "$SRC_CFG" "$K4_CFG" "$EXP_ID" <<'PY'
import json, re, sys
src, dst, exp_id = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(src))
cfg["npu_per_job"] = 4
flavor = cfg.get("flavor", "")
if flavor and "4xlarge" not in flavor:
    print(f"[k4] WARNING: flavor {flavor!r} is not a 4-card flavor "
          f"(expected a 4xlarge flavor)", file=sys.stderr)
elif not flavor:
    print("[k4] NOTE: flavor empty in config — the backend default_flavor "
          "must be a 4-card flavor", file=sys.stderr)
json.dump(cfg, open(dst, "w"), indent=2)
prefix = cfg["obs_prefix"].rstrip("/")
fuse = re.sub(r"^obs://[^/]+/(s\d+/)?", "/", prefix)
print(fuse)
print(f"{fuse}/exps/exp_{int(exp_id):04d}/result/mid_train.log")
PY
)"
FUSE_ROOT="$(printf '%s\n' "$_GEO" | sed -n 1p)"
WATCH_LOG="$(printf '%s\n' "$_GEO" | sed -n 2p)"
echo "[k4] repo: $REPO"
echo "[k4] wrote $K4_CFG (npu_per_job=4, isolated exp_${EXP_ID})"
echo "[k4] live log once training starts: tail -F '$WATCH_LOG'"

analyze() {
  python3 - "$M3_DIR" "$K4_DIR" "$FUSE_ROOT" "$EXP_ID" <<'PY'
import glob, json, os, re, sys

m3_dir, k4_dir, fuse_root, exp_id = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except OSError:
        return None

def log_stats(path):
    dts, toks, accum, total_m = [], [], None, None
    try:
        if not path:
            return dict(accum=None, dt_ms=None, tok_s=None, total_m=None)
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r"Grad accum steps: (\d+)", line)
                if m:
                    accum = int(m.group(1))
                m = re.search(r"dt: ([\d.]+)ms", line)
                if m:
                    dts.append(float(m.group(1)))
                m = re.search(r"tok/sec: ([\d,]+)", line)
                if m:
                    toks.append(int(m.group(1).replace(",", "")))
                m = re.search(r"total time: ([\d.]+)m", line)
                if m:
                    total_m = float(m.group(1))
    except OSError:
        pass
    tail = lambda xs: xs[-10:] if xs else []
    return dict(
        accum=accum,
        dt_ms=(sum(tail(dts)) / len(tail(dts))) if dts else None,
        tok_s=(sum(tail(toks)) / len(tail(toks))) if toks else None,
        total_m=total_m,
    )

def train_tokens(path):
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r"Total tokens: ([\d,]+)", line)
                if m:
                    return int(m.group(1).replace(",", ""))
    except OSError:
        pass
    return None

def gsm8k_per_rank(path):
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r"\[gsm8k_cot\] \d+/(\d+) examples", line)
                if m:
                    return int(m.group(1))
    except OSError:
        pass
    return None

def task_times(path):
    times, cur = {}, None
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r"Evaluating: (\S+)", line)
                if m:
                    cur = m.group(1)
                    continue
                m = re.search(r"time: ([\d.]+)s", line)
                if m and cur:
                    times[cur] = float(m.group(1))
                    cur = None
    except OSError:
        pass
    return times

def first(*paths):
    for p in paths:
        try:
            if p and os.path.isfile(p):
                return p
        except OSError:
            pass
    return None

def fmt(x, f="{:.1f}", na="?"):
    return na if x is None else f.format(x)

fuse_exp = os.path.join(fuse_root, "exps", f"exp_{exp_id:04d}") if fuse_root else ""
m3 = load(os.path.join(m3_dir, "meta.json"))
k4 = load(os.path.join(k4_dir, "meta.json"))
rr3 = load(os.path.join(m3_dir, "_remote_result.json")) or {}
rr4 = load(os.path.join(k4_dir, "_remote_result.json")) or {}
# local downloaded copies first (post-run); FUSE live logs as mid-run fallback
train_log = first(os.path.join(k4_dir, "mid_train.log"),
                  os.path.join(fuse_exp, "result", "mid_train.log"))
eval_log = first(os.path.join(k4_dir, "eval.log"),
                 os.path.join(fuse_exp, "result", "eval.log"))
ls3 = log_stats(os.path.join(m3_dir, "mid_train.log"))
ls4 = log_stats(train_log)

# ── semantics gate: did the dispatch actually carry the speedrun caps? ──
# signals arrive over time: mixture shards (dispatch side, before submit),
# spec.json (at submission, via FUSE), token budget (train end),
# gsm8k count/time (eval). A missing artifact is pending, not failure.
mix_n = len(glob.glob(os.path.join(k4_dir, "mixture_data", "*.parquet")))
spec = load(os.path.join(fuse_exp, "spec.json"))
spec_cap = None if spec is None else "--max-per-task" in (spec.get("eval_cmd") or [])
tt4 = train_tokens(train_log) if train_log else None
gk4 = gsm8k_per_rank(eval_log) if eval_log else None
gk_t = task_times(eval_log).get("gsm8k_cot") if eval_log else None

fails = []
print("=" * 64)
print(f"k=4 smoke vs M3 (exp_0000, npu_per_job=1)  [exp_{exp_id:04d}]")
print("=" * 64)
if k4 is None:
    print("[analyze] k4 meta.json not present — mid-run semantics check only")
    print("(rerun after the job finishes for the perf table + metric verdict)")
else:
    print(f"rc (mid/eval):      M3 {m3['mid_train_rc']}/{m3['eval_rc']}"
          f"   k4 {k4['mid_train_rc']}/{k4['eval_rc']}"
          f"   {'OK' if k4['mid_train_rc'] == 0 and k4['eval_rc'] == 0 else 'FAIL'}")
    print(f"grad accum steps:   M3 {ls3['accum']}   k4 {ls4['accum']}"
          f"   (expect k4 = M3/4 -> data parallel engaged)")
    print(f"mean dt (last 10):  M3 {fmt(ls3['dt_ms'])}ms"
          f"   k4 {fmt(ls4['dt_ms'])}ms")
    if ls3["tok_s"] and ls4["tok_s"]:
        r = ls4["tok_s"] / ls3["tok_s"]
        print(f"tok/sec (last 10):  M3 {ls3['tok_s']:,.0f}   k4 {ls4['tok_s']:,.0f}"
              f"   ratio {r:.2f}x (expect ~4x)")
    else:
        print(f"tok/sec:            M3 {fmt(ls3['tok_s'], '{:,.0f}')}"
              f"   k4 {fmt(ls4['tok_s'], '{:,.0f}')}")
    print(f"train total time:   M3 {fmt(ls3['total_m'], '{:.1f}')}m"
          f"   k4 {fmt(ls4['total_m'], '{:.1f}')}m")
    print(f"worker elapsed:     M3 {fmt(rr3.get('elapsed_seconds'))}s"
          f"   k4 {fmt(rr4.get('elapsed_seconds'))}s")
    print(f"master elapsed:     M3 {fmt(m3.get('elapsed_seconds'))}s"
          f"   k4 {fmt(k4.get('elapsed_seconds'))}s  (incl prep+upload+eval)")
print("-" * 64)
print("semantics gate (speedrun caps actually in effect?):")
if mix_n:
    kind = ("capped ~10M" if mix_n <= 6
            else "FULL POOL (flag lost!)" if mix_n >= 40 else "?")
    print(f"  mixture shards:        {mix_n} files  ({kind})")
    if mix_n >= 40:
        fails.append(f"mixture_data has {mix_n} shards: full pool selected")
else:
    print("  mixture shards:        pending (mixture_data not ready)")
if spec_cap is None:
    print("  spec --max-per-task:   pending (spec.json not visible yet)")
else:
    print(f"  spec --max-per-task:   {'present' if spec_cap else 'MISSING'}")
    if spec_cap is False:
        fails.append("spec.json eval_cmd lacks --max-per-task")
if tt4 is not None:
    kind = "capped ~10M" if tt4 < 20_000_000 else "FULL POOL (flag lost!)"
    print(f"  train token budget:    {tt4:,}  ({kind})")
    if tt4 >= 20_000_000:
        fails.append(f"train tokens {tt4:,}: full pool")
else:
    print("  train token budget:    pending (train not finished)")
if gk4 is not None:
    kind = "cap 100" if gk4 <= 30 else "full 1319 (cap lost!)"
    print(f"  gsm8k per-rank count:  {gk4}  ({kind})")
    if gk4 > 30:
        fails.append(f"gsm8k per-rank {gk4}: full set")
elif gk_t is not None:
    # cap 100 at k4 -> 25 examples/rank -> the every-80 progress line never
    # prints; fall back to the task wall time (capped ~150-350s, full ~1800s)
    kind = ("cap 100 (by task time)" if gk_t <= 600
            else "FULL SET (by task time)" if gk_t >= 1200 else "? (by task time)")
    print(f"  gsm8k eval time:       {gk_t:.0f}s  ({kind})")
    if gk_t >= 1200:
        fails.append(f"gsm8k_cot took {gk_t:.0f}s: full set")
else:
    print("  gsm8k per-rank count:  pending (eval not started)")
capped = (spec_cap is True) and (tt4 is not None and tt4 < 20_000_000)
if k4 is not None:
    if capped:
        d = k4.get("stem_metric", float("nan")) - m3.get("stem_metric", float("nan"))
        verdict = "PASS" if abs(d) <= 0.008 else "CHECK"
        print(f"  metric delta vs M3:   {d:+.4f}"
              f"  (band +-0.008; seeds differ -> ~0.003 sampling noise)  {verdict}")
    else:
        print("  metric delta vs M3:   NOT comparable (caps missing — rerun)")
    print("-" * 64)
    print(f"stem_metric:        M3 {m3.get('stem_metric', float('nan')):.4f}"
          f"   k4 {k4.get('stem_metric', float('nan')):.4f}")
    print(f"stem_nll:           M3 {m3.get('stem_nll', float('nan'))}"
          f"   k4 {k4.get('stem_nll', float('nan'))}")
print("=" * 64)
if fails:
    print("SEMANTICS FAIL — speedrun caps were lost again. Cancel the job now")
    print("(console URL was printed at submission) and resubmit via 'self run'.")
    for x in fails:
        print(f"  - {x}")
    sys.exit(1)
print("semantics: OK / pending — items above fill in as the job progresses.")
print("perf gate: rc 0/0; accum k4 = M3/4; tok/sec ~4x; train well under M3.")
PY
}

if [ "$mode" = "analyze" ]; then
  analyze
  exit 0
fi

# ── 2. recover the exact M3 dispatch command ────────────────────────────
base_cmd="$explicit_cmd"
if [ -z "$base_cmd" ]; then
  base_cmd=$(grep 'dispatch_remote\.py' "$HOME/.bash_history" 2>/dev/null \
             | grep -v -- '--check-assets' \
             | grep -E -- '--proxy-num-iterations[= ]50' | tail -1 || true)
fi
[ -n "$base_cmd" ] || die "M3 dispatch command not found in history — rerun as:
  bash scripts/diagnostics/k4_smoke.sh run '<the full M3 dispatch command>'"

cmd=$(python3 - "$base_cmd" "$K4_CFG" "$EXP_ID" "$EXP_NAME" "$OUT_DIR" <<'PY'
import os, shlex, sys
raw, cfg, exp_id, exp_name, out_dir = sys.argv[1:6]
seg = raw
if "&&" in raw:
    parts = [s for s in raw.split("&&") if "dispatch_remote.py" in s]
    seg = parts[-1] if parts else raw
argv = shlex.split(seg)

def setopt(flag, value):
    if flag in argv:                       # space form: replace value
        argv[argv.index(flag) + 1] = value
    else:
        argv.extend([flag, value])         # append; argparse last-wins

setopt("--remote-config", cfg)
setopt("--exp-id", exp_id)
setopt("--exp-name", exp_name)
setopt("--output-dir", out_dir)
# expanduser: shlex.join would quote bare ~/ paths and silently break
# tilde expansion when the command is re-executed via bash -c
argv = [os.path.expanduser(a) for a in argv]

has = lambda f: f in argv or any(a.startswith(f + "=") for a in argv)
missing = [f for f in ("--proxy-num-iterations", "--general-data-dir",
                       "--proxy-target-tokens", "--eval-max-per-task")
           if not has(f)]
for flag in missing:
    print(f"[k4] ERROR: command lacks {flag} — refusing to run "
          f"(M3/speedrun semantics broken; use 'self' mode)", file=sys.stderr)
if missing:
    sys.exit(1)
print(shlex.join(argv))
PY
) || die "command rejected: missing speedrun-semantics flags"

echo
echo "[k4] === command (run from $REPO) ==="
echo "  $cmd"
echo

if [ "$mode" = "dry-run" ]; then
  echo "[k4] dry-run — nothing executed"
  exit 0
fi

# --proxy-target-tokens 10M needs dispatch_remote >= 29295e8 (parse_token_count)
grep -q "parse_token_count" "$REPO/scripts/dispatch_remote.py" \
  || die "dispatch_remote.py predates 29295e8 (10M suffix rejected) — git pull first"

cd "$REPO"
bash -c "$cmd"
echo "[k4] dispatch finished — analyzing"
analyze
