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
#    - mid_train.log 里 "Grad accum steps:" 8 -> 2
#    - tok/sec ~4x M3; train 总时长 ~54m -> ~15-20m; 整 job 远低于 M3 的 76m
#    - stem_metric 接近 M3 的 0.1622 (抽样噪声带内, 不要求相等)
#  分片哈希与 M3 必然不同 (nproc 相关切分, 设计使然), 不比对哈希。
#
#  用法 (服务器上; 阻塞 ~30-40 分钟, 建议 tmux):
#    bash scripts/diagnostics/k4_smoke.sh dry-run     # 只打印确切命令
#    bash scripts/diagnostics/k4_smoke.sh run         # 执行
#    bash scripts/diagnostics/k4_smoke.sh run '<cmd>' # 显式传入 M3 原命令
#    bash scripts/diagnostics/k4_smoke.sh analyze     # 完成后对比 M3
#
#  M3 命令自动从 ~/.bash_history 恢复 (过滤掉 --check-assets 与缺
#  --proxy-num-iterations 的失败版本); 若在未关闭的会话里, 用 run '<cmd>'
#  显式传入。命令缺少 --proxy-num-iterations 50 或 --general-data-dir
#  时会打 WARNING (这两项是 M3 语义的关键)。
#
#  环境变量:
#    SRC_CFG=/tmp/remote_smoke.json   源 RemoteConfig (npu_per_job=1)
#    (REPO 自动取本脚本所在仓库根, 无需设置)
#
#  本脚本在 scripts/diagnostics/ 下, 不进指纹。
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC_CFG="${SRC_CFG:-/tmp/remote_smoke.json}"
K4_CFG="/tmp/remote_smoke_k4.json"
OUT_DIR="/tmp/remote_validation_k4"
EXP_ID="900"
EXP_NAME="remoteval-k4"
M3_DIR="$REPO/remote_validation/exp_0000"
K4_DIR="$OUT_DIR/exp_$(printf '%04d' "$EXP_ID")"

mode="${1:-run}"
explicit_cmd="${2:-}"

die() { echo "[k4] ERROR: $*" >&2; exit 1; }

[ -f "$SRC_CFG" ] || die "remote config not found: $SRC_CFG (set SRC_CFG=...)"
[ -f "$M3_DIR/meta.json" ] || die "M3 artifacts missing: $M3_DIR/meta.json"

# ── 1. k4 remote config (npu_per_job 1 -> 4) ────────────────────────────
WATCH_LOG=$(python3 - "$SRC_CFG" "$K4_CFG" "$EXP_ID" <<'PY'
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
print(f"{fuse}/exps/exp_{int(exp_id):04d}/result/mid_train.log")
PY
)
echo "[k4] repo: $REPO"
echo "[k4] wrote $K4_CFG (npu_per_job=4, isolated exp_${EXP_ID})"
echo "[k4] live log once training starts: tail -F '$WATCH_LOG'"

analyze() {
  python3 - "$M3_DIR" "$K4_DIR" <<'PY'
import json, os, re, sys

m3_dir, k4_dir = sys.argv[1], sys.argv[2]

def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except OSError:
        return None

def log_stats(path):
    dts, toks, accum, total_m = [], [], None, None
    try:
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

m3 = load(os.path.join(m3_dir, "meta.json"))
k4 = load(os.path.join(k4_dir, "meta.json"))
if k4 is None:
    print(f"[analyze] no k4 meta.json at {k4_dir} — job not finished?")
    sys.exit(1)
rr3 = load(os.path.join(m3_dir, "_remote_result.json")) or {}
rr4 = load(os.path.join(k4_dir, "_remote_result.json")) or {}
ls3 = log_stats(os.path.join(m3_dir, "mid_train.log"))
ls4 = log_stats(os.path.join(k4_dir, "mid_train.log"))

def fmt(x, f="{:.1f}", na="?"):
    return na if x is None else f.format(x)

print("=" * 64)
print("k=4 smoke vs M3 (exp_0000, npu_per_job=1)")
print("=" * 64)
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
print(f"stem_metric:        M3 {m3.get('stem_metric', float('nan')):.4f}"
      f"   k4 {k4.get('stem_metric', float('nan')):.4f}")
print(f"stem_nll:           M3 {m3.get('stem_nll', float('nan'))}"
      f"   k4 {k4.get('stem_nll', float('nan'))}")
print("(metric note: comparable only if both runs used the same "
      "--proxy-target-tokens; an uncapped run trains on the full "
      "weight-selected pool and the metric shifts — not a k=4 defect)")
print("=" * 64)
print("verdict criteria (perf gate): rc 0/0; accum k4 = M3/4; tok/sec ~4x;"
      " train time well under M3. Metric is informational only.")
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
for flag in ("--proxy-num-iterations", "--general-data-dir"):
    if not has(flag):
        print(f"[k4] WARNING: command lacks {flag} — M3 semantics broken!",
              file=sys.stderr)
print(shlex.join(argv))
PY
)

echo
echo "[k4] === command (run from $REPO) ==="
echo "  $cmd"
echo

if [ "$mode" = "dry-run" ]; then
  echo "[k4] dry-run — nothing executed"
  exit 0
fi

cd "$REPO"
bash -c "$cmd"
echo "[k4] dispatch finished — analyzing"
analyze
