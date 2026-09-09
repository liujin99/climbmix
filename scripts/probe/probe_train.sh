#!/usr/bin/env bash
# Multi-node probe C: real ws=16 (2 nodes x 8 NPU) mid_train benchmark.
# Runs on every node:
#   1. links the staged input mounts into nanochat's expected layout
#      (d28 base ckpt, tokenizer, mixture data)
#   2. extracts nanochat-npu from the code dir tar
#   3. resolves rendezvous (probe_common.sh) and launches mid_train
#      under multi-node torchrun: 50 steps, dbs=1, --load-optimizer=0
#      (cold optimizer — the ws!=8 shard mismatch is the point)
#   4. extracts timing/memory/checkpoint evidence into the output mount
#
# Env contract (driver): PROBE_OUT/PROBE_NNODES/PROBE_NPROC/PROBE_RDZV_*
#   PROBE_STEPS     iterations to run (default 50)
#   PROBE_TAG       model tag == d28 asset name (loads base_checkpoints/d28,
#                   writes mid_checkpoints/d28 in the container — throwaway)
set -u
CODE="${MA_CODE_DIR:-/home/ma-user/modelarts/user-job-dir}"
OUT="${PROBE_OUT:-/home/ma-user/modelarts/outputs/probe_out_0}"
IN="${PROBE_INPUT_BASE:-/home/ma-user/modelarts/inputs}"
BASE="${PROBE_BASE_DIR:-/home/ma-user/work/nanochat_base}"
NANOCHAT="${PROBE_NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
STEPS="${PROBE_STEPS:-50}"
TAG="${PROBE_TAG:-d28}"
export PROBE_OUT="$OUT"
mkdir -p "$OUT" "$BASE/mid_checkpoints" "$BASE/base_checkpoints"

. "$CODE/probe_common.sh"

# Boot-stage breadcrumbs -> output mount (the console log is NOT synced;
# run 20260908_201827's silent worker-3 left us no boot evidence).
# Bound HCCL connect timeout: EI0015 at ~10min instead of a 20min freeze.
BOOTLOG="$OUT/boot_$(hostname).log"
exec > >(tee -a "$BOOTLOG") 2>&1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"

# device-plane evidence (best-effort, into the boot log)
rdzv_dump_host_nets

# ── 1) assets: input mounts -> nanochat layout ──
ln -sfn "$IN/d28_0"      "$BASE/base_checkpoints/$TAG"
ln -sfn "$IN/tokenizer_0" "$BASE/tokenizer"
DATA="$NANOCHAT.probe_data"
rm -rf "$DATA"
ln -sfn "$IN/data_0" "$DATA"
echo "[probe C $(hostname)] assets: d28=$(ls "$IN"/d28_0/model_*.pt 2>/dev/null | head -1) data=$(ls "$IN"/data_0/*.parquet 2>/dev/null | wc -l) parquet files"

# ── 2) nanochat code from the code-dir tar (fresh channel) ──
if [ ! -d "$NANOCHAT/scripts" ]; then
  TAR=""
  for c in "$CODE" "$CODE/assets"; do
    [ -f "$c/nanochat-npu.tar.gz" ] && TAR="$c/nanochat-npu.tar.gz" && break
  done
  if [ -z "$TAR" ]; then
    echo "FATAL: no nanochat-npu.tar.gz in $CODE" | tee -a "$OUT/train_failure_$(hostname).log"
    exit 1
  fi
  mkdir -p "$(dirname "$NANOCHAT")"
  tar xzf "$TAR" -C "$(dirname "$NANOCHAT")" || exit 1
  echo "[probe C $(hostname)] extracted $TAR"
else
  echo "[probe C $(hostname)] nanochat already present"
fi

# ── 2.5) deps the vllm-ascend image lacks (verbatim the worker boot's
#         loop: import probe -> OFFLINE wheel from the code dir -> pip ->
#         internal mirror; rustbpe AND pyarrow both hit ModuleNotFoundError
#         live — the driver stages the run's wheels into the code dir) ──
for d in datasets dotenv=python-dotenv fastapi filelock huggingface_hub \
         jinja2 numpy pandas psutil pyarrow pydantic pytest requests \
         rustbpe sentence_transformers=sentence-transformers tiktoken \
         tokenizers tqdm transformers urllib3 uvicorn wandb yaml=pyyaml; do
  m=${d%%=*}; p=${d##*=}
  python3 -c "import $m" 2>/dev/null || {
    echo "[probe C $(hostname)] installing missing dep: $m"
    python3 -m pip install --no-index --find-links "$CODE" \
            --find-links "$CODE/assets" "$p" >/dev/null 2>&1 || \
    python3 -m pip install "$p" >/dev/null 2>&1 || \
    python3 -m pip install -i \
            http://repo.myhuaweicloud.com/repository/pypi/simple \
            --trusted-host repo.myhuaweicloud.com "$p" || true
  }
done

# ── 3) rendezvous + train (wrapper breadcrumbs: a hung non-master
#        node is otherwise invisible between import and the first
#        collective traceback) ──
rdzv_resolve || {
  echo "[probe C] FATAL: rendezvous failed on $(hostname)" \
       > "$OUT/train_failure_$(hostname).log"
  exit 1
}

export NANOCHAT_BASE_DIR="$BASE"
export PYTHONPATH="$NANOCHAT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
LOG="$OUT/mid_train_$(hostname).log"
cd "$NANOCHAT"

torchrun $(rdzv_torchrun_argv) "$CODE/probe_train_wrap.py" \
  --num-iterations="$STEPS" \
  --lr-scale=1.0 \
  --warmup-ratio=0.0 \
  --warmdown-ratio=0.9 \
  --core-metric-every=-1 \
  --device-batch-size=1 \
  --loader=flat \
  --sample-every=-1 \
  --eval-every=-1 \
  --load-optimizer=0 \
  --run=probe_c \
  --model-tag="$TAG" \
  --data-dir="$DATA" \
  > "$LOG" 2>&1
RC=$?

# ── 4) evidence extraction (every node; shards land on both) ──
MEAN_DT=$(grep -oE 'dt: [0-9.]+ms' "$LOG" | tail -30 | grep -oE '[0-9.]+' | \
           awk '{s+=$1; n+=1} END {if (n) printf "%.1f", s/n}')
DT_LAST=$(grep -oE 'dt: [0-9.]+ms' "$LOG" | tail -1)
PEAK=$(grep -i "peak memory" "$LOG" | tail -3)
CKPT_FILES=$(ls "$BASE/mid_checkpoints/$TAG" 2>/dev/null | tr '\n' ' ')
N_DT=$(grep -cE 'dt: [0-9.]+ms' "$LOG")
python3 - "$OUT/train_result_$(hostname).json" "$RC" "$MEAN_DT" "$N_DT" \
    "$DT_LAST" "$PEAK" "$CKPT_FILES" <<'PYEOF'
import json
import socket
import sys

path, rc, mean_dt, n_dt, dt_last, peak, ckpt = sys.argv[1:8]
json.dump({
    "hostname": socket.gethostname(),
    "rc": int(rc),
    "steps_logged": int(n_dt),
    "mean_dt_ms_last30": float(mean_dt) if mean_dt else None,
    "last_step_line_dt": dt_last,
    "peak_memory_lines": peak,
    "ckpt_files": ckpt.split(),
    "node_rank": __import__("os").environ.get("RDZV_NODE_RANK"),
    "rdzv_master": __import__("os").environ.get("RDZV_MASTER_ADDR"),
}, open(path, "w"), indent=2)
print("[probe C] wrote", path)
PYEOF

echo "[probe C $(hostname)] rc=$RC mean_dt=${MEAN_DT}ms steps=$N_DT"
exit "$RC"
