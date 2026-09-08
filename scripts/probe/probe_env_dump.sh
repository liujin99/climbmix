#!/usr/bin/env bash
# Multi-node probe A: per-node environment + identity dump.
# Runs IDENTICALLY on every node of a node_count>1 job. Answers:
#   1. does the gateway accept node_count>1 at all? (submit itself fails
#      otherwise — the driver reports that)
#   2. does the command run on EVERY node? (we count node_*.log files)
#   3. what node-identity env does the platform inject? (full env dump,
#      secrets masked)
#   4. is the output mount bi-directionally visible across nodes?
#      (rendezvous marker test: every node writes one, then polls)
#
# Env contract (set by the driver through the job's environments):
#   PROBE_OUT       container output mount dir (synced back to OBS)
#   PROBE_NNODES    expected node count
#   PROBE_RDZV_TIMEOUT  marker poll seconds (default 90)
#
# Output: $PROBE_OUT/node_$(hostname).log (everything) and
#         $PROBE_OUT/node_summary_$(hostname).json (machine-readable).
#
# Deliberately dependency-free (no probe_common.sh): it must run before
# we know ANYTHING about the platform's multi-node conventions.
set -u
OUT="${PROBE_OUT:-/home/ma-user/modelarts/outputs/probe_out_0}"
NNODES="${PROBE_NNODES:-2}"
RDZV_TIMEOUT="${PROBE_RDZV_TIMEOUT:-90}"
LOG="$OUT/node_$(hostname).log"
SUMMARY="$OUT/node_summary_$(hostname).json"
mkdir -p "$OUT"

mask_secrets() {
  # NAME=VALUE pairs whose NAME smells like a secret get the value masked
  sed -E 's/([^=]*(TOKEN|SECRET|PASS|KEY|CRED|AK|SK|SIGNATURE)[^=]*)=.+/\1=<masked>/I'
}

log() { echo "$@"; }

{
echo "==================== probe A: env dump ===================="
echo "time:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:    $(hostname)"
echo "uname:       $(uname -a)"
echo "nproc:       $(nproc)"
echo "free:        $(free -h | head -2 | tail -1)"
echo
echo "---- ips ----"
hostname -I 2>/dev/null || true
echo "---- ip route ----"
ip route 2>/dev/null || cat /proc/net/route 2>/dev/null || true
echo "---- /etc/hosts ----"
cat /etc/hosts 2>/dev/null || true
echo
echo "---- npu ----"
npu-smi info -l 2>/dev/null || npu-smi info 2>/dev/null | head -20 || true
ls -d /dev/dav* 2>/dev/null | head -8 || true
echo
echo "---- filesystem conventions ----"
echo "MA_CODE_DIR=${MA_CODE_DIR:-<unset>}"
for d in /home/ma-user/modelarts/inputs \
         /home/ma-user/modelarts/outputs \
         /home/ma-user/modelarts/user-job-dir; do
  echo "-- $d:"; ls "$d" 2>/dev/null | head -15 || echo "   (missing)"
done
echo
echo "---- env (secrets masked) ----"
env | sort | mask_secrets
echo
echo "---- python / torch ----"
python3 -c 'import sys; print("python", sys.version.split()[0])' 2>/dev/null
python3 -c 'import torch; print("torch", torch.__version__)' 2>/dev/null
python3 -c 'import torch_npu; print("torch_npu", torch_npu.__version__)' 2>/dev/null
which torchrun || true
} > "$LOG" 2>&1

log "[probe A] dump written to $LOG"

# ---- rendezvous primitive test: can nodes see each other's output files? --
RDZV_DIR="$OUT/rdzv_markers"
mkdir -p "$RDZV_DIR"
MY_IPS=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "${MY_IPS:-unknown}" > "$RDZV_DIR/node_$(hostname)"
start=$(date +%s)
seen=1
while [ $(( $(date +%s) - start )) -lt "$RDZV_TIMEOUT" ]; do
  seen=$(ls "$RDZV_DIR"/node_* 2>/dev/null | wc -l)
  [ "$seen" -ge "$NNODES" ] && break
  sleep 5
done
elapsed=$(( $(date +%s) - start ))
if [ "$seen" -ge "$NNODES" ]; then
  verdict="visible"
else
  verdict="NOT-visible (saw $seen/$NNODES after ${elapsed}s)"
fi
log "[probe A] rendezvous marker test: $verdict"

# compact machine-readable summary
python3 - "$SUMMARY" "$NNODES" "$seen" "$elapsed" "$verdict" <<'PYEOF'
import json
import os
import socket
import sys

summary_path, nnodes, seen, elapsed, verdict = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
    sys.argv[5])
try:
    import subprocess
    ips = subprocess.run(["hostname", "-I"], capture_output=True,
                         text=True).stdout.split()
except Exception:
    ips = []
summary = {
    "hostname": socket.gethostname(),
    "ips": ips,
    "nnodes_expected": nnodes,
    "rdzv_markers_seen": seen,
    "rdzv_poll_elapsed_s": elapsed,
    "rdzv_cross_node_output_visibility": verdict,
    # platform identity candidates (resolved values, None = unset)
    "platform_env": {
        "MASTER_ADDR": os.environ.get("MASTER_ADDR"),
        "MASTER_PORT": os.environ.get("MASTER_PORT"),
        "NODE_RANK": os.environ.get("NODE_RANK"),
        "WORLD_SIZE": os.environ.get("WORLD_SIZE"),
        "GROUP_RANK": os.environ.get("GROUP_RANK"),
        "TASK_INDEX": os.environ.get("TASK_INDEX"),
        "JOB_ID": os.environ.get("JOB_ID"),
        "MA_NUM_HOSTS": os.environ.get("MA_NUM_HOSTS"),
        "MA_HOST_RANK": os.environ.get("MA_HOST_RANK"),
        "VC_NODE_NUM": os.environ.get("VC_NODE_NUM"),
        "VC_TASK_INDEX": os.environ.get("VC_TASK_INDEX"),
    },
}
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print("[probe A] summary written to", summary_path)
PYEOF

log "[probe A] done"
exit 0
