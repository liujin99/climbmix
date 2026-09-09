#!/usr/bin/env bash
# Multi-node probe B: HCCL collective smoke + bandwidth measurement.
# Runs on every node; resolves rendezvous (probe_common.sh), then launches
# probe_hccl.py under torchrun with world_size = PROBE_NNODES * PROBE_NPROC.
# Rank 0 of node 0 writes hccl_result.json into the output mount.
set -u
CODE="${MA_CODE_DIR:-/home/ma-user/modelarts/user-job-dir}"
OUT="${PROBE_OUT:-/home/ma-user/modelarts/outputs/probe_out_0}"
export PROBE_OUT="$OUT"
mkdir -p "$OUT"

. "$CODE/probe_common.sh"

rdzv_resolve || {
  echo "[probe B] FATAL: rendezvous failed on $(hostname)" \
       > "$OUT/hccl_failure_$(hostname).json"
  exit 1
}

# Bound HCCL's own connect timeout (default ~20min): a hung node then
# fails at ~10min with EI0015 + breadcrumb evidence instead of freezing
# the job until the driver's runtime timeout cancels it.
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-600}"

# device-plane evidence for EI0015 postmortems (best-effort)
rdzv_dump_host_nets > "$OUT/host_nets_$(hostname).log" 2>&1 || true

cd "$CODE"
exec torchrun $(rdzv_torchrun_argv) "$CODE/probe_hccl.py" \
  > "$OUT/hccl_torchrun_$(hostname).log" 2>&1
