#!/usr/bin/env bash
# Shared rendezvous resolution for probes B (HCCL) and C (training).
# Sourced, not executed. Resolves a (master_addr, master_port, node_rank)
# triple for a torchrun multi-node launch and exports:
#   RDZV_MODE         platform | dns-sts | output-sync | static
#   RDZV_MASTER_ADDR  reachable IP of the rank-0 node
#   RDZV_MASTER_PORT  free TCP port on it
#   RDZV_NODE_RANK    this node's rank in [0, PROBE_NNODES)
#
# Env contract (driver sets; PROBE_OUT/PROBE_NNODES are mandatory):
#   PROBE_OUT         container output mount dir
#   PROBE_NNODES      node count of the job
#   PROBE_NPROC       NPUs per node (default 8)
#   PROBE_RDZV_MODE   auto | platform | dns-sts | output-sync | static
#                     (auto: platform, then k8s sts-dns, then output-sync)
#   PROBE_MASTER_ADDR / PROBE_MASTER_PORT / PROBE_NODE_RANK  (static mode)
#   PROBE_MASTER_PORT fallback port for non-static modes (default 29500)
#   PROBE_RDZV_TIMEOUT  election poll seconds (default 240)
#   PROBE_IP_OVERRIDE   use this IP instead of hostname -I's first
#   PROBE_HOSTS_FILE    hosts file for sts-dns derivation (default
#                       /etc/hosts; lookup is files-then-DNS, mirroring
#                       nsswitch — own FQDN hits the file, siblings hit
#                       the cluster DNS)
#
# Platform identity triplets, in priority order (probe A's env dump tells
# the driver which one actually exists; the driver pins PROBE_RDZV_MODE
# accordingly — this scan is the auto path):
#   MASTER_ADDR + MASTER_PORT + NODE_RANK     (torch-elastic style)
#   MA_MASTER_ADDR + MA_MASTER_PORT + MA_NODE_RANK
#   VC_MASTER_ADDR + VC_MASTER_PORT + VC_NODE_RANK
rdzv_log() { echo "[rdzv $(hostname)] $*"; }

rdzv_my_ip() {
  if [ -n "${PROBE_IP_OVERRIDE:-}" ]; then
    echo "$PROBE_IP_OVERRIDE"
  else
    hostname -I 2>/dev/null | awk '{print $1}'
  fi
}

# Sort dotted-quad IPs numerically (one per line).
rdzv_sort_ips() { sort -t. -k1,1n -k2,2n -k3,3n -k4,4n; }

rdzv_try_platform() {
  for p in "" MA_ VC_; do
    local a="${p}MASTER_ADDR" pt="${p}MASTER_PORT" r="${p}NODE_RANK"
    local addr="${!a:-}" port="${!pt:-}" rank="${!r:-}"
    if [ -n "$addr" ] && [ -n "$port" ] && [ -n "$rank" ]; then
      RDZV_MODE=platform
      RDZV_MASTER_ADDR="$addr"
      RDZV_MASTER_PORT="$port"
      RDZV_NODE_RANK="$rank"
      rdzv_log "platform env (${p:-plain}MASTER_*) -> $addr:$port rank=$rank"
      return 0
    fi
  done
  return 1
}

# Resolve a name to an IP: hosts file first, then DNS (mirrors nsswitch
# "files dns" — own pod FQDN comes from /etc/hosts, siblings from the
# cluster's per-pod DNS records).
rdzv_dns_lookup() {
  local name="$1" hosts="${PROBE_HOSTS_FILE:-/etc/hosts}" ip
  ip=$(awk -v n="$name" \
       '$1 !~ /^#/ && ($2 == n || $3 == n) {print $1; exit}' \
       "$hosts" 2>/dev/null)
  if [ -n "$ip" ]; then
    echo "$ip"
    return 0
  fi
  python3 - "$name" <<'PYLOOKUP' 2>/dev/null
import socket, sys
try:
    print(socket.gethostbyname(sys.argv[1]))
except Exception:
    sys.exit(1)
PYLOOKUP
}

# Kubernetes StatefulSet rendezvous: hostname is <svc>-worker-N and
# /etc/hosts carries the pod FQDN <host>.<svc>.<ns>.svc.cluster.local
# (k8s writes it for pods with hostname+subdomain set). With a governing
# headless service, per-pod DNS records resolve every worker FQDN from
# every node: master = worker-0 (resolved to an IP), rank = own worker
# number. Needs NO platform env and NO shared mount (probe A verifies
# the DNS capability; ModelArts runs jobs as exactly this StatefulSet).
rdzv_try_sts_dns() {
  local host svc rank fqdn domain master ip
  host=$(hostname)
  if [[ "$host" =~ ^(.*)-worker-([0-9]+)$ ]]; then
    svc="${BASH_REMATCH[1]}"
    rank="${BASH_REMATCH[2]}"
  else
    rdzv_log "sts-dns: hostname '$host' is not <svc>-worker-N"
    return 1
  fi
  fqdn=$(awk -v h="$host" \
         '((NF >= 3 && $3 == h) || (NF == 2 && $2 == h)) {print $2; exit}' \
         "${PROBE_HOSTS_FILE:-/etc/hosts}" 2>/dev/null)
  if [ -z "$fqdn" ] || [ "${fqdn#*.}" = "$fqdn" ]; then
    rdzv_log "sts-dns: no pod FQDN for '$host' in ${PROBE_HOSTS_FILE:-/etc/hosts}"
    return 1
  fi
  domain=${fqdn#*.}
  master="$svc-worker-0.$domain"
  ip=$(rdzv_dns_lookup "$master") || {
    rdzv_log "sts-dns: cannot resolve master $master"
    return 1
  }
  RDZV_MODE=dns-sts
  RDZV_MASTER_ADDR="$ip"
  RDZV_MASTER_PORT="${PROBE_MASTER_PORT:-29500}"
  RDZV_NODE_RANK="$rank"
  rdzv_log "sts-dns -> master $master ($ip):$RDZV_MASTER_PORT rank=$rank"
  return 0
}

# Election through the shared output mount: every node writes a marker with
# its IP, we poll until all NNODES markers exist, master = numerically
# smallest IP, my rank = position of my IP in the sorted list. Only works
# if the platform's output sync is bi-directional (probe A verified).
rdzv_elect_output_sync() {
  local my_ip timeout_s dir marker
  my_ip=$(rdzv_my_ip)
  timeout_s="${PROBE_RDZV_TIMEOUT:-240}"
  dir="$PROBE_OUT/rdzv_elect"
  marker="$dir/node_$(hostname)"
  mkdir -p "$dir"
  echo "$my_ip" > "$marker"
  local start=$(date +%s)
  while :; do
    local c
    c=$(ls "$dir"/node_* 2>/dev/null | wc -l)
    if [ "$c" -ge "${PROBE_NNODES:?}" ]; then break; fi
    if [ $(( $(date +%s) - start )) -gt "$timeout_s" ]; then
      rdzv_log "output-sync election TIMEOUT (saw $c/${PROBE_NNODES} markers)"
      return 1
    fi
    sleep 5
  done
  # FNR==1: first whitespace token of EACH marker (robust to missing
  # trailing newlines — cat would silently join files into one line)
  local sorted rank
  sorted=$(awk 'FNR==1{print $1}' "$dir"/node_* | rdzv_sort_ips)
  rank=$(printf '%s\n' "$sorted" | grep -n "^${my_ip}$" | head -1 | cut -d: -f1)
  rank=$(( rank - 1 ))
  RDZV_MODE=output-sync
  RDZV_MASTER_ADDR=$(printf '%s\n' "$sorted" | head -1)
  RDZV_MASTER_PORT="${PROBE_MASTER_PORT:-29500}"
  RDZV_NODE_RANK="$rank"
  rdzv_log "output-sync election -> master $RDZV_MASTER_ADDR:$RDZV_MASTER_PORT rank=$rank"
}

# Cross-node reachability check: connect to the master's rendezvous port
# BEFORE torchrun binds it. The honest pre-launch answer is "refused"
# (RST comes back: the network path is open, nothing is listening yet);
# "timeout" means packets are dropped — a node like that will hang the
# whole job exactly like the 20260908_201827 run (w3 -> w0, all healthy
# ranks die on HCCL EI0015 rank-list mismatch, the blocked node stays
# silent). Sets RDZV_TCP_RESULT/RDZV_TCP_RC/RDZV_TCP_MS; never fatal —
# the audit json carries the evidence.
rdzv_tcp_probe() {
  RDZV_TCP_RESULT=skipped RDZV_TCP_RC="" RDZV_TCP_MS=""
  [ -n "${PROBE_SKIP_TCP_PROBE:-}" ] && return 0
  local t0
  t0=$(date +%s%N)
  timeout "${PROBE_TCP_TIMEOUT:-5}" \
    bash -c "exec 3<>/dev/tcp/$RDZV_MASTER_ADDR/$RDZV_MASTER_PORT" 2>/dev/null
  RDZV_TCP_RC=$?
  RDZV_TCP_MS=$(( ( $(date +%s%N) - t0 ) / 1000000 ))
  case "$RDZV_TCP_RC" in
    0)   RDZV_TCP_RESULT=open ;;
    124) RDZV_TCP_RESULT=timeout ;;
    *)   RDZV_TCP_RESULT=refused ;;
  esac
  rdzv_log "tcp probe $RDZV_MASTER_ADDR:$RDZV_MASTER_PORT -> $RDZV_TCP_RESULT (rc=$RDZV_TCP_RC ${RDZV_TCP_MS}ms)"
}

# Best-effort device-plane evidence for HCCL EI0015 postmortems: the
# pod subnet in the rdzv json is NOT the fabric HCCL talks on. Device
# RoCE IPs live in /etc/hccn.conf (host file; usually invisible inside
# the container, but free to try) and npu-smi (bundled in some images).
# The EI0015 "rank num != rank list size" signature seen on 6.x/7.x
# pods implies duplicate/isolated device IPs -- this dump names them
# when the container can see them. Never fatal, always rc 0.
rdzv_dump_host_nets() {
  local f
  for f in /etc/hccn.conf /etc/ascend/ascend_install.info; do
    if [ -f "$f" ]; then
      echo "---- $f ----"
      cat "$f" 2>/dev/null || true
    fi
  done
  if command -v npu-smi >/dev/null 2>&1; then
    echo "---- npu-smi info ----"
    npu-smi info 2>/dev/null | head -40 || true
  fi
  if command -v ip >/dev/null 2>&1; then
    echo "---- ip -4 addr ----"
    ip -4 addr 2>/dev/null | grep -E '^[0-9]+:|inet ' || true
  fi
  return 0
}

# Materialize input-mount paths to LOCAL disk with a per-file timeout.
# Runs 20260908_201827 / _102807 / 20260909_1129xx: 1-2 nodes per job
# hang READING the obsfs input mounts (d28 model / tokenizer / parquet)
# while every other stage is provably healthy (wrap logs: device
# compute, store connect, init_process_group all ok) — the job then
# freezes ~20min until HCCL EI0015 kills the healthy ranks. Copy to
# local disk: a bad mount fails THIS node fast + visibly
# (train_failure marker) and training never touches obsfs. Glob in src;
# zero matches is a hard fail. Env: PROBE_MAT_TIMEOUT seconds per file
# (default 900).
probe_mat_cp() {
  local src_glob="$1" dst="$2" label="$3" n=0 t0 f
  t0=$(date +%s)
  mkdir -p "$dst"
  echo "[probe $(hostname)] materialize $label: $src_glob -> $dst"
  for f in $src_glob; do
    [ -f "$f" ] || continue
    if ! timeout "${PROBE_MAT_TIMEOUT:-900}" cp "$f" "$dst/" 2>/dev/null; then
      echo "[probe] FATAL: materialize $label FAILED on $(hostname): $f (read error or ${PROBE_MAT_TIMEOUT:-900}s timeout — obsfs?)" \
        | tee -a "${PROBE_OUT:-/tmp}/train_failure_$(hostname).log" >&2
      return 1
    fi
    n=$((n + 1))
  done
  if [ "$n" -eq 0 ]; then
    echo "[probe] FATAL: materialize $label matched NO files: $src_glob" \
      | tee -a "${PROBE_OUT:-/tmp}/train_failure_$(hostname).log" >&2
    return 1
  fi
  echo "[probe $(hostname)] $label: $n files local in $(( $(date +%s) - t0 ))s"
}

# Parquet pre-flight: open every shard, read metadata + EVERY row group,
# with a hard outer timeout. Run 20260909_154328 (heartbeat stacks):
# one node's 8 ranks spent 23min inside pyarrow ParquetFile.__init__ on
# the LOCAL materialized copies (~60s per read op; the 7.4GB model on
# the same disk read back in 9-14s from page cache) while healthy nodes
# finished in seconds — a sick per-node cold-read path. This check
# catches it BEFORE torchrun (visible train_failure marker, no 20min
# EI0015 burn) AND warms the page cache for every row group the
# distributed loaders will touch. Env: PROBE_PARQ_TOTAL (default 240s
# outer `timeout`), PROBE_PARQ_PER_FILE (default 60s soft, diagnostics).
probe_parquet_check() {
  local data_dir="$1"
  if [ ! -d "$data_dir" ]; then
    echo "[probe] FATAL: parquet pre-flight: no such dir: $data_dir" \
      | tee -a "${PROBE_OUT:-/tmp}/train_failure_$(hostname).log" >&2
    return 1
  fi
  timeout "${PROBE_PARQ_TOTAL:-240}" python3 - "$data_dir" \
      "${PROBE_PARQ_PER_FILE:-60}" <<'PYEOF'
import glob
import sys
import time

import pyarrow.parquet as pq

data_dir, per_file = sys.argv[1], float(sys.argv[2])
files = sorted(f for f in glob.glob(data_dir + "/*.parquet"))
assert files, "no .parquet files in " + data_dir
t_all = time.time()
for f in files:
    t0 = time.time()
    pf = pq.ParquetFile(f)
    n = pf.metadata.num_row_groups
    for rg in range(n):
        t = pf.read_row_group(rg)
        del t
    dt = time.time() - t0
    print(f"[probe parquet] {f}: {n} row groups in {dt:.1f}s", flush=True)
    if dt > per_file:
        sys.exit(f"SLOW: {f} took {dt:.0f}s (> {per_file:.0f}s budget)")
print(f"[probe parquet] OK: {len(files)} shards, all row groups "
      f"in {time.time() - t_all:.1f}s on {data_dir}")
PYEOF
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[probe] FATAL: parquet pre-flight FAILED on $(hostname) (rc=$rc) — sick local read path? (run 20260909_154328 crawled ~60s/op in ParquetFile.__init__ on LOCAL copies)" \
      | tee -a "${PROBE_OUT:-/tmp}/train_failure_$(hostname).log" >&2
    return 1
  fi
  return 0
}

rdzv_resolve() {
  local mode="${PROBE_RDZV_MODE:-auto}"
  case "$mode" in
    platform)
      rdzv_try_platform || { rdzv_log "FATAL: no platform MASTER_* env"; return 1; }
      ;;
    dns-sts)
      rdzv_try_sts_dns || { rdzv_log "FATAL: sts-dns derivation failed"; return 1; }
      ;;
    output-sync)
      rdzv_elect_output_sync || return 1
      ;;
    static)
      RDZV_MODE=static
      RDZV_MASTER_ADDR="${PROBE_MASTER_ADDR:?static mode needs PROBE_MASTER_ADDR}"
      RDZV_MASTER_PORT="${PROBE_MASTER_PORT:-29500}"
      RDZV_NODE_RANK="${PROBE_NODE_RANK:?static mode needs PROBE_NODE_RANK}"
      rdzv_log "static -> $RDZV_MASTER_ADDR:$RDZV_MASTER_PORT rank=$RDZV_NODE_RANK"
      ;;
    auto)
      rdzv_try_platform || rdzv_try_sts_dns || rdzv_elect_output_sync || {
        rdzv_log "FATAL: no platform env, sts-dns, or output-sync election"
        return 1
      }
      ;;
    *)
      rdzv_log "FATAL: unknown PROBE_RDZV_MODE=$mode"
      return 1
      ;;
  esac
  export RDZV_MODE RDZV_MASTER_ADDR RDZV_MASTER_PORT RDZV_NODE_RANK
  rdzv_tcp_probe
  # audit trail: what this node resolved (merges into the job's OBS output)
  if [ -n "${PROBE_OUT:-}" ]; then
    printf '{"hostname":"%s","mode":"%s","master":"%s","port":"%s","rank":"%s","my_ip":"%s","tcp":"%s","tcp_rc":"%s","tcp_ms":"%s"}\n' \
      "$(hostname)" "$RDZV_MODE" "$RDZV_MASTER_ADDR" "$RDZV_MASTER_PORT" \
      "$RDZV_NODE_RANK" "$(rdzv_my_ip)" \
      "$RDZV_TCP_RESULT" "${RDZV_TCP_RC:-}" "${RDZV_TCP_MS:-}" \
      > "$PROBE_OUT/rdzv_resolved_$(hostname).json" 2>/dev/null || true
  fi
  return 0
}

# Torchrun multi-node argv prefix (identical on every node; rank breaks the
# symmetry). $RDZV_* must be resolved first.
rdzv_torchrun_argv() {
  echo --nnodes="${PROBE_NNODES:?}" --node_rank="$RDZV_NODE_RANK" \
       --master_addr="$RDZV_MASTER_ADDR" --master_port="$RDZV_MASTER_PORT" \
       --nproc_per_node="${PROBE_NPROC:-8}"
}
