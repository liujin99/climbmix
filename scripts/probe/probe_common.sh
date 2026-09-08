#!/usr/bin/env bash
# Shared rendezvous resolution for probes B (HCCL) and C (training).
# Sourced, not executed. Resolves a (master_addr, master_port, node_rank)
# triple for a torchrun multi-node launch and exports:
#   RDZV_MODE         platform | output-sync | static
#   RDZV_MASTER_ADDR  reachable IP of the rank-0 node
#   RDZV_MASTER_PORT  free TCP port on it
#   RDZV_NODE_RANK    this node's rank in [0, PROBE_NNODES)
#
# Env contract (driver sets; PROBE_OUT/PROBE_NNODES are mandatory):
#   PROBE_OUT         container output mount dir
#   PROBE_NNODES      node count of the job
#   PROBE_NPROC       NPUs per node (default 8)
#   PROBE_RDZV_MODE   auto | platform | output-sync | static
#                     (auto: platform first, then output-sync, then static)
#   PROBE_MASTER_ADDR / PROBE_MASTER_PORT / PROBE_NODE_RANK  (static mode)
#   PROBE_MASTER_PORT fallback port for platform/output-sync (default 29500)
#   PROBE_RDZV_TIMEOUT  election poll seconds (default 240)
#   PROBE_IP_OVERRIDE   use this IP instead of hostname -I's first
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

rdzv_resolve() {
  local mode="${PROBE_RDZV_MODE:-auto}"
  case "$mode" in
    platform)
      rdzv_try_platform || { rdzv_log "FATAL: no platform MASTER_* env"; return 1; }
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
      rdzv_try_platform || rdzv_elect_output_sync || {
        rdzv_log "FATAL: no platform env and output-sync election failed"
        return 1
      }
      ;;
    *)
      rdzv_log "FATAL: unknown PROBE_RDZV_MODE=$mode"
      return 1
      ;;
  esac
  export RDZV_MODE RDZV_MASTER_ADDR RDZV_MASTER_PORT RDZV_NODE_RANK
  # audit trail: what this node resolved (merges into the job's OBS output)
  if [ -n "${PROBE_OUT:-}" ]; then
    printf '{"hostname":"%s","mode":"%s","master":"%s","port":"%s","rank":"%s","my_ip":"%s"}\n' \
      "$(hostname)" "$RDZV_MODE" "$RDZV_MASTER_ADDR" "$RDZV_MASTER_PORT" \
      "$RDZV_NODE_RANK" "$(rdzv_my_ip)" \
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
