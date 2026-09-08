#!/usr/bin/env python3
"""Multi-node probe driver (Phase 0 of the multi-node target-arm project).

Submits three probes against the REAL platform backend, in order, and
prints a go/no-go decision table at the end:

  A  identity    node_count=N job running a per-node env dump: does the
                 gateway accept N>1, does the command run on every node,
                 what node-identity env exists, is the output mount
                 cross-node visible (rendezvous primitive)
  B  HCCL        ws = N x 8 init_process_group(hccl) + 1GiB allreduce
                  bandwidth + latency. Rendezvous mode chosen from A's
                  findings (platform env > k8s StatefulSet DNS >
                  output-sync election > static)
  C  training    50-step real mid_train at ws=16 (2 nodes), dbs=1,
                  --load-optimizer=0 — measures s/step vs the 18.2s ws=8
                  anchor, peak memory, and cross-node checkpoint shards.
                  Opt-in: --with-train (needs --d28-uri/--tokenizer-uri/
                  --data-dir/--nanochat-dir). The vllm image lacks deps
                  nanochat needs (rustbpe, pyarrow): the driver stages
                  the run's wheels into the code dir (read-only share of
                  {obs_prefix}/assets) and probe_train.sh runs the same
                  offline dep boot the worker path uses.

Runs against a DEDICATED probe prefix (--obs-prefix, no default: probes
must never touch a production run's prefix). Requires a backend whose
job_api implements submit_raw (the climbmix-ma adapter; node_count
support >= the adapter commit that adds the parameter).

Usage (server, DEDICATED throwaway clone — never the running run's dir):
  cd ~/work
  git clone https://github.com/liujin99/climbmix.git climbmix-probe
  cd climbmix-probe
  git clone https://github.com/liujin99/climbmix-ma.git climbmix-ma-probe
  PYTHONPATH=$PWD/climbmix-ma-probe \
  python3 scripts/probe/run_multinode_probes.py \
      --remote-config /home/ma-user/work/climbmix/result/<run>/remote_config.json \
      --obs-prefix obs://bucket/.../probe_multinode_<ts> \
      --node-count 2
  # + probe C (pick a >=2*8-card idle window):
      --with-train --d28-uri obs://.../climbmix_resource_package/d28 \
      --tokenizer-uri obs://.../climbmix_resource_package/tokenizer \
      --data-dir <local dir with shard_*.parquet> \
      --nanochat-dir ~/work/nanochat-npu

Isolation: everything the probes write lives under --obs-prefix (a
DEDICATED probe prefix); the remote_config's obs_prefix is IGNORED.
Embedding caches, cluster caches, the production run's prefix and its
local result/ dirs are never touched (probe C READS the shared d28 /
tokenizer / resource-package assets via input mounts — read-only).
Scope: multi-node is for the 1.5B (d28) TARGET ARMS ONLY — the search
fleet, embedding waves, and proxy experiments stay single-node by
design (wave packing / independent jobs / cheap failure).

Safe to re-run: every probe's artifacts live under {prefix}/probe_X/<ts>/,
so re-runs never mix with earlier results (and --skip-a/--skip-b skip
earlier probes outright).
"""

import argparse
import importlib.util
import json
import os
import sys
import tarfile
import tempfile
import time
from typing import Dict, List, Optional

REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from climbmix.remote.backends import resolve_backend  # noqa: E402
from climbmix.remote.remote_executor import RemoteConfig  # noqa: E402

PROBE_DIR = os.path.join(REPO_ROOT, "scripts", "probe")
# container-side path conventions (mirror the climbmix-ma adapter's
# INPUT_BASE/OUTPUT_BASE/LOCAL_CODE_DIR; overridable for mock sims)
DEFAULT_INPUT_BASE = "/home/ma-user/modelarts/inputs"
DEFAULT_OUTPUT_BASE = "/home/ma-user/modelarts/outputs"
DEFAULT_CODE_DIR = "/home/ma-user/modelarts/user-job-dir"

# known platform identity env triplets, priority order (probe A scans for
# these; anything the platform actually injects shows up in its env dump)
PLATFORM_TRIPLETS = ("", "MA_", "VC_")


def load_spec(module_path: str, attr: str):
    """Import a function from a script by path (no package needed)."""
    spec = importlib.util.spec_from_file_location(
        "_probe_mod_" + os.path.basename(module_path).replace(".", "_"),
        module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, attr)


def parse_npu_env_block_shim():
    """dispatch_target_arm.parse_npu_env_block (the d28-proven env)."""
    src = os.path.join(REPO_ROOT, "scripts", "dispatch_target_arm.py")
    return load_spec(src, "parse_npu_env_block")


def build_probe_env(nnodes: int, nproc: int, rdzv_mode: str,
                    npu_env_script: str, extra: Optional[Dict[str, str]] = None
                    ) -> Dict[str, str]:
    """Job env for probes B/C: the npu_env block + rendezvous contract +
    the cross-node HCCL knobs the arm jobs use (dispatch_target_arm
    build_spec_env mirrors these)."""
    env = parse_npu_env_block_shim()(npu_env_script)
    env.update({
        "PROBE_NNODES": str(nnodes),
        "PROBE_NPROC": str(nproc),
        "PROBE_RDZV_MODE": rdzv_mode,
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "offline",
        "ASCEND_GLOBAL_LOG_LEVEL": "3",
        "PYTHONUNBUFFERED": "1",
        "NANOCHAT_DTYPE": "bfloat16",
        "PYTHONWARNINGS": "ignore::UserWarning:torch_npu",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_WHITELIST_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "HCCL_EXEC_TIMEOUT": "1200",
    })
    env.update(extra or {})
    return env


def shell_wrap(script: str) -> str:
    """The command string: resolve the staged code dir, run the script."""
    return (f'CODE=${{MA_CODE_DIR:-{DEFAULT_CODE_DIR}}}; '
            f'bash "$CODE/{script}"')


# ── submit + wait ──────────────────────────────────────────────────────────

def submit_probe(job_api, name: str, command: str, env: Dict[str, str],
                 prefix: str, tag: str, node_count: int, run_ts: str,
                 inputs: Optional[List[Dict]] = None,
                 input_base: str = DEFAULT_INPUT_BASE,
                 output_base: str = DEFAULT_OUTPUT_BASE) -> str:
    outputs = [{
        "name": "probe_out",
        "local_dir": f"{output_base}/probe_out_0",
        "access_method": "env",
        # per-run subdir: re-runs against the same prefix never mix old
        # and new node summaries in one result dir
        "remote": {"obs": {
            "obs_url": f"{prefix.rstrip('/')}/probe_{tag}/{run_ts}/result"}},
    }]
    ins = []
    for e in (inputs or []):
        e = dict(e)
        e["local_dir"] = f"{input_base}/{e['name']}_0"
        e.setdefault("access_method", "env")
        ins.append(e)
    print(f"  [{tag}] submitting {name} (node_count={node_count})...", flush=True)
    # code_dir MUST be the probe prefix's assets (the gateway validates it
    # as an existing OBS dir; submit_raw's default is the RemoteConfig's
    # obs_prefix — a PRODUCTION prefix we must never touch)
    job_id = job_api.submit_raw(name, command, env=env,
                                inputs=ins or None, outputs=outputs,
                                code_dir=f"{prefix.rstrip('/')}/assets/",
                                node_count=node_count)
    print(f"  [{tag}] job {job_id}", flush=True)
    return job_id


def wait_probe(job_api, job_id: str, tag: str, queue_timeout_s: float,
               runtime_timeout_s: float, poll_s: float) -> str:
    """Returns 'SUCCEEDED' | 'FAILED' | 'CANCELLED' | 'TIMEOUT'."""
    from climbmix.remote.job_api import JobStatus
    submitted = time.time()
    first_running = None
    while True:
        st = job_api.status(job_id)
        now = time.time()
        if st == JobStatus.RUNNING and first_running is None:
            first_running = now
            print(f"  [{tag}] RUNNING ({(now - submitted)/60:.0f}m queued)",
                  flush=True)
        if st.is_terminal:
            print(f"  [{tag}] terminal: {st.value}", flush=True)
            return st.value
        if first_running is None:
            if now - submitted > queue_timeout_s:
                job_api.cancel(job_id)
                print(f"  [{tag}] QUEUE TIMEOUT after "
                      f"{(now - submitted)/60:.0f}m — cancelled", flush=True)
                return "TIMEOUT"
            phase = "pending"
        else:
            if now - first_running > runtime_timeout_s:
                job_api.cancel(job_id)
                print(f"  [{tag}] RUNTIME TIMEOUT after "
                      f"{(now - first_running)/60:.0f}m — cancelled", flush=True)
                return "TIMEOUT"
            phase = f"running {(now - first_running)/60:.0f}m"
        print(f"  [{tag}] {phase} (job {job_id})", flush=True)
        time.sleep(poll_s)


def download_results(obs, uri: str, local_dir: str, wait_s: float = 300.0,
                     poll_s: float = 15.0, min_files: int = 1,
                     tag: str = "") -> List[str]:
    """Pull {uri}/** into local_dir, tolerating the output-sync lag:
    poll until at least min_files objects exist or wait_s elapses."""
    os.makedirs(local_dir, exist_ok=True)
    deadline = time.time() + wait_s
    pulled: List[str] = []
    while True:
        for obj in obs.list_objects(uri):
            rel = obj[len(uri.rstrip("/")):].lstrip("/")
            if not rel:
                continue
            dst = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            obs.download_file(obj, dst)
            pulled.append(dst)
        if len(pulled) >= min_files or time.time() > deadline:
            if not pulled:
                print(f"  [{tag}] WARN: no result files at {uri} after "
                      f"{wait_s:.0f}s", flush=True)
            return pulled
        print(f"  [{tag}] waiting for result files ({len(pulled)}/{min_files})"
              "...", flush=True)
        pulled = []
        time.sleep(poll_s)


# ── analysis ───────────────────────────────────────────────────────────────

def analyze_probe_a(local_dir: str, node_count: int) -> Dict:
    summaries = []
    if os.path.isdir(local_dir):
        for fn in sorted(os.listdir(local_dir)):
            if fn.startswith("node_summary_") and fn.endswith(".json"):
                try:
                    summaries.append(
                        json.load(open(os.path.join(local_dir, fn))))
                except (OSError, ValueError):
                    pass
    nodes_seen = len(summaries)
    out = {
        "nodes_seen": nodes_seen,
        "node_count_ok": nodes_seen >= node_count,
        "platform_triplet": None,
        "output_sync_visible": False,
        "ips": [],
        "sts_dns_ok": False,
        "sts_master_addr": None,
    }
    if not summaries:
        return out
    # platform triplet: present AND consistent on ALL nodes
    for p in PLATFORM_TRIPLETS:
        keys = (f"{p}MASTER_ADDR", f"{p}MASTER_PORT", f"{p}NODE_RANK")
        vals = [tuple(s.get("platform_env", {}).get(k) for k in keys)
                for s in summaries]
        if all(all(v is not None for v in t) for t in vals):
            out["platform_triplet"] = p
            break
    out["output_sync_visible"] = all(
        str(s.get("rdzv_cross_node_output_visibility", "")).startswith("visible")
        for s in summaries)
    out["ips"] = sorted({ip for s in summaries for ip in s.get("ips", [])})
    # k8s StatefulSet DNS rendezvous: pattern matched AND every worker
    # FQDN resolved on every node (a sibling resolution is the real
    # cross-node DNS test — own name may come from /etc/hosts alone)
    sts = [s.get("sts_dns") or {} for s in summaries]
    out["sts_dns_ok"] = all(bool(t.get("ok")) for t in sts)
    out["sts_master_addr"] = next(
        (t.get("master_addr") for t in sts if t.get("master_addr")), None)
    return out


def choose_rdzv_mode(a: Dict, args) -> str:
    if a["platform_triplet"] is not None:
        return "platform"
    if a.get("sts_dns_ok"):
        return "dns-sts"
    if a["output_sync_visible"]:
        return "output-sync"
    if args.master_addr:
        return "static"
    return "none"


def analyze_probe_b(local_dir: str) -> Optional[Dict]:
    path = os.path.join(local_dir, "hccl_result.json")
    if not os.path.isfile(path):
        return None
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def analyze_probe_c(local_dir: str) -> List[Dict]:
    out = []
    for fn in sorted(os.listdir(local_dir)):
        if fn.startswith("train_result_") and fn.endswith(".json"):
            try:
                out.append(json.load(open(os.path.join(local_dir, fn))))
            except (OSError, ValueError):
                pass
    return out


def stage_wheels(obs, sources: List[str], prefix: str,
                 tag: str = "C") -> List[str]:
    """Stage .whl files into {prefix}/assets/ (the job's code dir).

    Sources are OBS dir URIs (every *.whl under them), single .whl OBS
    URIs, or local .whl paths. The vllm-ascend image lacks deps the
    nanochat toolchain needs (rustbpe AND pyarrow both hit
    ModuleNotFoundError live); the worker boot installs them offline
    from the code dir — probe C's shell path has no boot, so the driver
    stages the wheels and probe_train.sh runs the same dep loop."""
    staged: List[str] = []
    for src in sources:
        dst_base = f"{prefix.rstrip('/')}/assets"
        if src.startswith("obs://"):
            objs = ([src] if src.endswith(".whl")
                    else [o for o in obs.list_objects(src)
                          if o.endswith(".whl")])
            for obj in objs:
                name = obj.rsplit("/", 1)[-1]
                tmp = os.path.join(tempfile.gettempdir(),
                                   f"probe_wheel_{name}")
                obs.download_file(obj, tmp)
                obs.upload_file(tmp, f"{dst_base}/{name}")
                staged.append(name)
        else:
            path = os.path.abspath(os.path.expanduser(src))
            if not (os.path.isfile(path) and path.endswith(".whl")):
                raise SystemExit(f"--wheel-from: not a .whl file: {path}")
            obs.upload_file(path,
                            f"{dst_base}/{os.path.basename(path)}")
            staged.append(os.path.basename(path))
    if staged:
        print(f"[{tag}] wheels -> assets: {', '.join(staged)}")
    return staged


def busbw_verdict(busbw: Optional[float]) -> str:
    if busbw is None:
        return "FAIL (no data)"
    if busbw < 2:
        return ("FAIL — ethernet-like; cross-node optimizer sync (~15GB/rank/"
                "step) would add ~10s+/step. Multi-node dead on this network.")
    if busbw < 10:
        return ("MARGINAL — partial RoCE; expect a visible per-step tax. "
                "Probe C's s/step decides.")
    return "PASS — RoCE-class interconnect."


def train_verdict(mean_dt_ms: Optional[float], ws: int) -> str:
    if mean_dt_ms is None:
        return "FAIL (no timing data)"
    s = mean_dt_ms / 1000.0
    base = 18.2
    thr = 11000 if ws == 16 else 7000
    if mean_dt_ms <= thr:
        return (f"PASS — {s:.1f}s/step vs {base}s ws=8 anchor "
                f"({base / s:.1f}x).")
    return (f"FAIL — {s:.1f}s/step > {thr / 1000.0}s gate for ws={ws} "
            f"({base / s:.1f}x vs ws=8 anchor).")


def print_decision_table(a: Optional[Dict], b: Optional[Dict],
                         c: List[Dict], node_count: int, rdzv_mode: str,
                         ws: int) -> None:
    print("\n" + "=" * 72)
    print("MULTI-NODE PHASE 0 DECISION TABLE")
    print("=" * 72)
    if a is not None:
        print(f"A  gateway accepted node_count={node_count}, command on "
              f"every node ......... "
              f"{'PASS' if a['node_count_ok'] else 'FAIL'} "
              f"({a['nodes_seen']} node dumps)")
        print(f"A  platform identity env .......................... "
              f"{a['platform_triplet'] or 'none found'}")
        print(f"A  k8s sts-dns rendezvous (worker FQDN) ........... "
              f"{'YES' if a.get('sts_dns_ok') else 'NO'}"
              + (f" (master={a['sts_master_addr']})"
                 if a.get('sts_dns_ok') and a.get('sts_master_addr')
                 else ""))
        print(f"A  cross-node output visibility (rdzv primitive) .. "
              f"{'YES' if a['output_sync_visible'] else 'NO'}")
        print(f"A  rendezvous mode chosen for B/C .................. "
              f"{rdzv_mode}")
    if b is not None:
        ok = b.get("init_ok")
        print(f"B  HCCL init (ws={b.get('world_size')}) .................... "
              f"{'PASS' if ok else 'FAIL'}"
              + (f" ({b.get('error', '')[:60]})" if not ok else ""))
        if ok:
            print(f"B  1GiB allreduce .................................... "
                  f"{b.get('allreduce_ms')} ms | busbw "
                  f"{b.get('busbw_gib_s')} GiB/s")
            print(f"B  small allreduce / barrier ........................ "
                  f"{b.get('small_allreduce_ms')} ms / "
                  f"{b.get('barrier_ms')} ms")
            print(f"B  interconnect verdict ............................ "
                  f"{busbw_verdict(b.get('busbw_gib_s'))}")
    for i, tr in enumerate(c):
        print(f"C  node {i} ({tr.get('hostname')}) rc={tr.get('rc')} "
              f"steps={tr.get('steps_logged')} "
              f"mean_dt={tr.get('mean_dt_ms_last30')}ms "
              f"ckpt_shards={len(tr.get('ckpt_files', []))}")
    if c:
        dts = [t["mean_dt_ms_last30"] for t in c
               if t.get("mean_dt_ms_last30")]
        rcs = [t.get("rc") for t in c]
        mean = sum(dts) / len(dts) if dts else None
        print(f"C  training verdict (ws={ws}) ....................... "
              f"{train_verdict(mean, ws)}")
        if all(r == 0 for r in rcs):
            n_shards = sum(len(t.get("ckpt_files", [])) for t in c)
            print(f"C  optim shards landed across nodes ................ "
                  f"{n_shards} files "
                  f"({'PASS' if n_shards > 0 else 'FAIL'})")
    print("=" * 72)
    go = []
    if a is not None and a["node_count_ok"]:
        go.append(True)
    if b is not None:
        go.append(bool(b.get("init_ok")) and
                  (b.get("busbw_gib_s") or 0) >= 2)
    if c:
        dts = [t["mean_dt_ms_last30"] for t in c
               if t.get("mean_dt_ms_last30")]
        thr = 11000 if ws == 16 else 7000
        go.append(all(t.get("rc") == 0 for t in c)
                  and bool(dts) and (sum(dts) / len(dts)) <= thr)
    verdict = "GO" if go and all(go) else (
        "CONDITIONAL" if go and any(go) else "NO-GO")
    print(f"OVERALL: {verdict}  "
          f"({'all gates green' if verdict == 'GO' else 'see rows above'})")
    print("=" * 72)


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="multi-node Phase 0 probes (A/B/C)")
    ap.add_argument("--remote-config", required=True,
                    help="RemoteConfig JSON (backend resolution + prefix)")
    ap.add_argument("--obs-prefix", required=True,
                    help="DEDICATED probe prefix (never a production run's)")
    ap.add_argument("--node-count", type=int, default=2)
    ap.add_argument("--nproc", type=int, default=8,
                    help="NPUs per node (flavor must match)")
    ap.add_argument("--skip-a", action="store_true")
    ap.add_argument("--skip-b", action="store_true")
    ap.add_argument("--with-train", action="store_true",
                    help="run probe C (needs d28/tokenizer/data/nanochat)")
    ap.add_argument("--master-addr", default="",
                    help="static rendezvous master IP (last-resort mode)")
    ap.add_argument("--rdzv-mode", default="",
                    help="force platform|dns-sts|output-sync|static "
                         "(default: auto from probe A)")
    ap.add_argument("--poll-s", type=float, default=30.0)
    ap.add_argument("--queue-timeout-min", type=float, default=60.0)
    ap.add_argument("--runtime-timeout-min-a", type=float, default=15.0)
    ap.add_argument("--runtime-timeout-min-b", type=float, default=15.0)
    ap.add_argument("--runtime-timeout-min-c", type=float, default=60.0)
    ap.add_argument("--probe-steps", type=int, default=50,
                    help="probe C training steps")
    # probe C asset/data inputs
    ap.add_argument("--d28-uri", default="",
                    help="OBS dir of the d28 base ckpt (model only is fine)")
    ap.add_argument("--tokenizer-uri", default="")
    ap.add_argument("--data-dir", default="",
                    help="local dir with shard_*.parquet for probe C")
    ap.add_argument("--max-shards", type=int, default=4,
                    help="cap parquet uploads for probe C (0 = all)")
    ap.add_argument("--nanochat-dir", default="",
                    help="local nanochat-npu checkout to tar for probe C")
    ap.add_argument("--nanochat-tar", default="",
                    help="existing nanochat-npu.tar.gz (alternative to "
                         "--nanochat-dir)")
    ap.add_argument("--wheel-from", action="append", default=None,
                    metavar="OBS_OR_PATH",
                    help="stage .whl files into the probe code dir for the "
                         "offline dep boot (OBS dir URI, single .whl OBS "
                         "URI, or local .whl path; repeatable). Default: "
                         "{remote_config obs_prefix}/assets — a READ-ONLY "
                         "share of the run's static wheels, same class as "
                         "the d28/tokenizer input mounts")
    ap.add_argument("--no-wheels", action="store_true",
                    help="skip wheel staging (probe_train.sh then relies "
                         "on pip / the internal mirror fallbacks)")
    ap.add_argument("--npu-env-script", default=os.path.join(
        REPO_ROOT, "runs", "lib", "npu_env.sh"))
    ap.add_argument("--download-dir", default="",
                    help="local dir for downloaded results (default: "
                         "cache/probe_multinode/<ts>)")
    args = ap.parse_args()

    if args.node_count < 2:
        raise SystemExit("--node-count must be >= 2 (multi-node probe)")

    remote_config = RemoteConfig.from_json_file(args.remote_config)
    bundle = resolve_backend(remote_config)
    job_api = bundle.make_job_api(remote_config)
    obs = bundle.make_obs_storage(remote_config)
    if not hasattr(job_api, "submit_raw"):
        raise SystemExit("backend job_api has no submit_raw — probes need "
                         "the climbmix-ma adapter (node_count support)")
    input_base = (getattr(bundle, "container_input_base", "")
                  or DEFAULT_INPUT_BASE)

    prefix = args.obs_prefix.rstrip("/")
    ts = time.strftime("%Y%m%d_%H%M%S")
    dl_root = args.download_dir or os.path.join(
        REPO_ROOT, "cache", "probe_multinode", ts)
    os.makedirs(dl_root, exist_ok=True)

    # ── upload probe scripts -> {prefix}/assets (the job's code dir) ──
    scripts = ["probe_env_dump.sh", "probe_common.sh", "probe_hccl.sh",
               "probe_hccl.py", "probe_train.sh"]
    for s in scripts:
        src = os.path.join(PROBE_DIR, s)
        if not os.path.isfile(src):
            raise SystemExit(f"probe script missing: {src}")
        obs.upload_file(src, f"{prefix}/assets/{s}")
    print(f"[probes] scripts -> {prefix}/assets")

    ws = args.node_count * args.nproc
    a = b = None
    c: List[Dict] = []
    rdzv_mode = args.rdzv_mode

    # ── probe A ──
    if not args.skip_a:
        env = {"PROBE_NNODES": str(args.node_count),
               "PROBE_RDZV_TIMEOUT": "90"}
        try:
            jid = submit_probe(job_api, f"probe-mn-a-{ts}",
                               shell_wrap("probe_env_dump.sh"), env,
                               prefix, "a", args.node_count, ts,
                               input_base=input_base)
        except RuntimeError as e:
            print(f"[A] SUBMIT REJECTED — the gateway/pool refused the "
                  f"multi-node job body (node_count={args.node_count}):\n"
                  f"    {e}\n"
                  f"    -> multi-node is NO-GO at the platform layer; check "
                  f"the error code against the backend's config\n"
                  f"       (status_map/transient patterns) before concluding.")
            print_decision_table(None, None, c, args.node_count,
                                 "n/a", ws)
            return 1
        st = wait_probe(job_api, jid, "a",
                        args.queue_timeout_min * 60,
                        args.runtime_timeout_min_a * 60, args.poll_s)
        files = download_results(
            obs, f"{prefix}/probe_a/{ts}/result", os.path.join(dl_root, "a"),
            min_files=args.node_count, tag="a")
        a = analyze_probe_a(os.path.join(dl_root, "a"), args.node_count)
        print(f"[A] {json.dumps(a, ensure_ascii=False)}")
        if st != "SUCCEEDED" or not a["node_count_ok"]:
            print(f"[A] probe A did not succeed cleanly (status={st}) — "
                  f"stopping before B/C")
            print_decision_table(a, None, c, args.node_count,
                                 rdzv_mode or "n/a", ws)
            return 1
        if not rdzv_mode:
            rdzv_mode = choose_rdzv_mode(a, args)
    else:
        print("[A] skipped (--skip-a)")
        if not rdzv_mode:
            rdzv_mode = "auto"

    # ── probe B ──
    if not args.skip_b:
        if rdzv_mode == "none":
            print("[B] SKIP: no rendezvous strategy (no platform env, no "
                  "sts-dns, no cross-node output visibility, no "
                  "--master-addr)")
            print_decision_table(a, None, c, args.node_count, rdzv_mode, ws)
            return 1
        env = build_probe_env(args.node_count, args.nproc, rdzv_mode,
                              args.npu_env_script)
        if rdzv_mode == "static":
            env["PROBE_MASTER_ADDR"] = args.master_addr
            env["PROBE_MASTER_PORT"] = "29500"
        jid = submit_probe(job_api, f"probe-mn-b-{ts}",
                           shell_wrap("probe_hccl.sh"), env,
                           prefix, "b", args.node_count, ts,
                           input_base=input_base)
        st = wait_probe(job_api, jid, "b",
                        args.queue_timeout_min * 60,
                        args.runtime_timeout_min_b * 60, args.poll_s)
        download_results(obs, f"{prefix}/probe_b/{ts}/result",
                         os.path.join(dl_root, "b"), min_files=1, tag="b")
        b = analyze_probe_b(os.path.join(dl_root, "b"))
        print(f"[B] {json.dumps(b, ensure_ascii=False)}")
        if st != "SUCCEEDED" or not (b or {}).get("init_ok"):
            print(f"[B] HCCL probe failed (status={st}) — stopping before C")
            print_decision_table(a, b, c, args.node_count, rdzv_mode, ws)
            return 1
    else:
        print("[B] skipped (--skip-b)")

    # ── probe C ──
    if args.with_train:
        missing = [k for k, v in (
            ("--d28-uri", args.d28_uri),
            ("--tokenizer-uri", args.tokenizer_uri),
            ("--data-dir", args.data_dir),
        ) if not v]
        if missing:
            raise SystemExit(f"probe C needs {missing}")
        # nanochat tar: reuse or build from a checkout
        tar_local = args.nanochat_tar
        if not tar_local:
            if not args.nanochat_dir:
                raise SystemExit("probe C needs --nanochat-dir or "
                                 "--nanochat-tar")
            print("[C] building nanochat tar from "
                  f"{args.nanochat_dir} (~300MB, one-time)")
            tar_local = build_nanochat_tar(args.nanochat_dir)
        obs.upload_file(tar_local, f"{prefix}/assets/nanochat-npu.tar.gz")
        # data: cap shards (per-run dir: re-runs never mix)
        data_uri = f"{prefix}/probe_c/{ts}/data"
        shards = sorted(
            f for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
        if args.max_shards:
            shards = shards[:args.max_shards]
        if not shards:
            raise SystemExit(f"no parquet shards in {args.data_dir}")
        for s in shards:
            obs.upload_file(os.path.join(args.data_dir, s),
                            f"{data_uri}/{s}")
        print(f"[C] uploaded {len(shards)} data shards + nanochat tar")
        # wheels for the offline dep boot (the vllm image lacks rustbpe/
        # pyarrow; the worker boot installs them from the code dir — the
        # probe's shell path needs them staged the same way)
        if not args.no_wheels:
            sources = args.wheel_from or [
                remote_config.obs_prefix.rstrip("/") + "/assets"]
            staged = stage_wheels(obs, sources, prefix, tag="C")
            if not staged:
                print(f"[C] WARN: no wheels found at {sources} — "
                      "probe_train.sh will fall back to pip/mirror "
                      "(rustbpe is NOT a pip-packageable dep; expect "
                      "ModuleNotFoundError)")
        env = build_probe_env(
            args.node_count, args.nproc, rdzv_mode, args.npu_env_script,
            extra={"PROBE_STEPS": str(args.probe_steps),
                   "PROBE_INPUT_BASE": input_base,
                   "PROBE_BASE_DIR": remote_config.container_base_dir,
                   "PROBE_NANOCHAT_DIR": remote_config.container_nanochat_dir})
        if rdzv_mode == "static":
            env["PROBE_MASTER_ADDR"] = args.master_addr
            env["PROBE_MASTER_PORT"] = "29500"
        inputs = [
            {"name": "d28", "access_method": "env",
             "remote": {"obs": {"obs_url": args.d28_uri}}},
            {"name": "tokenizer", "access_method": "env",
             "remote": {"obs": {"obs_url": args.tokenizer_uri}}},
            {"name": "data", "access_method": "env",
             "remote": {"obs": {"obs_url": data_uri}}},
        ]
        jid = submit_probe(job_api, f"probe-mn-c-{ts}",
                           shell_wrap("probe_train.sh"), env,
                           prefix, "c", args.node_count, ts, inputs=inputs,
                           input_base=input_base)
        st = wait_probe(job_api, jid, "c",
                        args.queue_timeout_min * 60,
                        args.runtime_timeout_min_c * 60, args.poll_s)
        download_results(obs, f"{prefix}/probe_c/{ts}/result",
                         os.path.join(dl_root, "c"),
                         min_files=args.node_count, tag="c")
        c = analyze_probe_c(os.path.join(dl_root, "c"))
    else:
        print("[C] skipped (pass --with-train to run it)")

    print_decision_table(a, b, c, args.node_count, rdzv_mode, ws)
    return 0


def build_nanochat_tar(nanochat_dir: str) -> str:
    """tar.gz of the nanochat-npu checkout (same layout ma_sync_code
    publishes: the repo root extracts to nanochat-npu/)."""
    nanochat_dir = os.path.abspath(nanochat_dir)
    if not os.path.isdir(os.path.join(nanochat_dir, "scripts")):
        raise SystemExit(f"not a nanochat-npu checkout: {nanochat_dir}")
    base = os.path.basename(nanochat_dir)
    if base != "nanochat-npu":
        raise SystemExit(
            f"checkout dir must be named 'nanochat-npu' (probe_train.sh "
            f"extracts into the container at that name): got {base!r}")
    out = os.path.join(tempfile.gettempdir(),
                       f"nanochat-npu-probe-{int(time.time())}.tar.gz")
    excluded_dirs = {".git", "__pycache__", ".ipynb_checkpoints",
                     "nanochat.egg-info"}
    with tarfile.open(out, "w:gz") as tf:
        for root, dirs, files in os.walk(nanochat_dir):
            dirs[:] = [d for d in dirs if d not in excluded_dirs]
            for f in files:
                full = os.path.join(root, f)
                tf.add(full, arcname=os.path.join(
                    base, os.path.relpath(full, nanochat_dir)))
    return out


if __name__ == "__main__":
    sys.exit(main())
