#!/usr/bin/env python3
"""dispatch_target_arm.py — run a d28 target arm (or the base anchor) as a
REMOTE 8-NPU job (prod2 B++).

Why remote arms: prod1's two arms ran sequentially on the master's 8 local
cards (~10h each, ~20h of the ~40h run). The search fleet already proved
the remote path (same pool, same image, same argv builders — the ONLY
difference vs a local experiment is WHERE it runs), so both arms become
remote jobs that OVERLAP the search (random) or each other's tails, and
the master's cards stay free for the k=8 local search slot.

Execution shapes (all marker-idempotent, safe to re-run):
  --arm random   full lifecycle: wait for the cluster cache + balanced
                 profile (the early nohup launch overlaps the search),
                 prepare the random-baseline data (flock'd against the
                 main script's Step 4/5 — both sides .done-double-check),
                 upload the mixture, submit, wait, land artifacts.
  --arm climb    data is already prepared by the main script's Steps 4-5
                 (verify + upload), then submit/wait/land.
  --arm base_eval_check
                 eval-only anchor job: evaluate the raw d28 base
                 checkpoint remotely (--model-type=base, the prod1-local
                 argv form) to verify the remote eval path reproduces the
                 local base score (0.1738). Lands eval_base_remote.csv;
                 touches NO arm markers (an anchor, not an arm).

Three-layer fallback contract with runs/run_climbmix.sh run_arm(): a
non-zero exit without .done_mid_train_<arm> makes the main script fall
back to the local torchrun path. A salvage path exists: a job whose
training succeeded but eval failed (result.json mid_train_rc == 0) lands
its checkpoint + logs + .done_mid_train_<arm> and exits non-zero — the
main script then skips retraining and evals locally.

Mutex: the per-arm lock ($OUTPUT_DIR/.dispatch_<arm>.lock) serializes
concurrent dispatches for the SAME arm (the early random dispatch and the
main script's Step-6 call); the second arriver sees the landed markers
and exits 0 without submitting. A prior FAILED attempt short-circuits
(exit 1, local fallback) unless --retry-failed.

Config sources (priority): CLI args > environment variables >
$OUTPUT_DIR/launch_env.json (written by run_climbmix.sh on every launch —
makes the separate nohup independent of the launching shell) +
$OUTPUT_DIR/remote_config.json (the search fleet's RemoteConfig).
"""

import argparse
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from climbmix.core.types import CLIMBConfig, DeviceConfig  # noqa: E402
from climbmix.remote.job_api import JobStatus, TransientSubmitError  # noqa: E402
from climbmix.pipeline.nanochat_cmds import (  # noqa: E402
    build_target_mid_train_cmd, build_target_eval_cmd)
from climbmix.remote.remote_executor import RemoteConfig, RemoteExecutor  # noqa: E402


# ── helpers ──────────────────────────────────────────────────────────────

def load_launch_env(output_dir: str) -> Dict[str, str]:
    path = os.path.join(output_dir, "launch_env.json")
    if not os.path.isfile(path):
        raise SystemExit(
            f"✗ launch_env.json not found at {path} — run via run_climbmix.sh "
            f"(it writes the snapshot every launch) or export the env vars")
    with open(path) as f:
        env = json.load(f)
    # CLI-time environment variables win (explicit override)
    for k in env:
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def parse_npu_env_block(path: str) -> Dict[str, str]:
    """Extract the export block of runs/lib/npu_env.sh (the d28-proven
    environment) as a dict — the remote arm job's spec.env carries exactly
    what the local subshell sources (TARGET_ENV_BLOCK=1 branch)."""
    env: Dict[str, str] = {}
    in_block = False
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("if ["):
                in_block = True
                continue
            if not in_block:
                continue
            if s == "else":
                in_block = False
                continue
            if s == "fi":
                break
            m = re.match(r"^export\s+([A-Za-z_][A-Za-z0-9_]*)=(.*)$", s)
            if m:
                k, v = m.group(1), m.group(2).strip()
                if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
                    v = v[1:-1]
                env[k] = v
    if not env:
        raise SystemExit(f"✗ could not parse any exports from {path}")
    return env


def build_spec_env(launch_env: Dict[str, str], job_env: Dict[str, str]) -> Dict[str, str]:
    """spec.env = npu_env block + the script-level exports that matter in
    the container (mirrors run_climbmix.sh's export block; node-level CANN
    paths are the container image's business — deliberately absent)."""
    env = parse_npu_env_block(
        os.path.join(launch_env.get("CLIMBMIX_DIR", REPO_ROOT),
                     "runs", "lib", "npu_env.sh"))
    env.update({
        "OMP_NUM_THREADS": "1",
        "WANDB_MODE": "offline",
        "ASCEND_GLOBAL_LOG_LEVEL": "3",
        "PYTHONUNBUFFERED": "1",
        "NANOCHAT_DTYPE": launch_env.get("NANOCHAT_DTYPE") or "bfloat16",
        "PYTHONWARNINGS": "ignore::UserWarning:torch_npu",
        "HCCL_CONNECT_TIMEOUT": "1200",
        "HCCL_WHITELIST_DISABLE": "1",
        "NCCL_IB_DISABLE": "1",
        "NCCL_SOCKET_IFNAME": "eth0",
        "HCCL_EXEC_TIMEOUT": "1200",
    })
    if launch_env.get("HF_ENDPOINT"):
        env["HF_ENDPOINT"] = launch_env["HF_ENDPOINT"]
    env.update({k: v for k, v in (job_env or {}).items() if v})
    return env


def run_logged(argv: List[str], env: Optional[Dict[str, str]] = None,
               label: str = "") -> None:
    print(f"  [{label}] $ {' '.join(argv)}" if label else f"  $ {' '.join(argv)}",
          flush=True)
    r = subprocess.run(argv, env=env)
    if r.returncode != 0:
        raise SystemExit(f"✗ command failed (rc={r.returncode}): {' '.join(argv)}")


def upload_dir_if_missing(obs, local_dir: str, obs_uri: str,
                          label: str) -> int:
    """Idempotent per-file upload (stat skip). Returns files uploaded."""
    if not os.path.isdir(local_dir):
        raise SystemExit(f"✗ {label}: local dir missing: {local_dir}")
    files = sorted(f for f in os.listdir(local_dir)
                   if os.path.isfile(os.path.join(local_dir, f)))
    if not files:
        raise SystemExit(f"✗ {label}: no files in {local_dir}")
    n = 0
    for f in files:
        uri = f"{obs_uri.rstrip('/')}/{f}"
        if not obs.stat(uri):
            obs.upload_file(os.path.join(local_dir, f), uri)
            n += 1
            print(f"  [{label}] uploaded {f} ({n}/{len(files)})", flush=True)
    if n == 0:
        print(f"  [{label}] all {len(files)} files already on OBS — skip")
    return n


def grep_node_info(console: str) -> List[str]:
    """NPU/node identity lines from the job console (audit record — arm
    scores are compared across nodes, so the node type is documented)."""
    seen: List[str] = []
    for line in console.splitlines():
        if re.search(r"910|Ascend|npu-smi|chip", line, re.I):
            line = line.strip()[:200]
            if line not in seen:
                seen.append(line)
        if len(seen) >= 5:
            break
    return seen


# ── job lifecycle ────────────────────────────────────────────────────────

class QueueTimedOut(Exception):
    pass


def submit_with_backoff(job_api, name: str, command: List[str],
                        env: Dict[str, str], asset_mounts: Optional[Dict[str, str]],
                        retry_timeout_s: float, label: str) -> str:
    backoff = 30.0
    deadline = time.time() + retry_timeout_s
    attempt = 0
    while True:
        attempt += 1
        try:
            return job_api.submit(name=name, command=command, env=env,
                                  asset_mounts=asset_mounts)
        except TransientSubmitError as e:
            if time.time() >= deadline:
                raise SystemExit(
                    f"✗ [{label}] submit still rejected after {attempt} "
                    f"attempts ({e}) — pool full for "
                    f"{retry_timeout_s/3600:.0f}h")
            print(f"  [{label}] submit rejected (attempt {attempt}: {e}) — "
                  f"backing off {backoff:.0f}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600.0)


def wait_job(job_api, obs, job_id: str, label: str, result_uri: str,
             poll_s: float, runtime_timeout_s: float,
             queue_timeout_s: float, heartbeat_s: float = 300.0):
    """Poll to terminal state. Two clocks (runtime from first RUNNING;
    queue patience from submission). Heartbeat prefers the WORKER's
    streamed log from OBS (real training progress) over the job console."""
    submitted_at = time.time()
    first_running: Optional[float] = None
    last_print = 0.0
    while True:
        st = job_api.status(job_id)
        if st.is_terminal:
            return st
        now = time.time()
        if first_running is None and st == JobStatus.RUNNING:
            first_running = now
        if first_running is None:
            if now - submitted_at > queue_timeout_s:
                job_api.cancel(job_id)
                raise QueueTimedOut(
                    f"[{label}] job {job_id} never started — queued "
                    f"{(now - submitted_at)/60:.0f}m (limit "
                    f"{queue_timeout_s/60:.0f}m); cancelled")
        elif now - first_running > runtime_timeout_s:
            job_api.cancel(job_id)
            raise SystemExit(
                f"✗ [{label}] job {job_id} timed out after "
                f"{(now - first_running)/60:.0f}m of RUNTIME (limit "
                f"{runtime_timeout_s/60:.0f}m); cancelled. Console tail:\n"
                f"{job_api.logs(job_id, 20)}")
        if now - last_print >= heartbeat_s:
            tail = ""
            try:
                for log_name in ("mid_train.log", "eval.log"):
                    try:
                        raw = obs.download_bytes(
                            f"{result_uri.rstrip('/')}/{log_name}")
                        lines = [l for l in
                                 raw.decode("utf-8", "replace").splitlines()
                                 if l.strip()]
                        if lines:
                            tail = lines[-1][:140]
                            break
                    except Exception:
                        continue
            except Exception:
                tail = ""
            if not tail:
                console_tail = (job_api.logs(job_id, 1) or "").strip()
                tail = (console_tail.splitlines()[-1][:140]
                        if console_tail else "(no output yet)")
            phase = "running" if first_running is not None else "queued"
            print(f"  [{label}] job {job_id} {st.value} {phase} "
                  f"{(now - submitted_at)/60:.0f}m | {tail}", flush=True)
            last_print = now
        time.sleep(poll_s)


def submit_and_wait(job_api, obs, base_name: str, command: List[str],
                    job_env: Dict[str, str], asset_mounts, remote: RemoteConfig,
                    label: str, result_uri: str, runtime_timeout_s: float):
    """Submit + wait with queue-timeout resubmission (arm jobs are
    must-deliver: queue patience = queue_timeout_s x (1 + attempts), no
    adaptive eviction)."""
    attempts = max(0, int(remote.queue_resubmit_attempts))
    for attempt in range(attempts + 1):
        name = base_name if attempt == 0 else f"{base_name[:59]}-r{attempt}"
        job_id = submit_with_backoff(
            job_api, name, command, job_env, asset_mounts,
            remote.submit_retry_timeout_s, label)
        print(f"  [{label}] submitted job {job_id}", flush=True)
        try:
            return job_id, wait_job(
                job_api, obs, job_id, label, result_uri,
                remote.poll_interval_s, runtime_timeout_s,
                remote.queue_timeout_s)
        except QueueTimedOut as e:
            if attempt >= attempts:
                raise SystemExit(f"✗ {e} — queue patience exhausted "
                                 f"({remote.queue_timeout_s/3600:.0f}h x "
                                 f"{attempts + 1})")
            print(f"  [{label}] {e} — resubmitting "
                  f"(try {attempt + 2}/{attempts + 1})", flush=True)
    raise AssertionError("unreachable")  # pragma: no cover


def download_result_json(obs, result_uri: str) -> Optional[Dict]:
    try:
        return json.loads(obs.download_bytes(
            f"{result_uri.rstrip('/')}/result.json").decode("utf-8"))
    except Exception:
        return None


def land_logs(obs, result_uri: str, output_dir: str, arm: str) -> None:
    for src, dst in ((f"{result_uri.rstrip('/')}/mid_train.log",
                      os.path.join(output_dir, f"mid_train_{arm}.log")),
                     (f"{result_uri.rstrip('/')}/eval.log",
                      os.path.join(output_dir, f"eval_{arm}.log"))):
        if obs.stat(src):
            obs.download_file(src, dst)
            print(f"  [{arm}] landed {os.path.basename(dst)}")


def land_checkpoint(obs, result_uri: str, nanochat_base_dir: str, tag: str) -> bool:
    ckpt_uri = f"{result_uri.rstrip('/')}/mid_checkpoint"
    objs = obs.list_objects(ckpt_uri)
    if not objs:
        return False
    dst = os.path.join(nanochat_base_dir, "mid_checkpoints", tag)
    if os.path.isdir(dst):
        import shutil
        shutil.rmtree(dst)
    os.makedirs(dst, exist_ok=True)
    for obj in objs:
        obs.download_file(obj, os.path.join(dst, obj.rsplit("/", 1)[-1]))
    print(f"  landed mid checkpoint -> {dst} ({len(objs)} files)")
    return True


def write_audit(path: str, payload: Dict) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


# ── main ─────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="dispatch a d28 target arm (or base anchor) as a remote job")
    p.add_argument("--arm", required=True,
                   choices=["random", "climb", "base_eval_check"])
    p.add_argument("--remote-config", default="",
                   help="RemoteConfig JSON (default: $OUTPUT_DIR/remote_config.json)")
    p.add_argument("--output-dir", default="",
                   help="result dir (default: env/launch_env OUTPUT_DIR)")
    p.add_argument("--data-dir", default="",
                   help="local mixed-data dir (default: $OUTPUT_DIR/<arm>_mixed)")
    p.add_argument("--tag", default="",
                   help="model tag (default: d<TARGET_DEPTH>_<arm>_<EXP_NAME>)")
    p.add_argument("--job-timeout-h", type=float, default=13.0,
                   help="remote RUNTIME timeout for the arm job (h, default 13)")
    p.add_argument("--d28-asset-uri", default="",
                   help="obs:// dir of the d28 base ckpt asset "
                        "(default: $REMOTE_D28_ASSET_URI or "
                        "{obs_prefix}/assets_big/d28)")
    p.add_argument("--arm-asset-mounts", default="",
                   help='JSON {"tokenizer":"obs://...","eval_stem":"obs://..."} '
                        "(default: reuse the search fleet's remote_config "
                        "asset_mounts entries)")
    p.add_argument("--no-d28-upload", action="store_true",
                   help="never upload the local d28 ckpt to the asset URI "
                        "(fail if the asset is missing)")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-attempt remote even if a prior dispatch failed")
    p.add_argument("--wait-cluster-min", type=float, default=120.0,
                   help="random arm: max minutes to wait for the cluster "
                        "cache + balanced profile (default 120)")
    p.add_argument("--cluster-poll-s", type=float, default=60.0)
    args = p.parse_args()

    # ── env + remote config ──
    output_dir = args.output_dir or os.environ.get("OUTPUT_DIR", "")
    if not output_dir:
        raise SystemExit("✗ --output-dir or $OUTPUT_DIR required")
    launch_env = load_launch_env(output_dir)
    exp_name = launch_env.get("EXP_NAME") or "main"
    nanochat_base_dir = launch_env["NANOCHAT_BASE_DIR"]
    nanochat_dir = launch_env["NANOCHAT_DIR"]
    target_depth = int(launch_env.get("TARGET_DEPTH") or 28)

    rc_path = args.remote_config or os.path.join(output_dir, "remote_config.json")
    if not os.path.isfile(rc_path):
        raise SystemExit(
            f"✗ remote config not found: {rc_path} (REMOTE_ENABLED=1 run?) — "
            f"the main script will fall back to local execution")
    remote = RemoteConfig.from_json_file(rc_path)
    # Arm jobs are whole-node 8-NPU jobs (d28 needs the full card group)
    # and never run the local parallel path — normalize the shape before
    # the executor init (which only publishes assets here; its hybrid
    # warnings would be noise).
    remote.npu_per_job = int(launch_env.get("NUM_NPU") or 8)
    remote.local_parallel = False

    arm = args.arm
    base_check = arm == "base_eval_check"

    # ── per-arm mutex + early exits ──
    os.makedirs(output_dir, exist_ok=True)
    lock_fh = open(os.path.join(output_dir, f".dispatch_{arm}.lock"), "w")
    fcntl.flock(lock_fh, fcntl.LOCK_EX)

    if not base_check:
        if os.path.isfile(os.path.join(output_dir, f".done_mid_train_{arm}")):
            print(f"  [{arm}] .done_mid_train_{arm} already present — nothing to do")
            return 0
        audit_path = os.path.join(output_dir, f"target_arm_{arm}.json")
        if (os.path.isfile(audit_path) and not args.retry_failed):
            try:
                prior = json.load(open(audit_path))
            except (OSError, ValueError):
                prior = {}
            if prior.get("status") not in (None, "SUCCEEDED"):
                print(f"  [{arm}] prior remote attempt recorded as "
                      f"{prior.get('status')} in {audit_path} — NOT retrying "
                      f"(use --retry-failed to force); local fallback takes over")
                return 1

    # ── data preparation (random arm only; climb expects Steps 4-5 done) ──
    if arm == "random":
        cluster_cache = os.path.join(output_dir, "cluster_cache.npz")
        balanced_profile = os.path.join(output_dir, "balanced_profile.json")
        deadline = time.time() + args.wait_cluster_min * 60.0
        while not (os.path.isfile(cluster_cache) and os.path.isfile(balanced_profile)):
            if time.time() > deadline:
                raise SystemExit(
                    f"✗ [random] cluster cache / balanced profile not found after "
                    f"{args.wait_cluster_min:.0f}m: {cluster_cache}, "
                    f"{balanced_profile} — is the main script running?")
            print(f"  [random] waiting for cluster cache + balanced profile "
                  f"({args.cluster_poll_s:.0f}s poll, "
                  f"{(deadline - time.time())/60:.0f}m left)", flush=True)
            time.sleep(args.cluster_poll_s)
        try:
            prof = json.load(open(balanced_profile))
            max_share = float(prof.get("max_share", 0.0))
            k_final = int(prof.get("K_final", 0))
            k_expected = int(launch_env.get("K_ENHANCED") or 0)
            print(f"  [random] balanced profile: K_final={k_final} "
                  f"(expected {k_expected}), max_share={max_share:.1%}, "
                  f"overflow={prof.get('overflow_assignments')}")
            if k_expected and k_final != k_expected:
                raise SystemExit(
                    f"✗ [random] balanced profile K_final={k_final} != "
                    f"K_ENHANCED={k_expected} — the search space changed; "
                    f"refusing to prep a mismatched baseline")
            if max_share > 0.15:
                raise SystemExit(
                    f"✗ [random] balanced profile max_share={max_share:.1%} "
                    f"> 15% — cluster structure gate would reject this run")
        except (OSError, ValueError) as e:
            raise SystemExit(f"✗ [random] unreadable balanced profile: {e}")

        # prep under the SAME lock the main script's Steps 4-5 use; both
        # sides .done-double-check, so the second arriver is a no-op.
        prep_lock = open(os.path.join(output_dir, ".random_arm.lock"), "w")
        fcntl.flock(prep_lock, fcntl.LOCK_EX)
        try:
            random_shards = os.path.join(output_dir, "random_shards")
            climbmix_dir = launch_env.get("CLIMBMIX_DIR", REPO_ROOT)
            if not os.path.isfile(os.path.join(random_shards, ".done")):
                run_logged([
                    "python3", os.path.join(climbmix_dir, "scripts",
                                            "prepare_random_baseline.py"),
                    "--data-dir", launch_env["DATA_DIR"],
                    "--output-dir", random_shards,
                    "--cluster-cache", cluster_cache,
                    "--schema", os.path.join(climbmix_dir, "config",
                                             "schema_stem.yaml"),
                    "--target-tokens", launch_env.get("TARGET_TOKENS") or "1B",
                    "--seed", "42",
                    "--num-npu", launch_env.get("NUM_NPU") or "8",
                ], label="random")
            else:
                print("  [random] baseline shards already prepared (.done)")
            random_mixed = os.path.join(output_dir, "random_mixed")
            if not os.path.isfile(os.path.join(random_mixed, ".done")):
                env_mix = dict(os.environ)
                env_mix["NANOCHAT_REPO"] = nanochat_dir
                run_logged([
                    "python3", os.path.join(climbmix_dir, "scripts",
                                            "mix_general_data.py"),
                    "--stem-dir", random_shards,
                    "--output-dir", random_mixed,
                    "--climbmix-dir", launch_env["GENERAL_DATA_DIR"],
                    "--stem-ratio", launch_env.get("STEM_RATIO") or "0.7",
                    "--num-workers", launch_env.get("NUM_NPU") or "8",
                    "--num-npu", launch_env.get("NUM_NPU") or "8",
                ], env=env_mix, label="random")
            else:
                print("  [random] mixture already mixed (.done)")
        finally:
            fcntl.flock(prep_lock, fcntl.LOCK_UN)

    # ── resolve local data dir + tag ──
    if base_check:
        data_dir = ""
        tag = args.tag or f"d{target_depth}"
    else:
        data_dir = args.data_dir or os.path.join(output_dir, f"{arm}_mixed")
        tag = args.tag or f"d{target_depth}_{arm}_{exp_name}"
        if not os.path.isfile(os.path.join(data_dir, ".done")):
            raise SystemExit(
                f"✗ [{arm}] mixed data not ready (no .done in {data_dir}) — "
                f"the main script will prep it and re-dispatch / fall back")

    # ── executor: publishes the worker bundle + nanochat code, and hands
    # us the backend's job_api / obs / worker-path convention (identical
    # to what the search fleet runs — that is the point).
    config = CLIMBConfig(
        nanochat_dir=nanochat_dir,
        nanochat_base_dir=nanochat_base_dir,
        output_dir=output_dir,
        experiment_name=f"{exp_name}_{arm}",
        device=DeviceConfig(device_type="npu",
                            npu_devices=int(launch_env.get("NUM_NPU") or 8)),
    )
    executor = RemoteExecutor(config, remote)
    job_api, obs = executor.job_api, executor.obs

    prefix = remote.obs_prefix.rstrip("/")
    arm_root = f"{prefix}/target_arms/{arm}"
    mixture_uri = f"{arm_root}/mixture_data"
    result_uri = f"{arm_root}/result"

    # ── d28 asset: ensure on OBS (one-time bootstrap) ──
    d28_uri = (args.d28_asset_uri
               or os.environ.get("REMOTE_D28_ASSET_URI")
               or launch_env.get("REMOTE_D28_ASSET_URI")
               or f"{prefix}/assets_big/d{target_depth}")
    local_d28 = os.path.join(nanochat_base_dir, "base_checkpoints",
                             f"d{target_depth}")
    have_asset = bool(obs.list_objects(d28_uri))
    if not have_asset:
        if args.no_d28_upload:
            raise SystemExit(f"✗ d28 asset missing on OBS ({d28_uri}) and "
                             f"--no-d28-upload given")
        if not glob.glob(os.path.join(local_d28, "model_*.pt")):
            raise SystemExit(
                f"✗ d28 asset missing on OBS ({d28_uri}) and no local ckpt at "
                f"{local_d28} — upload it or pass --d28-asset-uri")
        print(f"  [{arm}] one-time bootstrap: uploading d28 ckpt "
              f"{local_d28} -> {d28_uri} (several GB, patience)", flush=True)
        upload_dir_if_missing(obs, local_d28, d28_uri, "d28-asset")
    else:
        print(f"  [{arm}] d28 asset present at {d28_uri}")

    # ── per-launch asset mounts: d28 + tokenizer + eval_stem (REPLACES the
    # backend's global set — arm jobs must not stage the pool/stella) ──
    mounts: Dict[str, str] = {f"d{target_depth}": d28_uri}
    arm_mounts: Dict[str, str] = {}
    if args.arm_asset_mounts:
        arm_mounts = json.loads(args.arm_asset_mounts)
    search_mounts = remote.asset_mounts or {}
    for name in ("tokenizer", "eval_stem", "eval_bundle"):
        uri = (arm_mounts.get(name) or search_mounts.get(name))
        if uri:
            mounts[name] = uri
    for name in ("tokenizer", "eval_stem"):
        if name not in mounts:
            raise SystemExit(
                f"✗ asset mount '{name}' unresolvable — pass "
                f"--arm-asset-mounts or add it to the search fleet's "
                f"REMOTE_ASSET_MOUNTS")
    print(f"  [{arm}] asset mounts: {sorted(mounts)}")

    # ── upload mixture + build spec ──
    nproc = int(remote.npu_per_job or launch_env.get("NUM_NPU") or 8)
    spec_env = build_spec_env(launch_env, remote.job_env)
    work_dir = os.path.join(remote.container_work_root, "target_arms", arm)

    if base_check:
        container_data_dir = ""
        mid_cmd: List[str] = []
        eval_cmd = build_target_eval_cmd(
            model_tag=tag,
            eval_benchmarks=launch_env.get("EVAL_BENCHMARKS") or "stem",
            eval_max_per_task=launch_env.get("EVAL_MAX_PER_TASK") or "-1",
            device_batch_size=launch_env.get("EVAL_DEVICE_BATCH_SIZE") or "16",
            core_batch_size=launch_env.get("EVAL_CORE_BATCH_SIZE") or "8",
            model_type="base",
            nproc_per_node=nproc,
        )
    else:
        upload_dir_if_missing(obs, data_dir, mixture_uri, arm)
        container_data_dir = os.path.join(work_dir, "mixture_data")
        mid_cmd = build_target_mid_train_cmd(
            run_name=f"{arm}_mid",
            model_tag=tag,
            data_dir=container_data_dir,
            num_iterations=launch_env.get("TARGET_STEPS") or "1000",
            lr_scale=launch_env.get("TARGET_LR_SCALE") or "1.0",
            warmup=launch_env.get("TARGET_WARMUP") or "0.0",
            warmdown=launch_env.get("TARGET_WARMDOWN") or "0.9",
            core_metric_every=launch_env.get("CORE_METRIC_EVERY") or "-1",
            device_batch_size=launch_env.get("MID_DEVICE_BATCH_SIZE") or "1",
            loader=launch_env.get("MID_TRAIN_LOADER") or "flat",
            nproc_per_node=nproc,
        )
        eval_cmd = build_target_eval_cmd(
            model_tag=tag,
            eval_benchmarks=launch_env.get("EVAL_BENCHMARKS") or "stem",
            eval_max_per_task=launch_env.get("EVAL_MAX_PER_TASK") or "-1",
            device_batch_size=launch_env.get("EVAL_DEVICE_BATCH_SIZE") or "16",
            core_batch_size=launch_env.get("EVAL_CORE_BATCH_SIZE") or "8",
            nproc_per_node=nproc,
        )

    from climbmix.remote.exp_spec import ExpSpec
    spec = ExpSpec(
        experiment_name=f"{exp_name}_{arm}",
        model_tag=tag,
        nanochat_dir=remote.container_nanochat_dir,
        base_dir=remote.container_base_dir,
        work_dir=work_dir,
        base_ckpt_src=os.path.join(remote.container_base_dir,
                                   "base_checkpoints", f"d{target_depth}"),
        ckpt_src=(os.path.join(remote.container_base_dir, "base_checkpoints",
                               f"d{target_depth}") if base_check else ""),
        mixture_data_uri=mixture_uri if not base_check else "",
        result_uri=result_uri,
        mid_train_cmd=mid_cmd,
        eval_cmd=eval_cmd,
        eval_only=base_check,
        upload_checkpoint=not base_check,
        visible_devices=list(range(nproc)),
        env=spec_env,
    )
    spec_uri = f"{arm_root}/spec.json"
    # stale OBS artifacts from a previous attempt must not linger (same
    # discipline as the executor's exp cleanup)
    for uri in obs.list_objects(result_uri):
        obs.delete(uri)
    obs.upload_bytes(spec.to_json().encode("utf-8"), spec_uri)
    print(f"  [{arm}] spec -> {spec_uri}")

    # ── submit + wait ──
    worker_argv = [remote.container_python, remote.worker_path,
                   "--spec-uri", spec_uri,
                   "--storage", remote.storage_kind]
    if remote.storage_kind == "local":
        worker_argv += ["--storage-root", remote.storage_root]
    job_name = re.sub(r"[^A-Za-z0-9-]", "-",
                      f"climbmix-{exp_name}-arm-{arm}")[:63]
    t0 = time.time()
    job_id, status = submit_and_wait(
        job_api, obs, job_name, worker_argv, dict(remote.job_env), mounts,
        remote, arm, result_uri, args.job_timeout_h * 3600.0)
    elapsed = time.time() - t0
    console = ""
    try:
        console = job_api.logs(job_id, 400) or ""
    except Exception:
        pass
    print(f"  [{arm}] job {job_id} terminal: {status.value} "
          f"({elapsed/60:.0f}m total)")

    res = download_result_json(obs, result_uri) or {}
    mid_rc = int(res.get("mid_train_rc", -1))
    eval_rc = int(res.get("eval_rc", -1))
    land_logs(obs, result_uri, output_dir, arm)

    # ── land artifacts ──
    csv_landed = False
    if status == JobStatus.SUCCEEDED and eval_rc == 0:
        csv_uri = f"{result_uri.rstrip('/')}/eval_{tag}.csv"
        if obs.stat(csv_uri):
            if base_check:
                csv_dst = os.path.join(output_dir, "eval_base_remote.csv")
            else:
                csv_dst = os.path.join(output_dir, f"eval_{arm}.csv")
            obs.download_file(csv_uri, csv_dst)
            csv_landed = True
            print(f"  [{arm}] landed {os.path.basename(csv_dst)}")
        if not base_check:
            land_checkpoint(obs, result_uri, nanochat_base_dir, tag)
            for marker in (f".done_mid_train_{arm}", f".done_eval_{arm}"):
                open(os.path.join(output_dir, marker), "w").close()
            print(f"  [{arm}] landed .done_mid_train_{arm} + .done_eval_{arm}")
    elif (not base_check) and mid_rc == 0:
        # Salvage: training succeeded remotely (eval failed / job failed
        # later) — land the checkpoint + train marker; the main script
        # skips retraining and evals locally.
        if land_checkpoint(obs, result_uri, nanochat_base_dir, tag):
            open(os.path.join(output_dir, f".done_mid_train_{arm}"), "w").close()
            print(f"  [{arm}] SALVAGE: training succeeded (mid_train_rc=0, "
                  f"eval_rc={eval_rc}) — landed ckpt + .done_mid_train_{arm}; "
                  f"eval falls back to local")

    ok = bool(status == JobStatus.SUCCEEDED and eval_rc == 0 and mid_rc == 0
              and (base_check or csv_landed))
    audit = {
        "arm": arm,
        "tag": tag,
        "job_id": job_id,
        "job_name": job_name,
        "status": status.value,
        "ok": ok,
        "salvaged_train_only": (not base_check and mid_rc == 0
                                and not (eval_rc == 0 and status == JobStatus.SUCCEEDED)),
        "mid_train_rc": mid_rc,
        "eval_rc": eval_rc,
        "elapsed_seconds": round(elapsed, 1),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_uri": spec_uri,
        "result_uri": result_uri,
        "mixture_data_uri": mixture_uri if not base_check else None,
        "d28_asset_uri": d28_uri,
        "asset_mounts": mounts,
        "node_info": grep_node_info(console),
        "csv": (os.path.join(output_dir, "eval_base_remote.csv") if base_check
                else os.path.join(output_dir, f"eval_{arm}.csv")),
    }
    write_audit(os.path.join(output_dir, f"target_arm_{arm}.json"), audit)

    if not ok:
        print(f"✗ [{arm}] remote arm did not complete "
              f"(status={status.value}, mid_train_rc={mid_rc}, "
              f"eval_rc={eval_rc}) — see {output_dir}/mid_train_{arm}.log / "
              f"eval_{arm}.log; console tail:\n{console[-2000:]}")
        return 1
    print(f"✓ [{arm}] remote arm complete ({elapsed/60:.0f}m) — "
          f"audit: target_arm_{arm}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
