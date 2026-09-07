"""RemoteExecutor — runs proxy experiments as remote jobs, results land as
local exp_XXXX/ dirs.

Architecture (2026-08-28 production plan):
  - LOCAL host = scheduler + mixer: mixture preparation needs the cluster
    labels + the STEM pool, which live here. Shards (~1.5GB/exp) upload to OBS.
  - OBS = data plane: {obs_prefix}/exps/exp_XXXX/{spec.json, mixture_data/,
    result/} per experiment; {obs_prefix}/assets/ = worker code bundle.
  - Job backend = compute plane: one job per experiment
    (npu_per_job cards each, no cross-node collectives). The backend is
    resolved via the registry (backends.py): the built-in "mock"
    simulation or an out-of-tree platform adapter
    (RemoteConfig.backend_module / "climbmix.backends" entry point).
  - Dynamic submission: the shared pool fluctuates (10-200 cards), so a
    batch does NOT assume one fixed-size submission burst — capacity
    rejections (TransientSubmitError) back off and retry (a config is
    never burned by transient quota), in-flight jobs self-regulate to the
    real quota, and one iteration's jobs land in multiple submission
    rounds as capacity frees. Local mixture prep is semaphore-bounded
    (max_prep_parallel) so a high max_concurrent_jobs only buys more
    in-flight JOBS, not more concurrent preps.
  - Materialization: after a job succeeds, its result.json + logs + eval CSV
    download into the LOCAL exp_XXXX/ dir and the SHARED finalize path
    (ProxyRunner._finalize_exp) writes meta.json — the exact same shape a
    locally-executed experiment produces. Search resume (meta.json
    exact-weight match) and stage fingerprints need ZERO changes.

Remote resume levels (mirroring ProxyRunner):
  1. meta.json complete (rc=0/0, weights match) -> reuse, no job submitted.
  2. .remote_mid_ok note (train succeeded remotely, ckpt on OBS) ->
     eval-only job: the worker downloads the checkpoint from
     {result_uri}/mid_checkpoint and skips training.
  3. Fresh: prep locally, upload shards, full job.

Failure semantics match the local executor exactly: any per-experiment
exception becomes an inf/0.0 ProxyResult via run_batch (the bootstrapper
scores it NaN and a resume re-runs it).
"""

import glob
import hashlib
import json
import os
import queue
import re
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from climbmix.core.types import CLIMBConfig, MixtureConfig, ProxyResult
from climbmix.pipeline.proxy_runner import ProxyRunner
from climbmix.remote.exp_spec import ExpSpec, SPEC_VERSION
from climbmix.remote.backends import resolve_backend
from climbmix.remote.job_api import JobStatus, TransientSubmitError


class QueueTimeoutError(RuntimeError):
    """Job sat in the platform queue past queue_timeout_s and never
    started. Retrying is legitimate (infra event, not a property of the
    mixture): burning the config as a failed experiment would feed the
    predictor a fabricated inf/0.0 score for a mixture that was never
    trained."""


class ConfigEvictedError(RuntimeError):
    """Admission-mode eviction (ADAPTIVE_CONFIGS=1): this config's job sat
    PENDING past pending_grace_min while sibling jobs of the same batch
    were RUNNING — the realized pool is smaller than the probe admitted,
    so the wave is over-subscribed. The job is cancelled and the config is
    DROPPED from the iteration (the bootstrapper rewrites its pending list
    without it; a resume never re-runs it). Distinct from a failure: the
    mixture was never trained and never will be, so it must not surface as
    an inf/0.0 score either."""


class AdmissionController:
    """Per-batch adaptive admission state machine (prod2 B++).

    The bootstrapper sizes iterations in WAVES: expected configs e_i over
    an expected slot count S0 (remote max jobs + local slots) gives a wave
    budget w_i = max(1, round(e_i / S0)). The bootstrapper then samples a
    POOL sized to the upper bound (iter 1: w_i * S0 + reserve) or the last
    realized concurrency (iter >= 2), and this controller right-sizes what
    actually runs:

      - probe-truncate: once the first job has RUN for probe_delay_s (or
        probe_deadline_s elapsed since batch start), measure C_eff =
        remote RUNNING median + local slots; admit w_i * C_eff configs by
        dropping the never-submitted tail of the queue (guided iterations
        queue best-predicted first, so the tail is the worst — rank-head
        preservation).
      - straggler eviction: handled in _wait_job (needs per-job PENDING
        clocks); records evicted indices here.
      - rolling C_eff: run_batch exposes the measured concurrency so the
        NEXT iteration sizes directly from it.

    Pure decisions only — run_batch's monitor thread applies them, which
    makes the whole state machine unit-testable without any jobs. The
    drop-set is race-free by reconciliation: a marked index that was
    already picked by a worker simply runs and is excluded from the final
    dropped list (results non-None wins).
    """

    def __init__(self, wave_budget: int, expected_slots: int,
                 min_configs: int = 4,
                 allow_truncate: bool = True,
                 probe_delay_s: float = 1800.0,
                 probe_deadline_s: float = 2400.0,
                 pending_grace_s: float = 1800.0):
        self.wave_budget = max(1, int(wave_budget))
        self.expected_slots = max(1, int(expected_slots))
        self.min_configs = max(1, int(min_configs))
        self.allow_truncate = bool(allow_truncate)
        self.probe_delay_s = float(probe_delay_s)
        self.probe_deadline_s = float(probe_deadline_s)
        self.pending_grace_s = float(pending_grace_s)
        self.first_running_at: Optional[float] = None
        self.decided = False
        self.c_eff: Optional[int] = None
        self.drop_set: set = set()      # global indices dropped (truncate + evict)
        self.truncated: List[int] = []  # dropped before pickup (queue tail)
        self.evicted: List[int] = []    # submitted, cancelled while PENDING

    def note_running(self, now: float) -> None:
        if self.first_running_at is None:
            self.first_running_at = now

    def probe_ready(self, now: float, batch_started_at: float) -> bool:
        if self.decided:
            return False
        if self.first_running_at is not None:
            return now - self.first_running_at >= self.probe_delay_s
        return now - batch_started_at >= self.probe_deadline_s

    def decide(self, total_configs: int, n_local: int, running_now: int,
               local_slots: int, in_queue: int = 0) -> List[int]:
        """Probe-point truncation decision. Marks (and returns) the global
        indices to drop from the queue TAIL so the admitted count lands at
        wave_budget * C_eff. Empty list = admit everything (no truncation).

        Guardrails (docs/parallel_k_selection.md §5.2): C_eff < 2 disables
        truncation entirely (no usable pool signal — literal behavior, the
        queue's own 24h patience handles it); a pool smaller than
        min_configs is never truncated; the admitted count never drops
        below min_configs.
        """
        self.decided = True
        self.c_eff = int(running_now) + int(local_slots)
        if not self.allow_truncate:
            return []
        if self.c_eff < 2:
            return []
        total = int(total_configs)
        if total < self.min_configs:
            return []
        n_target = max(self.min_configs,
                       min(self.wave_budget * self.c_eff, total))
        need_drop = total - n_target - len(self.drop_set)
        if need_drop <= 0:
            return []
        # Highest unpicked-first: local slice ([0, n_local)) is always
        # admitted; remote picks are FIFO from the queue head, so the
        # highest indices are the ones still sitting in the queue.
        candidates = [g for g in range(n_local, total)
                      if g not in self.drop_set]
        drops = candidates[-need_drop:]
        self.drop_set.update(drops)
        self.truncated.extend(drops)
        return drops


@dataclass
class RemoteConfig:
    """All knobs are EXECUTION-SHAPE only (transport, quota, paths) — none
    change experiment semantics, so (like num_npu) they are excluded from the
    stage fingerprints. Semantic knobs (iterations, lr, eval caps, ...)
    already live in CLIMBConfig and are fingerprinted there."""

    # obs://bucket/... root for this experiment series. Layout below it:
    #   {prefix}/exps/exp_XXXX/{spec.json, mixture_data/, result/}
    #   {prefix}/assets/{remote_worker.py, nanochat_cmds.py}
    obs_prefix: str = ""

    # Which backend to construct when not injected (tests inject
    # job_api/obs): "mock" (built-in local simulation) or the name of an
    # out-of-tree platform backend, resolved via backends.py — either
    # backend_module below or a registered "climbmix.backends" entry
    # point.
    backend: str = "mock"

    # "package.module:attr" factory spec for an out-of-tree backend (attr
    # defaults to create_backend). Takes precedence over entry-point
    # lookup. Works with the backend repo cloned + PYTHONPATH — no
    # installation required.
    backend_module: str = ""

    # Path to the backend's platform config file (endpoint/IDs/auth —
    # schema is backend-defined; real values live OUTSIDE this public
    # repo, e.g. ~/.config/climbmix/). Passed through to the backend.
    platform_config: str = ""

    # ── container-side path conventions (baked into every ExpSpec) ──
    container_nanochat_dir: str = "/home/ma-user/work/nanochat-npu"
    container_base_dir: str = "/home/ma-user/work/nanochat_base"
    container_work_root: str = "/home/ma-user/work/climbmix_exp"
    # Container path of the proxy base checkpoint (d20). Empty = derived:
    # {container_base_dir}/base_checkpoints/d{proxy_depth}.
    container_base_ckpt_src: str = ""
    container_python: str = "python3"

    # Path of remote_worker.py AS SEEN BY THE JOB RUNTIME. Mock backend: the
    # local staged path (auto-derived when empty). Real backend: the
    # container path where the boot shell places the assets bundle.
    worker_path: str = ""

    # Worker storage backend: "local" (simulation: filesystem under
    # storage_root, same mapping as MockObsStorage) or "moxing" (real).
    storage_kind: str = "moxing"
    storage_root: str = ""

    # ── job resources ──
    image: str = ""
    flavor: str = ""
    npu_per_job: int = 1
    pool_name: str = ""

    # ── scheduling ──
    max_concurrent_jobs: int = 8
    # Dynamic submission (shared pool fluctuates 10-200 cards): a submit
    # rejected for capacity/quota is RETRIED with exponential backoff until
    # submit_retry_timeout_s — the config is never burned by transient
    # rejections, and in-flight jobs self-regulate to the real quota (an
    # iteration's configs submit in multiple rounds as capacity frees).
    submit_retry_timeout_s: float = 24 * 3600.0
    submit_retry_initial_s: float = 30.0
    submit_retry_max_s: float = 600.0
    # Local prep+upload concurrency for the REMOTE pipeline (semaphore).
    # Kept small so a high max_concurrent_jobs cannot make 1.5GB/exp
    # prep+upload runs stampede the master node; submit threads pull from
    # the prepped specs. (Local-slice prep is bounded by its own NPU slots.)
    max_prep_parallel: int = 4
    # Hybrid fleet: also run experiments on the LOCAL NPUs via the parent
    # ProxyRunner parallel path (npu_per_exp in [1, npu_devices], a divisor;
    # npu_per_exp == npu_devices = ONE whole-node slot, the parent's serial
    # full-card path — prod2 k=8 form). Configs[:n_local] run locally, the
    # rest remotely.
    local_parallel: bool = False

    # ── artifacts ──
    upload_checkpoint: bool = True    # worker -> OBS after successful train
    download_checkpoint: bool = False  # OBS -> local exp_dir (debug only)

    # ── polling ──
    poll_interval_s: float = 30.0
    status_print_interval_s: float = 300.0
    job_timeout_s: float = 6 * 3600.0
    # Queue-phase bound (submission → first RUNNING). Platform QUEUE time
    # does NOT burn job_timeout_s: shared pools can hold a job PENDING for
    # hours before cards free up (2026-09-04 prod pool: 0 idle at launch),
    # and cancelling a still-queued job because its runtime budget was
    # eaten by the queue is pure waste. This knob bounds the PENDING phase
    # alone (lost/zombie queue entries).
    queue_timeout_s: float = 24 * 3600.0
    # After a queue timeout: resubmit (fresh queue clock, job name -rN
    # suffix) up to this many times before giving up and burning the
    # config as a failed experiment. Total queue patience =
    # queue_timeout_s × (1 + attempts). 0 = old burn-immediately behavior.
    queue_resubmit_attempts: int = 2
    # Adaptive admission (ADAPTIVE_CONFIGS=1, prod2 B++): a SUBMITTED job
    # still PENDING after this many minutes while sibling jobs of the same
    # batch are RUNNING signals over-admission (pool shrank after the
    # probe) — the executor cancels it and the config is dropped from the
    # iteration permanently (the bootstrapper rewrites its pending list).
    # Only consulted in admission mode; arm jobs (dispatch_target_arm.py)
    # never enable it — must-deliver, queue patience applies instead.
    pending_grace_min: float = 30.0

    # Job-level env (HF_ENDPOINT=hf-mirror.com, ...). Passed to the job
    # process AND baked into the spec for the train/eval subprocesses.
    job_env: Dict[str, str] = field(default_factory=dict)

    # Local staging dir for the worker assets bundle (default:
    # {repo}/cache/remote_assets — outside the fingerprinted output dir).
    assets_stage_dir: str = ""

    # Platform wheel files (local paths, e.g. the aarch64 rustbpe wheel)
    # uploaded to {obs_prefix}/assets/ when absent — offline containers
    # pip-install missing deps from there. Upload-if-missing only: pin a
    # new wheel by bumping its filename (they are version-named anyway).
    code_wheels: List[str] = field(default_factory=list)

    # Per-launch direct asset mounts ({name: obs uri}) that REPLACE the
    # backend config's global obs.asset_mounts for THIS fleet's jobs only
    # (None = inherit the platform config's set, as before). Scopes a
    # search fleet to its own assets: job-class-specific mounts (embed
    # models, the parquet pool, target-depth checkpoints) stay out of
    # proxy jobs, which would otherwise stage them for nothing.
    asset_mounts: Optional[Dict[str, str]] = None

    @staticmethod
    def from_dict(d: Dict) -> "RemoteConfig":
        known = {f for f in RemoteConfig.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"RemoteConfig: unknown keys {sorted(unknown)}")
        cfg = RemoteConfig()
        for k, v in d.items():
            cur = getattr(cfg, k)
            if isinstance(cur, bool):
                if not isinstance(v, bool):
                    raise ValueError(f"RemoteConfig.{k} must be a bool, got {v!r}")
                setattr(cfg, k, v)
            elif cur is None or isinstance(cur, dict):
                # Optional[Dict]/Dict knobs (job_env always; asset_mounts
                # when present — None default must not hit type(cur)(v))
                if not isinstance(v, dict):
                    raise ValueError(f"RemoteConfig.{k} must be a dict, got {v!r}")
                setattr(cfg, k, dict(v))
            else:
                setattr(cfg, k, type(cur)(v))
        return cfg

    @staticmethod
    def from_json_file(path: str) -> "RemoteConfig":
        with open(path) as f:
            return RemoteConfig.from_dict(json.load(f))

    def validate(self) -> None:
        if not self.obs_prefix.startswith("obs://"):
            raise ValueError(
                f"RemoteConfig.obs_prefix must be an obs:// URI, got "
                f"{self.obs_prefix!r}")
        if not self.backend:
            raise ValueError("RemoteConfig.backend must be non-empty")
        if self.backend == "mock" or self.storage_kind == "local":
            if not self.storage_root:
                raise ValueError(
                    "RemoteConfig.storage_root is required for the mock/"
                    "local simulation backend")
        if self.npu_per_job < 1:
            raise ValueError("RemoteConfig.npu_per_job must be >= 1")
        if self.max_concurrent_jobs < 1:
            raise ValueError("RemoteConfig.max_concurrent_jobs must be >= 1")
        if self.submit_retry_timeout_s <= 0:
            raise ValueError("RemoteConfig.submit_retry_timeout_s must be > 0")
        if self.submit_retry_initial_s <= 0:
            raise ValueError("RemoteConfig.submit_retry_initial_s must be > 0")
        if self.submit_retry_max_s < self.submit_retry_initial_s:
            raise ValueError("RemoteConfig.submit_retry_max_s must be >= "
                             "submit_retry_initial_s")
        if self.max_prep_parallel < 1:
            raise ValueError("RemoteConfig.max_prep_parallel must be >= 1")
        if self.pending_grace_min <= 0:
            raise ValueError("RemoteConfig.pending_grace_min must be > 0")
        if self.asset_mounts is not None:
            for m_name, m_uri in self.asset_mounts.items():
                if (not m_name or not isinstance(m_uri, str)
                        or not m_uri.startswith("obs://")):
                    raise ValueError(
                        f"RemoteConfig.asset_mounts entries must be "
                        f"{{name: obs:// uri}}, got "
                        f"{m_name!r}: {m_uri!r}")


class RemoteExecutor(ProxyRunner):
    """ProxyRunner subclass whose execution backend is remote jobs
    (platform adapter or built-in mock, resolved via backends.py).

    Inherited unchanged from ProxyRunner: resume level 1 (meta.json), mixture
    preparation (cluster labels + pool + ClimbMix mixing), the command
    builders (nanochat_cmds — container paths passed as arguments), CSV
    parsing and meta.json writing (_finalize_exp). Overridden: the
    train+eval execution itself, the mid-train resume marker (remote analog),
    and run_batch (job submission + polling + hybrid fleet).
    """

    def __init__(
        self,
        config: CLIMBConfig,
        remote_config: RemoteConfig,
        job_api=None,
        obs=None,
    ):
        remote_config.validate()
        super().__init__(config)
        self.remote = remote_config
        self._obs_lock = threading.Lock()
        self._prep_sem = threading.BoundedSemaphore(remote_config.max_prep_parallel)
        # First hard submit error (bad image/auth): recorded so sibling
        # configs in the same batch burn fast instead of wasting prep.
        self._submit_hard_error: Optional[str] = None
        # Dynamic in-flight capacity (queue-consumer scheduling): workers
        # take configs from a queue only while inflight < cap_limit; a
        # capacity monitor thread (started per batch when the JobAPI
        # supports free_job_slots()) adjusts cap_limit as the shared pool
        # fluctuates. Shrinking NEVER kills in-flight jobs — the limit
        # floors at the current inflight count and only gates new pickups.
        self._cap_cond = threading.Condition()
        self._cap_limit = remote_config.max_concurrent_jobs
        self._cap_inflight = 0
        # Running-job registry (adaptive C_eff measurement): a shared
        # RUNNING counter + periodic samples, maintained by every _wait_job
        # poll (first RUNNING +1, terminal -1). Reset per batch.
        self._run_lock = threading.Lock()
        self._running_now = 0
        self._run_samples: List[int] = []
        # Per-batch adaptive admission state (None = literal behavior).
        self._admission: Optional["AdmissionController"] = None
        # Post-batch adaptive accounting — the bootstrapper reads these via
        # getattr() (last_effective_concurrency sizes the next iteration;
        # last_dropped_configs drive the pending rewrite; stats feed the
        # [Iter i] adaptive log line and the watch dashboard).
        self.last_effective_concurrency: Optional[int] = None
        self.last_dropped_configs: List[MixtureConfig] = []
        self.last_admission_stats: Dict[str, object] = {}

        # Backend bundle: resolves the job API + obs storage when they are
        # not injected. Kept None when both are injected (tests) — the
        # staged-path worker default then applies.
        bundle = None
        if job_api is None or obs is None:
            bundle = resolve_backend(remote_config)
            if job_api is None:
                job_api = bundle.make_job_api(remote_config)
            if obs is None:
                obs = bundle.make_obs_storage(remote_config)
        self.job_api = job_api
        self.obs = obs

        self._stage_assets()
        if not remote_config.worker_path:
            wp = bundle.default_worker_path if bundle is not None else ""
            if wp:
                # Out-of-tree backend: the platform delivers {obs_prefix}/
                # assets into its container code dir; the backend's boot
                # logic resolves the exact path at runtime.
                remote_config.worker_path = wp
            else:
                remote_config.worker_path = os.path.join(
                    self.assets_stage_dir, "remote_worker.py")
        self._ensure_assets_uploaded()
        self._ensure_code_synced()

        print(f"  [RemoteExecutor] backend={remote_config.backend} "
              f"obs_prefix={remote_config.obs_prefix} "
              f"npu_per_job={remote_config.npu_per_job} "
              f"max_concurrent_jobs={remote_config.max_concurrent_jobs}"
              + (f" + local x{self._local_slots()}" if remote_config.local_parallel else ""))

        # Hybrid-fleet guards (docs/parallel_k_selection.md: k stays
        # fleet-wide fixed for score comparability; the master node's NPUs
        # are part of the fleet by default).
        if remote_config.local_parallel:
            if self._local_slots() == 0:
                print(f"  [RemoteExecutor] WARNING: local_parallel=1 but the "
                      f"LOCAL NPUs will IDLE (npu_per_exp={self.npu_per_exp}, "
                      f"npu_devices={self.npu_devices}); set npu_per_exp to a "
                      f"divisor of npu_devices (<= npu_devices) so the "
                      f"master node joins the fleet")
            elif remote_config.npu_per_job != self.npu_per_exp:
                print(f"  [RemoteExecutor] WARNING: local slice runs "
                      f"npu_per_exp={self.npu_per_exp} card(s) per experiment "
                      f"but remote jobs run npu_per_job="
                      f"{remote_config.npu_per_job}; k should stay fleet-wide "
                      f"fixed for score comparability")
            elif self.npu_per_exp == self.npu_devices:
                print(f"  [RemoteExecutor] local whole-node slot: 1 x "
                      f"{self.npu_per_exp} NPU (serial full-card path) joins "
                      f"the fleet alongside remote jobs")

    # ── assets bundle (worker + shared cmds module) ──

    def _stage_assets(self) -> None:
        """Copy the worker bundle (remote_worker.py + nanochat_cmds.py +
        embed_worker.py) into the local staging dir (fresh on every init —
        the staged copy always matches the running code, which is what
        gets uploaded to OBS and executed in jobs)."""
        import climbmix
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(climbmix.__file__), "..", ".."))
        stage = self.remote.assets_stage_dir or os.path.join(
            repo_root, "cache", "remote_assets")
        os.makedirs(stage, exist_ok=True)
        self.assets_stage_dir = stage
        src_worker = os.path.join(repo_root, "scripts", "remote_worker.py")
        src_embed = os.path.join(repo_root, "scripts", "embed_worker.py")
        src_cmds = os.path.join(os.path.dirname(climbmix.__file__),
                                "pipeline", "nanochat_cmds.py")
        for src in (src_worker, src_embed, src_cmds):
            if not os.path.isfile(src):
                raise FileNotFoundError(f"remote asset missing: {src}")
            dst = os.path.join(stage, os.path.basename(src))
            # temp + atomic rename: concurrent dispatch processes share this
            # staging dir, and a plain copy2 lets another process upload a
            # half-written file to OBS
            tmp = f"{dst}.tmp.{os.getpid()}"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)

    def _ensure_assets_uploaded(self) -> None:
        """Upload the worker assets on EVERY init (idempotent overwrite).

        The staged copies always match the running code, so an
        unconditional refresh means worker-side changes take effect on
        the next launch. Uploading only when absent (the old behavior)
        would silently keep running stale worker code in jobs. Three
        tiny files — cheap to write on any backend."""
        assets_uri = f"{self.remote.obs_prefix.rstrip('/')}/assets"
        with self._obs_lock:
            for name in ("remote_worker.py", "embed_worker.py",
                         "nanochat_cmds.py"):
                self.obs.upload_file(
                    os.path.join(self.assets_stage_dir, name),
                    f"{assets_uri}/{name}")
            print(f"  [RemoteExecutor] worker assets fresh -> {assets_uri}")
            # Some gateways validate EVERY input mount as an existing OBS
            # dir, and the adapter mounts {prefix}/assets_big — a fresh
            # prefix would fail its first submit until that dir exists.
            # A placeholder file is inert: the boot links only known
            # asset names, workers never read unknown entries.
            marker = (f"{self.remote.obs_prefix.rstrip('/')}"
                      "/assets_big/.climbmix_placeholder")
            if not self.obs.stat(marker):
                self.obs.upload_bytes(
                    b"placeholder - big assets are direct mounts or "
                    b"uploaded per docs/remote_setup.md", marker)

    # ── nanochat code channel (auto-publish on every init) ──

    def _git(self, repo: str, *args: str) -> str:
        r = subprocess.run(["git", "-C", repo, *args],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed:\n{r.stderr}")
        return r.stdout.strip()

    def _ensure_code_synced(self) -> None:
        """Publish the local nanochat-npu tree to {prefix}/assets (auto).

        nanochat-npu is the one frequently-changing job input the worker
        bundle does not cover, and job containers have no internet. The
        master's tree (self.nanochat_dir, validated by ProxyRunner) is
        tarred (excluding .git/__pycache__/pyc), its sha256 stored beside
        the upload as a .sha256 sidecar, and the tar uploaded only when
        the remote sidecar differs — a clean tree at the same commit is a
        zero-transfer no-op via a local marker, so every dispatch can
        call this unconditionally. A dirty tree always rebuilds (the
        marker is only trusted for clean trees: git's status output does
        not fingerprint file contents). Wheels (remote.code_wheels, e.g.
        the aarch64 rustbpe wheel for offline pip) upload when absent.

        Version policy stays with the master: this publishes the tree AS
        IT SITS (no git pull) — the operator decides when to pull."""
        name = os.path.basename(os.path.normpath(self.nanochat_dir))
        if name != "nanochat-npu":
            print(f"  [RemoteExecutor] WARNING: nanochat dir basename is "
                  f"{name!r}, not 'nanochat-npu' — the boot extracts the "
                  f"tar's top dir as-is and expects that name")
        assets_uri = f"{self.remote.obs_prefix.rstrip('/')}/assets"
        tar_uri = f"{assets_uri}/{name}.tar.gz"
        sha_uri = tar_uri + ".sha256"
        marker_path = os.path.join(self.assets_stage_dir,
                                   ".code_sync_marker.json")

        # A non-git dir (mock/test layouts) degrades to always-rebuild:
        # no HEAD/dirty signal to trust, but the sha comparison still
        # prevents re-UPLOADS. Real deployments are git clones.
        try:
            head = self._git(self.nanochat_dir, "rev-parse", "--short", "HEAD")
            dirty = bool(self._git(self.nanochat_dir, "status", "--porcelain"))
        except (RuntimeError, OSError):
            head, dirty = "nogit", True
        print(f"  [RemoteExecutor] nanochat-npu @ {head}"
              + ("  (DIRTY — tar rebuilt, uncommitted edits ride along)"
                 if dirty else "  (clean)"))

        def _remote_sha() -> str:
            try:
                return self.obs.download_bytes(sha_uri).decode().strip()
            except Exception:
                return ""

        # Fast path: clean tree at the same commit this host already
        # published, and the remote still carries that exact tar.
        if not dirty and os.path.isfile(marker_path):
            try:
                marker = json.load(open(marker_path))
            except (OSError, ValueError):
                marker = {}
            if (marker.get("clean") and marker.get("head") == head
                    and marker.get("sha")
                    and _remote_sha() == marker["sha"]):
                print(f"  [RemoteExecutor] code already published "
                      f"(sha256:{marker['sha'][:16]}…) — skip")
                self._sync_wheels(assets_uri)
                return

        with tempfile.TemporaryDirectory(prefix="climbmix_sync_") as td:
            tar_path = os.path.join(td, f"{name}.tar.gz")
            r = subprocess.run(
                ["tar", "czf", tar_path, "--exclude=.git",
                 "--exclude=__pycache__", "--exclude=*.pyc",
                 "-C", os.path.dirname(
                     os.path.normpath(self.nanochat_dir)), name])
            if r.returncode != 0:
                raise RuntimeError("tar of nanochat-npu failed")
            with open(tar_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()[:16]
            with self._obs_lock:
                if _remote_sha() == sha:
                    print(f"  [RemoteExecutor] remote already at "
                          f"sha256:{sha}… — skip tar upload")
                else:
                    self.obs.upload_file(tar_path, tar_uri)
                    self.obs.upload_bytes((sha + "\n").encode(), sha_uri)
                    print(f"  [RemoteExecutor] nanochat code published -> "
                          f"{tar_uri} (sha256:{sha}…)")
            # marker written regardless: it records what this tree
            # publishes, clean or dirty (a clean tree may re-trust it;
            # a dirty one never does)
            tmp = marker_path + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump({"head": head, "clean": not dirty, "sha": sha}, f)
            os.replace(tmp, marker_path)
        self._sync_wheels(assets_uri)

    def _sync_wheels(self, assets_uri: str) -> None:
        for wheel in self.remote.code_wheels:
            wheel = os.path.abspath(os.path.expanduser(wheel))
            if not os.path.isfile(wheel):
                raise FileNotFoundError(
                    f"RemoteConfig.code_wheels: wheel not found: {wheel}")
            dest = f"{assets_uri}/{os.path.basename(wheel)}"
            if not self.obs.stat(dest):
                self.obs.upload_file(wheel, dest)
                print(f"  [RemoteExecutor] wheel -> {dest}")

    # ── OBS helpers ──

    def _exp_obs_prefix(self, experiment_id: int) -> str:
        return (f"{self.remote.obs_prefix.rstrip('/')}/exps/"
                f"exp_{experiment_id:04d}")

    def _upload_dir(self, local_dir: str, obs_uri: str) -> List[str]:
        files = sorted(
            f for f in os.listdir(local_dir)
            if os.path.isfile(os.path.join(local_dir, f)))
        for f in files:
            self.obs.upload_file(os.path.join(local_dir, f),
                                 f"{obs_uri.rstrip('/')}/{f}")
        return files

    # ── remote mid-train resume marker (level 2) ──

    @staticmethod
    def _remote_mid_marker_path(exp_dir: str) -> str:
        return os.path.join(exp_dir, ".remote_mid_ok")

    def _write_remote_mid_marker(self, exp_dir: str, mixture_config: MixtureConfig,
                                 model_tag: str, result_uri: str) -> None:
        payload = {
            "weights_sha256": self._weights_sha256(mixture_config),
            "model_tag": model_tag,
            "num_iterations": self.proxy_num_iterations,
            "mid_checkpoint_uri": f"{result_uri.rstrip('/')}/mid_checkpoint",
        }
        with open(self._remote_mid_marker_path(exp_dir), "w") as f:
            json.dump(payload, f, indent=2)

    def _load_remote_mid_marker(self, exp_dir: str, mixture_config: MixtureConfig,
                                model_tag: str) -> Optional[str]:
        """Returns the mid-checkpoint OBS URI if a previous REMOTE mid_train
        of THIS config succeeded and its checkpoint is still on OBS (fail-
        safe: any mismatch/corruption/missing object -> None)."""
        path = self._remote_mid_marker_path(exp_dir)
        if not os.path.isfile(path):
            return None
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
        if not isinstance(data, dict):
            return None
        if data.get("weights_sha256") != self._weights_sha256(mixture_config):
            return None
        if data.get("model_tag") != model_tag:
            return None
        uri = data.get("mid_checkpoint_uri")
        if not isinstance(uri, str) or not uri:
            return None
        try:
            if not self.obs.list_objects(uri):
                return None
        except Exception:
            return None
        return uri

    def _upload_local_mid_ckpt(self, model_tag: str, result_uri: str) -> None:
        """Rare case: a previous LOCAL run trained this exp (marker valid
        against the local checkpoint); upload it so the remote job can do
        eval-only."""
        ckpt_dir = os.path.join(self.nanochat_base_dir,
                                "mid_checkpoints", model_tag)
        if not glob.glob(os.path.join(ckpt_dir, "model_*.pt")):
            raise FileNotFoundError(
                f"local mid checkpoint vanished between marker check and "
                f"upload: {ckpt_dir}")
        self._upload_dir(ckpt_dir, f"{result_uri.rstrip('/')}/mid_checkpoint")

    # ── dynamic capacity management ──

    def _probe_slots(self) -> Optional[int]:
        """free_job_slots() when the JobAPI supports capacity queries,
        else None. The value is adapter-normalized: the number of jobs
        that could be submitted RIGHT NOW (the adapter divides free cards
        by npu_per_job itself)."""
        probe = getattr(self.job_api, "free_job_slots", None)
        if probe is None:
            return None
        try:
            return int(probe())
        except Exception:
            return None

    def _adjust_capacity_limit(self, slots: Optional[int]) -> None:
        if slots is None:
            return
        with self._cap_cond:
            target = max(0, min(self.remote.max_concurrent_jobs, slots))
            # floor at inflight: a shrink never kills running jobs, it only
            # stops NEW pickups until capacity returns
            self._cap_limit = max(target, self._cap_inflight)
            self._cap_cond.notify_all()

    def _capacity_monitor(self, stop_event: threading.Event) -> None:
        """Periodically probe the pool and resize the in-flight limit while
        a batch runs. The shared pool fluctuates (10-200 cards): growth
        wakes queued workers immediately (new jobs start as capacity
        appears — the user-facing 'dynamically detect idle cards' behavior);
        shrink just queues pickups."""
        while not stop_event.is_set():
            self._adjust_capacity_limit(self._probe_slots())
            stop_event.wait(self.remote.poll_interval_s)

    def _acquire_slot(self) -> None:
        with self._cap_cond:
            while self._cap_inflight >= self._cap_limit:
                self._cap_cond.wait()
            self._cap_inflight += 1

    def _release_slot(self) -> None:
        with self._cap_cond:
            self._cap_inflight -= 1
            self._cap_cond.notify_all()

    # ── dynamic submission ──

    def _submit_with_retry(self, name: str, command: List[str],
                           env: Dict[str, str], experiment_id: int) -> str:
        """submit() with exponential backoff on TransientSubmitError.

        The shared NPU pool fluctuates, so rejections are expected: back off
        and retry until submit_retry_timeout_s. Retrying threads hold a pool
        slot but no resources — as sibling jobs finish and free quota, a
        retry lands, which is exactly how one iteration's configs end up
        submitted in multiple rounds. A hard (non-transient) error is
        recorded for fast-fail of siblings and raised immediately."""
        backoff = self.remote.submit_retry_initial_s
        deadline = time.time() + self.remote.submit_retry_timeout_s
        attempt = 0
        last_warn = 0.0
        while True:
            attempt += 1
            try:
                return self.job_api.submit(name=name, command=command,
                                           env=env,
                                           asset_mounts=self.remote.asset_mounts)
            except TransientSubmitError as e:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"submit for experiment {experiment_id} still "
                        f"rejected after {attempt} attempts over "
                        f"{self.remote.submit_retry_timeout_s/60:.0f}m "
                        f"(retry timeout); last error: {e}") from e
                now = time.time()
                if attempt == 1 or now - last_warn >= 300.0:
                    print(f"  [Exp {experiment_id}] submit rejected "
                          f"(attempt {attempt}: {e}) — capacity full? "
                          f"backing off {backoff:.0f}s "
                          f"(retry deadline in {(deadline - now)/60:.0f}m)")
                    last_warn = now
                time.sleep(backoff)
                backoff = min(backoff * 2, self.remote.submit_retry_max_s)
            except Exception as e:
                if self._submit_hard_error is None:
                    self._submit_hard_error = str(e)
                raise

    # ── job lifecycle ──

    def _job_name(self, experiment_id: int) -> str:
        raw = f"climbmix-{self.experiment_name}-exp{experiment_id:04d}"
        return re.sub(r"[^A-Za-z0-9-]", "-", raw)[:63]

    def _worker_argv(self, spec_uri: str) -> List[str]:
        argv = [
            self.remote.container_python,
            self.remote.worker_path,
            "--spec-uri", spec_uri,
            "--storage", self.remote.storage_kind,
        ]
        if self.remote.storage_kind == "local":
            argv += ["--storage-root", self.remote.storage_root]
        return argv

    def _submit_and_wait(self, experiment_id: int,
                         spec_uri: str) -> Tuple[str, JobStatus]:
        """Submit the worker job and wait for a terminal status. A QUEUE
        timeout (never started) resubmits with a fresh queue clock up to
        queue_resubmit_attempts times — the mixture was never trained, so
        the config must not be burned for an infra event. Runtime
        timeouts and other failures propagate immediately.
        Returns (final job_id, terminal status)."""
        attempts = max(0, self.remote.queue_resubmit_attempts)
        for attempt in range(attempts + 1):
            name = self._job_name(experiment_id)
            if attempt > 0:
                name = f"{name[:59]}-r{attempt}"
            job_id = self._submit_with_retry(
                name=name,
                command=self._worker_argv(spec_uri),
                env=dict(self.remote.job_env),
                experiment_id=experiment_id,
            )
            print(f"  [Exp {experiment_id}] submitted job {job_id} "
                  f"(spec: {spec_uri})")

            try:
                return job_id, self._wait_job(job_id, experiment_id)
            except QueueTimeoutError as e:
                if attempt >= attempts:
                    raise
                print(f"  [Exp {experiment_id}] {e}\n"
                      f"  [Exp {experiment_id}] resubmitting "
                      f"(try {attempt + 2}/{attempts + 1}, fresh queue clock)")
        raise AssertionError("unreachable")  # pragma: no cover

    def _wait_job(self, job_id: str, experiment_id: int,
                  timeout: Optional[float] = None) -> JobStatus:
        """Poll until the job reaches a terminal state; RETURNS the status
        (does not raise on FAILED — the result.json the worker uploads on
        known-stage failures carries the precise rc's, and the caller needs
        it to write the eval-only resume markers). Raises only on timeout
        (job cancelled).

        Two clocks: job_timeout_s measures RUNTIME (first RUNNING →
        terminal — platform queue time does not burn it; shared pools can
        hold a job PENDING for hours, 2026-09-04 prod pool had 0 idle
        cards at launch), queue_timeout_s bounds the PENDING phase alone
        (lost/zombie queue entries).

        Adaptive mode adds: (a) fleet RUNNING accounting — first RUNNING
        increments the shared counter (and feeds the admission probe's
        C_eff), terminal decrements it, every poll appends a sample; (b)
        straggler eviction — a job still PENDING past pending_grace_min
        while siblings RUN is over-admission: cancel + ConfigEvictedError
        (the config is dropped from the iteration, never re-run)."""
        timeout = timeout if timeout is not None else self.remote.job_timeout_s
        queue_timeout = self.remote.queue_timeout_s
        submitted_at = time.time()
        first_running_at: Optional[float] = None
        counted_running = False
        last_print = 0.0
        while True:
            st = self.job_api.status(job_id)
            now = time.time()
            if st.is_terminal:
                if counted_running:
                    with self._run_lock:
                        self._running_now -= 1
                return st
            if first_running_at is None and st == JobStatus.RUNNING:
                first_running_at = now
                counted_running = True
                with self._run_lock:
                    self._running_now += 1
                adm = self._admission
                if adm is not None:
                    adm.note_running(now)
            if counted_running:
                # Sample the fleet's concurrent RUNNING count for C_eff
                # (median over the steady-state window at batch end).
                with self._run_lock:
                    self._run_samples.append(self._running_now)
                    if len(self._run_samples) > 2400:
                        del self._run_samples[:1200]
            if first_running_at is None:
                # queue phase (PENDING/UNKNOWN): submission → start clock
                if now - submitted_at > queue_timeout:
                    self.job_api.cancel(job_id)
                    raise QueueTimeoutError(
                        f"remote job {job_id} (exp {experiment_id}) never "
                        f"started — queued {(now - submitted_at)/60:.0f}m "
                        f"(limit {queue_timeout/60:.0f}m); cancelled. "
                        f"logs (tail):\n{self.job_api.logs(job_id, 20)}")
                # Straggler eviction (adaptive only): PENDING past grace
                # while siblings run = over-admission (the realized pool
                # is smaller than the probe admitted).
                adm = self._admission
                if (adm is not None and adm.pending_grace_s > 0
                        and now - submitted_at > adm.pending_grace_s):
                    with self._run_lock:
                        fleet_running = self._running_now
                    if fleet_running > 0:
                        self.job_api.cancel(job_id)
                        raise ConfigEvictedError(
                            f"remote job {job_id} (exp {experiment_id}) sat "
                            f"PENDING {(now - submitted_at)/60:.0f}m > grace "
                            f"{adm.pending_grace_s/60:.0f}m while siblings "
                            f"ran — over-admission; config dropped from the "
                            f"iteration (never re-run)")
            elif now - first_running_at > timeout:
                self.job_api.cancel(job_id)
                raise RuntimeError(
                    f"remote job {job_id} (exp {experiment_id}) timed out "
                    f"after {(now - first_running_at)/60:.0f}m of RUNTIME "
                    f"(limit {timeout/60:.0f}m, queue time excluded); "
                    f"cancelled. logs (tail):\n{self.job_api.logs(job_id, 20)}")
            if now - last_print >= self.remote.status_print_interval_s:
                tail = self.job_api.logs(job_id, 1).strip()
                phase = "running" if first_running_at is not None else "queued"
                print(f"  [Exp {experiment_id}] job {job_id} {st.value} "
                      f"{phase} {(now - submitted_at)/60:.0f}m | {tail[:120]}")
                last_print = now
            time.sleep(self.remote.poll_interval_s)

    # ── single remote experiment (runs inside a worker thread) ──

    def _run_remote_experiment(
        self,
        mixture_config: MixtureConfig,
        experiment_id: int,
        output_dir: Optional[str] = None,
    ) -> ProxyResult:
        output_dir = output_dir or self.config.output_dir
        exp_dir = os.path.join(output_dir, f"exp_{experiment_id:04d}")
        meta_path = os.path.join(exp_dir, "meta.json")
        model_tag = f"climbmix_{self.experiment_name}_{experiment_id:04d}"
        t_start = time.time()
        exp_obs = self._exp_obs_prefix(experiment_id)
        result_uri = f"{exp_obs}/result"
        mix_uri = f"{exp_obs}/mixture_data"

        # Resume 1: completed experiment — identical semantics to local.
        reused = self._load_completed_result(meta_path, mixture_config)
        if reused is not None:
            print(f"\n  [Exp {experiment_id}] Reusing completed experiment "
                  f"(tag={model_tag}, weights match, rc=0/0)")
            shutil.rmtree(os.path.join(exp_dir, "mixture_data"), ignore_errors=True)
            return reused

        # Resume 2: training already done?
        eval_only = False
        if self._load_mid_train_marker(exp_dir, mixture_config, model_tag):
            print(f"\n  [Exp {experiment_id}] mid_train already complete "
                  f"(local marker, tag={model_tag}) — uploading ckpt, eval-only job")
            self._upload_local_mid_ckpt(model_tag, result_uri)
            eval_only = True
        elif self._load_remote_mid_marker(exp_dir, mixture_config, model_tag):
            print(f"\n  [Exp {experiment_id}] mid_train already complete "
                  f"(remote marker, tag={model_tag}) — eval-only job")
            eval_only = True

        if not eval_only:
            # Fresh (re)run: clear partial state, prep + upload the shards.
            if self._submit_hard_error is not None:
                # A sibling already hit a hard submit error (bad image/auth):
                # every further submission would fail identically — burn this
                # config now instead of wasting prep+upload on it.
                raise RuntimeError(
                    f"submission broken since hard error: "
                    f"{self._submit_hard_error}")
            if os.path.isdir(exp_dir):
                shutil.rmtree(exp_dir)
            os.makedirs(exp_dir, exist_ok=True)
            # Stale OBS artifacts from a PREVIOUS attempt must not
            # linger: the worker uploads result.json/logs only AFTER a
            # stage finishes, so until then the result prefix still
            # holds the old attempt's files — a stale mid_train.log
            # there reads as THIS attempt's failure (live: user tailed
            # the previous job's log while the new one was training).
            for uri in (f"{result_uri}/result.json",
                        f"{result_uri}/mid_train.log",
                        f"{result_uri}/eval.log",
                        f"{result_uri}/eval_{model_tag}.csv"):
                if self.obs.stat(uri):
                    self.obs.delete(uri)
            for uri in self.obs.list_objects(mix_uri):
                self.obs.delete(uri)
            print(f"\n  [Exp {experiment_id}] Starting REMOTE proxy experiment "
                  f"(d{self.proxy_depth}, tag={model_tag}, "
                  f"npu_per_job={self.remote.npu_per_job})")
            print(f"  [Exp {experiment_id}] Preparing mixture-weighted data "
                  f"({self.stem_ratio*100:.0f}% STEM + "
                  f"{(1-self.stem_ratio)*100:.0f}% general)...")
            mixture_data_dir = os.path.join(exp_dir, "mixture_data")
            # Prep+upload under the semaphore: bounded local load while the
            # submit threads (max_concurrent_jobs of them) stay free to
            # submit/wait — prepped specs feed submissions continuously.
            with self._prep_sem:
                self._prepare_mixture_data(
                    mixture_config, experiment_id, mixture_data_dir,
                    nproc_per_node=self.remote.npu_per_job)
                self._upload_dir(mixture_data_dir, mix_uri)
            # The OBS copy is the source of truth for the job; free local disk.
            shutil.rmtree(mixture_data_dir, ignore_errors=True)

        # Commands are built by the SHARED builders with CONTAINER paths —
        # the job runs exactly the argv a local executor would run.
        container_mix_dir = os.path.join(
            self.remote.container_work_root,
            f"exp_{experiment_id:04d}", "mixture_data")
        mid_cmd = self._build_mid_train_cmd(
            model_tag, container_mix_dir,
            nproc_per_node=self.remote.npu_per_job, master_port=None)
        eval_cmd = self._build_eval_cmd(
            model_tag, nproc_per_node=self.remote.npu_per_job, master_port=None)

        base_ckpt_src = self.remote.container_base_ckpt_src or os.path.join(
            self.remote.container_base_dir, "base_checkpoints",
            f"d{self.proxy_depth}")
        spec = ExpSpec(
            experiment_id=experiment_id,
            experiment_name=self.experiment_name,
            model_tag=model_tag,
            weights=mixture_config.mixture_weights.weights.tolist(),
            nanochat_dir=self.remote.container_nanochat_dir,
            base_dir=self.remote.container_base_dir,
            work_dir=os.path.join(self.remote.container_work_root,
                                  f"exp_{experiment_id:04d}"),
            base_ckpt_src=base_ckpt_src,
            mixture_data_uri=mix_uri,
            result_uri=result_uri,
            mid_train_cmd=mid_cmd,
            eval_cmd=eval_cmd,
            eval_only=eval_only,
            upload_checkpoint=self.remote.upload_checkpoint,
            visible_devices=list(range(self.remote.npu_per_job)),
            env=dict(self.remote.job_env),
        )
        spec_uri = f"{exp_obs}/spec.json"
        self.obs.upload_bytes(spec.to_json().encode("utf-8"), spec_uri)

        job_id, job_status = self._submit_and_wait(experiment_id, spec_uri)

        # ── materialize the result into exp_dir ──
        # The worker uploads result.json even for KNOWN-stage failures
        # (train/eval rc != 0 — job status FAILED); it is missing only for
        # infrastructure failures (node death, worker crash, timeout).
        try:
            res_bytes = self.obs.download_bytes(f"{result_uri}/result.json")
            res = json.loads(res_bytes.decode("utf-8"))
        except Exception as e:
            raise RuntimeError(
                f"remote job {job_id} for experiment {experiment_id} "
                f"({job_status.value}) produced no readable result.json at "
                f"{result_uri}/result.json ({type(e).__name__}: {e}); "
                f"logs (tail):\n{self.job_api.logs(job_id, 20)}") from e
        with open(os.path.join(exp_dir, "_remote_result.json"), "w") as f:
            f.write(json.dumps(res, indent=2))

        for log_name in ("mid_train.log", "eval.log"):
            uri = f"{result_uri}/{log_name}"
            if self.obs.stat(uri):
                self.obs.download_file(uri, os.path.join(exp_dir, log_name))

        mid_rc = int(res.get("mid_train_rc", -1))
        eval_rc = int(res.get("eval_rc", -1))

        # Markers BEFORE the failure raise: a train-ok/eval-failed job must
        # resume eval-only, not retrain (mirrors the local marker write that
        # happens right after a successful mid_train).
        if mid_rc == 0:
            if not eval_only:
                self._write_mid_train_marker(exp_dir, mixture_config, model_tag)
            if self.remote.upload_checkpoint:
                self._write_remote_mid_marker(exp_dir, mixture_config,
                                              model_tag, result_uri)
        if mid_rc != 0 or eval_rc != 0:
            raise RuntimeError(
                f"remote job {job_id} for experiment {experiment_id} failed "
                f"(mid_train_rc={mid_rc}, eval_rc={eval_rc}); logs (tail):\n"
                f"{self.job_api.logs(job_id, 20)}")

        csv_uri = f"{result_uri}/eval_{model_tag}.csv"
        csv_path: Optional[str] = None
        if self.obs.stat(csv_uri):
            csv_path = os.path.join(exp_dir, f"eval_{model_tag}.csv")
            self.obs.download_file(csv_uri, csv_path)

        if self.remote.download_checkpoint:
            ckpt_uri = f"{result_uri}/mid_checkpoint"
            if self.obs.list_objects(ckpt_uri):
                dst = os.path.join(exp_dir, "mid_checkpoint")
                os.makedirs(dst, exist_ok=True)
                for obj in self.obs.list_objects(ckpt_uri):
                    name = obj.rsplit("/", 1)[-1]
                    self.obs.download_file(obj, os.path.join(dst, name))

        # Shared tail with the local executor: parse CSV, write meta.json.
        return self._finalize_exp(
            exp_dir=exp_dir, model_tag=model_tag,
            mixture_config=mixture_config, experiment_id=experiment_id,
            csv_path=csv_path, mid_rc=mid_rc, eval_rc=eval_rc,
            t_start=t_start, copy_ckpt=False,
        )

    # ── batch orchestration ──

    def _local_slots(self) -> int:
        # npu_per_exp == npu_devices = ONE whole-node slot: the parent's
        # run_batch takes the serial full-card path (one experiment on all
        # cards at a time) — prod2's k=8 form. Proper divisors slice as
        # before; anything else idles.
        if (self.npu_per_exp and self.npu_per_exp <= self.npu_devices
                and self.npu_devices % self.npu_per_exp == 0):
            return self.npu_devices // self.npu_per_exp
        return 0

    def _remote_worker_loop(
        self,
        q: "queue.Queue[int]",
        results: List[Optional[ProxyResult]],
        remote_configs: List[MixtureConfig],
        offset: int,
        experiment_id_base: int,
        output_dir: Optional[str],
    ) -> None:
        """Queue consumer: pick the next queued config, wait for an
        in-flight SLOT (dynamic capacity), run its full lifecycle
        (prep -> submit -> wait -> materialize), release the slot, repeat.
        Queue items are GLOBAL results indices (offset + remote index) so
        local-slice slots are never clobbered. Taking from the queue only
        after acquiring a slot means queued configs are NEVER prepped early
        (no OBS/disk pileup for the ~90 configs waiting behind a 16-card
        pool), and a finished job frees its slot instantly — the next
        config starts with ZERO backoff delay. Per-exp failure keeps the
        burn semantics (inf/0.0 result)."""
        while True:
            try:
                gidx = q.get_nowait()
            except queue.Empty:
                return
            adm = self._admission
            if adm is not None and gidx in adm.drop_set:
                # Adaptive truncation dropped this config before pickup —
                # zero cost, results slot stays None (the bootstrapper's
                # dropped-configs handling filters it out).
                continue
            exp_id = experiment_id_base + gidx
            try:
                self._acquire_slot()
                try:
                    results[gidx] = self._run_remote_experiment(
                        remote_configs[gidx - offset], exp_id, output_dir)
                finally:
                    self._release_slot()
            except ConfigEvictedError as e:
                # Over-admission eviction (never trained, never re-run):
                # NOT a failure — results slot stays None and the config is
                # recorded as evicted for the admission stats.
                print(f"  [Exp {exp_id}] EVICTED: {e}")
                results[gidx] = None
                if self._admission is not None:
                    self._admission.evicted.append(gidx)
                    self._admission.drop_set.add(gidx)
            except Exception as e:
                print(f"  [Exp {exp_id}] FAILED: {e}")
                results[gidx] = ProxyResult(
                    mixture_config=remote_configs[gidx - offset],
                    validation_loss=float("inf"),
                    validation_accuracy=0.0,
                    validation_nll=float("inf"),
                    per_task_accuracies={},
                    per_task_nlls={},
                    metadata={"experiment_id": exp_id, "error": str(e)},
                )

    def _admission_monitor(
        self,
        stop: threading.Event,
        adm: "AdmissionController",
        batch_started_at: float,
        remote_q: "queue.Queue[int]",
        n_total: int,
        n_local: int,
    ) -> None:
        """Applies the AdmissionController's probe decision once the probe
        window closes (first RUNNING + probe_delay_s, or probe_deadline_s
        after batch start). Drops are marked in the shared drop_set —
        workers skip marked indices at pickup, and the end-of-batch
        reconciliation excludes any marked index that ran anyway."""
        while not stop.is_set() and not adm.decided:
            if stop.wait(self.remote.poll_interval_s):
                break
            now = time.time()
            if not adm.probe_ready(now, batch_started_at):
                continue
            with self._run_lock:
                running = self._running_now
            local_slots = self._local_slots() if n_local > 0 else 0
            drops = adm.decide(n_total, n_local, running, local_slots,
                               remote_q.qsize())
            if drops:
                print(f"  [Adaptive] probe: C_eff={adm.c_eff} "
                      f"(remote RUNNING={running} + local={local_slots}) x "
                      f"budget {adm.wave_budget} wave(s) -> admit "
                      f"{n_total - len(adm.drop_set)}/{n_total}; dropped "
                      f"{len(drops)} queued config(s) from the tail")
            else:
                reason = ("truncation disabled" if not adm.allow_truncate
                          else f"C_eff={adm.c_eff} < 2")
                print(f"  [Adaptive] probe: {reason} -> admit all "
                      f"{n_total} (literal behavior, queue patience applies)")

    def _compute_effective_concurrency(self, has_local: bool) -> Optional[int]:
        """Median remote RUNNING over the steady-state window + local
        slots. None when no job ever ran (no usable measurement)."""
        with self._run_lock:
            samples = list(self._run_samples)
        if not samples:
            return None
        window = samples[-120:]
        if len(window) >= 8:
            window = window[len(window) // 2:]  # drop the startup ramp
        med = int(round(statistics.median(window)))
        return med + (self._local_slots() if has_local else 0)

    def run_batch(
        self,
        configs: List[MixtureConfig],
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        experiment_id_base: int = 0,
        admission: Optional[Dict[str, object]] = None,
    ) -> List[Optional[ProxyResult]]:
        """Same contract as ProxyRunner.run_batch (probed by the bootstrapper
        for experiment_id_base). Mixed fleet when local_parallel: the first
        _local_slots() configs run via the parent's local parallel path, the
        rest as remote jobs; all concurrent, results merged in input order.

        Remote side is a DYNAMIC queue: workers (up to max_concurrent_jobs)
        pull configs only while the in-flight limit allows it. The limit
        starts from a synchronous capacity probe (16 free cards, k=2 ->
        8 jobs start immediately) and a monitor thread keeps re-probing:
        capacity appearing mid-run wakes queued workers instantly (a 98-exp
        iteration with a 16-card pool drains as cards free up / the pool
        grows, across as many submission rounds as the pool dictates).
        Without capacity queries (free_job_slots -> None) the limit simply
        stays at max_concurrent_jobs and submit-rejected backoff handles
        over-admission.

        admission (prod2 B++, optional dict): {wave_budget, expected_slots,
        min_configs, allow_truncate} — activates the AdmissionController
        (probe-truncate + straggler eviction). CONTRACT CHANGE in that
        mode: dropped configs come back as None entries in the results
        list (never trained, never re-run — the bootstrapper rewrites its
        pending list from last_dropped_configs); without admission the
        return value is exactly the old list-of-ProxyResult."""
        if self.remote.local_parallel and self._local_slots() > 0:
            n_local = self._local_slots()
            local_configs = configs[:n_local]
            remote_configs = configs[n_local:]
        else:
            local_configs = []
            remote_configs = list(configs)
        offset = len(local_configs)

        results: List[Optional[ProxyResult]] = [None] * len(configs)

        # Per-batch state resets: dynamic capacity, running registry,
        # adaptive accounting.
        with self._cap_cond:
            self._cap_limit = self.remote.max_concurrent_jobs
            self._cap_inflight = 0
        with self._run_lock:
            self._running_now = 0
            self._run_samples = []
        self._admission = None
        self.last_dropped_configs = []
        self.last_admission_stats = {}
        adm: Optional[AdmissionController] = None
        if admission:
            adm = AdmissionController(
                wave_budget=int(admission.get("wave_budget", 1)),
                expected_slots=int(admission.get("expected_slots", 1)),
                min_configs=int(admission.get("min_configs", 4)),
                allow_truncate=bool(admission.get("allow_truncate", True)),
                # probe windows are overridable purely for fast tests
                probe_delay_s=float(admission.get("probe_delay_s", 1800.0)),
                probe_deadline_s=float(
                    admission.get("probe_deadline_s", 2400.0)),
                pending_grace_s=self.remote.pending_grace_min * 60.0,
            )
            self._admission = adm
            print(f"  [Adaptive] wave budget {adm.wave_budget} "
                  f"(expected {adm.expected_slots} slots, pool "
                  f"{len(configs)} configs, grace "
                  f"{self.remote.pending_grace_min:.0f}m)")

        initial_slots = self._probe_slots()
        self._adjust_capacity_limit(initial_slots)
        remote_q: "queue.Queue[int]" = queue.Queue()
        for i in range(len(remote_configs)):
            remote_q.put(offset + i)  # GLOBAL results index
        stop = threading.Event()
        monitor = None
        if initial_slots is not None:
            monitor = threading.Thread(
                target=self._capacity_monitor, args=(stop,),
                name="remote-capacity-monitor", daemon=True)
            monitor.start()
        adm_monitor = None
        if adm is not None:
            adm_monitor = threading.Thread(
                target=self._admission_monitor,
                args=(stop, adm, time.time(), remote_q, len(configs),
                      len(local_configs)),
                name="remote-admission-monitor", daemon=True)
            adm_monitor.start()

        # Worker threads at the UPPER bound: when the pool grows mid-batch
        # the monitor raises the limit and parked workers wake up to pick
        # queued configs. With few configs, surplus workers find an empty
        # queue and exit immediately.
        n_threads = self.remote.max_concurrent_jobs + (1 if local_configs else 0)
        try:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                local_future = None
                if local_configs:
                    local_future = pool.submit(
                        super().run_batch, local_configs,
                        data_dir=data_dir, output_dir=output_dir,
                        experiment_id_base=experiment_id_base)
                for _ in range(self.remote.max_concurrent_jobs):
                    pool.submit(
                        self._remote_worker_loop, remote_q, results,
                        remote_configs, offset, experiment_id_base,
                        output_dir)
                if local_future is not None:
                    for j, r in enumerate(local_future.result()):
                        results[j] = r
        finally:
            stop.set()
            if monitor is not None:
                monitor.join(timeout=self.remote.poll_interval_s * 3)
            if adm_monitor is not None:
                adm_monitor.join(timeout=self.remote.poll_interval_s * 3)

        # ── post-batch adaptive accounting (bootstrapper reads these) ──
        self.last_effective_concurrency = self._compute_effective_concurrency(
            bool(local_configs))
        if self._admission is not None:
            adm = self._admission
            # Reconciliation: only indices whose results are None count as
            # dropped — a marked index a worker had already picked simply
            # ran and is excluded (over-admission self-corrects to the
            # realized set, never silently discards a trained result).
            dropped = sorted(g for g in adm.drop_set
                             if g < len(results) and results[g] is None)
            self.last_dropped_configs = [configs[g] for g in dropped]
            self.last_admission_stats = {
                "c_eff": adm.c_eff,
                "wave_budget": adm.wave_budget,
                "expected_slots": adm.expected_slots,
                "truncated": [g for g in adm.truncated
                              if g < len(results) and results[g] is None],
                "evicted": [g for g in adm.evicted
                            if g < len(results) and results[g] is None],
                "ran_despite_mark": [g for g in sorted(adm.drop_set)
                                     if g < len(results)
                                     and results[g] is not None],
            }
            self._admission = None

        return results
