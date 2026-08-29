"""RemoteExecutor — runs proxy experiments as ModelArts jobs, results land as
local exp_XXXX/ dirs.

Architecture (2026-08-28 production plan):
  - LOCAL host = scheduler + mixer: mixture preparation needs the cluster
    labels + the STEM pool, which live here. Shards (~1.5GB/exp) upload to OBS.
  - OBS = data plane: {obs_prefix}/exps/exp_XXXX/{spec.json, mixture_data/,
    result/} per experiment; {obs_prefix}/assets/ = worker code bundle.
  - ModelArts Job API = compute plane: one job per experiment
    (npu_per_job cards each, no cross-node collectives).
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
import json
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from climbmix.core.types import CLIMBConfig, MixtureConfig, ProxyResult
from climbmix.pipeline.proxy_runner import ProxyRunner
from climbmix.remote.exp_spec import ExpSpec, SPEC_VERSION
from climbmix.remote.job_api import JobStatus, MockJobAPI, TransientSubmitError
from climbmix.remote.modelarts_job_api import LOCAL_CODE_DIR, ModelArtsJobAPI
from climbmix.remote.obs import MockObsStorage, ModelArtsObsStorage


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

    # Which JobAPI/ObsStorage to construct when not injected (tests inject
    # mocks): "modelarts" (real) or "mock" (local simulation).
    backend: str = "modelarts"

    # Path to the ModelArts platform config JSON (gateway endpoint, IDs,
    # image, auth — the repo is PUBLIC so real values live outside it in
    # ~/.config/climbmix/remote_ma.json; schema: config/
    # remote_ma.example.json). Empty = default resolution
    # ($CLIMBMIX_MA_CONFIG then ~/.config/climbmix/remote_ma.json).
    ma_config: str = ""

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
    # ProxyRunner parallel path (requires npu_per_exp in [1, npu_devices)).
    # Configs[:n_local] run locally, the rest remotely.
    local_parallel: bool = False

    # ── artifacts ──
    upload_checkpoint: bool = True    # worker -> OBS after successful train
    download_checkpoint: bool = False  # OBS -> local exp_dir (debug only)

    # ── polling ──
    poll_interval_s: float = 30.0
    status_print_interval_s: float = 300.0
    job_timeout_s: float = 6 * 3600.0

    # Job-level env (HF_ENDPOINT=hf-mirror.com, ...). Passed to the job
    # process AND baked into the spec for the train/eval subprocesses.
    job_env: Dict[str, str] = field(default_factory=dict)

    # Local staging dir for the worker assets bundle (default:
    # {repo}/cache/remote_assets — outside the fingerprinted output dir).
    assets_stage_dir: str = ""

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
            elif isinstance(cur, dict):
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
        if self.backend not in ("modelarts", "mock"):
            raise ValueError(f"RemoteConfig.backend must be modelarts|mock")
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


class RemoteExecutor(ProxyRunner):
    """ProxyRunner subclass whose execution backend is ModelArts jobs.

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

        if job_api is not None:
            self.job_api = job_api
        elif remote_config.backend == "mock":
            self.job_api = MockJobAPI()
        else:
            self.job_api = ModelArtsJobAPI(remote_config)
        if obs is not None:
            self.obs = obs
        elif remote_config.backend == "mock":
            self.obs = MockObsStorage(remote_config.storage_root)
        else:
            self.obs = ModelArtsObsStorage(remote_config.ma_config or None)

        self._stage_assets()
        if not remote_config.worker_path:
            if remote_config.backend == "mock":
                remote_config.worker_path = os.path.join(
                    self.assets_stage_dir, "remote_worker.py")
            else:
                # Platform copies code_dir ({obs_prefix}/assets) into
                # LOCAL_CODE_DIR; the boot shell resolves the exact path at
                # runtime (basename nesting varies).
                remote_config.worker_path = os.path.join(
                    LOCAL_CODE_DIR, "remote_worker.py")
        self._ensure_assets_uploaded()

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
                      f"proper divisor of npu_devices (< npu_devices) so the "
                      f"master node joins the fleet")
            elif remote_config.npu_per_job != self.npu_per_exp:
                print(f"  [RemoteExecutor] WARNING: local slice runs "
                      f"npu_per_exp={self.npu_per_exp} card(s) per experiment "
                      f"but remote jobs run npu_per_job="
                      f"{remote_config.npu_per_job}; k should stay fleet-wide "
                      f"fixed for score comparability")

    # ── assets bundle (worker + shared cmds module) ──

    def _stage_assets(self) -> None:
        """Copy remote_worker.py + nanochat_cmds.py into the local staging
        dir (fresh on every init — the staged copy always matches the running
        code, which is what gets uploaded to OBS and executed in jobs)."""
        import climbmix
        repo_root = os.path.normpath(
            os.path.join(os.path.dirname(climbmix.__file__), "..", ".."))
        stage = self.remote.assets_stage_dir or os.path.join(
            repo_root, "cache", "remote_assets")
        os.makedirs(stage, exist_ok=True)
        self.assets_stage_dir = stage
        src_worker = os.path.join(repo_root, "scripts", "remote_worker.py")
        src_cmds = os.path.join(os.path.dirname(climbmix.__file__),
                                "pipeline", "nanochat_cmds.py")
        for src in (src_worker, src_cmds):
            if not os.path.isfile(src):
                raise FileNotFoundError(f"remote asset missing: {src}")
            dst = os.path.join(stage, os.path.basename(src))
            shutil.copy2(src, dst)

    def _ensure_assets_uploaded(self) -> None:
        assets_uri = f"{self.remote.obs_prefix.rstrip('/')}/assets"
        with self._obs_lock:
            if not self.obs.stat(f"{assets_uri}/remote_worker.py"):
                self.obs.upload_file(
                    os.path.join(self.assets_stage_dir, "remote_worker.py"),
                    f"{assets_uri}/remote_worker.py")
                self.obs.upload_file(
                    os.path.join(self.assets_stage_dir, "nanochat_cmds.py"),
                    f"{assets_uri}/nanochat_cmds.py")
                print(f"  [RemoteExecutor] uploaded worker assets -> {assets_uri}")

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
                return self.job_api.submit(name=name, command=command, env=env)
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

    def _wait_job(self, job_id: str, experiment_id: int,
                  timeout: Optional[float] = None) -> JobStatus:
        """Poll until the job reaches a terminal state; RETURNS the status
        (does not raise on FAILED — the result.json the worker uploads on
        known-stage failures carries the precise rc's, and the caller needs
        it to write the eval-only resume markers). Raises only on timeout
        (job cancelled)."""
        timeout = timeout if timeout is not None else self.remote.job_timeout_s
        t0 = time.time()
        last_print = 0.0
        while True:
            st = self.job_api.status(job_id)
            if st.is_terminal:
                return st
            elapsed = time.time() - t0
            if elapsed > timeout:
                self.job_api.cancel(job_id)
                raise RuntimeError(
                    f"remote job {job_id} (exp {experiment_id}) timed out "
                    f"after {elapsed/60:.0f}m (limit {timeout/60:.0f}m); "
                    f"cancelled. logs (tail):\n{self.job_api.logs(job_id, 20)}")
            now = time.time()
            if now - last_print >= self.remote.status_print_interval_s:
                tail = self.job_api.logs(job_id, 1).strip()
                print(f"  [Exp {experiment_id}] job {job_id} {st.value} "
                      f"{elapsed/60:.0f}m | {tail[:120]}")
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

        job_id = self._submit_with_retry(
            name=self._job_name(experiment_id),
            command=self._worker_argv(spec_uri),
            env=dict(self.remote.job_env),
            experiment_id=experiment_id,
        )
        print(f"  [Exp {experiment_id}] submitted job {job_id} "
              f"(spec: {spec_uri})")

        job_status = self._wait_job(job_id, experiment_id)

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
        if (self.npu_per_exp and self.npu_per_exp < self.npu_devices
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
            exp_id = experiment_id_base + gidx
            try:
                self._acquire_slot()
                try:
                    results[gidx] = self._run_remote_experiment(
                        remote_configs[gidx - offset], exp_id, output_dir)
                finally:
                    self._release_slot()
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

    def run_batch(
        self,
        configs: List[MixtureConfig],
        data_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        experiment_id_base: int = 0,
    ) -> List[ProxyResult]:
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
        over-admission."""
        if self.remote.local_parallel and self._local_slots() > 0:
            n_local = self._local_slots()
            local_configs = configs[:n_local]
            remote_configs = configs[n_local:]
        else:
            local_configs = []
            remote_configs = list(configs)
        offset = len(local_configs)

        results: List[Optional[ProxyResult]] = [None] * len(configs)

        # (re)set per-batch dynamic capacity; synchronous initial probe
        with self._cap_cond:
            self._cap_limit = self.remote.max_concurrent_jobs
            self._cap_inflight = 0
        initial_slots = self._probe_slots()
        self._adjust_capacity_limit(initial_slots)
        stop = threading.Event()
        monitor = None
        if initial_slots is not None:
            monitor = threading.Thread(
                target=self._capacity_monitor, args=(stop,),
                name="remote-capacity-monitor", daemon=True)
            monitor.start()

        remote_q: "queue.Queue[int]" = queue.Queue()
        for i in range(len(remote_configs)):
            remote_q.put(offset + i)  # GLOBAL results index

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

        return results
