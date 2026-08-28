"""JobAPI — thin job-submission interface for remote experiment execution.

Two implementations:
  - MockJobAPI: executes the command argv as a LOCAL subprocess (used by the
    laptop simulation /tests; combined with MockObsStorage it exercises the
    full RemoteExecutor -> worker -> storage -> materialization stack with
    zero mocks inside the worker itself).
  - ModelArtsJobAPI: real ModelArts training-job adapter (M1 deliverable —
    the SDK/auth specifics get filled in after the environment survey; the
    interface is final).

Contract: submit() takes the WORKER argv (python remote_worker.py --spec-uri
obs://... --storage ...) plus env; the adapter is responsible for making that
argv run in the target environment (e.g. wrapping it into the job's boot
shell with asset download steps).
"""

import os
import subprocess
import threading
import time
import uuid
from enum import Enum
from typing import Dict, List, Optional, Protocol, runtime_checkable


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED,
                        JobStatus.CANCELLED)


class TransientSubmitError(Exception):
    """submit() rejected for a RETRYABLE reason — capacity/quota exhausted,
    API throttling, transient service errors. The caller should back off and
    retry: shared NPU pools fluctuate (10-200 cards), so a rejected job now
    often fits minutes later. Hard failures (bad image/auth/argv) keep
    raising RuntimeError — retrying those is pointless."""


@runtime_checkable
class JobAPI(Protocol):
    def submit(self, name: str, command: List[str],
               env: Optional[Dict[str, str]] = None,
               workdir: Optional[str] = None) -> str: ...
    def status(self, job_id: str) -> JobStatus: ...
    def logs(self, job_id: str, tail: int = 50) -> str: ...
    def cancel(self, job_id: str) -> None: ...


class MockJobAPI:
    """Local in-process fake: runs the submitted argv via subprocess.Popen in
    a daemon thread and reports the process state as job status.

    stdout+stderr are captured to a per-job file so logs() works like a real
    job console. Used with MockObsStorage (same local root) the simulation is
    end-to-end: the REAL remote_worker.py runs as a REAL subprocess against a
    filesystem-backed fake OBS.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        self._log_dir = log_dir or os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "mock_jobs")
        os.makedirs(self._log_dir, exist_ok=True)
        self.submit_count = 0
        self.submit_attempts = 0
        # Test injection knobs (capacity simulation):
        #   fail_submits_remaining: next N submit ATTEMPTS raise
        #     TransientSubmitError ("pool full") — exercises backoff/retry.
        #   fail_submits_hard: next N submit attempts raise RuntimeError —
        #     a hard fleet error (bad image/auth).
        self.fail_submits_remaining = 0
        self.fail_submits_hard = 0

    def submit(self, name: str, command: List[str],
               env: Optional[Dict[str, str]] = None,
               workdir: Optional[str] = None) -> str:
        with self._lock:
            self.submit_attempts += 1
            if self.fail_submits_remaining > 0:
                self.fail_submits_remaining -= 1
                raise TransientSubmitError(
                    f"mock capacity exhausted ({self.fail_submits_remaining} "
                    f"rejections queued after this one)")
            if self.fail_submits_hard > 0:
                self.fail_submits_hard -= 1
                raise RuntimeError("mock hard submit error (bad image)")
        job_id = f"mock-{uuid.uuid4().hex[:12]}"
        log_path = os.path.join(self._log_dir, f"{job_id}.log")
        popen_env = os.environ.copy()
        if env:
            popen_env.update(env)
        with open(log_path, "w") as log_f:
            proc = subprocess.Popen(
                command, cwd=workdir, env=popen_env,
                stdout=log_f, stderr=subprocess.STDOUT)
        with self._lock:
            self._jobs[job_id] = {
                "name": name, "proc": proc, "log_path": log_path,
                "rc": None, "cancelled": False,
            }
            self.submit_count += 1
        return job_id

    def _get(self, job_id: str) -> dict:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"unknown job_id: {job_id}")
            return self._jobs[job_id]

    def status(self, job_id: str) -> JobStatus:
        job = self._get(job_id)
        if job["cancelled"]:
            return JobStatus.CANCELLED
        rc = job["proc"].poll()
        if rc is None:
            return JobStatus.RUNNING
        job["rc"] = rc
        return JobStatus.SUCCEEDED if rc == 0 else JobStatus.FAILED

    def logs(self, job_id: str, tail: int = 50) -> str:
        job = self._get(job_id)
        try:
            with open(job["log_path"], "rb") as f:
                data = f.read().decode("utf-8", errors="replace")
        except OSError:
            return "(no logs)"
        lines = data.splitlines()
        return "\n".join(lines[-tail:]) if tail > 0 else data

    def cancel(self, job_id: str) -> None:
        job = self._get(job_id)
        job["cancelled"] = True
        if job["proc"].poll() is None:
            job["proc"].kill()
            job["proc"].wait(timeout=30)

    # Test helper: block until every submitted job reaches a terminal state
    # (the executor's poll loop normally does this; useful for assertions).
    def wait_all(self, timeout: float = 120.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                jobs = list(self._jobs.values())
            if all(j["proc"].poll() is not None or j["cancelled"]
                   for j in jobs):
                return
            time.sleep(0.05)


class ModelArtsJobAPI:
    """Real ModelArts training-job adapter — SKELETON (M1 deliverable).

    The submit host needs (environment survey, docs/remote_setup.md):
      - SDK choice: ma-sdk (modelarts) vs raw REST vs moxing job helpers
      - auth: AK/SK + project_id + region (env MA_AK / MA_SK /
        MA_PROJECT_ID / MA_REGION, or ~/.ma_creds)
      - pool/flavor: dedicated pool name + Ascend 910B4 flavor
      - image: SWR URI of the baked image (torch_npu + CANN 8.5.1 + pyarrow)

    submit() composes the job's boot shell around the worker argv:
      download assets bundle (cached in /home/ma-user/work) ->
      python remote_worker.py --spec-uri ... --storage moxing
    Error mapping (M1): API responses meaning quota/capacity/throttling
    raise TransientSubmitError (executor backs off and retries — the pool
    fluctuates); everything else raises RuntimeError (hard, burns the
    config like any experiment failure).
    The interface below is final; only the SDK calls are missing.
    """

    def __init__(self):
        missing = [k for k in ("MA_AK", "MA_SK", "MA_PROJECT_ID", "MA_REGION")
                   if not os.environ.get(k)]
        if missing:
            raise NotImplementedError(
                f"ModelArtsJobAPI: real adapter pending M1 environment survey "
                f"(docs/remote_setup.md). Missing env config: {missing}. "
                f"For local simulation use MockJobAPI.")

    def submit(self, name: str, command: List[str],
               env: Optional[Dict[str, str]] = None,
               workdir: Optional[str] = None) -> str:
        raise NotImplementedError("ModelArtsJobAPI.submit: fill in after M1")

    def status(self, job_id: str) -> JobStatus:
        raise NotImplementedError("ModelArtsJobAPI.status: fill in after M1")

    def logs(self, job_id: str, tail: int = 50) -> str:
        raise NotImplementedError("ModelArtsJobAPI.logs: fill in after M1")

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError("ModelArtsJobAPI.cancel: fill in after M1")
