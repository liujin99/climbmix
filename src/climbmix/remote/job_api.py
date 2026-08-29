"""JobAPI — thin job-submission interface for remote experiment execution.

Two kinds of implementations:
  - MockJobAPI (this file): executes the command argv as a LOCAL subprocess
    (used by the laptop simulation /tests; combined with MockObsStorage it
    exercises the full RemoteExecutor -> worker -> storage -> materialization
    stack with zero mocks inside the worker itself).
  - Real platform backends live OUT of this repo (access-restricted
    adapter repositories; see backends.py and docs/remote_setup.md
    "Writing a backend"): they implement this protocol over their
    platform's job-submission API and carry the platform identity values
    in a config file outside the public repo.

Contract: submit() takes the WORKER argv (python remote_worker.py --spec-uri
obs://... --storage ...) plus env; the adapter is responsible for making that
argv run in the target environment (an adapter may wrap it into a boot
shell that bootstraps assets, then execs the argv).
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
    # Optional capacity probe (dynamic scheduling): the number of jobs that
    # could be submitted RIGHT NOW without a quota rejection, or None when
    # the backend has no query API (executor then falls back to
    # submit-rejected backoff only). Implementations normalize cards ->
    # jobs themselves (free_cards // npu_per_job) since only they know the
    # job shape.
    def free_job_slots(self) -> Optional[int]: ...


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
        # Dynamic capacity model (tests simulate the fluctuating shared
        # pool): max_job_slots = the pool's concurrent-job capacity. None
        # disables the model entirely (free_job_slots() -> None, submit is
        # never capacity-rejected) — the plain behavioral fake. With a value,
        # submit raises TransientSubmitError when the pool is full and
        # free_job_slots() reports live headroom; the value may be MUTATED
        # while jobs run (pool grows/shrinks).
        self.max_job_slots: Optional[int] = None
        self.max_inflight = 0  # observed in-flight peak (diagnostic)
        # Test injection knobs (capacity simulation):
        #   fail_submits_remaining: next N submit ATTEMPTS raise
        #     TransientSubmitError ("pool full") — exercises backoff/retry.
        #   fail_submits_hard: next N submit attempts raise RuntimeError —
        #     a hard fleet error (bad image/auth).
        self.fail_submits_remaining = 0
        self.fail_submits_hard = 0

    def _inflight_locked(self) -> int:
        n = 0
        for job in self._jobs.values():
            if job["cancelled"]:
                continue
            if job["proc"].poll() is None:
                n += 1
            else:
                job["rc"] = job["proc"].returncode
        return n

    def free_job_slots(self) -> Optional[int]:
        with self._lock:
            if self.max_job_slots is None:
                return None
            return max(0, self.max_job_slots - self._inflight_locked())

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
            if (self.max_job_slots is not None
                    and self._inflight_locked() >= self.max_job_slots):
                raise TransientSubmitError(
                    f"mock pool full: {self.max_job_slots}-job capacity "
                    f"reached")
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
            self.max_inflight = max(self.max_inflight,
                                    self._inflight_locked())
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
