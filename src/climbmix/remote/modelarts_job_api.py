"""ModelArtsJobAPI — training-job adapter over the internal CSB/ROMA gateway.

REST shape (learned from the submit-host tooling, 2026-08-29):
  POST   {endpoint}/v2/{project_id}/training-jobs        create
  GET    {endpoint}/v2/{project_id}/training-jobs/{id}    status
  DELETE {endpoint}/v2/{project_id}/training-jobs/{id}    cancel
Auth: IAM token (iam_token.IamTokenProvider — JWT/password auth, ~24h
validity, transparent mid-run rollover; HTTP 401 triggers exactly one
force-refreshed retry).

SECURITY (the repo is PUBLIC): every platform-specific value — gateway
endpoint, project/workspace IDs, pool id, SWR image + repo id, and the auth
credentials — lives OUTSIDE the repo in a JSON config file:

    RemoteConfig.ma_config   (highest precedence, a file path)
    $CLIMBMIX_MA_CONFIG
    ~/.config/climbmix/remote_ma.json   (default)

config/remote_ma.example.json documents the schema with placeholders ONLY.
scripts/check_repo_secrets.py guards against accidental commits.

submit() wraps the worker argv into a boot shell that:
  1. bootstraps the one-time big assets from OBS (marker-file caching per
     node: a reused pool node skips the re-download) — the nanochat-npu
     repo tarball, every d{depth} base checkpoint found under assets_big
     (auto-discovery, so proxy_depth needs no extra knob), tokenizer and
     the eval datasets, landing at the container_* paths of RemoteConfig;
  2. execs the worker argv unchanged (spec.json is the single source of
     experiment truth — platform job parameters are deliberately unused).
Job code delivery: v2 `code_dir` = {obs_prefix}/assets (the two-file worker
bundle RemoteExecutor auto-uploads), which the platform copies into
local_code_dir; the boot shell resolves the exact worker path at runtime
(the platform may nest the copy under the code_dir basename).

submit_raw() skips the boot shell — the M1 hello-world / error-code
calibration tool (scripts/ma_hello_world.py) runs platform commands
directly.

Status mapping: the v2 status table below is best-effort and gets
CALIBRATED against real responses (M1 hello-world + M3 concurrency wave);
unknown codes map to UNKNOWN, which the executor treats as non-terminal —
the safe default. logs() returns a pointer string in v1: the worker uploads
full logs to result_uri even on failure, so console access is a nicety.
"""

import json
import os
import shlex
import threading
from typing import Any, Dict, List, Optional

from climbmix.remote.iam_token import IamTokenProvider
from climbmix.remote.job_api import JobStatus, TransientSubmitError

# Platform copies code_dir's content here (basename nesting resolved at
# runtime by the boot shell).
LOCAL_CODE_DIR = "/home/ma-user/modelarts/user-job-dir"

DEFAULT_MA_CONFIG_PATH = os.path.join(
    os.path.expanduser("~"), ".config", "climbmix", "remote_ma.json")

# Required config keys (endpoint/project/workspace/image are platform
# identity; auth is handled by the token provider). Flavor and pool may
# instead come from RemoteConfig (execution-shape knobs).
_REQUIRED_KEYS = ("endpoint", "project_id", "workspace_id", "auth")

# v2 training-job status table — CALIBRATE (M1 hello-world): compare these
# against what the gateway actually returns; unknown -> UNKNOWN (safe).
_INT_STATUS = {
    1: JobStatus.PENDING,     # creating
    2: JobStatus.PENDING,     # pending / queued server-side
    3: JobStatus.RUNNING,
    4: JobStatus.SUCCEEDED,
    5: JobStatus.FAILED,
    6: JobStatus.CANCELLED,   # terminating
    7: JobStatus.CANCELLED,   # terminated
    8: JobStatus.FAILED,      # abnormal
}
_STR_STATUS = {
    "creating": JobStatus.PENDING,
    "pending": JobStatus.PENDING,
    "queued": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "completed": JobStatus.SUCCEEDED,
    "succeeded": JobStatus.SUCCEEDED,
    "success": JobStatus.SUCCEEDED,
    "finished": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "error": JobStatus.FAILED,
    "abnormal": JobStatus.FAILED,
    "terminating": JobStatus.CANCELLED,
    "terminated": JobStatus.CANCELLED,
    "cancelled": JobStatus.CANCELLED,
}

# Submit-rejection texts that mean RETRYABLE (pool full / quota / throttle).
# Heuristic until M1/M3 calibrate the real error codes — extend freely, the
# executor backs off and retries TransientSubmitError forever (24h net).
_TRANSIENT_MESSAGE_PATTERNS = (
    "quota", "capacity", "insufficient", "resource", "busy",
    "too many", "limit", "throttl", "concurrent", "occupied",
)


def load_ma_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load + validate the platform config. Raises with the missing keys
    (never prints values)."""
    path = path or os.environ.get("CLIMBMIX_MA_CONFIG") or DEFAULT_MA_CONFIG_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ModelArts platform config not found: {path}. Copy "
            f"config/remote_ma.example.json there and fill it in (the repo "
            f"is public — real values stay out of it). See "
            f"docs/remote_setup.md §1.")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a JSON object")
    missing = [k for k in _REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required keys: {missing} "
                         f"(schema: config/remote_ma.example.json)")
    return cfg


def to_platform_obs_uri(uri: str) -> str:
    """obs://bucket/a/b -> /bucket/a/b — the form the gateway's job fields
    (code_dir, inputs/outputs) use. Bare /bucket/... URIs pass through."""
    if uri.startswith("obs://"):
        return "/" + uri[len("obs://"):]
    if uri.startswith("/"):
        return uri
    raise ValueError(f"not an OBS URI: {uri!r}")


def resolve_image_flavor_pool(remote_config, ma_config: Dict[str, Any]):
    """Effective (image_url, image_repo_id, flavor, pool_id):
    RemoteConfig knobs win (per-launch shape), config file is the default
    (platform identity). Raises a fail-fast message when nothing resolves
    image or flavor."""
    image = remote_config.image or str(ma_config.get("image_url") or "")
    repo_id = str(ma_config.get("image_repo_id") or "")
    flavor = remote_config.flavor or str(
        ma_config.get("default_flavor") or "")
    pool = remote_config.pool_name or str(ma_config.get("pool_id") or "")
    if not image:
        raise ValueError(
            "no image resolved: set RemoteConfig.image (REMOTE_IMAGE) or "
            "image_url in the platform config")
    if not flavor:
        raise ValueError(
            "no flavor resolved: set RemoteConfig.flavor (REMOTE_FLAVOR) or "
            "default_flavor in the platform config")
    if not repo_id:
        raise ValueError(
            "image_repo_id missing from the platform config (required by "
            "the gateway even for custom SWR images)")
    return image, repo_id, flavor, pool


class ModelArtsJobAPI:
    """Real training-job adapter (see module docstring). Constructed by
    RemoteExecutor with the RemoteConfig; tests inject a fake session +
    token provider for fully-offline coverage."""

    def __init__(self, remote_config=None, ma_config=None, session=None,
                 token_provider=None):
        self.rc = remote_config
        if ma_config is None:
            path = ""
            if remote_config is not None and getattr(
                    remote_config, "ma_config", ""):
                path = remote_config.ma_config
            ma_config = load_ma_config(path or None)
        self.cfg = ma_config
        self.provider = token_provider or IamTokenProvider()
        self._token_lock = threading.Lock()
        self._token: Optional[str] = None
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.verify_tls = bool(ma_config.get("verify_tls", False))
        self.timeout_s = float(ma_config.get("timeout_s", 60))
        self._unknown_status_warned = set()

    # ── URL / token plumbing ──────────────────────────────────────────────

    def _jobs_url(self) -> str:
        return (f"{self.cfg['endpoint'].rstrip('/')}/v2/"
                f"{self.cfg['project_id']}/training-jobs")

    def _get_token(self, force: bool = False) -> str:
        with self._token_lock:
            self._token = self.provider.get_token(
                self.cfg["auth"], force_refresh=force)
            return self._token

    def _request(self, method: str, url: str, json_body=None):
        """One authenticated request; HTTP 401/403 retries EXACTLY once
        with a force-refreshed token (clock-skew safety net — the
        provider's 5-min-early expiry already covers the 24h rollover)."""
        headers = {
            "Content-Type": "application/json",
            "region": str(self.cfg.get("region") or ""),
            "Authorization": self._get_token(),
        }
        resp = self.session.request(
            method, url, json=json_body, headers=headers,
            timeout=self.timeout_s, verify=self.verify_tls)
        if resp.status_code in (401, 403):
            self.provider.invalidate_cached_token(self.cfg["auth"])
            headers["Authorization"] = self._get_token(force=True)
            resp = self.session.request(
                method, url, json=json_body, headers=headers,
                timeout=self.timeout_s, verify=self.verify_tls)
        return resp

    @staticmethod
    def _error_text(resp) -> str:
        try:
            return json.dumps(resp.json())[:500]
        except Exception:
            return getattr(resp, "text", "")[:500]

    def _raise_submit_error(self, resp) -> None:
        """Map a failed submit to TransientSubmitError (retry: capacity/
        throttle/service) or RuntimeError (hard: bad image/auth/spec).
        Heuristic table until M1/M3 calibrate the gateway's real codes."""
        code = resp.status_code
        text = self._error_text(resp)
        if code in (401, 403):
            raise RuntimeError(
                f"submit rejected: auth failed even after token refresh "
                f"(HTTP {code}) — check the auth section of the platform "
                f"config. Body: {text}")
        if code == 429 or code >= 500:
            raise TransientSubmitError(
                f"submit rejected (HTTP {code}, service/throttle): {text}")
        lower = text.lower()
        if code >= 400 and any(p in lower for p in _TRANSIENT_MESSAGE_PATTERNS):
            raise TransientSubmitError(
                f"submit rejected (HTTP {code}, capacity/quota pattern "
                f"matched): {text}")
        raise RuntimeError(f"submit failed (HTTP {code}): {text}")

    # ── JobAPI protocol ───────────────────────────────────────────────────

    def free_job_slots(self) -> Optional[int]:
        # No quota-usage query API known on this gateway (2026-08-29); the
        # executor's dynamic scheduler falls back to submit-rejected
        # backoff, which is fully functional. Revisit if M1/M3 surfaces one.
        return None

    def submit(self, name: str, command: List[str],
               env: Optional[Dict[str, str]] = None,
               workdir: Optional[str] = None) -> str:
        shell = self._boot_shell(command)
        return self.submit_raw(name, shell, env=env)

    def submit_raw(self, name: str, shell_command: str,
                   env: Optional[Dict[str, str]] = None) -> str:
        """Submit an arbitrary shell command (no boot shell, no assets).
        The M1 calibration tool uses this; submit() is the worker path."""
        body = self._build_job_body(name, shell_command, env or {})
        resp = self._request("POST", self._jobs_url(), json_body=body)
        if resp.status_code != 200:
            self._raise_submit_error(resp)
        try:
            data = resp.json()
            job_id = data["metadata"]["id"]
        except Exception as e:
            raise RuntimeError(
                f"submit returned HTTP 200 but no metadata.id — body: "
                f"{self._error_text(resp)}") from e
        print(f"  [ModelArts] submitted job {job_id} "
              f"({data['metadata'].get('name', name)}); console: "
              f"https://console.huaweicloud.com/modelarts/?region="
              f"{self.cfg.get('region', '')}#/training/detail/{job_id}")
        return str(job_id)

    def status(self, job_id: str) -> JobStatus:
        resp = self._request("GET", f"{self._jobs_url()}/{job_id}")
        if resp.status_code != 200:
            print(f"  [ModelArts] status({job_id}) HTTP "
                  f"{resp.status_code} — treating as UNKNOWN (poll "
                  f"continues): {self._error_text(resp)}")
            return JobStatus.UNKNOWN
        try:
            data = resp.json()
        except Exception:
            return JobStatus.UNKNOWN
        raw = data.get("status", data.get("metadata", {}).get("status"))
        st = None
        if isinstance(raw, bool):
            st = None
        elif isinstance(raw, int):
            st = _INT_STATUS.get(raw)
        elif isinstance(raw, str):
            st = _STR_STATUS.get(raw.strip().lower())
            if st is None and raw.strip().lstrip("-").isdigit():
                st = _INT_STATUS.get(int(raw.strip()))
        if st is None:
            if raw not in self._unknown_status_warned:
                print(f"  [ModelArts] status({job_id}): unmapped raw value "
                      f"{raw!r} -> UNKNOWN — calibrate _INT_STATUS/"
                      f"_STR_STATUS (docs/remote_setup.md M1)")
                self._unknown_status_warned.add(raw)
            return JobStatus.UNKNOWN
        return st

    def logs(self, job_id: str, tail: int = 50) -> str:
        # v1: the gateway's log API is not wired; the WORKER uploads
        # mid_train.log / eval.log / result.json to the experiment's
        # result_uri even on failure, which is the primary debugging path.
        return (f"(platform logs not wired; console: "
                f"https://console.huaweicloud.com/modelarts/?region="
                f"{self.cfg.get('region', '')}#/training/detail/{job_id}; "
                f"worker logs land at {{result_uri}}/mid_train.log and "
                f"{{result_uri}}/eval.log)")

    def cancel(self, job_id: str) -> None:
        resp = self._request("DELETE", f"{self._jobs_url()}/{job_id}")
        if resp.status_code == 200:
            return
        if resp.status_code == 404:
            print(f"  [ModelArts] cancel({job_id}): 404 — already gone")
            return
        print(f"  [ModelArts] cancel({job_id}) failed (HTTP "
              f"{resp.status_code}): {self._error_text(resp)}")

    # ── job body construction ─────────────────────────────────────────────

    def _build_job_body(self, name: str, shell_command: str,
                        env: Dict[str, str]) -> Dict[str, Any]:
        image, repo_id, flavor, pool = resolve_image_flavor_pool(
            self.rc, self.cfg)
        environments = {"MOX_OBS_SIGNATURE": "obs"}
        environments.update(env)
        resource: Dict[str, Any] = {
            "flavor_id": flavor,
            "node_count": 1,
            "policy": "regular",
        }
        if pool:
            resource["pool_id"] = pool
        return {
            "kind": "job",
            "metadata": {
                "name": name.lower()[:63],
                "description": "climbmix proxy experiment",
                "workspace_id": self.cfg["workspace_id"],
            },
            "algorithm": {
                "engine": {"image_url": image, "image_repo_id": repo_id},
                "command": shell_command,
                "parameters": [],
                "environments": environments,
                "code_dir": to_platform_obs_uri(
                    f"{self.rc.obs_prefix.rstrip('/')}/assets"),
                "local_code_dir": LOCAL_CODE_DIR,
                "working_dir": LOCAL_CODE_DIR,
            },
            "inputs": [],
            "outputs": [],
            "spec": {
                "resource": resource,
                "log_export_path": {"obs_url": ""},
            },
        }

    # ── boot shell ────────────────────────────────────────────────────────

    _BOOT_ASSETS_PY = """python3 - <<'PYEOF'
import os
import sys
import moxing as mox

ASSETS = @@ASSETS@@
NANOCHAT_DIR = @@NANOCHAT@@
BASE = @@BASE@@

def _mark(d):
    return os.path.join(d, ".climbmix_asset_ok")

def _have(d):
    return os.path.exists(_mark(d))

def fetch_dir(src, dst):
    if _have(dst):
        print("[boot] cached:", dst)
        return
    os.makedirs(dst, exist_ok=True)
    mox.file.copy_parallel(src, dst)
    open(_mark(dst), "w").write("ok")
    print("[boot] fetched:", src, "->", dst)

def fetch_repo():
    if _have(NANOCHAT_DIR):
        print("[boot] cached:", NANOCHAT_DIR)
        return
    parent = os.path.dirname(NANOCHAT_DIR)
    os.makedirs(parent, exist_ok=True)
    tmp = os.path.join(parent, ".nanochat-npu.tar.gz")
    mox.file.copy(ASSETS + "/nanochat-npu.tar.gz", tmp)
    rc = os.system("tar xzf " + tmp + " -C " + parent)
    os.remove(tmp)
    if rc != 0:
        sys.exit("[boot] FATAL: tar extract failed for nanochat-npu.tar.gz")
    open(_mark(NANOCHAT_DIR), "w").write("ok")
    print("[boot] fetched repo ->", NANOCHAT_DIR)

fetch_repo()
# Auto-discovered d{depth} base checkpoints: whatever depths live under
# assets_big land under {base}/base_checkpoints/ — the run's proxy_depth
# simply uses the one it needs.
try:
    entries = mox.file.list_directory(ASSETS)
except Exception as e:
    sys.exit("[boot] FATAL: cannot list " + ASSETS + " (" + str(e) + ")")
for entry in entries:
    name = os.path.basename(str(entry).rstrip("/"))
    if name.startswith("d") and name[1:].isdigit():
        fetch_dir(ASSETS + "/" + name,
                  os.path.join(BASE, "base_checkpoints", name))
for name in ("tokenizer", "eval_bundle", "eval_stem"):
    fetch_dir(ASSETS + "/" + name, os.path.join(BASE, name))
print("[boot] assets ready")
PYEOF"""

    def _boot_shell(self, command: List[str]) -> str:
        """Assets bootstrap + exec the worker argv. command = [python,
        worker_path, ...args] as built by RemoteExecutor._worker_argv; the
        worker script path is resolved at RUNTIME (the platform may nest
        the code_dir copy under its basename)."""
        assets = f"{self.rc.obs_prefix.rstrip('/')}/assets_big"
        boot_py = (self._BOOT_ASSETS_PY
                   .replace("@@ASSETS@@", json.dumps(assets))
                   .replace("@@NANOCHAT@@", json.dumps(
                       self.rc.container_nanochat_dir))
                   .replace("@@BASE@@", json.dumps(
                       self.rc.container_base_dir)))
        parts = ["set -e", boot_py]
        if len(command) >= 2:
            # [python, worker_path, *args] -> runtime-resolved script path
            parts.append(f'CODE={shlex.quote(LOCAL_CODE_DIR)}')
            parts.append('W="$CODE/remote_worker.py"')
            parts.append('[ -f "$W" ] || W="$CODE/assets/remote_worker.py"')
            tail = " ".join(shlex.quote(a) for a in command[2:])
            parts.append(f'exec {shlex.quote(command[0])} "$W" {tail}')
        else:
            parts.append("exec " + " ".join(
                shlex.quote(a) for a in command))
        return "\n".join(parts) + "\n"
