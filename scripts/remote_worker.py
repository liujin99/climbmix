#!/usr/bin/env python3
"""Remote experiment worker — runs INSIDE a remote job container.

Standalone by design: stdlib only + nanochat_cmds.py (the shared command/env/
eval-dir module co-located in the assets bundle). The submit host (climbmix
RemoteExecutor) uploads the ExpSpec JSON and the mixture shards to OBS, then
submits a job whose command runs THIS script. The spec carries the fully
built torchrun commands (constructed on the submit host by the same shared
builders the local executor uses), so the remote job runs exactly the argv a
local experiment would run.

Flow:
  1. Download spec.json + (unless eval_only) mixture shards from OBS.
  2. eval_only: download the mid checkpoint from {result_uri}/mid_checkpoint.
     Otherwise: symlink the base checkpoint, run mid_train (torchrun).
     On success optionally upload the mid checkpoint (enables eval-only
     resume + post-hoc debugging).
  3. Build the private eval base dir (symlink farm — same code as local) and
     run base_eval.
  4. Claim the CSV out of the private dir, upload it + logs + result.json to
     {result_uri}/.

Exit code: 0 iff mid_train_rc == 0 and eval_rc == 0 (mirrors the local
executor's fail-fast). result.json is uploaded even on failure, with the
error message, so the submit host can report precisely.

Storage backends:
  --storage local    filesystem under --storage-root with the obs:// mapping
                     convention (simulation/tests; same convention as the
                     submit-side MockObsStorage)
  --storage moxing   moxing OBS SDK (real)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

# nanochat_cmds.py sits next to this file in the assets bundle — importing it
# gives the EXACT command/env/eval-dir semantics of the local executor.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanochat_cmds  # noqa: E402

SPEC_VERSION = 1
HEARTBEAT_S = 300  # print progress to the job console every 5 minutes


# ── storage backends ──────────────────────────────────────────────────────

class LocalStorage:
    """obs://bucket/a/b -> {root}/bucket/a/b (same convention as the
    submit-side MockObsStorage)."""

    def __init__(self, root: str):
        if not root:
            raise SystemExit("--storage-root is required for --storage local")
        self.root = os.path.abspath(root)

    def _local(self, uri: str) -> str:
        rest = uri[len("obs://"):]
        bucket, _, key = rest.partition("/")
        return os.path.join(self.root, bucket, key)

    def download_file(self, uri: str, local_path: str) -> None:
        src = self._local(uri)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"obs object not found: {uri}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(src, local_path)

    def upload_file(self, local_path: str, uri: str) -> None:
        dst = self._local(uri)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(local_path, dst)

    def list_objects(self, uri: str):
        path = self._local(uri)
        if not os.path.isdir(path):
            return []
        return sorted(os.path.join(uri.rstrip("/"), f)
                      for f in os.listdir(path))

    def download_dir(self, uri: str, local_dir: str) -> None:
        for obj in self.list_objects(uri):
            name = obj.rsplit("/", 1)[-1]
            self.download_file(obj, os.path.join(local_dir, name))

    def upload_dir(self, local_dir: str, uri: str) -> None:
        for f in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, f)
            if os.path.isfile(p):
                self.upload_file(p, f"{uri.rstrip('/')}/{f}")


class MoxingStorage:
    """moxing OBS SDK backend."""

    def __init__(self):
        import moxing as mox  # noqa: F401  (lazy — only in real containers)
        self.mox = mox

    def download_file(self, uri: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.mox.file.copy(uri, local_path)

    def upload_file(self, local_path: str, uri: str) -> None:
        self.mox.file.make_dirs(os.path.dirname(uri))
        self.mox.file.copy(local_path, uri)

    def list_objects(self, uri: str):
        uri = uri.rstrip("/")
        if not self.mox.file.exists(uri):
            return []
        return sorted(f"{uri}/{f}" for f in self.mox.file.list_directory(uri))

    def download_dir(self, uri: str, local_dir: str) -> None:
        os.makedirs(local_dir, exist_ok=True)
        for obj in self.list_objects(uri):
            self.download_file(obj, os.path.join(
                local_dir, obj.rsplit("/", 1)[-1]))

    def upload_dir(self, local_dir: str, uri: str) -> None:
        for f in sorted(os.listdir(local_dir)):
            p = os.path.join(local_dir, f)
            if os.path.isfile(p):
                self.upload_file(p, f"{uri.rstrip('/')}/{f}")


def get_storage(kind: str, root: str):
    if kind == "local":
        return LocalStorage(root)
    if kind == "moxing":
        return MoxingStorage()
    raise SystemExit(f"unknown --storage backend: {kind!r}")


# ── subprocess helpers (heartbeat mirrors the local executor) ─────────────

def _tail_last_line(log_path: str, max_chars: int = 120) -> str:
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 8192))
            chunk = f.read().decode("utf-8", errors="replace")
        lines = [l.strip() for l in chunk.splitlines() if l.strip()]
        if not lines:
            return "(no output yet)"
        last = lines[-1]
        return last[:max_chars] + ("..." if len(last) > max_chars else "")
    except OSError:
        return "(log not readable)"


def run_cmd(cmd, log_path: str, cwd: str, env) -> int:
    print(f"[worker] {' '.join(cmd)}\n[worker] log: {log_path}", flush=True)
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env,
                                stdout=log_f, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                proc.wait(timeout=HEARTBEAT_S)
                break
            except subprocess.TimeoutExpired:
                print(f"[worker] running {(time.time()-t0)/60:.0f}m | "
                      f"{_tail_last_line(log_path)}", flush=True)
    if proc.returncode != 0:
        print(f"[worker] FAILED (exit code {proc.returncode})", flush=True)
    else:
        print(f"[worker] Completed (exit code 0)", flush=True)
    return proc.returncode


# ── main ──────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="climbmix remote experiment worker")
    p.add_argument("--spec-uri", required=True,
                   help="obs:// URI of the ExpSpec JSON")
    p.add_argument("--storage", default="moxing",
                   choices=["local", "moxing"])
    p.add_argument("--storage-root", default="",
                   help="local backend only: filesystem root of the fake OBS")
    p.add_argument("--spec-local", default="",
                   help="override: read the spec from a local file instead of "
                        "--spec-uri (bootstrap/debug)")
    args = p.parse_args()

    storage = get_storage(args.storage, args.storage_root)

    # Per-PROCESS spec path: two workers must never share it (the local
    # simulation runs several jobs on one host; a fixed /tmp path let one
    # worker read another's spec — real containers are isolated but this
    # stays correct either way).
    spec_path = args.spec_local
    if not spec_path:
        spec_path = os.path.join(
            tempfile.gettempdir(), f"climbmix_spec_{os.getpid()}.json")
        storage.download_file(args.spec_uri, spec_path)
    with open(spec_path) as f:
        s = json.load(f)

    if s.get("spec_version") != SPEC_VERSION:
        print(f"[worker] FATAL: spec_version {s.get('spec_version')!r} != "
              f"{SPEC_VERSION} — assets bundle and submit host disagree",
              flush=True)
        return 2

    tag = s["model_tag"]
    base = s["base_dir"]
    work = s["work_dir"]
    result_uri = s["result_uri"].rstrip("/")
    os.makedirs(work, exist_ok=True)

    res = {
        "spec_version": SPEC_VERSION,
        "experiment_id": s["experiment_id"],
        "model_tag": tag,
        "mid_train_rc": -1,
        "eval_rc": -1,
        "elapsed_seconds": 0.0,
        "csv": None,
        "checkpoint_uploaded": False,
        "error": None,
    }
    t0 = time.time()

    def finish(exit_code: int) -> int:
        res["elapsed_seconds"] = time.time() - t0
        try:
            local = os.path.join(work, "result.json")
            with open(local, "w") as f:
                json.dump(res, f, indent=2)
            storage.upload_file(local, f"{result_uri}/result.json")
        except Exception:
            traceback.print_exc()
        return exit_code

    try:
        eval_only = bool(s.get("eval_only", False))
        # eval_only's PREMISE is a previously-successful training (the submit
        # host verified the marker + checkpoint); rc 0, not -1.
        mid_rc = 0
        res["mid_train_rc"] = 0 if eval_only else -1

        if eval_only:
            ckpt_dir = os.path.join(base, "mid_checkpoints", tag)
            print(f"[worker] eval-only: downloading mid checkpoint -> {ckpt_dir}",
                  flush=True)
            storage.download_dir(f"{result_uri}/mid_checkpoint", ckpt_dir)
        else:
            mix_dir = os.path.join(work, "mixture_data")
            print(f"[worker] downloading mixture data -> {mix_dir}", flush=True)
            storage.download_dir(s["mixture_data_uri"], mix_dir)

            # Base checkpoint symlink (same semantics as the local executor's
            # _symlink_base_checkpoint: base_checkpoints/{tag} -> d{depth}).
            base_dst = os.path.join(base, "base_checkpoints", tag)
            if not os.path.exists(base_dst):
                os.makedirs(os.path.dirname(base_dst), exist_ok=True)
                try:
                    os.symlink(s["base_ckpt_src"], base_dst)
                    print(f"[worker] symlink {base_dst} -> {s['base_ckpt_src']}",
                          flush=True)
                except FileExistsError:
                    # Concurrent creation (shared-base scenarios); only
                    # acceptable if it points at the SAME source.
                    if os.path.realpath(base_dst) != os.path.realpath(s["base_ckpt_src"]):
                        raise
            else:
                # A stale link from a previous attempt pointing elsewhere
                # would silently train from the wrong checkpoint.
                if os.path.islink(base_dst) and \
                        os.path.realpath(base_dst) != os.path.realpath(s["base_ckpt_src"]):
                    raise RuntimeError(
                        f"{base_dst} exists but points at "
                        f"{os.path.realpath(base_dst)}, expected "
                        f"{s['base_ckpt_src']}")

            env = nanochat_cmds.build_subprocess_env(
                s["nanochat_dir"], base,
                device_ids=s.get("visible_devices") or [0],
                extra_env=s.get("env") or None)
            mid_rc = run_cmd(s["mid_train_cmd"],
                             os.path.join(work, "mid_train.log"),
                             cwd=s["nanochat_dir"], env=env)
            res["mid_train_rc"] = mid_rc
            if mid_rc != 0:
                storage.upload_file(os.path.join(work, "mid_train.log"),
                                    f"{result_uri}/mid_train.log")
                return finish(mid_rc)
            if s.get("upload_checkpoint", True):
                ckpt_dir = os.path.join(base, "mid_checkpoints", tag)
                print(f"[worker] uploading mid checkpoint -> "
                      f"{result_uri}/mid_checkpoint", flush=True)
                storage.upload_dir(ckpt_dir, f"{result_uri}/mid_checkpoint")
                res["checkpoint_uploaded"] = True

        # Eval in a PRIVATE base dir (symlink farm — identical to local).
        eval_base = nanochat_cmds.make_eval_base_dir(base, work, tag)
        env = nanochat_cmds.build_subprocess_env(
            s["nanochat_dir"], base,
            device_ids=s.get("visible_devices") or [0],
            base_dir_override=eval_base,
            extra_env=s.get("env") or None)
        eval_rc = run_cmd(s["eval_cmd"], os.path.join(work, "eval.log"),
                          cwd=s["nanochat_dir"], env=env)
        res["eval_rc"] = eval_rc

        csv_path = nanochat_cmds.claim_eval_csv(work, tag, eval_base)
        if csv_path is not None:
            res["csv"] = os.path.basename(csv_path)
            storage.upload_file(csv_path, f"{result_uri}/{res['csv']}")

        for log_name in ("mid_train.log", "eval.log"):
            lp = os.path.join(work, log_name)
            if os.path.isfile(lp):
                storage.upload_file(lp, f"{result_uri}/{log_name}")

        if mid_rc == 0 and eval_rc == 0:
            return finish(0)
        return finish(1)

    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return finish(3)


if __name__ == "__main__":
    sys.exit(main())
