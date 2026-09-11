#!/usr/bin/env python3
"""Remote experiment worker — runs INSIDE a remote job container.

Standalone by design: stdlib only + nanochat_cmds.py (the shared command/env/
eval-dir module co-located in the assets bundle). The submit host (climbmix
RemoteExecutor) uploads the ExpSpec JSON and the mixture shards to OBS, then
submits a job whose command runs THIS script. The spec carries the fully
built torchrun commands (constructed on the submit host by the same shared
builders the local executor uses), so the remote job runs exactly the argv a
local experiment would run.

Flow (single-node, spec node_count=1 — today's form):
  1. Download spec.json + (unless eval_only) mixture shards from OBS.
  2. eval_only: download the mid checkpoint from {result_uri}/mid_checkpoint.
     Otherwise: symlink the base checkpoint, run mid_train (torchrun).
     On success optionally upload the mid checkpoint (enables eval-only
     resume + post-hoc debugging).
  3. Build the private eval base dir (symlink farm — same code as local) and
     run base_eval.
  4. Claim the CSV out of the private dir, upload it + logs + result.json to
     {result_uri}/.

Multi-node (spec v3 node_count>1, the Phase-1 d28 target arms): the SAME
worker runs on EVERY node of the job. Each node resolves its rendezvous
(platform MASTER_* env or the k8s StatefulSet DNS the job runs as) and
retargets the TRAIN torchrun launcher prefix (nanochat_cmds); the argv
after "-m" stays pinned by the spec. Roles: every node trains, then the
MODEL RELAY (2026-09-10, rev 2) decides the eval world. The trained
model exists ONLY on node 0 — save_checkpoint's rank-0 write lands in
that node's local nanochat_base — and the platform offers NO cross-node
file path: the output mount is per-node local with one-way local->OBS
sync (probe A 20260908, both probe jobs: "NOT-visible (saw 1/2 after
90s)" on every node) and containers carry no OBS SDK. But the k8s
network IS bidirectional (proven by every HCCL step and the train
rendezvous itself), so node 0 serves model_*.pt + meta_*.json on
master_port+2 and every non-master node pulls them into its own base
dir (a ~3GB transfer takes seconds on the pod network). All peers
delivered => the eval torchrun spans all 8*node_count ranks (rendezvous
port = train port + 1; base_eval's per-sample striding + all_reduce
aggregation is world-size invariant, so scores match the 8-rank form
~node_count x faster). Any relay miss (peer failed, timeout, relay
disabled via CLIMBMIX_EVAL_RELAY=0) => automatic fallback to node-0-only
8-rank eval — the pre-relay behavior; the relay can only add speed,
never take away correctness. Each node uses its own private
_eval_base[_node{r}] dir and uploads mid_train[_node{r}].log +
eval[_node{r}].log. Node 0 additionally owns the eval CSV, the
checkpoint upload and result.json (uploaded after eval so non-master
ranks do not idle at the eval rendezvous). The job succeeds only if
every node exits 0.

Progress visibility: a daemon thread streams the IN-PROGRESS mid_train.log
and eval.log to {result_uri} every log_stream_s seconds (spec field, default
30). With the mount/local storage backends upload_file lands on the OUTPUT
MOUNT, so the platform's periodic sync publishes the tail to OBS while the
stage is still running — watchable from the submit host through its mount
within about a minute. Best-effort by design: a failed stream never kills
training, and the post-stage uploads remain the authoritative copies.

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
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback

# nanochat_cmds.py sits next to this file in the assets bundle — importing it
# gives the EXACT command/env/eval-dir semantics of the local executor.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nanochat_cmds  # noqa: E402

SPEC_VERSION = 3  # exp_spec.SPEC_VERSION (v3: +node_count/master_port — keep in lockstep)
HEARTBEAT_S = 300  # print progress to the job console every 5 minutes
LOG_STREAM_S = 30  # default in-progress log upload period (spec-tunable)


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


def start_log_streamer(storage, work: str, result_uri: str,
                       period_s: float,
                       names=("mid_train.log", "eval.log", "embed.log")):
    """Best-effort live log upload (see the module docstring). Returns the
    stop() closure; the thread is a daemon, so early-exit paths never need
    to call it explicitly. names is overridable so multi-node workers can
    stream their per-node train log (mid_train_node{r}.log) alongside the
    master-owned mid_train.log."""
    if period_s <= 0:
        return lambda: None
    stop = threading.Event()

    def _loop():
        while not stop.wait(period_s):
            for name in names:
                lp = os.path.join(work, name)
                if not os.path.isfile(lp):
                    continue
                try:
                    storage.upload_file(lp, f"{result_uri}/{name}")
                except Exception:
                    pass  # progress visibility only — never fatal

    t = threading.Thread(target=_loop, daemon=True,
                         name="climbmix-log-stream")
    t.start()

    def _stop():
        stop.set()
        t.join(timeout=5)

    return _stop


# ── multi-node model relay (Phase 1.5 rev 2, 2026-09-10) ───────────────────
# Why this exists: the trained model file exists ONLY on node 0 (the
# training's global rank 0 saves into ITS node-local nanochat_base), and
# the platform provides no cross-node file path — output mounts are
# per-node local with one-way local->OBS sync (probe A 20260908:
# "NOT-visible (saw 1/2 after 90s)" on every node of both probe jobs)
# and containers carry no OBS SDK. The k8s pod network, however, is
# bidirectional and battle-proven (HCCL collectives + the train TCPStore
# rendezvous both ride it). So the model crosses nodes the same way the
# gradients do: over TCP.
#
# Protocol (port = master_port + 2; train uses master_port, the eval
# rendezvous master_port + 1):
#   client -> "GET <rank>\n"
#   server -> "OK <nfiles>\n", then per file "<name> <size>\n" + raw bytes
#   client -> "DONE\n"                    (after verifying every size)
#   server -> "EVAL 32\n" | "EVAL 8\n"    (once ALL peers are done or the
#                                          serve deadline passes)
# Only model_*.pt + meta_*.json are served — eval loads those; the
# per-rank optim_* shards already exist on every node (each rank wrote
# its own) and are useless to eval.
#
# Failure semantics: ANY miss (connect failure, short read, missing DONE,
# deadline, CLIMBMIX_EVAL_RELAY=0) degrades to the node-0-only 8-rank
# eval — the behavior that shipped before the relay. The relay can only
# add speed, never take away correctness.

RELAY_SERVE_TIMEOUT_S = float(
    os.environ.get("CLIMBMIX_RELAY_TIMEOUT_S", "300"))
RELAY_CONNECT_TIMEOUT_S = float(
    os.environ.get("CLIMBMIX_RELAY_CONNECT_S", "180"))
RELAY_ENABLED = os.environ.get(
    "CLIMBMIX_EVAL_RELAY", "1").strip().lower() not in ("0", "false", "no")


def _relay_files(tag_dir: str):
    """The files eval needs from the training node: the consolidated
    model + its meta. Sorted for deterministic protocol order."""
    names = []
    for fn in sorted(os.listdir(tag_dir)):
        if (fn.startswith("model_") and fn.endswith(".pt")) or \
                (fn.startswith("meta_") and fn.endswith(".json")):
            names.append(fn)
    return names


class _RelayServer:
    """Node 0 side. start() spawns the listener; wait() blocks until every
    non-master peer finished its pull (-> 32) or the deadline passes
    (-> 8), then broadcasts the verdict to the completed peers."""

    def __init__(self, tag_dir: str, port: int, node_count: int):
        self.tag_dir = tag_dir
        self.port = port
        self.peers = node_count - 1
        self.done = set()        # ranks that finished their pull
        self.failed = set()      # ranks whose connection broke mid-transfer
        self.socks = {}          # rank -> socket (kept open for verdict)
        self.world = 8
        self.decided = threading.Event()
        self.listener = None
        self._lock = threading.Lock()

    def start(self):
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("", self.port))
        self.listener.listen(self.peers + 2)
        self.listener.settimeout(1.0)
        threading.Thread(target=self._accept_loop, daemon=True,
                         name="climbmix-relay-accept").start()

    def _accept_loop(self):
        while not self.decided.is_set():
            try:
                conn, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve_one, args=(conn,),
                             daemon=True, name="climbmix-relay-conn").start()

    def _serve_one(self, conn: socket.socket):
        try:
            conn.settimeout(RELAY_SERVE_TIMEOUT_S)
            f = conn.makefile("rwb", buffering=0)
            req = f.readline().decode(errors="replace").strip()
            rank = int(req.split()[-1])  # "GET <rank>"
            files = _relay_files(self.tag_dir)
            f.write(f"OK {len(files)}\n".encode())
            for name in files:
                path = os.path.join(self.tag_dir, name)
                f.write(f"{name} {os.path.getsize(path)}\n".encode())
                with open(path, "rb") as fh:
                    while True:
                        chunk = fh.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
            ack = f.readline().decode(errors="replace").strip()
            if ack != "DONE":
                raise IOError(f"peer {rank} acked {ack!r}, expected DONE")
            with self._lock:
                self.done.add(rank)
                self.socks[rank] = conn
            # Hold the socket until the verdict is decided, then deliver
            # it on this same connection.
            if not self.decided.wait(RELAY_SERVE_TIMEOUT_S):
                return  # deadline while waiting — client times out too
            try:
                f.write(f"EVAL {self.world}\n".encode())
            except OSError:
                pass
        except Exception as e:
            # A broken transfer can never complete later — record the peer
            # as failed so wait() can decide the fallback immediately
            # instead of burning the whole deadline.
            try:
                rank = int(locals().get("req", "?").split()[-1])
            except (ValueError, IndexError):
                rank = None
            if rank is not None:
                with self._lock:
                    self.failed.add(rank)
            print(f"[worker] relay: peer transfer failed: {e}", flush=True)
            try:
                conn.close()
            except OSError:
                pass

    def wait(self) -> int:
        deadline = time.time() + RELAY_SERVE_TIMEOUT_S
        while time.time() < deadline:
            with self._lock:
                if len(self.done) >= self.peers:
                    break
                if self.failed:  # a broken peer => 32 is unreachable now
                    break
            time.sleep(0.5)
        with self._lock:
            self.world = 32 if (len(self.done) >= self.peers
                                and not self.failed) else 8
        self.decided.set()
        try:
            self.listener.close()
        except OSError:
            pass
        return self.world


def _relay_pull(master_addr: str, port: int, rank: int, tag_dir: str,
                log=print):
    """Non-master node: pull model/meta from node 0 into tag_dir. Returns
    the eval world (32 = every peer delivered) or None on ANY failure —
    the caller falls back to exiting after its train stage."""
    connect_deadline = time.time() + RELAY_CONNECT_TIMEOUT_S
    sock = None
    while sock is None and time.time() < connect_deadline:
        try:
            sock = socket.create_connection((master_addr, port), timeout=10)
        except OSError:
            time.sleep(2.0)
    if sock is None:
        log(f"[worker] relay: cannot reach {master_addr}:{port} in "
            f"{RELAY_CONNECT_TIMEOUT_S:.0f}s — falling back (eval on node 0)")
        return None
    try:
        # Generous cap: covers the server's full serve deadline even if
        # this node finished its own pull early and waits for stragglers.
        sock.settimeout(RELAY_SERVE_TIMEOUT_S + 120)
        f = sock.makefile("rwb", buffering=0)
        f.write(f"GET {rank}\n".encode())
        header = f.readline().decode(errors="replace").strip()
        if not header.startswith("OK"):
            raise IOError(f"relay header {header!r}")
        n = int(header.split()[1])
        os.makedirs(tag_dir, exist_ok=True)
        for _ in range(n):
            meta = f.readline().decode(errors="replace").strip()
            name, size_s = meta.rsplit(" ", 1)
            size = int(size_s)
            dst = os.path.join(tag_dir, name)
            tmp = dst + ".relay_tmp"
            got = 0
            with open(tmp, "wb") as out:
                while got < size:
                    chunk = f.read(min(1 << 20, size - got))
                    if not chunk:
                        raise IOError(f"relay EOF mid-file {name}")
                    out.write(chunk)
                    got += len(chunk)
            if got != size:
                raise IOError(f"relay short read {name}: {got} != {size}")
            os.replace(tmp, dst)
            log(f"[worker] relay: landed {name} ({size:,} B)")
        f.write(b"DONE\n")
        verdict = f.readline().decode(errors="replace").strip()
        if not verdict.startswith("EVAL"):
            raise IOError(f"relay verdict {verdict!r}")
        world = int(verdict.split()[1])
        log(f"[worker] relay: verdict eval world = {world} ranks")
        return world
    except Exception as e:
        log(f"[worker] relay: pull failed ({e}) — falling back")
        return None
    finally:
        try:
            sock.close()
        except OSError:
            pass


# ── embed dispatch (TODO E — kind == "embed" specs) ───────────────────────

def run_embed(s: dict, storage, spec_path: str) -> int:
    """Embed unit: delegate to embed_worker.py (same assets bundle), then
    upload its outputs. The child's console output lands in embed.log —
    streamed to {result_uri} by the log streamer while it runs."""
    work = s["work_dir"]
    os.makedirs(work, exist_ok=True)
    result_uri = s["result_uri"].rstrip("/")

    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, os.path.join(here, "embed_worker.py"),
           "--spec-path", spec_path, "--work-dir", work]
    rc = run_cmd(cmd, os.path.join(work, "embed.log"), cwd=work,
                 env=dict(os.environ))

    # embed_worker writes result.json (kind=embed: embed_rc/docs/error) into
    # the work dir on every exit path — upload it plus the stage artifacts.
    result_local = os.path.join(work, "result.json")
    if os.path.isfile(result_local):
        storage.upload_file(result_local, f"{result_uri}/result.json")
    for name in ("embed.log", "manifest.json", "partial_block.npz"):
        local = os.path.join(work, name)
        if os.path.isfile(local):
            storage.upload_file(local, f"{result_uri}/{name}")
    return rc


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

    # Embed units (TODO E): a different spec shape entirely — dispatch
    # before any experiment-field extraction (model_tag etc. are absent).
    if s.get("kind") == "embed":
        work = s["work_dir"]
        result_uri = s["result_uri"].rstrip("/")
        os.makedirs(work, exist_ok=True)
        stop_stream = start_log_streamer(
            storage, work, result_uri,
            float(s.get("log_stream_s", LOG_STREAM_S)))
        try:
            return run_embed(s, storage, spec_path)
        finally:
            stop_stream()

    tag = s["model_tag"]
    base = s["base_dir"]
    work = s["work_dir"]
    result_uri = s["result_uri"].rstrip("/")
    os.makedirs(work, exist_ok=True)

    # ── multi-node (spec v3): node_count > 1 runs THIS worker on every
    # node of the job. node_rank/master_addr resolve HERE (platform env /
    # StatefulSet DNS — unknowable on the submit host), and the torchrun
    # launcher prefix is swapped accordingly; the training argv after
    # "-m" stays pinned by the spec. Node roles: EVERY node trains; node 0
    # additionally evals + owns result.json; non-master nodes exit with
    # their train rc after uploading their own log.
    node_count = int(s.get("node_count", 1) or 1)
    master_port = int(s.get("master_port", 29500) or 29500)
    node_rank = 0
    master_addr = ""
    if node_count > 1:
        master_addr, master_port, node_rank = \
            nanochat_cmds.resolve_multinode_rendezvous(
                node_count, master_port, log=print)
        s["mid_train_cmd"] = nanochat_cmds.retarget_torchrun_multinode(
            s["mid_train_cmd"], node_count, node_rank,
            master_addr, master_port)
        # The eval cmd is retargeted LATER, only if the model relay
        # delivers to every peer (see the relay block below) — the
        # rendezvous port is train port + 1 (the train TCPStore may sit
        # in TIME_WAIT when eval starts).
        print(f"[worker] multi-node job: node {node_rank}/{node_count}, "
              f"master {master_addr}:{master_port} "
              f"(eval rdzv port {master_port + 1}, "
              f"relay {master_port + 2})", flush=True)
    train_log_name = ("mid_train.log" if node_rank == 0
                      else f"mid_train_node{node_rank}.log")
    # Non-master eval log name is only USED when the relay delivers
    # (32-rank eval); in the fallback the node exits after train and the
    # streamer simply never sees this file.
    eval_log_name = ("eval.log" if node_rank == 0
                     else f"eval_node{node_rank}.log")

    # Live progress: stream the in-progress logs to the result prefix
    # while the stages run (0/absent disables — final uploads only).
    stream_names = ["mid_train.log", "eval.log", "embed.log"]
    for log_name in (train_log_name, eval_log_name):
        if log_name not in stream_names:
            stream_names.append(log_name)
    stop_stream = start_log_streamer(
        storage, work, result_uri,
        float(s.get("log_stream_s", LOG_STREAM_S)), tuple(stream_names))

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
        if eval_only and node_count > 1:
            # N standalone evals would race one CSV and N result.json
            # writers would clobber each other — refuse the shape.
            raise RuntimeError(
                "eval_only specs are single-node only (node_count="
                f"{node_count})")
        # eval_only's PREMISE is a previously-successful training (the submit
        # host verified the marker + checkpoint); rc 0, not -1.
        mid_rc = 0
        res["mid_train_rc"] = 0 if eval_only else -1

        if eval_only:
            ckpt_dir = os.path.join(base, "mid_checkpoints", tag)
            ckpt_src = s.get("ckpt_src") or ""
            if ckpt_src:
                # Evaluate an EXISTING container checkpoint (the d28 base
                # anchor): symlink it as mid_checkpoints/{tag} instead of
                # downloading. Fail fast with the path if the mount did not
                # land — a dangling symlink would surface as a confusing
                # mid-training loader error minutes later.
                if not os.path.exists(ckpt_src):
                    raise RuntimeError(
                        f"ckpt_src missing in container: {ckpt_src} — "
                        f"check asset_mounts and the backend's boot mapping")
                os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
                if not os.path.lexists(ckpt_dir):
                    os.symlink(ckpt_src, ckpt_dir)
                print(f"[worker] eval-only: ckpt_src symlink {ckpt_dir} "
                      f"-> {ckpt_src}", flush=True)
            else:
                print(f"[worker] eval-only: downloading mid checkpoint -> {ckpt_dir}",
                      flush=True)
                storage.download_dir(f"{result_uri}/mid_checkpoint", ckpt_dir)
        else:
            mix_dir = os.path.join(work, "mixture_data")
            print(f"[worker] downloading mixture data -> {mix_dir}", flush=True)
            storage.download_dir(s["mixture_data_uri"], mix_dir)

            # Base checkpoint symlink (same semantics as the local executor's
            # _symlink_base_checkpoint: base_checkpoints/{tag} -> d{depth}).
            # Fail fast when the asset mount did not land — a dangling
            # symlink would die minutes later inside mid_train's loader
            # with an opaque checkpoint error.
            if not os.path.exists(s["base_ckpt_src"]):
                raise RuntimeError(
                    f"base ckpt asset missing in container: "
                    f"{s['base_ckpt_src']} — check asset_mounts "
                    f"(name 'd<N>' -> {{BASE}}/base_checkpoints/d<N>) and "
                    f"the backend's boot mapping")
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
                             os.path.join(work, train_log_name),
                             cwd=s["nanochat_dir"], env=env)
            res["mid_train_rc"] = mid_rc
            if mid_rc != 0:
                # torch elastic teardown makes a train failure visible on
                # EVERY node, so nobody proceeds to the eval rendezvous —
                # no straggler hangs at init_process_group.
                try:
                    storage.upload_file(os.path.join(work, train_log_name),
                                        f"{result_uri}/{train_log_name}")
                except Exception:
                    traceback.print_exc()  # log delivery is best-effort
                if node_count > 1 and node_rank != 0:
                    stop_stream()
                    return mid_rc
                return finish(mid_rc)

        # ── model relay (multi-node; see the relay section above). Any
        # miss degrades to eval_world=8. Disabled entirely by
        # CLIMBMIX_EVAL_RELAY=0. eval_only specs never get here (they
        # are single-node only — refused at the top of this try).
        eval_world = 8
        if node_count > 1 and mid_rc == 0 and RELAY_ENABLED:
            tag_dir = os.path.join(base, "mid_checkpoints", tag)
            if node_rank == 0:
                try:
                    srv = _RelayServer(tag_dir, master_port + 2, node_count)
                    srv.start()
                    print(f"[worker] relay: serving "
                          f"{', '.join(_relay_files(tag_dir))} to "
                          f"{node_count - 1} peers on :{master_port + 2}",
                          flush=True)
                    eval_world = srv.wait()
                except Exception:
                    traceback.print_exc()
                    eval_world = 8
                print(f"[worker] relay: eval world = "
                      f"{8 * node_count if eval_world == 32 else 8} ranks",
                      flush=True)
            else:
                eval_world = _relay_pull(
                    master_addr, master_port + 2, node_rank, tag_dir) or 8

        if node_count > 1 and node_rank != 0:
            if eval_world != 32:
                # Relay fallback: train was this node's LAST stage. Land
                # the train log and release the cards back to the pool
                # while node 0 runs the 8-rank eval.
                try:
                    storage.upload_file(os.path.join(work, train_log_name),
                                        f"{result_uri}/{train_log_name}")
                except Exception:
                    traceback.print_exc()  # log delivery is best-effort
                stop_stream()
                return 0
            # 32-rank world: this node joins the retargeted eval. The
            # CSV (rank-0-only writer), result.json and the checkpoint
            # upload stay master-owned — a second writer would clobber.
            s["eval_cmd"] = nanochat_cmds.retarget_torchrun_multinode(
                s["eval_cmd"], node_count, node_rank,
                master_addr, master_port + 1)
            eval_base = nanochat_cmds.make_eval_base_dir(
                base, work, tag, subdir=f"_eval_base_node{node_rank}")
            env = nanochat_cmds.build_subprocess_env(
                s["nanochat_dir"], base,
                device_ids=s.get("visible_devices") or [0],
                base_dir_override=eval_base,
                extra_env=s.get("env") or None)
            eval_rc = run_cmd(s["eval_cmd"],
                              os.path.join(work, eval_log_name),
                              cwd=s["nanochat_dir"], env=env)
            res["eval_rc"] = eval_rc
            stop_stream()
            for log_name in (train_log_name, eval_log_name):
                lp = os.path.join(work, log_name)
                if os.path.isfile(lp):
                    try:
                        storage.upload_file(lp, f"{result_uri}/{log_name}")
                    except Exception:
                        traceback.print_exc()  # best-effort
            return 0 if (mid_rc == 0 and eval_rc == 0) \
                else (mid_rc or eval_rc)

        # Eval in a PRIVATE base dir (symlink farm — identical to local).
        # Single-node jobs and relay-fallback runs use the spec's original
        # single-node 8-rank --standalone argv; relay-delivered runs
        # retarget to the full-node-count world (score-identical by
        # construction: base_eval's per-sample striding + all_reduce).
        if eval_world == 32:
            s["eval_cmd"] = nanochat_cmds.retarget_torchrun_multinode(
                s["eval_cmd"], node_count, node_rank,
                master_addr, master_port + 1)
        eval_base = nanochat_cmds.make_eval_base_dir(base, work, tag)
        env = nanochat_cmds.build_subprocess_env(
            s["nanochat_dir"], base,
            device_ids=s.get("visible_devices") or [0],
            base_dir_override=eval_base,
            extra_env=s.get("env") or None)
        eval_rc = run_cmd(s["eval_cmd"], os.path.join(work, eval_log_name),
                          cwd=s["nanochat_dir"], env=env)
        res["eval_rc"] = eval_rc
        # stages done — the post-stage uploads below are authoritative
        stop_stream()

        for log_name in (train_log_name, eval_log_name):
            lp = os.path.join(work, log_name)
            if os.path.isfile(lp):
                try:
                    storage.upload_file(lp, f"{result_uri}/{log_name}")
                except Exception:
                    traceback.print_exc()  # log delivery is best-effort

        # Node 0 (or any single-node run): land the artifacts. The
        # checkpoint upload sits AFTER eval so the eval starts on a quiet
        # machine (no GB-scale upload competing for the mount); the
        # mid_rc==0 guard keeps eval-failure paths from losing the
        # checkpoint (an eval-only retry needs it).
        if not eval_only and s.get("upload_checkpoint", True) \
                and res["mid_train_rc"] == 0:
            ckpt_dir = os.path.join(base, "mid_checkpoints", tag)
            print(f"[worker] uploading mid checkpoint -> "
                  f"{result_uri}/mid_checkpoint", flush=True)
            storage.upload_dir(ckpt_dir, f"{result_uri}/mid_checkpoint")
            res["checkpoint_uploaded"] = True

        csv_path = nanochat_cmds.claim_eval_csv(work, tag, eval_base,
                                                eval_rc=eval_rc)
        if csv_path is not None:
            res["csv"] = os.path.basename(csv_path)
            storage.upload_file(csv_path, f"{result_uri}/{res['csv']}")

        if mid_rc == 0 and eval_rc == 0:
            return finish(0)
        return finish(1)

    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        # A crash between train success and the post-eval upload must not
        # lose the trained checkpoint — an eval-only retry on the landed
        # mid_checkpoint is far cheaper than retraining (or the local
        # 10h fallback). Best-effort: the crash may be storage-related.
        if not eval_only and s.get("upload_checkpoint", True) \
                and res["mid_train_rc"] == 0 and not res["checkpoint_uploaded"]:
            try:
                ckpt_dir = os.path.join(base, "mid_checkpoints", tag)
                storage.upload_dir(ckpt_dir, f"{result_uri}/mid_checkpoint")
                res["checkpoint_uploaded"] = True
            except Exception:
                traceback.print_exc()
        # result.json is master-owned: a non-master finish() would clobber
        # node 0's copy on OBS (last-writer-wins).
        if node_count > 1 and node_rank != 0:
            return 3
        return finish(3)


if __name__ == "__main__":
    sys.exit(main())
