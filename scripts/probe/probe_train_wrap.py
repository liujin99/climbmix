"""Probe C wrapper: stage breadcrumbs around mid_train's hang points.

Runs 20260908_201827 / 20260909_102807: some nodes' 8 ranks freeze with
ZERO output -- mid_train only print0's on world-rank 0, so a hung
non-master node is invisible between `import torch_npu` and the first
collective traceback (set_device vs init_process_group vs HCCL comm
build all look identical). torchrun entrypoint -> this wrapper -> runpy
mid_train (unmodified). Marks every stage a rank can hang at:

  device open + tiny compute       wedged NPU
  TCP connect MASTER_ADDR:PORT     store unreachable (the pod-net tcp
                                   probe runs pre-listen; this one runs
                                   when the store SHOULD be listening)
  init_process_group enter/done    patched
  every collective enter/done      patched (first 3 + every 50th call
                                   of each name; HCCL builds its
                                   communicator inside the FIRST one --
                                   the EI0015 "rank num != rank list
                                   size" hang)
  loader generator next()          patched -- run 20260909_145450: the
                                   hung node's ranks cleared load_model
                                   /get_token_bytes/loader-FACTORY (the
                                   factory call only CREATES the
                                   generator; the body -- parquet reads
                                   (_document_batches), rustbpe encode,
                                   H2D copies -- runs on the FIRST
                                   next(), mid_train.py:247). Traces
                                   next() on the loader factories and
                                   _document_batches.
  HEARTBEAT stack dumps            faulthandler.dump_traceback_later
                                   (a C thread -- dumps even when the
                                   main thread starves the GIL inside a
                                   native call) + a timestamped marker
                                   thread. If markers STOP but dumps
                                   continue: the hang holds the GIL
                                   (rustbpe/pyarrow-style native call);
                                   if both stop: the process is gone.

Each mark prints AND appends to $PROBE_OUT/wrap_rank{N}.log, which the
output mount syncs back. Never sends bytes on the store connection
(connect + immediate close; c10d TCPStore drops closed clients).
"""

import os
import runpy
import socket
import sys
import threading
import time


def bc(msg):
    rank = os.environ.get("RANK", "?")
    line = f"{time.strftime('%H:%M:%S')} rank {rank} {msg}"
    print(f"[wrap {line}", flush=True)
    out = os.environ.get("PROBE_OUT", "")
    if out:
        try:
            with open(os.path.join(out, f"wrap_rank{rank}.log"), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


bc(f"start pid={os.getpid()} host={socket.gethostname()}")


def _start_heartbeat():
    """Periodic all-threads stack dumps into the wrap log.

    dump_traceback_later runs on a faulthandler C thread: it fires even
    if the main thread is wedged inside a native call that never
    releases the GIL (the Python marker thread would starve -- that
    divergence is itself diagnostic). The file object must stay open for
    the process lifetime (faulthandler keeps its fd).
    """
    try:
        import faulthandler
    except ImportError:  # pragma: no cover
        bc("heartbeat unavailable: no faulthandler")
        return
    try:
        period = float(os.environ.get("PROBE_HEARTBEAT_S", "60"))
    except ValueError:
        period = 60.0
    if period <= 0:
        bc("heartbeat disabled (PROBE_HEARTBEAT_S<=0)")
        return
    out = os.environ.get("PROBE_OUT", "")
    rank = os.environ.get("RANK", "?")
    if not out:
        return
    try:
        f = open(os.path.join(out, f"wrap_rank{rank}.log"), "a")
        f.write(f"{time.strftime('%H:%M:%S')} rank {rank} "
                f"HEARTBEAT armed: stack dumps every {period:.0f}s\n")
        f.flush()
        faulthandler.dump_traceback_later(period, repeat=True, file=f)

        def _marker():
            i = 0
            while True:
                time.sleep(period)
                i += 1
                try:
                    with open(os.path.join(out, f"wrap_rank{rank}.log"),
                              "a") as m:
                        m.write(f"{time.strftime('%H:%M:%S')} rank {rank} "
                                f"HEARTBEAT #{i} (stack below)\n")
                except OSError:
                    pass

        threading.Thread(target=_marker, name="probe-heartbeat",
                         daemon=True).start()
        bc(f"heartbeat every {period:.0f}s (faulthandler C-thread dumps)")
    except (OSError, ValueError) as e:
        bc(f"heartbeat setup FAILED: {type(e).__name__}: {e}")


_start_heartbeat()

import torch  # noqa: E402

try:
    import torch_npu  # noqa: F401
    bc("torch_npu imported")
    dev = int(os.environ.get("LOCAL_RANK", "0"))
    bc(f"set_device npu:{dev}")
    torch.npu.set_device(dev)
    x = torch.ones(8, device=f"npu:{dev}")
    bc(f"device compute ok ({float((x * 2).sum())})")
except SystemExit:
    raise
except BaseException as e:
    bc(f"device stage FAILED: {type(e).__name__}: {e}")
    sys.exit(1)

ma, mp = os.environ.get("MASTER_ADDR", ""), os.environ.get("MASTER_PORT", "")
bc(f"store tcp {ma}:{mp}")
try:
    with socket.create_connection((ma, int(mp)), timeout=10):
        pass
    bc("store tcp connect ok")
except BaseException as e:
    bc(f"store tcp connect FAILED: {type(e).__name__}: {e}")

# breadcrumb-patch the torch.distributed entry points mid_train hits
# (module-attribute patch -- every `dist.X(...)` call site in nanochat
# resolves through the same module object).
import torch.distributed as dist  # noqa: E402

_counts = {}
_NAMES = ("init_process_group", "all_reduce", "reduce_scatter",
          "all_gather", "broadcast", "barrier", "destroy_process_group")

# All-rank arrival gate at the FIRST data collective. EI0015's
# communicator-creation timeout is ~20min and HCCL_CONNECT_TIMEOUT does
# not shorten it on this CANN — runs 20260908_201827 .. a3cd7236 all
# burned the full 20min waiting for ranks that never arrive. Before the
# first real collective, every rank marks the c10d store; if not all
# world_size marks appear within PROBE_FIRSTCOLL_TIMEOUT_S (300), raise
# immediately — the hung ranks' heartbeat stacks name the frame, and a
# 25min job becomes a ~10min one.
_gate_armed = {"v": True}


def _first_collective_gate(timeout_s):
    if not _gate_armed["v"]:
        return
    _gate_armed["v"] = False
    try:
        ws = dist.get_world_size()
        rank = dist.get_rank()
        store = dist.distributed_c10d._get_default_store()
    except BaseException as e:
        bc(f"gate unavailable (proceeding ungated): "
           f"{type(e).__name__}: {e}")
        return
    lws = int(os.environ.get("LOCAL_WORLD_SIZE", "8") or 8)
    bc(f"gate: rank {rank}/{ws} at first collective — waiting for all "
       f"(<= {timeout_s:.0f}s)")
    t0 = time.time()
    try:
        store.set(f"probe/arr/{rank}", "1")
    except BaseException as e:
        bc(f"gate store.set FAILED: {type(e).__name__}: {e}")
        return
    while True:
        missing = []
        for r in range(ws):
            try:
                store.get(f"probe/arr/{r}")
            except BaseException:
                missing.append(r)
        if not missing:
            bc(f"gate: all {ws} ranks arrived in {time.time() - t0:.0f}s")
            return
        if time.time() - t0 > timeout_s:
            nodes = sorted({r // lws for r in missing})
            msg = (f"gate FAILED: {len(missing)}/{ws} ranks never reached "
                   f"the first collective within {timeout_s:.0f}s (missing "
                   f"ranks {missing[:12]}{'...' if len(missing) > 12 else ''}"
                   f", node(s) {nodes}) — hung BEFORE any HCCL call; their "
                   f"wrap_rank*.log heartbeats name the exact frame. "
                   f"Aborting instead of waiting ~20min for EI0015")
            bc(msg)
            raise RuntimeError(msg)
        time.sleep(2)


def _patch(name):
    fn = getattr(dist, name, None)
    if not callable(fn):
        return

    def wrapped(*a, **k):
        c = _counts.get(name, 0)
        _counts[name] = c + 1
        log = c < 3 or (c + 1) % 50 == 0
        if log:
            bc(f"dist.{name} #{c + 1} enter")
        if name not in ("init_process_group", "destroy_process_group"):
            _first_collective_gate(
                float(os.environ.get("PROBE_FIRSTCOLL_TIMEOUT_S", "300")))
        try:
            r = fn(*a, **k)
        except BaseException as e:
            if c < 3:
                bc(f"dist.{name} #{c + 1} RAISED {type(e).__name__}")
            raise
        if log:
            bc(f"dist.{name} #{c + 1} ok")
        return r

    setattr(dist, name, wrapped)


for _n in _NAMES:
    _patch(_n)
_patched = [n for n in _NAMES if callable(getattr(dist, n, None))]
bc("patches applied: " + ",".join(_patched))

# nanochat boot-stage patches. Run 20260909_1129xx: the hung nodes'
# ranks cleared init_process_group and then went silent — the gap is
# model/tokenizer/data loading (all obsfs reads on the input mounts).
# mid_train binds these names via from-imports at runpy time, so
# patching the source modules NOW (before runpy) intercepts every call
# site. nanochat/__init__ is empty — importing has no side effects.
#
# Run 20260909_145450: with inputs materialized to local disk the
# hang MOVED — hung-node ranks cleared load_model/get_token_bytes and
# the loader FACTORY call ("done in 0s" — that only creates the
# generator object), then went silent until EI0015 killed the healthy
# ranks at the first collective. The generator BODY (parquet reads,
# rustbpe encode, buffer assembly, H2D copy) runs on the FIRST next()
# at mid_train.py:247 — so the loader factories and _document_batches
# get traced-generator patches (next # breadcrumbs), not just
# enter/done.
try:
    import nanochat.checkpoint_manager as _cm
    import nanochat.dataloader as _dl
    import nanochat.tokenizer as _tok

    def _bc_patch(mod, name):
        fn = getattr(mod, name, None)
        if not callable(fn):
            return

        def wrapped(*a, **k):
            bc(f"{name} enter")
            t0 = time.time()
            try:
                r = fn(*a, **k)
            except BaseException as e:
                bc(f"{name} RAISED {type(e).__name__} "
                   f"after {time.time() - t0:.0f}s")
                raise
            bc(f"{name} done in {time.time() - t0:.0f}s")
            return r

        setattr(mod, name, wrapped)

    def _trace_generator(gen, label):
        """Wrap a generator: breadcrumb each next() (first 3 + 50ths)."""
        n = 0
        while True:
            n += 1
            log = n <= 3 or n % 50 == 0
            if log:
                bc(f"{label} next #{n} enter")
            t0 = time.time()
            try:
                v = next(gen)
            except StopIteration:
                if log:
                    bc(f"{label} next #{n} StopIteration")
                raise
            except BaseException as e:
                bc(f"{label} next #{n} RAISED {type(e).__name__} "
                   f"after {time.time() - t0:.0f}s")
                raise
            if log:
                bc(f"{label} next #{n} ok in {time.time() - t0:.1f}s")
            yield v

    def _patch_gen_factory(mod, name, label):
        """Patch a GENERATOR factory: trace every next() of the body."""
        fn = getattr(mod, name, None)
        if not callable(fn):
            return

        def wrapped(*a, **k):
            bc(f"{name} enter (generator body runs on FIRST next)")
            t0 = time.time()
            gen = fn(*a, **k)
            bc(f"{name} generator created in {time.time() - t0:.0f}s")
            return _trace_generator(gen, label)

        setattr(mod, name, wrapped)

    for _m, _nms in ((_cm, ("load_model", "load_optimizer_state")),
                     (_tok, ("get_token_bytes",))):
        for _nm in _nms:
            _bc_patch(_m, _nm)
    _patch_gen_factory(_dl, "tokenizing_distributed_data_loader_flat",
                       "flat_loader")
    _patch_gen_factory(
        _dl, "tokenizing_distributed_data_loader_with_state_flat",
        "flat_loader_with_state")
    _patch_gen_factory(_dl, "_document_batches", "doc_batches")
    bc("nanochat boot-stage patches applied "
       "(loader generators + _document_batches traced)")
except BaseException as e:
    bc(f"nanochat patches unavailable: {type(e).__name__}: {e}")

# pyarrow breadcrumbs. Run 20260909_154328 (heartbeat stacks): the hung
# node's ranks crawled ~60s PER READ OP inside ParquetFile.__init__ on
# LOCAL files (healthy nodes: seconds). Patch the class methods — every
# `pq.ParquetFile(...)` call site resolves through the same class object
# regardless of import order. __init__: every call (few per rank).
# read_row_group: first 3 + every 50th (one call per rank per row group).
try:
    import pyarrow.parquet as _pq  # noqa: E402

    _pf_init = _pq.ParquetFile.__init__
    _rrg = _pq.ParquetFile.read_row_group
    _rrg_n = [0]

    def _pf_init_w(self, *a, **k):
        src = a[0] if a else k.get("source", "?")
        bc(f"ParquetFile.__init__ enter: {src}")
        t0 = time.time()
        try:
            _pf_init(self, *a, **k)
        except BaseException as e:
            bc(f"ParquetFile.__init__ RAISED {type(e).__name__} "
               f"after {time.time() - t0:.1f}s: {src}")
            raise
        bc(f"ParquetFile.__init__ ok in {time.time() - t0:.1f}s: {src} "
           f"({self.metadata.num_row_groups} row groups)")

    def _rrg_w(self, *a, **k):
        c = _rrg_n[0]
        _rrg_n[0] = c + 1
        log = c < 3 or (c + 1) % 50 == 0
        if log:
            bc(f"read_row_group #{c + 1} enter")
        t0 = time.time()
        r = _rrg(self, *a, **k)
        dt = time.time() - t0
        if log or dt > 10:
            bc(f"read_row_group #{c + 1} ok in {dt:.1f}s"
               + ("" if log else " (SLOW)"))
        return r

    _pq.ParquetFile.__init__ = _pf_init_w
    _pq.ParquetFile.read_row_group = _rrg_w
    bc("pyarrow patches applied (ParquetFile.__init__ + read_row_group)")
except BaseException as e:
    bc(f"pyarrow patches unavailable: {type(e).__name__}: {e}")

bc("runpy scripts.mid_train " + " ".join(sys.argv[1:]))
sys.argv = ["scripts.mid_train"] + sys.argv[1:]
try:
    runpy.run_module("scripts.mid_train", run_name="__main__")
except SystemExit as e:
    bc(f"mid_train exited {e.code!r}")
    raise
except BaseException as e:
    bc(f"mid_train raised {type(e).__name__}: {e}")
    raise
