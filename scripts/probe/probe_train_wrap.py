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
