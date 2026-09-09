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

Each mark prints AND appends to $PROBE_OUT/wrap_rank{N}.log, which the
output mount syncs back. Never sends bytes on the store connection
(connect + immediate close; c10d TCPStore drops closed clients).
"""

import os
import runpy
import socket
import sys
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
