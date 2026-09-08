"""Multi-node probe B: HCCL init + allreduce bandwidth + latency.

Runs under torchrun (probe_hccl.sh) with world_size = nnodes * nproc.
Rank 0 writes hccl_result.json into $PROBE_OUT (the platform's output
mount — synced back to OBS). Any rank's failure writes an error record
so the driver can distinguish "HCCL broken" from "job broken".
"""

import json
import os
import socket
import time


def write_result(out_dir, payload):
    payload = dict(payload)
    payload.setdefault("hostname", socket.gethostname())
    path = os.path.join(out_dir, "hccl_result.json")
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    print(f"[probe B rank {payload.get('rank')}] wrote {path}")


def main():
    out_dir = os.environ.get("PROBE_OUT", ".")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))

    import torch
    try:
        import torch_npu  # noqa: F401  (registers the npu backend)
    except ImportError as e:
        if rank == 0:
            write_result(out_dir, {"init_ok": False,
                                   "error": f"torch_npu import failed: {e}"})
        raise SystemExit(1)

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    try:
        torch.npu.set_device(local_rank % torch.npu.device_count())
    except Exception as e:
        write_result(out_dir, {"init_ok": False, "rank": rank,
                               "error": f"set_device failed: {e}"})
        raise SystemExit(1)

    import torch.distributed as dist

    t0 = time.time()
    try:
        dist.init_process_group(backend="hccl")
    except Exception as e:
        write_result(out_dir, {"init_ok": False, "rank": rank, "world_size": world,
                               "error": f"init_process_group failed: {e}"})
        raise SystemExit(1)
    init_s = time.time() - t0
    print(f"[probe B rank {rank}/{world}] hccl init ok in {init_s:.1f}s")

    dev = f"npu:{local_rank}"

    # ── large allreduce: 1 GiB fp32, 2 warmup + 10 timed ──
    n = 256 * 1024 * 1024  # 268,435,456 fp32 = 1 GiB
    x = torch.ones(n, dtype=torch.float32, device=dev)
    for _ in range(2):
        dist.all_reduce(x)
    torch.npu.synchronize()
    t0 = time.time()
    iters = 10
    for _ in range(iters):
        dist.all_reduce(x)
    torch.npu.synchronize()
    ar_ms = (time.time() - t0) / iters * 1000.0
    algbw_gib = (n * 4) / (ar_ms / 1000.0) / 2**30
    busbw_gib = algbw_gib * 2 * (world - 1) / world    # ring allreduce model
    print(f"[probe B rank {rank}] 1GiB allreduce: {ar_ms:.1f} ms | "
          f"algbw {algbw_gib:.2f} GiB/s | busbw {busbw_gib:.2f} GiB/s")

    # ── small allreduce latency: 4 floats ──
    y = torch.ones(4, dtype=torch.float32, device=dev)
    for _ in range(10):
        dist.all_reduce(y)
    torch.npu.synchronize()
    t0 = time.time()
    n_lat = 200
    for _ in range(n_lat):
        dist.all_reduce(y)
        torch.npu.synchronize()
    small_ms = (time.time() - t0) / n_lat * 1000.0

    # ── barrier latency ──
    for _ in range(10):
        dist.barrier()
    t0 = time.time()
    n_bar = 100
    for _ in range(n_bar):
        dist.barrier()
    bar_ms = (time.time() - t0) / n_bar * 1000.0

    if rank == 0:
        write_result(out_dir, {
            "init_ok": True, "rank": rank, "world_size": world,
            "init_s": round(init_s, 2),
            "allreduce_ms": round(ar_ms, 1), "allreduce_gib": 1,
            "algbw_gib_s": round(algbw_gib, 2),
            "busbw_gib_s": round(busbw_gib, 2),
            "small_allreduce_ms": round(small_ms, 3),
            "barrier_ms": round(bar_ms, 3),
        })
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
