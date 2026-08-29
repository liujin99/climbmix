"""FAISS clustering throughput microbenchmark.

Answers two questions for a given host before/while running the cluster
stage at scale:
  1. thread sweet spot — faiss IndexFlatIP search (the exact shape of one
     K-means train iteration: n x K x dim inner-product assignment) swept
     over thread counts. SGEMM is compute-bound: hyperthreaded hosts peak
     at physical-core thread counts and REGRESS beyond (measured on a
     8C/16T x86: 8 threads 253 GFLOP/s, 16 threads 177). If the sweet spot
     differs from the code default (min(cpu,24)), pin it with
     CLIMBMIX_CLUSTER_THREADS.
  2. process fan-out scaling — when intra-process threads plateau below
     the physical core count (observed on a 192-vCPU aarch64 host: 64
     threads requested, ~23 cores busy), multiprocessing over row ranges
     recovers the rest: each child gets its own OpenMP + BLAS pools.

This script pins BLAS pools to 1 thread BEFORE importing numpy/faiss
(mirroring the production fix in cluster_embeddings_faiss: faiss OpenMP
supplies outer parallelism; nested BLAS pools thrash — unpinned measured
2.5x slower on 8C/16T). Children are spawn-started: fork after the parent
has run OpenMP regions deadlocks in libgomp.

Usage:
  python3 scripts/diagnostics/cluster_bench.py                   # sweep + fanout 4,8
  python3 scripts/diagnostics/cluster_bench.py --sweep 8,16,32,64 --fanout ""
  python3 scripts/diagnostics/cluster_bench.py --fanout 4,8,16 --threads 24
"""

import os

# Pin ALL BLAS pools to 1 thread BEFORE importing numpy/faiss, mirroring the
# production fix in cluster_embeddings_faiss. Two libraries are involved:
# numpy's OpenBLAS (pthread build, reads OPENBLAS_NUM_THREADS) and faiss's
# bundled OpenBLAS (OpenMP build — ignores OPENBLAS_NUM_THREADS, reads only
# OMP_NUM_THREADS). OMP_NUM_THREADS=1 also caps faiss's OpenMP pool at init,
# but the sweep re-raises it per point via faiss.omp_set_num_threads() (the
# two runtimes are separate libraries; the runtime call only touches faiss's).
# Unpinned, every sgemm under every OpenMP worker spawns a full BLAS pool —
# nested oversubscription, measured 2.5x slower on 8C/16T and the suspected
# cause of the 64-threads-requested / 23-cores-busy plateau on the 192-vCPU
# aarchend host.
for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import multiprocessing as mp
import platform
import time

import numpy as np

import faiss

GFLOP_PER_ROW = None  # k * d * 2 / 1e9, set in main

_WORKER_X = None
_WORKER_IDX = None


def _make_problem(n, k, d, seed):
    rs = np.random.RandomState(seed)
    x = rs.randn(n, d).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    c = rs.randn(k, d).astype(np.float32)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    return x, c


def _cpu_banner():
    print(f"cpu_count={os.cpu_count()} arch={platform.machine()}")
    try:
        with open("/proc/cpuinfo") as f:
            names = {ln.split(":", 1)[1].strip()
                     for ln in f if ln.startswith("model name")}
        if names:
            print(f"model: {sorted(names)[0]}")
    except OSError:
        pass
    print(f"faiss {faiss.__version__}; BLAS pinned via env "
          f"(OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
          f"OMP={os.environ.get('OMP_NUM_THREADS')}); "
          f"blas_threshold={faiss.cvar.distance_compute_blas_threshold} "
          f"query_bs={faiss.cvar.distance_compute_blas_query_bs} "
          f"db_bs={faiss.cvar.distance_compute_blas_database_bs}")
    try:
        import threadpoolctl
        pools = [(i.get("internal_api"), i.get("num_threads"))
                 for i in threadpoolctl.threadpool_info()]
        print(f"threadpools: {pools}")
    except ImportError:
        pass
    for var in ("OMP_NUM_THREADS", "CLIMBMIX_CLUSTER_THREADS"):
        if os.environ.get(var):
            print(f"env: {var}={os.environ[var]}")


def _default_sweep(cpu):
    cands = {max(1, cpu // 8), max(1, cpu // 4), max(1, cpu // 2), cpu}
    if cpu > 64:
        cands.add(64)
    return sorted(cands)


def _pin_probe():
    """Replicate the PRODUCTION pin path and report whether it clobbers
    faiss's OpenMP pool: omp_set_num_threads(t) FIRST, then
    threadpool_limits(1, 'blas') — the exact order cluster_embeddings_faiss
    uses. Wheels bundling an OpenMP-built OpenBLAS (libopenblaso) sharing
    libgomp with faiss get clobbered here: openblas_set_num_threads(1)
    forwards to omp_set_num_threads(1) on the shared runtime. The env-pin
    this script applies BEFORE import does NOT hit this (order reversed),
    so without this probe the sweep measures a configuration production
    never runs. Production re-asserts around the pin (embedding_cluster
    _blas_single_thread); a CLOBBERED reading here means that re-assert is
    load-bearing on this host."""
    if not hasattr(faiss, "omp_set_num_threads"):
        print("\npin probe: faiss has no OpenMP — SKIP")
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        print("\npin probe: threadpoolctl missing — production pin is a "
              "no-op; SKIP")
        return
    t = min(os.cpu_count() or 1, 24)
    faiss.omp_set_num_threads(t)
    before = faiss.omp_get_max_threads()
    with threadpool_limits(limits=1, user_api="blas"):
        inside = faiss.omp_get_max_threads()
    after = faiss.omp_get_max_threads()
    if inside == before and after == before:
        print(f"\npin probe: set {t} -> inside pin {inside}, after exit "
              f"{after} [OK]")
    else:
        print(f"\npin probe: set {t} -> inside pin {inside}, after exit "
              f"{after} [CLOBBERED — the pin itself collapses faiss's pool; "
              f"production _blas_single_thread re-asserts around it]")
    faiss.omp_set_num_threads(1)


def bench_threads(x, c, sweep, repeats=1):
    n, k, d = x.shape[0], c.shape[0], x.shape[1]
    gflop = n * k * d * 2 / 1e9
    print(f"\n== single-process thread sweep "
          f"(search {n}x{k}x{d}, {gflop:.0f} GFLOP/call) ==")
    print(f"{'threads':>8} {'best_s':>8} {'GFLOP/s':>9}")
    best = (None, 0.0)
    idx = faiss.IndexFlatIP(d)
    idx.add(c)
    for t in sweep:
        faiss.omp_set_num_threads(t)
        dt = min(_timed(idx, x) for _ in range(repeats))
        g = gflop / dt
        mark = ""
        if g > best[1]:
            best = (t, g)
            mark = "  <- best"
        print(f"{t:>8} {dt:>8.2f} {g:>9.0f}{mark}")
    print(f"sweet spot: {best[0]} threads ({best[1]:.0f} GFLOP/s)")
    if best[0] is not None:
        print(f"  if it differs from the code default (min(cpu,24)), "
              f"pin with: export CLIMBMIX_CLUSTER_THREADS={best[0]}")


def _timed(idx, x):
    t0 = time.time()
    idx.search(x, 1)
    return time.time() - t0


def _fanout_init(n_child, k, d, seed, tpp, blas_threshold):
    global _WORKER_X, _WORKER_IDX
    faiss.omp_set_num_threads(tpp)
    faiss.cvar.distance_compute_blas_threshold = blas_threshold
    x, c = _make_problem(n_child, k, d, seed)
    _WORKER_X = x
    _WORKER_IDX = faiss.IndexFlatIP(d)
    _WORKER_IDX.add(c)


def _fanout_child(_):
    t0 = time.time()
    _WORKER_IDX.search(_WORKER_X, 1)
    return time.time() - t0


def bench_fanout(n, k, d, procs_list, threads_per_proc=None,
                 blas_threshold=1, repeats=1):
    print(f"\n== process fan-out (spawn; each child builds n/procs rows, "
          f"searches {repeats}x; blas_threshold forced to {blas_threshold} "
          f"so slices use the same gemm path as the full call; wall includes "
          f"~seconds of fixed spawn/build cost) ==")
    print(f"{'procs':>6} {'thr/proc':>9} {'total_thr':>10} {'wall_s':>8} "
          f"{'GFLOP/s':>9}")
    cpu = os.cpu_count() or 1
    ctx = mp.get_context("spawn")
    for p in procs_list:
        if p < 1 or p > n:
            continue
        tpp = threads_per_proc or max(1, cpu // p)
        n_child = n // p
        t0 = time.time()
        with ctx.Pool(p, initializer=_fanout_init,
                      initargs=(n_child, k, d, 42 + p, tpp, blas_threshold)) as pool:
            for _ in range(repeats):
                pool.map(_fanout_child, range(p))
        dt = time.time() - t0
        gflop = n_child * p * k * d * 2 / 1e9 * repeats
        print(f"{p:>6} {tpp:>9} {p * tpp:>10} {dt:>8.2f} {gflop / dt:>9.0f}")


def main():
    global GFLOP_PER_ROW
    ap = argparse.ArgumentParser(
        description="faiss clustering throughput microbenchmark")
    ap.add_argument("--n", type=int, default=256000,
                    help="docs (default 256000 = K-means train subsample at K_init=1000)")
    ap.add_argument("--k", type=int, default=1000, help="centroids")
    ap.add_argument("--d", type=int, default=1024, help="embedding dim")
    ap.add_argument("--sweep", type=str, default="",
                    help="comma-separated thread counts (default: adaptive)")
    ap.add_argument("--fanout", type=str, default="4,8",
                    help="process counts for the fan-out probe ("" = skip)")
    ap.add_argument("--threads", type=int, default=0,
                    help="fixed threads/proc for --fanout (default cpu//procs)")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    GFLOP_PER_ROW = args.k * args.d * 2 / 1e9
    _cpu_banner()
    _pin_probe()
    # uniform gemm path across sizes (small n would otherwise take the seq
    # path and measure the wrong kernel)
    faiss.cvar.distance_compute_blas_threshold = 1

    x, c = _make_problem(args.n, args.k, args.d, seed=42)
    cpu = os.cpu_count() or 1
    sweep = ([int(t) for t in args.sweep.split(",") if t.strip()]
             if args.sweep else _default_sweep(cpu))
    bench_threads(x, c, sweep, repeats=args.repeats)

    if args.fanout.strip():
        procs = [int(p) for p in args.fanout.split(",") if p.strip()]
        bench_fanout(args.n, args.k, args.d, procs,
                     threads_per_proc=args.threads or None,
                     repeats=args.repeats)


if __name__ == "__main__":
    main()
