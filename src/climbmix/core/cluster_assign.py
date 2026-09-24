"""Chunked (memory-bounded) cluster assignment for at-scale embedding pools.

At production scale (116M docs x 1024 dim fp32 ~= 475 GB) the embedding
matrix lives in a disk memmap and can never be an in-RAM ndarray. The
original in-memory cluster path (`cluster_embeddings_faiss` in
embedding_cluster.py) materializes full-matrix intermediates —
``np.isnan(embeddings)`` (~119 GB bool), ``np.nan_to_num`` (~475 GB copy),
``embeddings == 0`` (~119 GB bool) — and hands the whole matrix to
``index.search``. On a small pool that is fine; at scale it OOMs.

Assignment (each row's nearest centroid by inner product) is ROW-INDEPENDENT:
chunking changes only the I/O batching, never a value. Every function here
produces elementwise-identical results to the in-memory path — guarded by
tests asserting exact equality, not approximation.

Chunk sizing is memory-adaptive and cgroup-aware: /proc/meminfo reports the
HOST's memory, which is wrong inside containers (a 1.5 TB host may cap a
job at 64 GB). We read the cgroup limit too (v2 ``memory.max``, v1
``memory.limit_in_bytes``) and trust the smaller figure, then use 1/8 of
it, clamped to [64 MB, 8 GB]: above ~8 GB a chunk buys nothing (total I/O
and flops are unchanged; faiss saturates threads at million-row batches),
while a mis-read limit's blast radius stays bounded. CLIMB_ASSIGN_CHUNK_GB
overrides for deliberate tuning.
"""

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt

_MIN_CHUNK_BYTES = 64 * 1024 * 1024    # 64 MB — sequential-read floor
_MAX_CHUNK_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB — throughput saturates past this
_MEM_FRACTION = 8                      # use 1/8 of the trustworthy limit


def read_mem_available_bytes(proc_meminfo: str = "/proc/meminfo") -> Optional[int]:
    """MemAvailable from /proc/meminfo, or None if unreadable.

    MemAvailable (not MemFree): includes reclaimable page cache — the
    number the OOM killer effectively enforces against.
    """
    try:
        with open(proc_meminfo) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def read_cgroup_limit_bytes() -> Optional[int]:
    """The process's cgroup memory limit, or None when unlimited/unknown.

    v2: /sys/fs/cgroup/memory.max ("max" = unlimited).
    v1: /sys/fs/cgroup/memory/memory.limit_in_bytes (a huge sentinel
    (~2^63 / ~9E18 on 4GiB-page hosts) means unlimited).
    Walks neither hierarchy beyond the unified/first-mount file: jobs that
    cap memory always expose it there; uncapped hosts return None and we
    fall back to host memory.
    """
    candidates = (
        ("/sys/fs/cgroup/memory.max", "max"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", None),
    )
    for path, unlimited_token in candidates:
        try:
            with open(path) as f:
                raw = f.read().strip()
        except OSError:
            continue
        if unlimited_token is not None and raw == unlimited_token:
            return None
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit > 1 << 62:  # v1 "unlimited" sentinel
            return None
        return limit
    return None


def choose_chunk_rows(
    dim: int,
    itemsize: int = 4,
    mem_available: Optional[int] = None,
    cgroup_limit: Optional[int] = None,
) -> int:
    """Rows per assignment chunk for a (n, dim) fp array.

    budget = min(MemAvailable, cgroup limit)/8, clamped to [64 MB, 8 GB];
    CLIMB_ASSIGN_CHUNK_GB (float GB) overrides the budget entirely.
    The result covers whole rows and is at least 1.
    """
    env = os.environ.get("CLIMB_ASSIGN_CHUNK_GB", "").strip()
    if env:
        try:
            budget = int(float(env) * 1024 ** 3)
        except ValueError:
            budget = None
    else:
        budget = None
    if budget is None:
        signals = [s for s in (mem_available
                               if mem_available is not None
                               else read_mem_available_bytes(),
                               cgroup_limit
                               if cgroup_limit is not None
                               else read_cgroup_limit_bytes())
                   if s and s > 0]
        budget = min(signals) if signals else _MIN_CHUNK_BYTES
        budget //= _MEM_FRACTION
        budget = max(_MIN_CHUNK_BYTES, min(_MAX_CHUNK_BYTES, budget))
    row_bytes = max(1, dim * itemsize)
    return max(1, budget // row_bytes)


def _scan_thread_count() -> int:
    """Worker count for the block-parallel prescan. Mirrors
    embedding_cluster._cluster_thread_count (24 = measured sweet spot on
    the 192-vCPU fleet host; more collapses on NUMA traffic) — duplicated
    here to keep cluster_assign import-free of embedding_cluster (which
    imports THIS module). CLIMBMIX_CLUSTER_THREADS overrides both."""
    env = os.environ.get("CLIMBMIX_CLUSTER_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return min(os.cpu_count() or 1, 24)


def _scan_block_worker(task):
    """One block of the anomaly prescan, in a worker process.

    Opens its own read-only memmap (memmaps never cross processes) and
    runs the exact per-chunk logic of the serial scan_row_anomalies —
    row-independent, so block-parallel results are bit-identical.
    task = (path, rows, dim, chunk_rows, global_start). Returns
    (global_start, n_nan, n_inf, zero_mask_for_this_block).
    """
    path, rows, dim, chunk_rows, gstart = task
    block = np.load(path, mmap_mode="r")
    if tuple(block.shape) != (rows, dim):
        raise ValueError(f"{path}: shape {tuple(block.shape)} != "
                         f"({rows}, {dim})")
    zero_mask = np.zeros(rows, dtype=bool)
    n_nan = 0
    n_inf = 0
    for lo in range(0, rows, chunk_rows):
        c = np.asarray(block[lo:lo + chunk_rows])
        nan_rows = np.isnan(c).any(axis=1)
        inf_rows = np.isinf(c).any(axis=1)
        n_nan += int(nan_rows.sum())
        n_inf += int(inf_rows.sum())
        bad = nan_rows | inf_rows
        if bad.any():
            # identical to nan_to_num(...)-then-==0-all semantics: a row is
            # flagged when every element was non-finite OR it is all zeros
            all_nonfinite = ~(np.isfinite(c).any(axis=1))
            zero_mask[lo:lo + chunk_rows] = (
                all_nonfinite | (c == 0).all(axis=1))
        else:
            zero_mask[lo:lo + chunk_rows] = (c == 0).all(axis=1)
    return gstart, n_nan, n_inf, zero_mask


def _scan_sharded_parallel(cache, chunk_rows: int, tag: str):
    """Block-parallel scan_row_anomalies for a ShardedEmbeddingCache.

    63 independent read-only block memmaps = a natural ProcessPool fan-out
    (the embed_merge per-block validation uses the same pattern). At
    prod5 scale the serial single-thread scan costs ~15-20min of pure
    memory traffic; ~24 workers bring it to ~2-3min. Any pool failure
    falls back to the serial loop (identical results, just slower)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    blocks = cache.manifest["blocks"]
    n_docs = cache.shape[0]
    dim = cache.dim
    tasks = [(os.path.join(cache.cache_dir, b["file"]),
              int(b["rows"]), dim, chunk_rows, int(b["global_start"]))
             for b in blocks]
    workers = max(1, min(_scan_thread_count(), len(tasks) or 1))
    # The caller sized chunk_rows for ONE process (cgroup-aware); N worker
    # processes each materializing that chunk would multiply the footprint.
    # Chunking is value-irrelevant (row-independent scan) — shrink per
    # worker so the TOTAL stays within the caller's budget.
    tasks = [(p, r, d, max(1, cr // workers), g) for p, r, d, cr, g in tasks]
    zero_mask = np.zeros(n_docs, dtype=bool)
    n_nan = n_inf = 0
    done_rows = 0
    mark = max(1, len(tasks) // 4)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_scan_block_worker, t): t for t in tasks}
        for done, fut in enumerate(as_completed(futs), start=1):
            gstart, b_nan, b_inf, mask = fut.result()
            n_nan += b_nan
            n_inf += b_inf
            zero_mask[gstart:gstart + len(mask)] = mask
            done_rows += len(mask)
            if len(tasks) > 4 and (done % mark == 0 or done == len(tasks)):
                print(f"{tag} prescan {done_rows:,}/{n_docs:,} rows "
                      f"({100.0 * done / len(tasks):.0f}%, parallel "
                      f"x{workers})", flush=True)
    return n_nan, n_inf, zero_mask


def scan_row_anomalies(
    embeddings: np.memmap,
    chunk_rows: int,
    tag: str = "[Cluster]",
) -> Tuple[int, int, npt.NDArray[np.bool_]]:
    """Chunked equivalent of the in-memory NaN/Inf/zero-row prescan.

    Returns (n_nan_rows, n_inf_rows, zero_mask_after_sanitize) where
    zero_mask_after_sanitize matches the ORIGINAL semantics: a row is
    flagged when it is all zeros OR entirely NaN/Inf (those become zeros
    under np.nan_to_num). A mixed row like [1.0, NaN] is NOT flagged —
    identical to the in-memory ``nan_to_num``-then-``== 0``-all behavior.
    Memory: one chunk + the (n_docs,) bool mask (~1 bit/8 per row).

    Sharded caches take the block-parallel path (see
    _scan_sharded_parallel); anything else runs the serial chunk loop.
    Streaming caches deliberately stay serial: forked workers would
    inherit a dead OBS client (the merge builds one client per process
    for exactly this reason), and the network — not CPU — is the
    bottleneck, so block-parallelism buys nothing there anyway.
    """
    from climbmix.core.embedding_cache import (
        ShardedEmbeddingCache, StreamingShardedEmbeddingCache)
    if (isinstance(embeddings, ShardedEmbeddingCache)
            and not isinstance(embeddings, StreamingShardedEmbeddingCache)):
        try:
            return _scan_sharded_parallel(embeddings, chunk_rows, tag)
        except Exception as e:
            # a pool failure is never a correctness problem — the serial
            # loop computes the identical mask, just slower
            print(f"{tag} prescan parallel path failed ({e!r}) — "
                  "falling back to the serial scan", flush=True)
    n_docs = embeddings.shape[0]
    zero_mask = np.zeros(n_docs, dtype=bool)
    n_nan = 0
    n_inf = 0
    n_chunks = (n_docs + chunk_rows - 1) // chunk_rows
    for ci in range(n_chunks):
        start = ci * chunk_rows
        chunk = np.asarray(embeddings[start:start + chunk_rows])
        nan_rows = np.isnan(chunk).any(axis=1)
        inf_rows = np.isinf(chunk).any(axis=1)
        n_nan += int(nan_rows.sum())
        n_inf += int(inf_rows.sum())
        bad = nan_rows | inf_rows
        if bad.any():
            # What the row becomes after nan_to_num(nan=0, posinf=0, neginf=0):
            # flagged iff every element was non-finite (finite elements stay).
            all_nonfinite = ~(np.isfinite(chunk).any(axis=1))
            zero_mask[start:start + chunk_rows] = all_nonfinite | (chunk == 0).all(axis=1)
        else:
            zero_mask[start:start + chunk_rows] = (chunk == 0).all(axis=1)
        if n_chunks > 4 and ci % max(1, n_chunks // 4) == 0:
            print(f"{tag} prescan {start:,}/{n_docs:,} rows "
                  f"({100.0 * (ci + 1) / n_chunks:.0f}%)")
    return n_nan, n_inf, zero_mask


def sanitize_memmap_to(
    src: np.memmap,
    dst_path: str,
    chunk_rows: int,
    tag: str = "[Cluster]",
) -> np.memmap:
    """Write a nan/inf-sanitized side-car copy of ``src`` (r+ memmap).

    The in-memory path does ``np.nan_to_num(embeddings, ...)`` — a full
    in-RAM copy. At scale the bounded equivalent is a disk copy produced
    chunk-by-chunk (peak RAM = one chunk). Only taken when the prescan
    found non-finite rows, which our embed pipeline already prevents —
    this is belt-and-suspenders for foreign/corrupted caches.
    """
    n_docs, dim = src.shape
    dst = np.memmap(dst_path, dtype=np.float32, mode="w+",
                    shape=(n_docs, dim))
    n_chunks = (n_docs + chunk_rows - 1) // chunk_rows
    for ci in range(n_chunks):
        start = ci * chunk_rows
        chunk = np.asarray(src[start:start + chunk_rows])
        dst[start:start + chunk_rows] = np.nan_to_num(
            chunk, nan=0.0, posinf=0.0, neginf=0.0)
    dst.flush()
    print(f"{tag} Sanitized copy written: {dst_path} "
          f"(non-finite -> 0, chunked; peak RAM = one chunk)")
    return dst


def assign_in_chunks(
    index,
    embeddings,
    chunk_rows: int,
    tag: str = "[Cluster]",
) -> npt.NDArray[np.int64]:
    """Chunked equivalent of ``index.search(embeddings, 1)`` label extraction.

    ``index`` is anything with ``.search(x, k)`` (faiss IndexFlatIP in
    production; fakes in tests). Row-independent exact argmax — the labels
    are elementwise identical to the single-call form. Memory: one chunk +
    the (n_docs,) int64 label array (~8 bytes/row).
    """
    n_docs = embeddings.shape[0]
    labels = np.empty(n_docs, dtype=np.int64)
    n_chunks = (n_docs + chunk_rows - 1) // chunk_rows
    for ci in range(n_chunks):
        start = ci * chunk_rows
        chunk = np.asarray(embeddings[start:start + chunk_rows])
        _, lab = index.search(chunk, 1)
        labels[start:start + chunk_rows] = lab.reshape(-1)
        if n_chunks > 4 and (ci % max(1, n_chunks // 4) == 0 or ci == n_chunks - 1):
            print(f"{tag} assign {start + len(chunk):,}/{n_docs:,} rows "
                  f"({100.0 * (ci + 1) / n_chunks:.0f}%)")
    return labels


def sidecar_path_for(memmap_obj: np.memmap) -> str:
    """Scratch path for the sanitized side-car: beside the source memmap
    when its directory still exists, else the system tempdir."""
    fname = getattr(memmap_obj, "filename", None)
    if fname:
        d = os.path.dirname(str(fname))
        if os.path.isdir(d):
            return os.path.join(d, "embedding_sanitized.tmp")
    return os.path.join(os.path.abspath(tempfile.gettempdir()),
                        "embedding_sanitized.tmp")
