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
    """
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
