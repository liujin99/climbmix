"""Atomic file I/O helpers shared across the pipeline.

Every persistent artifact that gates a resume/skip decision (caches, search
state, shard directories, .done markers) must be written atomically:
write to a temp name, then os.replace() onto the final name. A crash
mid-write then leaves either the old complete file or an orphan .tmp —
never a half-finished file that looks "complete" to the skip logic on the
next restart.
"""

import hashlib
import json
import math
import os


def shard_content_key(directory: str, length: int = 12) -> str:
    """Content-addressed key over the shard_*.parquet set in a directory.

    sha1 of sorted "name:size" lines — any change in the shard set (count,
    names, sizes) moves the key. Cheap: one listdir + stat pass, no file
    content is read (name+size is the identity contract between pipeline
    stages, same blind spot class as the stage fingerprints: a same-size
    content swap is not detected).

    Consumers:
      - dispatch_target_arm: OBS mixture dir keying (retry with a different
        budget/weights/seed must NOT stat-skip onto stale OBS shards — the
        prod4 2026-09-16 landmine: mixture_uri had no content key, so a
        3B retry would silently train on the 6B shards uploaded earlier)
      - prepare_random_baseline / mix_general_data: .done staleness guards
        (the local twins of the same bug — a .done skip that does not
        compare the config identity silently reuses old derived data).
    """
    if not os.path.isdir(directory):
        raise ValueError(f"not a directory: {directory}")
    entries = []
    for name in os.listdir(directory):
        if name.startswith("shard_") and name.endswith(".parquet"):
            entries.append((name, os.path.getsize(os.path.join(directory, name))))
    if not entries:
        raise ValueError(f"no shard_*.parquet in {directory}")
    entries.sort()
    h = hashlib.sha1()
    for name, size in entries:
        h.update(f"{name}:{size}\n".encode("utf-8"))
    return h.hexdigest()[:length]


def _finite_json(obj):
    """Recursively replace non-finite floats with None.

    json.dump's default writes literal NaN/Infinity/-Infinity, which Python's
    json.load reads back but jq and every strict RFC-8259 parser reject
    ("not valid json"). null round-trips as None; callers that need the
    non-finite semantics map None back to float('nan') on load.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


def _clear_tmp(tmp_path: str) -> None:
    try:
        os.remove(tmp_path)
    except FileNotFoundError:
        pass


# ── Stage-1 product naming (⑬r) ──────────────────────────────────────────
# The Stage-1 product (116M docs → 15 macro labels + cluster profiles)
# used to live in files named cluster_cache.npz / cluster_info_cache.json
# at two tiers (run dir + pool stage1_key_<hash>/) — "cache" named the
# role, not the content, and read as a checksum to humans. New writes
# use content names; legacy names keep loading (existing run dirs and
# copied seeds), and _save paths supersede the legacy pair.
STAGE1_NPZ = "macro_labels.npz"
STAGE1_JSON = "macro_info.json"
STAGE1_NPZ_LEGACY = "cluster_cache.npz"
STAGE1_JSON_LEGACY = "cluster_info_cache.json"


def stage1_pair(directory: str):
    """(npz_path, json_path) for a Stage-1 product directory: the
    content-named pair when present, else the legacy pair, else the
    content-named paths (fresh-write targets)."""
    npz = os.path.join(directory, STAGE1_NPZ)
    jsn = os.path.join(directory, STAGE1_JSON)
    if os.path.exists(npz) and os.path.exists(jsn):
        return npz, jsn
    legacy_npz = os.path.join(directory, STAGE1_NPZ_LEGACY)
    legacy_jsn = os.path.join(directory, STAGE1_JSON_LEGACY)
    if os.path.exists(legacy_npz) and os.path.exists(legacy_jsn):
        return legacy_npz, legacy_jsn
    return npz, jsn


def supersede_legacy_stage1(directory: str) -> None:
    """Remove a legacy-named pair after a content-named save (both load,
    but a stale same-tier copy invites exactly the confusion ⑬r fixed)."""
    for name in (STAGE1_NPZ_LEGACY, STAGE1_JSON_LEGACY):
        p = os.path.join(directory, name)
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def atomic_savez(path: str, **arrays) -> None:
    """np.savez with tmp+rename so a crash can never leave a truncated npz."""
    import numpy as np

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp.npz"
    _clear_tmp(tmp)
    try:
        np.savez(tmp, **arrays)
        os.replace(tmp, path)
    except BaseException:
        _clear_tmp(tmp)
        raise


def atomic_save_npy(path: str, array) -> None:
    """Raw .npy write with tmp+rename (same crash semantics as atomic_savez).

    np.save streams C-contiguous arrays to disk via tofile(), so a
    pool-sized embeddings array (475 GB at the full pool) writes without
    an in-RAM copy; reads mmap instead of materializing. fsync before
    the rename: merge blocks live on FUSE mounts where a crash must
    never leave a renamed-but-unflushed block (resume trusts block
    files at face value).
    """
    import numpy as np

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp.npy"
    _clear_tmp(tmp)
    try:
        np.save(tmp, np.ascontiguousarray(array))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        _clear_tmp(tmp)
        raise


def atomic_save_embeddings(path: str, embeddings) -> None:
    """Embeddings cache writer, dispatched by extension.

    .npy is the canonical format (streamable write, mmap-able read);
    .npz keeps working for legacy paths and callers that pass a .npz
    name explicitly (tests pin that behavior)."""
    if path.endswith(".npy"):
        atomic_save_npy(path, embeddings)
    else:
        atomic_savez(path, embeddings=embeddings)



def atomic_write_json(path: str, obj, default=None, indent=None) -> None:
    """json.dump with tmp+rename."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    _clear_tmp(tmp)
    try:
        with open(tmp, "w") as f:
            json.dump(_finite_json(obj), f, default=default, indent=indent)
        os.replace(tmp, path)
    except BaseException:
        _clear_tmp(tmp)
        raise


def atomic_write_parquet(path: str, table) -> None:
    """pq.write_table with tmp+rename. Accepts a pyarrow Table or pandas DataFrame."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not isinstance(table, pa.Table):
        table = pa.Table.from_pandas(table, preserve_index=False)

    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp.parquet"
    _clear_tmp(tmp)
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        _clear_tmp(tmp)
        raise


def load_json_state(path: str):
    """Load a JSON state file; returns None if missing or corrupt.

    A corrupt file means the previous run died mid-write (pre-atomic era or
    an fsync-less crash); callers should treat it as "no state" and rebuild
    rather than crash.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def file_lock(lock_path: str):
    """Exclusive advisory lock (fcntl.flock) held for the duration of the
    context. Cross-process: two runs embedding the SAME pool serialize —
    the second waits, then finds the first run's cache complete instead of
    racing it. The lock file itself is empty and permanent (its presence
    costs nothing; correctness relies on flock, not on file contents).
    """
    import fcntl
    from contextlib import contextmanager

    @contextmanager
    def _lock():
        directory = os.path.dirname(lock_path) or "."
        os.makedirs(directory, exist_ok=True)
        f = open(lock_path, "w")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            f.close()

    return _lock()
