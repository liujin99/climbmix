"""Atomic file I/O helpers shared across the pipeline.

Every persistent artifact that gates a resume/skip decision (caches, search
state, shard directories, .done markers) must be written atomically:
write to a temp name, then os.replace() onto the final name. A crash
mid-write then leaves either the old complete file or an orphan .tmp —
never a half-finished file that looks "complete" to the skip logic on the
next restart.
"""

import json
import os


def _clear_tmp(tmp_path: str) -> None:
    try:
        os.remove(tmp_path)
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


def atomic_write_json(path: str, obj, default=None, indent=None) -> None:
    """json.dump with tmp+rename."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    _clear_tmp(tmp)
    try:
        with open(tmp, "w") as f:
            json.dump(obj, f, default=default, indent=indent)
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
