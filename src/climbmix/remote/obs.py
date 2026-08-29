"""ObsStorage — thin object-storage interface for the remote data plane.

Implementations:
  - MockObsStorage (this file): maps obs://bucket/prefix/obj to
    <root>/bucket/prefix/obj on the LOCAL filesystem. Paired with the
    worker's `--storage local` backend (same mapping convention) this
    gives a fully-functional fake OBS for laptop simulation and tests —
    the worker code path is 100% real.
  - Real object-store adapters live OUT of this repo with the other
    platform backend pieces (see backends.py); the worker's `--storage
    moxing` mode talks to the same SDK directly inside job containers.
"""

import os
import shutil
from typing import List, Optional, Protocol, runtime_checkable


def parse_obs_uri(uri: str) -> tuple:
    """obs://bucket/a/b -> ("bucket", "a/b"). Raises on malformed input."""
    prefix = "obs://"
    if not uri.startswith(prefix):
        raise ValueError(f"not an obs:// URI: {uri!r}")
    rest = uri[len(prefix):]
    if not rest or rest.startswith("/"):
        raise ValueError(f"malformed obs URI (empty bucket): {uri!r}")
    parts = rest.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


@runtime_checkable
class ObsStorage(Protocol):
    def upload_file(self, local_path: str, obs_uri: str) -> None: ...
    def download_file(self, obs_uri: str, local_path: str) -> None: ...
    def upload_bytes(self, data: bytes, obs_uri: str) -> None: ...
    def download_bytes(self, obs_uri: str) -> bytes: ...
    def list_objects(self, obs_uri: str) -> List[str]: ...
    def stat(self, obs_uri: str) -> bool: ...
    def delete(self, obs_uri: str) -> None: ...


class MockObsStorage:
    """Filesystem-backed fake OBS. obs://bucket/a/b maps to
    <root>/bucket/a/b. The remote worker's `--storage local` backend uses the
    SAME convention (root passed via --storage-root), so submit side and
    worker side see one coherent storage."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def _local(self, obs_uri: str) -> str:
        bucket, key = parse_obs_uri(obs_uri)
        return os.path.join(self.root, bucket, key)

    def upload_file(self, local_path: str, obs_uri: str) -> None:
        dst = self._local(obs_uri)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(local_path, dst)

    def download_file(self, obs_uri: str, local_path: str) -> None:
        src = self._local(obs_uri)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"obs object not found: {obs_uri}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        shutil.copy2(src, local_path)

    def upload_bytes(self, data: bytes, obs_uri: str) -> None:
        dst = self._local(obs_uri)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)

    def download_bytes(self, obs_uri: str) -> bytes:
        src = self._local(obs_uri)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"obs object not found: {obs_uri}")
        with open(src, "rb") as f:
            return f.read()

    def list_objects(self, obs_uri: str) -> List[str]:
        path = self._local(obs_uri)
        if not os.path.isdir(path):
            return []
        return sorted(
            os.path.join(obs_uri.rstrip("/"), f)
            for f in os.listdir(path)
        )

    def stat(self, obs_uri: str) -> bool:
        return os.path.exists(self._local(obs_uri))

    def delete(self, obs_uri: str) -> None:
        p = self._local(obs_uri)
        if os.path.isdir(p):
            shutil.rmtree(p)
        elif os.path.isfile(p):
            os.remove(p)
