"""ObsStorage — thin object-storage interface for the remote data plane.

Two implementations:
  - MockObsStorage: maps obs://bucket/prefix/obj to <root>/bucket/prefix/obj
    on the LOCAL filesystem. Paired with the worker's `--storage local`
    backend (same mapping convention) this gives a fully-functional fake OBS
    for laptop simulation and tests — the worker code path is 100% real.
  - ModelArtsObsStorage: moxing adapter (mox.file.*) for both the submit
    host and the job containers. Optional ak/sk come from the platform
    config (config/remote_ma.example.json schema; see modelarts_job_api).
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


class ModelArtsObsStorage:
    """Real OBS adapter over the moxing SDK (mox.file.*).

    Works on both the submit host (spec/shard upload, result download) and
    inside job containers (the worker's `--storage moxing` uses the same
    SDK directly). The ma_config "obs" section is OPTIONAL: ModelArts-
    managed hosts have credentials injected, and the internal buckets are
    reachable without explicit ak/sk; set them only when the plain import
    fails auth (moxing reads the classic ACCESS_KEY_ID/SECRET_ACCESS_KEY
    env vars — set here at construction, before the first call).

    ma_config: dict, or a path (default: RemoteConfig.ma_config /
    $CLIMBMIX_MA_CONFIG / ~/.config/climbmix/remote_ma.json — same
    resolution as ModelArtsJobAPI).
    """

    def __init__(self, ma_config=None):
        if ma_config is None:
            from climbmix.remote.modelarts_job_api import load_ma_config
            ma_config = load_ma_config()
        elif isinstance(ma_config, str):
            from climbmix.remote.modelarts_job_api import load_ma_config
            ma_config = load_ma_config(ma_config)
        obs_cfg = dict(ma_config.get("obs") or {})
        if obs_cfg.get("ak"):
            # Classic moxing credential convention (no-ops when the host
            # already has ModelArts-injected auth).
            os.environ.setdefault("ACCESS_KEY_ID", str(obs_cfg["ak"]))
            os.environ.setdefault("SECRET_ACCESS_KEY", str(obs_cfg["sk"]))
        try:
            import moxing as mox
        except ImportError as e:
            raise RuntimeError(
                "ModelArtsObsStorage requires the moxing SDK on this host "
                "(ModelArts images ship it; elsewhere check the M1 survey "
                "step in docs/remote_setup.md §1). For laptop simulation "
                "use MockObsStorage.") from e
        self.mox = mox

    # moxing URI convention: obs://bucket/key (bare /bucket/key is the
    # GATEWAY's form for job fields — never passed here).

    def upload_file(self, local_path: str, obs_uri: str) -> None:
        parent = obs_uri.rsplit("/", 1)[0]
        self.mox.file.make_dirs(parent)
        self.mox.file.copy(local_path, obs_uri)

    def download_file(self, obs_uri: str, local_path: str) -> None:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self.mox.file.copy(obs_uri, local_path)

    def upload_bytes(self, data: bytes, obs_uri: str) -> None:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".upload")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            self.upload_file(tmp, obs_uri)
        finally:
            os.remove(tmp)

    def download_bytes(self, obs_uri: str) -> bytes:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".download")
        os.close(fd)
        try:
            self.download_file(obs_uri, tmp)
            with open(tmp, "rb") as f:
                return f.read()
        finally:
            os.remove(tmp)

    def list_objects(self, obs_uri: str) -> List[str]:
        uri = obs_uri.rstrip("/")
        if not self.mox.file.exists(uri):
            return []
        out = []
        for entry in self.mox.file.list_directory(uri):
            name = os.path.basename(str(entry).rstrip("/"))
            if not name:
                continue
            try:
                if self.mox.file.is_directory(f"{uri}/{name}"):
                    continue
            except Exception:
                pass  # listing a file's neighbors is best-effort
            out.append(f"{uri}/{name}")
        return sorted(out)

    def stat(self, obs_uri: str) -> bool:
        return bool(self.mox.file.exists(obs_uri))

    def delete(self, obs_uri: str) -> None:
        try:
            self.mox.file.remove(obs_uri)
        except Exception:
            self.mox.file.remove(obs_uri, recursive=True)
