"""Backend registry — binds out-of-tree execution backends to RemoteExecutor.

RemoteExecutor is platform-neutral: it speaks the JobAPI (compute plane)
and ObsStorage (data plane) protocols. The core ships exactly ONE backend:

  - "mock" — local simulation (MockJobAPI subprocesses + filesystem-backed
    fake OBS); powers the laptop end-to-end tests and any dry run.

Every real platform backend lives OUT of this repo (typically in a
separate, access-restricted repository holding the gateway REST client,
the auth/token plumbing, and the deployment's platform identity values).
A backend packages itself as a BackendBundle:

  make_job_api(remote_config)  -> JobAPI implementation (submit/status/
                                  logs/cancel/free_job_slots)
  make_obs_storage(remote_config) -> ObsStorage implementation
  default_worker_path          -> container path of remote_worker.py under
                                  the platform's code-delivery convention
                                  ("" = executor derives the locally staged
                                  path — the mock behavior)
  validate(remote_config)      -> optional fail-fast hook: resolve/verify
                                  the backend's platform config at launch
                                  (None = nothing to check)

Registration (either works):
  1. RemoteConfig.backend_module = "pkg.mod:attr" — attr is a callable
     create_backend(remote_config) -> BackendBundle. Works with the
     backend repo simply cloned and on PYTHONPATH (no installation).
  2. pip-installed backends expose an entry point in group
     "climbmix.backends" whose name equals RemoteConfig.backend
     (auto-discovered; backend_module is then unnecessary).

Resolution order: built-in "mock" -> backend_module -> entry points. An
explicit backend_module always wins over entry-point lookup. Unresolvable
names raise with setup guidance. See docs/remote_setup.md ("Writing a
backend") for the full contract.
"""

import importlib
from dataclasses import dataclass
from typing import Callable, Optional

from climbmix.remote.job_api import JobAPI, MockJobAPI
from climbmix.remote.obs import MockObsStorage, ObsStorage


@dataclass
class BackendBundle:
    """Everything a backend provides beyond the two protocol objects: the
    factories (called with the RemoteConfig), the worker-path convention,
    and an optional launch-time validation hook."""

    make_job_api: Callable[[object], JobAPI]
    make_obs_storage: Callable[[object], ObsStorage]
    default_worker_path: str = ""
    validate: Optional[Callable[[object], None]] = None


def create_mock_backend(remote_config):
    """Built-in simulation bundle: local subprocess jobs + a filesystem
    fake OBS rooted at RemoteConfig.storage_root."""

    def make_job_api(rc):
        return MockJobAPI()

    def make_obs_storage(rc):
        return MockObsStorage(rc.storage_root)

    return BackendBundle(make_job_api=make_job_api,
                         make_obs_storage=make_obs_storage)


def _load_from_module_spec(spec: str) -> Callable:
    module_name, _, attr = spec.partition(":")
    if not module_name:
        raise ValueError(
            f"invalid backend_module spec {spec!r} — expected "
            f"'package.module:attr' (attr defaults to create_backend)")
    mod = importlib.import_module(module_name)
    factory = getattr(mod, attr or "create_backend", None)
    if not callable(factory):
        raise ValueError(
            f"backend_module {spec!r}: {module_name}.{attr or 'create_backend'} "
            f"is not callable")
    return factory


def _resolve_entry_point(name: str) -> Callable:
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="climbmix.backends")
    except Exception:
        eps = ()
    for ep in eps:
        if ep.name == name:
            return ep.load()
    raise ValueError(
        f"backend {name!r} is not built in and no backend_module is set. "
        f"Either install the backend package (it registers the "
        f"'climbmix.backends' entry point) or point RemoteConfig."
        f"backend_module at its create_backend factory, e.g. "
        f"backend_module='my_backend:create_backend'. "
        f"Built-in backends: mock")


def resolve_backend(remote_config) -> BackendBundle:
    """Resolve RemoteConfig.backend into a BackendBundle (see the module
    docstring for the resolution order). Raises ValueError with setup
    guidance when the backend cannot be resolved."""
    name = str(getattr(remote_config, "backend", "mock") or "mock")
    module_spec = str(getattr(remote_config, "backend_module", "") or "")
    if name == "mock" and not module_spec:
        return create_mock_backend(remote_config)
    factory = (_load_from_module_spec(module_spec) if module_spec
               else _resolve_entry_point(name))
    bundle = factory(remote_config)
    if not isinstance(bundle, BackendBundle):
        raise TypeError(
            f"backend {name!r}: factory must return a "
            f"climbmix.remote.backends.BackendBundle, got "
            f"{type(bundle).__name__}")
    return bundle
