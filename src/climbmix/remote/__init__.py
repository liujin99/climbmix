"""Remote experiment execution (job backend + OBS data plane).

Local host = scheduler + mixer: it prepares each experiment's mixture shards
locally (cluster labels + pool live here), uploads them to OBS, submits a
remote job per experiment, polls, and materializes the results back as
LOCAL exp_XXXX/ directories — byte-identical in shape to locally-executed
experiments, so the search resume logic (meta.json exact-weight match) and
the stage fingerprints need ZERO changes.

Modules:
  - exp_spec.py            ExpSpec: the JSON contract between submit host and worker
  - job_api.py             JobAPI protocol + MockJobAPI (local simulation)
  - obs.py                 ObsStorage protocol + MockObsStorage (fake OBS)
  - backends.py            BackendBundle + registry: binds out-of-tree
                           platform backends (or the built-in mock) to the
                           executor — the platform-specific implementation
                           lives in a separate, access-restricted repo
  - remote_executor.py     RemoteConfig + RemoteExecutor(ProxyRunner)
"""

from climbmix.remote.exp_spec import ExpSpec
from climbmix.remote.remote_executor import RemoteConfig, RemoteExecutor

__all__ = ["ExpSpec", "RemoteConfig", "RemoteExecutor"]
