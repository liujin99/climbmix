"""Remote experiment execution (ModelArts Job API + OBS data plane).

Local host = scheduler + mixer: it prepares each experiment's mixture shards
locally (cluster labels + pool live here), uploads them to OBS, submits a
ModelArts job per experiment, polls, and materializes the results back as
LOCAL exp_XXXX/ directories — byte-identical in shape to locally-executed
experiments, so the search resume logic (meta.json exact-weight match) and
the stage fingerprints need ZERO changes.

Modules:
  - exp_spec.py            ExpSpec: the JSON contract between submit host and worker
  - job_api.py             JobAPI protocol + MockJobAPI (local simulation);
                           ModelArtsJobAPI is re-exported lazily
  - modelarts_job_api.py   ModelArtsJobAPI (real): CSB/ROMA gateway REST +
                           IAM token auth + boot-shell composition; platform
                           values live in ~/.config/climbmix/remote_ma.json
                           (the repo is public — see config/remote_ma.example.json)
  - iam_token.py           IAM token fetch/cache (JWT + password auth, 24h
                           rollover via a 5-min-early expiry buffer)
  - obs.py                 ObsStorage protocol + MockObsStorage +
                           ModelArtsObsStorage (moxing)
  - remote_executor.py     RemoteConfig + RemoteExecutor(ProxyRunner)
"""

from climbmix.remote.exp_spec import ExpSpec
from climbmix.remote.remote_executor import RemoteConfig, RemoteExecutor

__all__ = ["ExpSpec", "RemoteConfig", "RemoteExecutor"]
