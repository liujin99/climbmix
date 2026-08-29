#!/usr/bin/env python3
"""M1 hello-world job — proves the gateway/token/pool/image path end to end.

Submits a tiny platform job via ModelArtsJobAPI.submit_raw (no boot shell,
no assets): npu-smi + a moxing read of the run's OBS prefix, then polls to
a terminal state. This calibrates, on the REAL gateway:

  - auth (token fetch + 24h rollover + 401 retry path)
  - the v2 status table (modelarts_job_api._INT_STATUS/_STR_STATUS —
    compare what this script logs against the console's job state)
  - submit-rejection error codes/texts (TransientSubmitError mapping)
  - DELETE/cancel behavior
  - that the image sees NPUs + can reach OBS (moxing)

Usage (server):
    python3 scripts/ma_hello_world.py --remote-config <remote_config.json>

The remote_config.json is the one run_climbmix.sh generates in the output
dir (or hand-write the minimal one from config/remote_ma.example.json +
obs_prefix/backend knobs). Polls every 10s; Ctrl-C cancels the job.
"""

import argparse
import os
import sys
import time

try:
    from climbmix.remote.job_api import JobStatus
    from climbmix.remote.modelarts_job_api import ModelArtsJobAPI
    from climbmix.remote.remote_executor import RemoteConfig
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from climbmix.remote.job_api import JobStatus
    from climbmix.remote.modelarts_job_api import ModelArtsJobAPI
    from climbmix.remote.remote_executor import RemoteConfig

HELLO_CMD = (
    "npu-smi info && "
    "python3 -c \"import moxing as mox; "
    "print('moxing OK, version', getattr(mox, '__version__', '?')); "
    "print('obs prefix visible:', mox.file.exists('%(obs_prefix)s'))\""
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--remote-config", required=True,
                   help="RemoteConfig JSON (run_climbmix.sh output or "
                        "hand-written)")
    p.add_argument("--poll-s", type=float, default=10.0)
    args = p.parse_args()

    rc = RemoteConfig.from_json_file(args.remote_config)
    api = ModelArtsJobAPI(rc)

    obs_prefix = rc.obs_prefix.rstrip("/")
    cmd = HELLO_CMD % {"obs_prefix": obs_prefix}
    print("[hello] submitting calibration job "
          f"(obs_prefix={obs_prefix}) ...")
    job_id = api.submit_raw("climbmix-hello", cmd, env=dict(rc.job_env))
    print(f"[hello] job {job_id} — watch the console link above; this "
          f"script polls every {args.poll_s:.0f}s and prints every raw "
          f"status transition it maps")

    try:
        last = None
        while True:
            st = api.status(job_id)
            if st is not last:
                print(f"[hello] {time.strftime('%H:%M:%S')} status -> "
                      f"{st.value}")
                last = st
            if st.is_terminal:
                print(f"[hello] terminal: {st.value}")
                print(f"[hello] CALIBRATE: if this disagrees with the "
                      f"console state, fix _INT_STATUS/_STR_STATUS in "
                      f"src/climbmix/remote/modelarts_job_api.py")
                return 0 if st == JobStatus.SUCCEEDED else 1
            time.sleep(args.poll_s)
    except KeyboardInterrupt:
        print("\n[hello] interrupted — cancelling job")
        api.cancel(job_id)
        return 130


if __name__ == "__main__":
    sys.exit(main())
