#!/usr/bin/env python3
"""prod2 runtime verification — target-arm argv parity / adaptive sizing /
local whole-node slot / admission + eviction / spec v2 / dispatch smoke.

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_prod2_runtime.py
Exit 0 = all checks pass. Complements test_prod2_fixes.py (22 checks,
balanced/acc-only/no-signal) — rerun both.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from climbmix.core.types import (  # noqa: E402
    CLIMBConfig, MixtureConfig, MixtureWeights, ProxyResult)
from climbmix.core.iterative_bootstrapper import IterativeBootstrapper  # noqa: E402
from climbmix.pipeline.nanochat_cmds import (  # noqa: E402
    build_target_mid_train_cmd, build_target_eval_cmd)
from climbmix.remote.exp_spec import ExpSpec, SPEC_VERSION  # noqa: E402
from climbmix.remote.job_api import JobStatus  # noqa: E402
from climbmix.remote.remote_executor import (  # noqa: E402
    AdmissionController, ConfigEvictedError, RemoteConfig, RemoteExecutor)

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ═══════════════════════════════════════════════════════════════════════
# 1. argv parity: runs/lib/target_arm.sh vs the python builders
# ═══════════════════════════════════════════════════════════════════════
ENVV = {
    "TARGET_STEPS": "1000", "TARGET_LR_SCALE": "1.0",
    "TARGET_WARMUP": "0.0", "TARGET_WARMDOWN": "0.9",
    "CORE_METRIC_EVERY": "-1", "MID_DEVICE_BATCH_SIZE": "1",
    "MID_TRAIN_LOADER": "flat", "EVAL_BENCHMARKS": "stem",
    "EVAL_MAX_PER_TASK": "-1", "EVAL_DEVICE_BATCH_SIZE": "16",
    "EVAL_CORE_BATCH_SIZE": "8", "NUM_NPU": "8",
}
TAG, NAME = "d28_x_test", "testarm"

with tempfile.TemporaryDirectory(prefix="prod2rt_") as td:
    stub_bin = os.path.join(td, "bin")
    os.makedirs(stub_bin)
    argv_file = os.path.join(td, "argv.txt")
    stub = os.path.join(stub_bin, "torchrun")
    with open(stub, "w") as f:
        f.write('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$ARGV_FILE"\nexit 0\n')
    os.chmod(stub, 0o755)

    out_dir = os.path.join(td, "out")
    nc_dir = os.path.join(td, "nanochat")
    base_dir = os.path.join(td, "base")
    d28 = os.path.join(base_dir, "base_checkpoints", "d28")
    for d in (out_dir, nc_dir, d28, os.path.join(base_dir, "base_eval")):
        os.makedirs(d, exist_ok=True)
    with open(os.path.join(base_dir, "base_eval", "mid_model_000100.csv"), "w") as f:
        f.write("STEM,0.5,0.5,1.0\n")

    env = dict(os.environ)
    env.update(ENVV)
    env.update({
        "PATH": stub_bin + os.pathsep + env["PATH"],
        "ARGV_FILE": argv_file,
        "CLIMBMIX_DIR": REPO,
        "NANOCHAT_DIR": nc_dir,
        "NANOCHAT_BASE_DIR": base_dir,
        "OUTPUT_DIR": out_dir,
        "TARGET_BASE_CKPT": d28,
    })

    def run_lib(fn):
        r = subprocess.run(
            ["bash", "-c",
             f'source "{REPO}/runs/lib/target_arm.sh"; {fn}'],
            env=env, cwd=REPO, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
        return r

    r = run_lib(f'target_arm_train /data/mixed {TAG} {NAME}')
    check("parity: target_arm_train exit 0", r.returncode == 0)
    # the stub's "$@" excludes argv[0] — re-attach the command name
    shell_train = ["torchrun"] + open(argv_file).read().splitlines()
    py_train = build_target_mid_train_cmd(
        run_name=f"{NAME}_mid", model_tag=TAG, data_dir="/data/mixed",
        num_iterations=ENVV["TARGET_STEPS"], lr_scale=ENVV["TARGET_LR_SCALE"],
        warmup=ENVV["TARGET_WARMUP"], warmdown=ENVV["TARGET_WARMDOWN"],
        core_metric_every=ENVV["CORE_METRIC_EVERY"],
        device_batch_size=ENVV["MID_DEVICE_BATCH_SIZE"],
        loader=ENVV["MID_TRAIN_LOADER"], nproc_per_node=8)
    check("parity: train argv token-identical (shell == python)",
          shell_train == py_train,
          f"shell={shell_train}" if shell_train != py_train else "")
    check("parity: train argv shape (torchrun/mid_train/13 flags)",
          py_train[0] == "torchrun" and "-m" in py_train
          and "scripts.mid_train" in py_train
          and "--sample-every=-1" in py_train
          and "--eval-every=-1" in py_train
          and "--load-optimizer" not in " ".join(py_train))
    check("parity: no --device-type in target train (shell-proven form)",
          "--device-type" not in py_train)
    check("parity: train log tee'd",
          os.path.isfile(os.path.join(out_dir, f"mid_train_{NAME}.log")))
    link = os.path.join(base_dir, "base_checkpoints", TAG)
    check("parity: base symlink cleaned up after train",
          not os.path.lexists(link))

    r = run_lib(f'target_arm_eval {TAG} {NAME}')
    check("parity: target_arm_eval exit 0", r.returncode == 0)
    shell_eval = ["torchrun"] + open(argv_file).read().splitlines()
    py_eval = build_target_eval_cmd(
        model_tag=TAG, eval_benchmarks=ENVV["EVAL_BENCHMARKS"],
        eval_max_per_task=ENVV["EVAL_MAX_PER_TASK"],
        device_batch_size=ENVV["EVAL_DEVICE_BATCH_SIZE"],
        core_batch_size=ENVV["EVAL_CORE_BATCH_SIZE"], nproc_per_node=8)
    check("parity: eval argv token-identical (shell == python)",
          shell_eval == py_eval,
          f"shell={shell_eval}" if shell_eval != py_eval else "")
    check("parity: eval CSV archived to eval_<name>.csv",
          os.path.isfile(os.path.join(out_dir, f"eval_{NAME}.csv")))

    # base-check variant (model_type=base)
    py_base = build_target_eval_cmd(
        model_tag="d28", eval_benchmarks="stem", eval_max_per_task="-1",
        device_batch_size="16", core_batch_size="8", model_type="base")
    check("parity: base-eval argv uses --model-type=base",
          "--model-type=base" in py_base and "--model-type=mid" not in py_base)

# ═══════════════════════════════════════════════════════════════════════
# 2. local whole-node slot (_local_slots)
# ═══════════════════════════════════════════════════════════════════════
def bare_executor(npu_per_exp, npu_devices):
    ex = object.__new__(RemoteExecutor)
    ex.npu_per_exp = npu_per_exp
    ex.npu_devices = npu_devices
    return ex

check("local_slots: k=8 on 8 cards -> 1 whole-node slot",
      bare_executor(8, 8)._local_slots() == 1)
check("local_slots: k=4 on 8 cards -> 2 slices",
      bare_executor(4, 8)._local_slots() == 2)
check("local_slots: k=1 on 8 cards -> 8 slices",
      bare_executor(1, 8)._local_slots() == 8)
check("local_slots: k=7 on 8 cards -> 0 (non-divisor idles)",
      bare_executor(7, 8)._local_slots() == 0)
check("local_slots: k=16 on 8 cards -> 0 (oversize idles)",
      bare_executor(16, 8)._local_slots() == 0)

# ═══════════════════════════════════════════════════════════════════════
# 3. AdmissionController.decide (pure truncation math + guardrails)
# ═══════════════════════════════════════════════════════════════════════
adm = AdmissionController(wave_budget=2, expected_slots=11)
drops = adm.decide(total_configs=26, n_local=1, running_now=10, local_slots=1)
check("admission: iter1 pool 26, C_eff=11, budget 2 -> admit 22, drop 4 tail",
      drops == [22, 23, 24, 25] and len(adm.drop_set) == 4 and adm.c_eff == 11,
      f"drops={drops}")

adm2 = AdmissionController(wave_budget=1, expected_slots=11)
drops2 = adm2.decide(total_configs=11, n_local=1, running_now=10, local_slots=1)
check("admission: n_target == total -> no drops",
      drops2 == [] and not adm2.drop_set)

adm3 = AdmissionController(wave_budget=2, expected_slots=11)
drops3 = adm3.decide(total_configs=26, n_local=1, running_now=0, local_slots=1)
check("admission: C_eff=1 < 2 guardrail -> no truncation",
      drops3 == [] and adm3.c_eff == 1)

adm4 = AdmissionController(wave_budget=2, expected_slots=11, allow_truncate=False)
drops4 = adm4.decide(total_configs=26, n_local=1, running_now=10, local_slots=1)
check("admission: allow_truncate=False -> literal behavior",
      drops4 == [] and adm4.c_eff == 11)

adm5 = AdmissionController(wave_budget=1, expected_slots=2, min_configs=4)
drops5 = adm5.decide(total_configs=3, n_local=0, running_now=2, local_slots=0)
check("admission: pool < min_configs -> never truncated",
      drops5 == [])

adm6 = AdmissionController(wave_budget=1, expected_slots=2, min_configs=4)
drops6 = adm6.decide(total_configs=26, n_local=1, running_now=1, local_slots=1)
# C_eff=2, target = max(4, min(2, 26)) = 4 -> drop 22 (pool collapsed case)
check("admission: tiny C_eff clamps admitted to min_configs=4",
      len(drops6) == 22 and 26 - len(adm6.drop_set) == 4)

adm7 = AdmissionController(wave_budget=1, expected_slots=11)
check("admission: probe deadline fallback (no RUNNING yet) fires",
      not adm7.decided and adm7.probe_ready(time.time(), time.time() - 9999))
adm7.decide(5, 1, 3, 1)
check("admission: decided latches after decide()",
      adm7.decided and not adm7.probe_ready(time.time(), time.time() - 9999))

t0 = time.time()
adm8 = AdmissionController(wave_budget=1, expected_slots=11,
                           probe_delay_s=1800.0, probe_deadline_s=2400.0)
check("admission: probe not ready before delay",
      not adm8.probe_ready(t0 + 100, t0))
check("admission: probe ready at first-running + delay",
      adm8.first_running_at is None  # not yet noted
      and adm8.probe_ready(t0 + 2400, t0))
adm8.note_running(t0 + 60)
check("admission: probe follows first RUNNING clock",
      not adm8.probe_ready(t0 + 60 + 1700, t0)
      and adm8.probe_ready(t0 + 60 + 1800, t0))

# ═══════════════════════════════════════════════════════════════════════
# 4. _wait_job: eviction + running accounting
# ═══════════════════════════════════════════════════════════════════════
class FakeAPI:
    def __init__(self, seq):
        self.seq = seq
        self.i = 0
        self.cancelled = []

    def status(self, job_id):
        st = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return st

    def cancel(self, job_id):
        self.cancelled.append(job_id)

    def logs(self, job_id, tail=50):
        return ""


def bare_wait_job(api, adm, running_now=0):
    ex = object.__new__(RemoteExecutor)
    ex.remote = RemoteConfig(poll_interval_s=0.005,
                             status_print_interval_s=9999.0,
                             job_timeout_s=30.0, queue_timeout_s=30.0)
    ex.job_api = api
    ex._run_lock = threading.Lock()
    ex._running_now = running_now
    ex._run_samples = []
    ex._admission = adm
    return ex._wait_job("job-1", 7)


# eviction: PENDING past grace while a sibling runs
ev_adm = AdmissionController(wave_budget=1, expected_slots=2,
                             pending_grace_s=0.15)
api_ev = FakeAPI([JobStatus.PENDING])
evicted = False
try:
    bare_wait_job(api_ev, ev_adm, running_now=1)
except ConfigEvictedError:
    evicted = True
check("eviction: PENDING past grace + sibling RUNNING -> cancel + raise",
      evicted and api_ev.cancelled == ["job-1"])

# no eviction without a running sibling: queue patience applies instead
# (bounded by a tiny queue_timeout so the test finishes fast)
api_q = FakeAPI([JobStatus.PENDING])
ex_q = object.__new__(RemoteExecutor)
ex_q.remote = RemoteConfig(poll_interval_s=0.005, status_print_interval_s=9999.0,
                           job_timeout_s=30.0, queue_timeout_s=0.4)
ex_q.job_api = api_q
ex_q._run_lock = threading.Lock()
ex_q._running_now = 0
ex_q._run_samples = []
ex_q._admission = ev_adm
q_err = None
try:
    ex_q._wait_job("job-1", 7)
except Exception as e:
    q_err = type(e).__name__
check("eviction: no sibling RUNNING -> stays queued (QueueTimeout, not evicted)",
      q_err == "QueueTimeoutError" and api_q.cancelled == ["job-1"])

# running accounting: PENDING -> RUNNING -> SUCCEEDED counts +1 then -1
run_adm = AdmissionController(wave_budget=1, expected_slots=2)
api_run = FakeAPI([JobStatus.PENDING, JobStatus.RUNNING,
                   JobStatus.RUNNING, JobStatus.SUCCEEDED])
ex = object.__new__(RemoteExecutor)
ex.remote = RemoteConfig(poll_interval_s=0.005, status_print_interval_s=9999.0,
                         job_timeout_s=30.0, queue_timeout_s=30.0)
ex.job_api = api_run
ex._run_lock = threading.Lock()
ex._running_now = 0
ex._run_samples = []
ex._admission = run_adm
st = ex._wait_job("job-2", 8)
check("accounting: terminal status returned",
      st == JobStatus.SUCCEEDED and ex._running_now == 0)
check("accounting: RUNNING samples recorded + probe clock noted",
      len(ex._run_samples) >= 2 and run_adm.first_running_at is not None)

# ═══════════════════════════════════════════════════════════════════════
# 5. spec v2 + worker/embed version lockstep
# ═══════════════════════════════════════════════════════════════════════
spec = ExpSpec(model_tag="d28_random_x", ckpt_src="/nc/base_checkpoints/d28")
rt = ExpSpec.from_json(spec.to_json())
check("spec: v2 roundtrip preserves ckpt_src",
      rt.ckpt_src == "/nc/base_checkpoints/d28" and rt.spec_version == 2)
try:
    ExpSpec.from_dict({"spec_version": 1})
    check("spec: v1 rejected by v2 parser", False)
except ValueError:
    check("spec: v1 rejected by v2 parser", True)

# remote_worker imports nanochat_cmds from ITS OWN directory (the staged
# assets bundle layout) — mirror the staging dir exactly
import shutil  # noqa: E402
_stage = tempfile.mkdtemp(prefix="prod2worker_")
shutil.copy(os.path.join(REPO, "scripts", "remote_worker.py"), _stage)
shutil.copy(os.path.join(REPO, "src", "climbmix", "pipeline",
                         "nanochat_cmds.py"), _stage)
sys.path.insert(0, _stage)
import remote_worker  # noqa: E402
check("spec: remote_worker SPEC_VERSION in lockstep",
      remote_worker.SPEC_VERSION == SPEC_VERSION == 2)
embed_src = open(os.path.join(REPO, "scripts", "embed_dispatch.py")).read()
check("spec: embed_dispatch spec_version bumped to 2",
      '"spec_version": 2' in embed_src)

# RemoteConfig roundtrip with the new knob
rc = RemoteConfig.from_dict({"obs_prefix": "obs://b/p", "backend": "mock",
                             "storage_kind": "local", "storage_root": "/tmp/x",
                             "pending_grace_min": 45.0})
check("remote_config: pending_grace_min roundtrip", rc.pending_grace_min == 45.0)
try:
    RemoteConfig.from_dict({"obs_prefix": "obs://b/p", "backend": "mock",
                            "storage_kind": "local", "storage_root": "/tmp/x",
                            "pending_grace_min": 0}).validate()
    check("remote_config: pending_grace_min must be > 0", False)
except ValueError:
    check("remote_config: pending_grace_min must be > 0", True)

# ═══════════════════════════════════════════════════════════════════════
# 6. bootstrapper adaptive: presample / admission kwargs / dropped rewrite
# ═══════════════════════════════════════════════════════════════════════
class FakeAdaptiveRunner:
    """Emulates the RemoteExecutor's adaptive contract: records the
    admission kwargs, optionally drops the tail as None, exposes the
    post-batch accounting attrs."""

    def __init__(self, drop_tail=0, c_eff=11):
        self.remote = types.SimpleNamespace(max_concurrent_jobs=10,
                                            local_parallel=True)
        self.drop_tail = drop_tail
        self.last_effective_concurrency = c_eff
        self.last_dropped_configs = []
        self.last_admission_stats = {}
        self.calls = []

    def _local_slots(self):
        return 1

    def run_batch(self, configs, experiment_id_base=0, admission=None):
        self.calls.append({"n": len(configs), "admission": admission,
                           "base": experiment_id_base})
        results = []
        for i, c in enumerate(configs):
            acc = {"mmlu_stem": 0.40 + 0.02 * ((i * 7) % 5)}
            results.append(ProxyResult(
                mixture_config=c, validation_loss=0.0,
                validation_accuracy=0.5, validation_nll=1.0,
                per_task_accuracies=acc, per_task_nlls={},
                metadata={"experiment_id": experiment_id_base + i}))
        if admission and self.drop_tail:
            for i in range(self.drop_tail):
                results[len(results) - 1 - i] = None
            self.last_dropped_configs = [
                configs[len(configs) - 1 - i] for i in range(self.drop_tail)]
            self.last_admission_stats = {
                "c_eff": 11, "wave_budget": admission["wave_budget"],
                "truncated": list(range(len(configs) - self.drop_tail,
                                        len(configs))),
                "evicted": []}
        return results


def make_bs(state_path, adaptive=True, per_iter=(20, 10, 10)):
    cfg = CLIMBConfig(val_tasks=["mmlu_stem"])
    cfg.search.configs_per_iter = list(per_iter)
    cfg.search.adaptive_configs = adaptive
    cluster_tokens = np.array([600, 600], dtype=np.int64)
    labels = np.array([0, 0, 0, 1])
    return IterativeBootstrapper(cfg, cluster_tokens, labels,
                                 state_path=state_path)


with tempfile.TemporaryDirectory(prefix="prod2bs_") as td:
    # iter 1: e=20, S0=11 -> w=2 -> presample 2*11+4=26; executor drops 4
    fake = FakeAdaptiveRunner(drop_tail=4, c_eff=11)
    bs = make_bs(os.path.join(td, "search_state.json"))
    bs.run_iteration(1, 20, fake)
    c1 = fake.calls[0]
    check("adaptive iter1: presamples w*S0+4 = 26", c1["n"] == 26, f"n={c1['n']}")
    check("adaptive iter1: admission kwargs (w=2, S0=11, truncate on)",
          c1["admission"] == {"wave_budget": 2, "expected_slots": 11,
                              "min_configs": 4, "allow_truncate": True})
    check("adaptive iter1: dropped 4 -> 22 accumulated",
          len(bs._accumulated_configs) == 22)
    state = json.load(open(os.path.join(td, "search_state.json")))
    check("adaptive iter1: pending rewritten to admitted 22 (atomic, resume-safe)",
          len(state["pending"]["configs"]) == 22)
    check("adaptive iter1: realized_configs_per_iter persisted",
          state.get("realized_configs_per_iter") == [22])
    check("adaptive iter1: last_c_eff persisted",
          state.get("last_c_eff") == 11)
    check("adaptive iter1: predictor fitted (guided iter2 ahead)",
          bs._predictor is not None)

    # iter 2: e=10, S0=11 -> w=1; C_eff_prev=11 -> n_sample = 11 exactly
    fake2 = FakeAdaptiveRunner(drop_tail=0, c_eff=9)
    bs.run_iteration(2, 10, fake2)
    c2 = fake2.calls[0]
    check("adaptive iter2: sizes directly from C_eff_prev (1*11=11)",
          c2["n"] == 11, f"n={c2['n']}")
    check("adaptive iter2: admission still active (eviction guard)",
          c2["admission"] is not None and c2["admission"]["wave_budget"] == 1)
    check("adaptive iter2: allow_truncate follows C_eff_prev >= 2",
          c2["admission"]["allow_truncate"] is True)
    check("adaptive iter2: C_eff updates to the new measurement",
          bs._last_c_eff == 9)

    # guardrail path: C_eff_prev=1 disables truncation for the next round
    fake3 = FakeAdaptiveRunner(drop_tail=0, c_eff=1)
    bs2 = make_bs(os.path.join(td, "state2.json"))
    bs2.run_iteration(1, 20, FakeAdaptiveRunner(drop_tail=0, c_eff=1))
    fake3b = FakeAdaptiveRunner(drop_tail=0, c_eff=5)
    bs2.run_iteration(2, 10, fake3b)
    check("adaptive guardrail: C_eff_prev=1 -> allow_truncate=False",
          fake3b.calls[0]["admission"]["allow_truncate"] is False)

    # non-adaptive runner (no 'remote' attr): literal behavior, no admission
    class PlainRunner:
        def run_batch(self, configs, experiment_id_base=0):
            self.n = len(configs)
            return [ProxyResult(
                mixture_config=c, validation_loss=0.0, validation_accuracy=0.5,
                validation_nll=1.0, per_task_accuracies={"mmlu_stem": 0.4},
                per_task_nlls={}, metadata={}) for c in configs]
    plain = PlainRunner()
    bs3 = make_bs(os.path.join(td, "state3.json"))
    bs3.config.search.adaptive_configs = False
    bs3.run_iteration(1, 20, plain)
    check("adaptive off: literal count, no admission kwarg",
          plain.n == 20)

    # _expected_fleet_slots
    check("S0: remote max jobs + local slot",
          IterativeBootstrapper._expected_fleet_slots(fake) == 11)
    check("S0: plain runner -> None (adaptive no-op)",
          IterativeBootstrapper._expected_fleet_slots(plain) is None)

# ═══════════════════════════════════════════════════════════════════════
# 7. dispatch smoke + shell syntax
# ═══════════════════════════════════════════════════════════════════════
r = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts", "dispatch_target_arm.py"),
     "--help"], capture_output=True, text=True)
check("dispatch: --help exits 0", r.returncode == 0
      and "--arm" in r.stdout and "base_eval_check" in r.stdout)
r = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts", "dispatch_target_arm.py"),
     "--arm", "bogus"], capture_output=True, text=True)
check("dispatch: unknown arm rejected", r.returncode != 0)

for sh in ("runs/run_climbmix.sh", "runs/lib/target_arm.sh",
           "scripts/diagnostics/prod2_watch.sh"):
    r = subprocess.run(["bash", "-n", os.path.join(REPO, sh)],
                       capture_output=True, text=True)
    check(f"shell: bash -n {sh}", r.returncode == 0, r.stderr[:200])

# run_climbmix.sh wiring: fingerprint + defaults + guard
src = open(os.path.join(REPO, "runs", "run_climbmix.sh")).read()
check("shell: adaptive_configs in FP_SEARCH_PARAMS",
      'adaptive_configs=$ADAPTIVE_CONFIGS' in src)
check("shell: TARGET_ARM_MODE not fingerprinted (execution shape)",
      re.search(r'"target_arm_mode=', src) is None)
check("shell: MERGE_STRATEGY defaults to balanced",
      'MERGE_STRATEGY="${MERGE_STRATEGY:-balanced}"' in src)
check("shell: REMOTE_LOCAL_PARALLEL defaults to 0 (opt-in slot)",
      'REMOTE_LOCAL_PARALLEL="${REMOTE_LOCAL_PARALLEL:-0}"' in src)
check("shell: K_CLUSTER_MAX follows K_ENHANCED",
      'K_CLUSTER_MAX="${K_CLUSTER_MAX:-$K_ENHANCED}"' in src)
check("shell: hybrid guard allows == (whole-node slot)",
      '"$NPU_PER_EXP" -gt "$NUM_NPU"' in src
      and '"$NPU_PER_EXP" -ge "$NUM_NPU"' not in src)
check("shell: dispatch three-layer in run_arm",
      "dispatch_target_arm.py" in src and "target_arm_train" in src
      and "TARGET_ARM_MODE" in src)
check("shell: launch_env.json snapshot written",
      "launch_env.json" in src)

# ── summary ───────────────────────────────────────────────────────────────
print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
