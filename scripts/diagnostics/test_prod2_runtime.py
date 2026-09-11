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
    build_target_mid_train_cmd, build_target_eval_cmd,
    retarget_torchrun_multinode, resolve_multinode_rendezvous)
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

    # cold-optimizer variant (TARGET_ARM_NODES>1 runs derive
    # TARGET_LOAD_OPTIMIZER=0): the flag must reach shell AND python at
    # the same position — a remote-cold vs local-warm arm pair would be
    # an unfair verdict, so both paths share the semantics.
    env["TARGET_LOAD_OPTIMIZER"] = "0"
    r = run_lib(f'target_arm_train /data/mixed {TAG} {NAME}')
    check("parity: target_arm_train exit 0 (load-optimizer=0)",
          r.returncode == 0)
    shell_train_cold = ["torchrun"] + open(argv_file).read().splitlines()
    py_train_cold = build_target_mid_train_cmd(
        run_name=f"{NAME}_mid", model_tag=TAG, data_dir="/data/mixed",
        num_iterations=ENVV["TARGET_STEPS"], lr_scale=ENVV["TARGET_LR_SCALE"],
        warmup=ENVV["TARGET_WARMUP"], warmdown=ENVV["TARGET_WARMDOWN"],
        core_metric_every=ENVV["CORE_METRIC_EVERY"],
        device_batch_size=ENVV["MID_DEVICE_BATCH_SIZE"],
        loader=ENVV["MID_TRAIN_LOADER"], load_optimizer="0",
        nproc_per_node=8)
    check("parity: cold-optimizer argv token-identical (shell == python)",
          shell_train_cold == py_train_cold,
          f"shell={shell_train_cold}" if shell_train_cold != py_train_cold else "")
    check("parity: --load-optimizer=0 sits after --eval-every=-1, before --run",
          py_train_cold.index("--eval-every=-1")
          < py_train_cold.index("--load-optimizer=0")
          < py_train_cold.index(f"--run={NAME}_mid"))
    check("parity: cold argv differs from warm only by the one flag",
          py_train_cold == py_train[:py_train.index(f"--run={NAME}_mid")]
          + ["--load-optimizer=0"]
          + py_train[py_train.index(f"--run={NAME}_mid"):])
    del env["TARGET_LOAD_OPTIMIZER"]

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
#    2026-09-09 floor semantics: configs_per_iter = per-iteration MINIMUM;
#    two tiers — floor (never truncated/evicted) + overshoot (dynamic).
# ═══════════════════════════════════════════════════════════════════════
adm = AdmissionController(wave_budget=2, expected_slots=11)
drops = adm.decide(total_configs=26, n_local=1, running_now=10, local_slots=1)
# C_eff=11 -> target = max(4, min(2*11+5, 26)) = 26 -> admit all
check("admission: iter1 pool 26, C_eff=11, buffer 5 -> admit all (drain fill)",
      drops == [] and not adm.drop_set and adm.c_eff == 11,
      f"drops={drops}")

# 2026-09-09 regression: prod2 iter1 truncated 26->22 (exact w*C_eff) and
# idled 9 slots for 2.75h through the batch drain; floor 20 + buffer now
# admits the whole pool — the 4 former "reserve" configs fill the drain.
admF = AdmissionController(wave_budget=2, expected_slots=11, floor_configs=20)
dropsF = admF.decide(total_configs=26, n_local=1, running_now=10, local_slots=1)
check("admission: floor 20 + buffer rescues the 26->22 drain incident",
      dropsF == [] and admF.floor_configs == 20 and admF.admit_buffer == 5)

admT = AdmissionController(wave_budget=2, expected_slots=11, floor_configs=20)
dropsT = admT.decide(total_configs=26, n_local=1, running_now=1, local_slots=1)
# tight pool: C_eff=2 -> target = max(20, min(2*2+5, 26)) = 20 -> drop 6 tail
check("admission: tight pool C_eff=2 truncates to exactly the floor",
      dropsT == [20, 21, 22, 23, 24, 25] and len(admT.drop_set) == 6)

admB = AdmissionController(wave_budget=2, expected_slots=11)
dropsB = admB.decide(total_configs=60, n_local=1, running_now=10, local_slots=1)
# C_eff=11 -> target = max(4, min(27, 60)) = 27 -> drop 33 tail
check("admission: oversized pool still truncates to w*C_eff + buffer",
      len(dropsB) == 33 and dropsB[0] == 27 and dropsB[-1] == 59)

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

# two-tier pure decisions: is_floor / evict_pending / evict_tail
adm2t = AdmissionController(wave_budget=1, expected_slots=11, floor_configs=10)
check("floor: tier membership by global index",
      adm2t.is_floor(0) and adm2t.is_floor(9) and not adm2t.is_floor(10))
check("floor: defaults to min_configs when not passed",
      AdmissionController(wave_budget=1, expected_slots=11).floor_configs == 4)
check("floor: clamps up below min_configs",
      AdmissionController(wave_budget=1, expected_slots=2, min_configs=4,
                          floor_configs=2).floor_configs == 4)
check("evict_pending: overshoot past grace + sibling RUNNING",
      adm2t.evict_pending(False, 2000.0, 1) is True)
check("evict_pending: floor exempt even far past grace",
      adm2t.evict_pending(True, 99999.0, 5) is False)
check("evict_pending: no sibling running -> patient (queue timeout bounds)",
      adm2t.evict_pending(False, 99999.0, 0) is False)
check("evict_tail: overshoot sole-remaining past tail grace",
      adm2t.evict_tail(False, 1000.0) is True)
check("evict_tail: floor exempt",
      adm2t.evict_tail(True, 99999.0) is False)
check("evict_tail: within grace",
      adm2t.evict_tail(False, 899.0) is False)

# floor-guaranteed pool sizing (2026-09-09 regression: a drain-poisoned
# C_eff_prev=2 shrank prod2 iter2's pool to 4; the floor keeps >= e_i)
check("pool_size: iter1 e=20, w*S0=22 -> 26",
      IterativeBootstrapper._adaptive_pool_size(20, 22) == 26)
check("pool_size: iter2 e=10, poisoned C_eff_prev=2 -> floor 10 + 4",
      IterativeBootstrapper._adaptive_pool_size(10, 2) == 14)
check("pool_size: floor dominates a small capacity term",
      IterativeBootstrapper._adaptive_pool_size(20, 12) == 24)

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


def bare_wait_job(api, adm, running_now=0, active=1, is_floor=False,
                  queue_timeout=30.0):
    ex = object.__new__(RemoteExecutor)
    ex.remote = RemoteConfig(poll_interval_s=0.005,
                             status_print_interval_s=9999.0,
                             job_timeout_s=30.0,
                             queue_timeout_s=queue_timeout)
    ex.job_api = api
    ex._run_lock = threading.Lock()
    ex._running_now = running_now
    ex._run_samples = []
    ex._active_lifecycles = active
    ex._admission = adm
    return ex._wait_job("job-1", 7, is_floor=is_floor)


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
ex_q._active_lifecycles = 1
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
ex._active_lifecycles = 1
ex._admission = run_adm
st = ex._wait_job("job-2", 8)
check("accounting: terminal status returned",
      st == JobStatus.SUCCEEDED and ex._running_now == 0)
check("accounting: RUNNING samples recorded + probe clock noted",
      len(ex._run_samples) >= 2 and run_adm.first_running_at is not None)

# tail-kill: sole remaining lifecycle, PENDING past tail grace -> evicted
# (over-admission grace disabled so only the tail path can fire)
tk_adm = AdmissionController(wave_budget=1, expected_slots=2,
                             pending_grace_s=999.0, tail_grace_s=0.1)
api_tk = FakeAPI([JobStatus.PENDING])
tk_evicted = False
try:
    bare_wait_job(api_tk, tk_adm, running_now=0, active=1)
except ConfigEvictedError:
    tk_evicted = True
check("tail-kill: sole remaining PENDING past grace -> cancel + raise",
      tk_evicted and api_tk.cancelled == ["job-1"])

# tail-kill: floor tier exempt -> queue patience bounds the wait instead
api_tkf = FakeAPI([JobStatus.PENDING])
tkf_err = None
try:
    bare_wait_job(api_tkf, tk_adm, running_now=0, active=1, is_floor=True,
                  queue_timeout=0.4)
except Exception as e:
    tkf_err = type(e).__name__
check("tail-kill: floor tier stays queued (QueueTimeout, not evicted)",
      tkf_err == "QueueTimeoutError" and api_tkf.cancelled == ["job-1"])

# tail-kill: siblings still active -> never fires (clock resets on >1)
api_ns = FakeAPI([JobStatus.PENDING])
ns_err = None
try:
    bare_wait_job(api_ns, tk_adm, running_now=0, active=3, queue_timeout=0.4)
except Exception as e:
    ns_err = type(e).__name__
check("tail-kill: siblings still active -> QueueTimeout, not evicted",
      ns_err == "QueueTimeoutError" and api_ns.cancelled == ["job-1"])

# honest C_eff: whole-batch median, drain-phase immune (2026-09-09: the
# last-120 window measured the drain (median 1) and shrank prod2 iter2
# to a pool of 4)
def bare_ceff(samples, has_local=False):
    ex = object.__new__(RemoteExecutor)
    ex._run_lock = threading.Lock()
    ex._run_samples = list(samples)
    return ex._compute_effective_concurrency(has_local)

check("c_eff: steady 3000x10 + drain 330x1 -> 10 (drain immune)",
      bare_ceff([10] * 3000 + [1] * 330) == 10)
check("c_eff: short homogeneous batch -> its own concurrency",
      bare_ceff([3] * 990) == 3)
check("c_eff: no samples -> None", bare_ceff([]) is None)
check("c_eff: startup ramp dropped",
      bare_ceff([1, 1, 1, 1, 8, 8, 8, 8]) == 8)

# ═══════════════════════════════════════════════════════════════════════
# 5. spec v3 + worker/embed version lockstep + multi-node helpers
# ═══════════════════════════════════════════════════════════════════════
spec = ExpSpec(model_tag="d28_random_x", ckpt_src="/nc/base_checkpoints/d28")
rt = ExpSpec.from_json(spec.to_json())
check("spec: v3 roundtrip preserves ckpt_src",
      rt.ckpt_src == "/nc/base_checkpoints/d28" and rt.spec_version == 3)
spec_mn = ExpSpec(model_tag="d28_climb_x", node_count=4, master_port=29600)
rt_mn = ExpSpec.from_json(spec_mn.to_json())
check("spec: v3 roundtrip preserves node_count/master_port",
      rt_mn.node_count == 4 and rt_mn.master_port == 29600)
_d3 = spec_mn.to_dict()
del _d3["node_count"], _d3["master_port"]
check("spec: node_count/master_port default when absent",
      ExpSpec.from_dict(_d3).node_count == 1
      and ExpSpec.from_dict(_d3).master_port == 29500)
try:
    ExpSpec.from_dict({"spec_version": 2})
    check("spec: v2 rejected by v3 parser", False)
except ValueError:
    check("spec: v2 rejected by v3 parser", True)
try:
    ExpSpec.from_dict({"spec_version": 1})
    check("spec: v1 rejected by v3 parser", False)
except ValueError:
    check("spec: v1 rejected by v3 parser", True)

# retarget_torchrun_multinode: prefix swap only, module tail pinned
_mn_base = build_target_mid_train_cmd(
    run_name="climb_mid", model_tag="d28_x", data_dir="/d",
    num_iterations="2000", lr_scale="1.0", warmup="0.0", warmdown="0.9",
    core_metric_every="-1", device_batch_size="1", loader="flat",
    load_optimizer="0", nproc_per_node=8)
_mn = retarget_torchrun_multinode(_mn_base, 4, 2, "10.3.0.1", 29500)
check("retarget: prefix shape (probe-proven flag order)",
      _mn[:6] == ["torchrun", "--nnodes=4", "--node_rank=2",
                  "--master_addr=10.3.0.1", "--master_port=29500",
                  "--nproc_per_node=8"])
check("retarget: module tail byte-identical to the spec's argv",
      _mn[6:] == _mn_base[_mn_base.index("-m"):])
check("retarget: single-node argv keeps --standalone until retargeted",
      "--standalone" in _mn_base and "--standalone" not in _mn)
# The retarget FUNCTION still rewrites any torchrun argv; train is its
# only live caller now — the worker stopped retargeting eval on
# 2026-09-10 (nodes r>0 cannot see node 0's model file: per-node output
# mounts, probe A 20260908 NOT-visible).
_ev_base = build_target_eval_cmd(
    model_tag="d28_x", eval_benchmarks="stem", eval_max_per_task="-1",
    device_batch_size="16", core_batch_size="8", nproc_per_node=8)
_ev = retarget_torchrun_multinode(_ev_base, 4, 3, "10.3.0.1", 29501)
check("retarget: eval argv would retarget if asked (standalone dropped)",
      _ev[:6] == ["torchrun", "--nnodes=4", "--node_rank=3",
                  "--master_addr=10.3.0.1", "--master_port=29501",
                  "--nproc_per_node=8"]
      and "--standalone" not in _ev
      and _ev[6:] == _ev_base[_ev_base.index("-m"):])
for _bad in (["python3", "-m", "x"],
             ["torchrun", "-m", "x"],
             ["torchrun", "--standalone", "-m", "x"]):
    try:
        retarget_torchrun_multinode(_bad, 2, 0, "a", 1)
        check(f"retarget: rejects malformed prefix {_bad[:2]!r}", False)
    except ValueError:
        check(f"retarget: rejects malformed prefix {_bad[:2]!r}", True)
try:
    retarget_torchrun_multinode(_mn_base, 4, 4, "a", 1)
    check("retarget: rank outside [0, node_count) rejected", False)
except ValueError:
    check("retarget: rank outside [0, node_count) rejected", True)

# resolve_multinode_rendezvous: platform env first, then sts-dns
check("rdzv: platform env triplet wins (addr/port/rank)",
      resolve_multinode_rendezvous(
          4, 29500, env={"MASTER_ADDR": "10.1.1.1", "MASTER_PORT": "1234",
                         "NODE_RANK": "3"}) == ("10.1.1.1", 1234, 3))
check("rdzv: MA_ prefixed platform triplet",
      resolve_multinode_rendezvous(
          4, 29500, env={"MA_MASTER_ADDR": "10.2.2.2", "MA_MASTER_PORT": "77",
                         "MA_NODE_RANK": "1"}) == ("10.2.2.2", 77, 1))
check("rdzv: VC_ prefixed platform triplet",
      resolve_multinode_rendezvous(
          4, 29500, env={"VC_MASTER_ADDR": "10.2.2.3", "VC_MASTER_PORT": "78",
                         "VC_NODE_RANK": "0"}) == ("10.2.2.3", 78, 0))
with tempfile.NamedTemporaryFile("w", suffix="hosts", delete=False) as _hf:
    _hf.write("# fake /etc/hosts (sts pods with hostname+subdomain)\n")
    _hf.write("10.5.0.1 jobA-worker-0.jobA.ns.svc.cluster.local jobA-worker-0\n")
    _hf.write("10.5.0.2 jobA-worker-2.jobA.ns.svc.cluster.local jobA-worker-2\n")
    _hosts_path = _hf.name
check("rdzv: sts-dns derives master ip + own rank",
      resolve_multinode_rendezvous(
          4, 29500, env={}, hostname="jobA-worker-2",
          hosts_file=_hosts_path) == ("10.5.0.1", 29500, 2))
for _bad_host, _why in (("plainhost", "non-sts hostname"),
                        ("jobB-worker-2", "fqdn for another svc")):
    try:
        resolve_multinode_rendezvous(
            4, 29500, env={}, hostname=_bad_host, hosts_file=_hosts_path)
        check(f"rdzv: {_why} raises", False)
    except RuntimeError:
        check(f"rdzv: {_why} raises", True)
try:
    resolve_multinode_rendezvous(
        2, 29500, env={}, hostname="jobA-worker-2", hosts_file=_hosts_path)
    check("rdzv: rank >= node_count raises", False)
except RuntimeError:
    check("rdzv: rank >= node_count raises", True)
try:
    resolve_multinode_rendezvous(
        4, 29500, env={"MASTER_ADDR": "a", "MASTER_PORT": "1",
                       "NODE_RANK": "9"})
    check("rdzv: platform rank >= node_count raises", False)
except RuntimeError:
    check("rdzv: platform rank >= node_count raises", True)
os.unlink(_hosts_path)

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
      remote_worker.SPEC_VERSION == SPEC_VERSION == 3)
embed_src = open(os.path.join(REPO, "scripts", "embed_dispatch.py")).read()
check("spec: embed_dispatch spec_version bumped to 3",
      '"spec_version": 3' in embed_src)
# 2026-09-09 latent bug: the v2 bump missed embed_worker, so every embed
# unit since 9d6c810 would die at ITS version gate ("2 != 1") — the whole
# assets bundle must move on the shared counter together.
embed_wsrc = open(os.path.join(REPO, "scripts", "embed_worker.py")).read()
check("spec: embed_worker SPEC_VERSION in lockstep (v2-bump miss fixed)",
      re.search(r"^SPEC_VERSION = 3", embed_wsrc, re.M) is not None)

# ── multi-node worker simulation (spec v3, node_count=2) ──────────────────
# Runs the REAL remote_worker.py as subprocesses (node 0 + node 1)
# against a LocalStorage fake OBS, with a stub torchrun recording argv.
# Rendezvous goes through the platform-env path (MASTER_ADDR/MASTER_PORT/
# NODE_RANK in the worker's inherited env) — no hostname tricks needed.
# The stub is rank-faithful: only a rank-0 (or unretargeted) train call
# creates model+meta (the rank-0 consolidated save); non-zero ranks
# create just their optim shard — so the model file reaching node 1's
# base dir is PROOF the TCP relay delivered it.
with tempfile.TemporaryDirectory(prefix="prod2mn_") as mn_td:
    mn_bin = os.path.join(mn_td, "bin")
    os.makedirs(mn_bin)
    stub_tr = os.path.join(mn_bin, "torchrun")
    with open(stub_tr, "w") as f:
        f.write(
            "#!/usr/bin/env bash\n"
            'printf \'%s\\n\' "$@" >> "$ARGV_REC"\n'
            'echo "=== call ===" >> "$ARGV_REC"\n'
            "# rank-faithful mid_train emulation (train calls carry\n"
            "# --num-iterations; eval calls do not)\n"
            'tag=""; train=0; rank=""\n'
            'for a in "$@"; do case "$a" in '
            '--model-tag=*) tag="${a#--model-tag=}";; '
            '--num-iterations=*) train=1;; '
            '--node_rank=*) rank="${a#--node_rank=}";; esac; done\n'
            'if [ "$train" = 1 ] && [ -n "$tag" ]; then\n'
            '  mkdir -p "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag"\n'
            '  if [ -z "$rank" ] || [ "$rank" = 0 ]; then\n'
            '    echo ckpt > "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag/'
            'model_000100.pt"\n'
            '    echo meta > "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag/'
            'meta_000100.json"\n'
            '  else\n'
            '    echo optim > "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag/'
            'optim_000100_rank$rank.pt"\n'
            '  fi\n'
            "fi\n"
            "exit 0\n")
    os.chmod(stub_tr, 0o755)

    mn_root = os.path.join(mn_td, "fakeobs")       # LocalStorage root
    mn_base0 = os.path.join(mn_td, "nc_base0")     # node 0 spec base_dir
    mn_base1 = os.path.join(mn_td, "nc_base1")     # node 1 spec base_dir
    mn_work = os.path.join(mn_td, "nc_work")       # spec work_dir
    mn_ncdir = os.path.join(mn_td, "nanochat")     # spec nanochat_dir (cwd)
    for d in (mn_root, mn_work, mn_ncdir):
        os.makedirs(d, exist_ok=True)
    for b in (mn_base0, mn_base1):
        os.makedirs(os.path.join(b, "tokenizer"), exist_ok=True)
    mn_ckpt_src = os.path.join(mn_td, "d28_asset")
    os.makedirs(mn_ckpt_src)
    with open(os.path.join(mn_ckpt_src, "model_000000.pt"), "w") as f:
        f.write("base")
    result_uri = "obs://b/p/arms/x/result"
    mixture_uri = "obs://b/p/arms/x/mixture_data"
    mix_local = os.path.join(mn_root, "b", "p", "arms", "x", "mixture_data")
    os.makedirs(mix_local)
    with open(os.path.join(mix_local, "shard_00000.parquet"), "w") as f:
        f.write("data")

    def mn_spec(node_count, node_rank=0, base=None):
        spec = ExpSpec(
            model_tag="d28_mn_x", nanochat_dir=mn_ncdir,
            base_dir=base or mn_base0,
            work_dir=mn_work, base_ckpt_src=mn_ckpt_src,
            mixture_data_uri=mixture_uri, result_uri=result_uri,
            mid_train_cmd=build_target_mid_train_cmd(
                run_name="climb_mid", model_tag="d28_mn_x",
                data_dir=os.path.join(mn_work, "mixture_data"),
                num_iterations="50", lr_scale="1.0", warmup="0.0",
                warmdown="0.9", core_metric_every="-1",
                device_batch_size="1", loader="flat",
                load_optimizer="0", nproc_per_node=8),
            eval_cmd=build_target_eval_cmd(
                model_tag="d28_mn_x", eval_benchmarks="stem",
                eval_max_per_task="-1", device_batch_size="16",
                core_batch_size="8", nproc_per_node=8),
            node_count=node_count, log_stream_s=0,
        )
        path = os.path.join(mn_td,
                            f"spec_{node_count}_{node_rank}_{id(base)}.json")
        with open(path, "w") as f:
            f.write(spec.to_json())
        return path

    def _worker_env(node_rank, extra_env=None):
        env = dict(os.environ)
        env.update({"PATH": mn_bin + os.pathsep + env["PATH"],
                    "ARGV_REC": os.path.join(mn_td,
                                             f"argv_rec_n{node_rank}.txt"),
                    "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29511",
                    "NODE_RANK": str(node_rank)})
        if extra_env:
            env.update(extra_env)
        return env

    def run_worker(spec_path, node_rank, extra_env=None):
        return subprocess.run(
            [sys.executable, os.path.join(_stage, "remote_worker.py"),
             "--spec-uri", "obs://unused/spec.json",
             "--spec-local", spec_path, "--storage", "local",
             "--storage-root", mn_root],
            env=_worker_env(node_rank, extra_env),
            capture_output=True, text=True, timeout=180)

    def spawn_worker(spec_path, node_rank, extra_env=None):
        return subprocess.Popen(
            [sys.executable, os.path.join(_stage, "remote_worker.py"),
             "--spec-uri", "obs://unused/spec.json",
             "--spec-local", spec_path, "--storage", "local",
             "--storage-root", mn_root],
            env=_worker_env(node_rank, extra_env),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def rec_of(node_rank):
        p = os.path.join(mn_td, f"argv_rec_n{node_rank}.txt")
        return open(p).read() if os.path.isfile(p) else ""

    res_local = os.path.join(mn_root, "b", "p", "arms", "x", "result",
                             "result.json")

    def fresh_scenario():
        """The stub APPENDS to shared argv recs and LocalStorage keeps
        every prior upload — isolate the argv checks and the
        log-presence checks per scenario."""
        import shutil as _sh
        for r in (0, 1):
            try:
                os.remove(os.path.join(mn_td, f"argv_rec_n{r}.txt"))
            except FileNotFoundError:
                pass
        _sh.rmtree(os.path.join(mn_root, "b", "p", "arms", "x", "result"),
                   ignore_errors=True)

    # single-node sanity first: node_count=1 runs the argv VERBATIM
    r0 = run_worker(mn_spec(1), 0)
    check("worker sim: single-node exit 0", r0.returncode == 0,
          r0.stdout[-400:] + r0.stderr[-400:])
    rec1 = rec_of(0)
    check("worker sim: single-node argv untouched (--standalone kept)",
          "--standalone" in rec1 and "--nnodes" not in rec1)
    check("worker sim: single-node lands result.json",
          json.load(open(res_local))["mid_train_rc"] == 0)

    # ── multi-node WITH relay: nodes run CONCURRENTLY (node 1's pull
    # must overlap node 0's serve window — the real job shape). Real
    # localhost sockets: MASTER_PORT 29511 -> eval rdzv 29512, relay
    # 29513.
    fresh_scenario()
    relay_env = {"CLIMBMIX_RELAY_TIMEOUT_S": "45",
                 "CLIMBMIX_RELAY_CONNECT_S": "30"}
    p_n0 = spawn_worker(mn_spec(2, 0, base=mn_base0), 0, relay_env)
    p_n1 = spawn_worker(mn_spec(2, 1, base=mn_base1), 1, relay_env)
    out_n0 = p_n0.communicate(timeout=180)
    out_n1 = p_n1.communicate(timeout=180)
    check("worker sim(relay): node0 exit 0", p_n0.returncode == 0,
          (out_n0[0] or "")[-400:] + (out_n0[1] or "")[-400:])
    check("worker sim(relay): node1 exit 0", p_n1.returncode == 0,
          (out_n1[0] or "")[-400:] + (out_n1[1] or "")[-400:])
    rec_n0, rec_n1 = rec_of(0), rec_of(1)
    check("worker sim(relay): node0 train retargeted (--nnodes=2 rank 0)",
          "--nnodes=2" in rec_n0 and "--node_rank=0" in rec_n0
          and "--master_port=29511" in rec_n0
          and "--standalone" not in rec_n0.split("=== call ===")[0])
    check("worker sim(relay): node0 eval retargeted (port 29512, rank 0)",
          rec_n0.count("=== call ===") == 2
          and "--nnodes=2" in rec_n0.split("=== call ===")[1]
          and "--node_rank=0" in rec_n0.split("=== call ===")[1]
          and "--master_port=29512" in rec_n0.split("=== call ===")[1]
          and "--standalone" not in rec_n0.split("=== call ===")[1])
    check("worker sim(relay): node1 joins eval (2 calls, rank 1, 29512)",
          rec_n1.count("=== call ===") == 2
          and "--node_rank=1" in rec_n1
          and "--master_port=29511" in rec_n1.split("=== call ===")[0]
          and "--master_port=29512" in rec_n1.split("=== call ===")[1]
          and "--standalone" not in rec_n1.split("=== call ===")[1])
    # RELAY PROOF: the stub wrote only an optim shard on node 1 — these
    # files can only exist if the TCP relay delivered them.
    n1_tag = os.path.join(mn_base1, "mid_checkpoints", "d28_mn_x")
    n0_tag = os.path.join(mn_base0, "mid_checkpoints", "d28_mn_x")
    check("worker sim(relay): model+meta RELAYED to node1 (bytes match)",
          os.path.isfile(os.path.join(n1_tag, "model_000100.pt"))
          and os.path.isfile(os.path.join(n1_tag, "meta_000100.json"))
          and open(os.path.join(n1_tag, "model_000100.pt"), "rb").read()
          == open(os.path.join(n0_tag, "model_000100.pt"), "rb").read()
          and os.path.isfile(os.path.join(n1_tag,
                                          "optim_000100_rank1.pt")))
    res_json = json.load(open(res_local))
    check("worker sim(relay): master owns result.json (train+eval rc 0)",
          res_json["mid_train_rc"] == 0 and res_json["eval_rc"] == 0
          and res_json["csv"] is None)  # stub eval writes no CSV
    n0_ckpt = os.path.join(mn_root, "b", "p", "arms", "x", "result",
                           "mid_checkpoint", "model_000100.pt")
    check("worker sim(relay): master ckpt landed after eval",
          os.path.isfile(n0_ckpt))

    # ── multi-node relay FALLBACK (CLIMBMIX_EVAL_RELAY=0): node-0-only
    # 8-rank eval — the pre-relay behavior, which any relay miss must
    # reproduce exactly.
    fresh_scenario()
    fb_env = {"CLIMBMIX_EVAL_RELAY": "0"}
    fb_base0 = os.path.join(mn_td, "fb_base0")
    fb_base1 = os.path.join(mn_td, "fb_base1")
    for b in (fb_base0, fb_base1):
        os.makedirs(os.path.join(b, "tokenizer"), exist_ok=True)
    r_n0f = run_worker(mn_spec(2, 0, base=fb_base0), 0, fb_env)
    check("worker sim(fallback): node0 exit 0", r_n0f.returncode == 0,
          r_n0f.stdout[-400:] + r_n0f.stderr[-400:])
    with open(res_local) as f:
        res_before_fb = f.read()   # AFTER node0's own rewrite
    r_n1f = run_worker(mn_spec(2, 1, base=fb_base1), 1, fb_env)
    check("worker sim(fallback): node1 exit 0", r_n1f.returncode == 0,
          r_n1f.stdout[-400:] + r_n1f.stderr[-400:])
    rec_n0f, rec_n1f = rec_of(0), rec_of(1)
    check("worker sim(fallback): node0 eval UNRETARGETED (standalone)",
          rec_n0f.count("=== call ===") == 2
          and "--standalone" in rec_n0f.split("=== call ===")[1]
          and "--nnodes" not in rec_n0f.split("=== call ===")[1])
    check("worker sim(fallback): node1 trains ONLY (no eval call)",
          rec_n1f.count("=== call ===") == 1
          and "--node_rank=1" in rec_n1f
          and "base_eval" not in rec_n1f)
    fb_tag1 = os.path.join(fb_base1, "mid_checkpoints", "d28_mn_x")
    check("worker sim(fallback): no model file reached node1",
          not os.path.exists(os.path.join(fb_tag1, "model_000100.pt"))
          and os.path.isfile(os.path.join(fb_tag1,
                                          "optim_000100_rank1.pt")))
    node1_train_log = os.path.join(mn_root, "b", "p", "arms", "x", "result",
                                   "mid_train_node1.log")
    node1_eval_log = os.path.join(mn_root, "b", "p", "arms", "x", "result",
                                  "eval_node1.log")
    check("worker sim(fallback): node1 train log landed, no eval log",
          os.path.isfile(node1_train_log)
          and not os.path.exists(node1_eval_log))
    # node 1 must not touch master-owned artifacts: result.json stays
    # byte-identical to node 0's copy.
    with open(res_local) as f:
        check("worker sim(fallback): node1 does NOT touch result.json",
              f.read() == res_before_fb)

    # eval_only + node_count>1 must refuse (N evals would race one CSV)
    spec_eo = ExpSpec.from_json(open(mn_spec(2)).read())
    spec_eo.eval_only = True
    eo_path = os.path.join(mn_td, "spec_eo.json")
    with open(eo_path, "w") as f:
        f.write(spec_eo.to_json())
    r_eo = run_worker(eo_path, 0)
    check("worker sim: eval_only + node_count=2 refuses",
          r_eo.returncode != 0 and "single-node" in r_eo.stdout + r_eo.stderr)

# private eval dir must carry base_checkpoints/{tag} when the base dir
# has it: --model-type=base (the remote d28 anchor) loads through
# nanochat load_model("base") -> {base_dir}/base_checkpoints/{tag}.
# 2026-09-09: the remote anchor exited 1 — make_eval_base_dir linked
# only mid_checkpoints, so base_eval saw an empty private dir.
import nanochat_cmds as staged_cmds  # noqa: E402  (the staged copy jobs run)
_bd = tempfile.mkdtemp(prefix="prod2evalbase_")
for _sub in ("mid_checkpoints/d28", "base_checkpoints/d28", "tokenizer",
             "eval_bundle", "eval_stem"):
    os.makedirs(os.path.join(_bd, _sub), exist_ok=True)
_eb = staged_cmds.make_eval_base_dir(_bd, tempfile.mkdtemp(prefix="prod2exp_"),
                                     "d28")
check("eval_base: base_checkpoints/{tag} linked (anchor model-type=base)",
      os.path.isdir(os.path.join(_eb, "base_checkpoints", "d28"))
      and os.path.realpath(os.path.join(_eb, "base_checkpoints", "d28"))
      == os.path.join(_bd, "base_checkpoints", "d28"))
check("eval_base: mid_checkpoints/{tag} still linked",
      os.path.isdir(os.path.join(_eb, "mid_checkpoints", "d28")))
_bd2 = tempfile.mkdtemp(prefix="prod2evalbase2_")
os.makedirs(os.path.join(_bd2, "mid_checkpoints", "d20_x7"))
os.makedirs(os.path.join(_bd2, "tokenizer"))
_eb2 = staged_cmds.make_eval_base_dir(_bd2, tempfile.mkdtemp(), "d20_x7")
check("eval_base: mid-only exp unaffected (no base_checkpoints link)",
      not os.path.exists(os.path.join(_eb2, "base_checkpoints")))
# make_eval_base_dir's subdir param (capability; the multi-node caller
# is gone since the 2026-09-10 revert — eval is node-0 only): an
# isolated farm at a custom path.
_eb3 = staged_cmds.make_eval_base_dir(
    _bd, tempfile.mkdtemp(prefix="prod2exp_"), "d28",
    subdir="_eval_base_node1")
check("eval_base: custom subdir honored (same farm, own path)",
      os.path.basename(_eb3) == "_eval_base_node1"
      and os.path.realpath(os.path.join(_eb3, "mid_checkpoints", "d28"))
      == os.path.join(_bd, "mid_checkpoints", "d28"))

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
    check("adaptive iter1: admission kwargs (w=2, S0=11, floor 20, truncate on)",
          c1["admission"] == {"wave_budget": 2, "expected_slots": 11,
                              "min_configs": 4, "floor_configs": 20,
                              "allow_truncate": True})
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

    # iter 2: e=10, S0=11 -> w=1; C_eff_prev=11 -> n_sample = max(10,11)+4 = 15
    # (floor guarantee + overshoot headroom, never the old bare w*C_eff_prev)
    fake2 = FakeAdaptiveRunner(drop_tail=0, c_eff=9)
    bs.run_iteration(2, 10, fake2)
    c2 = fake2.calls[0]
    check("adaptive iter2: floor 10 + C_eff_prev 11 + reserve 4 = 15",
          c2["n"] == 15, f"n={c2['n']}")
    check("adaptive iter2: floor_configs passed through",
          c2["admission"]["floor_configs"] == 10)
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
    # 2026-09-09 regression: the poisoned C_eff_prev must not shrink the
    # pool below the floor (prod2 iter2 sampled 4 with C_eff_prev=2)
    check("adaptive guardrail: poisoned C_eff_prev=1 -> floor still 10+4=14",
          fake3b.calls[0]["n"] == 14, f"n={fake3b.calls[0]['n']}")

    # compact profile (ADAPTIVE_COMPACT=1): time-boxed — pool = e_i+4 (no
    # capacity fill), admit buffer pinned 0. Greedy's S0 fill forced every
    # iteration to >= 2 waves; compact lands exactly w_i.
    bs_c = make_bs(os.path.join(td, "state_c.json"))
    bs_c.config.search.adaptive_compact = True
    fake_c = FakeAdaptiveRunner(drop_tail=2, c_eff=11)
    bs_c.run_iteration(1, 20, fake_c)
    cc1 = fake_c.calls[0]
    check("compact iter1: pool = e_i+4 = 24 (no S0 fill)",
          cc1["n"] == 24, f"n={cc1['n']}")
    check("compact iter1: buffer pinned 0, floor still passed",
          cc1["admission"] == {"wave_budget": 2, "expected_slots": 11,
                               "min_configs": 4, "floor_configs": 20,
                               "allow_truncate": True,
                               "admit_buffer_configs": 0})
    check("compact iter1: probe target w*C_eff+0 = 22 -> 22 accumulated",
          len(bs_c._accumulated_configs) == 22)
    fake_c2 = FakeAdaptiveRunner(drop_tail=3, c_eff=11)
    bs_c.run_iteration(2, 10, fake_c2)
    cc2 = fake_c2.calls[0]
    check("compact iter2: pool = 14 regardless of C_eff_prev=11",
          cc2["n"] == 14, f"n={cc2['n']}")
    check("compact iter2: probe target = C_eff+0 = 11 -> 11 accumulated",
          len(bs_c._accumulated_configs) == 22 + 11)

    # controller: explicit buffer override vs the slots//2 default
    adm_buf = AdmissionController(wave_budget=1, expected_slots=11,
                                  floor_configs=10, admit_buffer_configs=0)
    check("controller: explicit buffer 0 wins over slots//2",
          adm_buf.admit_buffer == 0)
    adm_def = AdmissionController(wave_budget=1, expected_slots=11,
                                  floor_configs=10)
    check("controller: default buffer stays slots//2",
          adm_def.admit_buffer == 5)
    drops = adm_buf.decide(total_configs=14, n_local=1, running_now=10,
                           local_slots=1)
    check("controller: buffer 0 -> admit exactly w*C_eff (11 of 14)",
          sorted(drops) == [11, 12, 13], f"drops={sorted(drops)}")

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
      and "--arm" in r.stdout and "base_eval_check" in r.stdout
      and "--node-count" in r.stdout)
r = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts", "dispatch_target_arm.py"),
     "--arm", "bogus"], capture_output=True, text=True)
check("dispatch: unknown arm rejected", r.returncode != 0)
# node_count power-of-2 guard (ws=8*n must divide d28's power-of-2 dims;
# ws=24 failed live, run 601cdb67) — validated before any config is needed
for bad in ("3", "0", "-2", "5"):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "dispatch_target_arm.py"),
         "--arm", "random", "--node-count", bad],
        capture_output=True, text=True)
    check(f"dispatch: node_count={bad} rejected (power of 2 only)",
          r.returncode != 0 and "power of 2" in r.stdout + r.stderr)
r = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts", "dispatch_target_arm.py"),
     "--arm", "base_eval_check", "--node-count", "4"],
    capture_output=True, text=True)
check("dispatch: base_eval_check refuses multi-node",
      r.returncode != 0 and "single-node" in r.stdout + r.stderr)

# Full main() walk with a mock backend + a fabricated launch_env: the
# multi-node resolution (node_count from launch_env, base_check guard,
# load-optimizer cross-check, the resolution print) must complete BEFORE
# any use of base_check — the 2026-09-10 smoke hit UnboundLocalError at
# exactly that spot (checks above never enter main()'s body).
_mn_td = tempfile.mkdtemp(prefix="prod2disp_")
with open(os.path.join(_mn_td, "launch_env.json"), "w") as f:
    json.dump({"EXP_NAME": "x", "NUM_NPU": "8",
               "NANOCHAT_BASE_DIR": os.path.join(_mn_td, "base"),
               "NANOCHAT_DIR": os.path.join(_mn_td, "nc"),
               "TARGET_ARM_NODES": "4", "TARGET_LOAD_OPTIMIZER": "0"}, f)
with open(os.path.join(_mn_td, "remote_config.json"), "w") as f:
    json.dump({"obs_prefix": "obs://b/p", "backend": "mock",
               "storage_kind": "local",
               "storage_root": os.path.join(_mn_td, "fakeobs")}, f)
r = subprocess.run(
    [sys.executable, os.path.join(REPO, "scripts",
                                  "dispatch_target_arm.py"),
     "--arm", "climb", "--output-dir", _mn_td],
    capture_output=True, text=True, timeout=120)
_out = r.stdout + r.stderr
check("dispatch: main() passes multi-node resolution (no UnboundLocalError)",
      "UnboundLocalError" not in _out
      and "multi-node arm: node_count=4" in _out
      and "mixed data not ready" in _out,
      _out[-300:])

for sh in ("runs/run_climbmix.sh", "runs/lib/target_arm.sh",
           "scripts/diagnostics/prod2_watch.sh"):
    r = subprocess.run(["bash", "-n", os.path.join(REPO, sh)],
                       capture_output=True, text=True)
    check(f"shell: bash -n {sh}", r.returncode == 0, r.stderr[:200])

# run_climbmix.sh wiring: fingerprint + defaults + guard
src = open(os.path.join(REPO, "runs", "run_climbmix.sh")).read()
check("shell: adaptive_configs in FP_SEARCH_PARAMS",
      'adaptive_configs=$ADAPTIVE_CONFIGS' in src)
check("shell: adaptive_compact in FP_SEARCH_PARAMS",
      'adaptive_compact=$ADAPTIVE_COMPACT' in src)
check("shell: ADAPTIVE_COMPACT default 0 + forwarded",
      'ADAPTIVE_COMPACT="${ADAPTIVE_COMPACT:-0}"' in src
      and 'ADAPTIVE_ARGS+=(--adaptive-compact)' in src)
check("shell: TARGET_ARM_MODE not fingerprinted (execution shape)",
      re.search(r'"target_arm_mode=', src) is None)
check("shell: TARGET_ARM_NODES default 1",
      'TARGET_ARM_NODES="${TARGET_ARM_NODES:-1}"' in src)
check("shell: TARGET_ARM_NODES not fingerprinted (execution shape)",
      re.search(r'"target_arm_nodes=', src) is None)
check("shell: nodes>1 derives TARGET_LOAD_OPTIMIZER=0",
      re.search(r'\[ "\$TARGET_ARM_NODES" -gt 1 \]; then\s*'
                r'TARGET_LOAD_OPTIMIZER=0', src) is not None)
check("shell: target_load_optimizer in FP_TARGET_PARAMS (semantic)",
      '"target_load_optimizer=$TARGET_LOAD_OPTIMIZER"' in src)
check("shell: TARGET_ARM_NODES/LOAD_OPTIMIZER in launch_env.json",
      src.count("TARGET_ARM_NODES") >= 3
      and src.count("TARGET_LOAD_OPTIMIZER") >= 4)
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
# target_arm.sh fallback shares the optimizer semantics (fair verdict:
# a remote-cold arm vs local-warm arm pair would be an unfair comparison)
tasrc = open(os.path.join(REPO, "runs", "lib", "target_arm.sh")).read()
check("shell: target_arm_train emits --load-optimizer=0 from env",
      'lo_extra+=(--load-optimizer=0)' in tasrc
      and '${TARGET_LOAD_OPTIMIZER:-1}' in tasrc
      and 'lo_extra[@]+"${lo_extra[@]}"' in tasrc)

# ── summary ───────────────────────────────────────────────────────────────
print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
