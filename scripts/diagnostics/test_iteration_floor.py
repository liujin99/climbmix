#!/usr/bin/env python3
"""Iteration-floor fail-loud verification (prod3 iter-1 lesson).

prod3 iter-1 (2026-09-14): of a 20-floor iteration only 9 configs produced
measurable scores (10 STEM-supply guard rejections + 5 single-pass
violations + 1 success on 16 fresh; 8 weekend-done reused). The loop
advanced silently with a WARNING and thinned every later predictor refit —
CONFIGS_PER_ITER is the per-iteration data contract and must be enforced.

This file locks the contract from both sides:
  1. an iteration whose valid points fall below its floor raises
     RuntimeError (naming the guard families to check), NOT a warning;
  2. a healthy iteration (all finite) and an at-floor iteration (exactly
     floor valid) still advance;
  3. the raise happens AFTER pending-config persistence — resume re-runs
     exactly the failed iteration (no state corruption).
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from climbmix.core.iterative_bootstrapper import IterativeBootstrapper
from climbmix.core.types import (
    CLIMBConfig, MixtureConfig, MixtureWeights, ProxyResult, SearchConfig,
)

_failures = []
_n = 0


def check(name, cond, extra=""):
    global _n
    _n += 1
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {extra}" if extra and not cond else ""))
    if not cond:
        _failures.append(name)


class FakeRunner:
    """Returns healthy ProxyResults plus `n_error` errored ones (metadata
    error + unmeasured per-task dicts — exactly what the guard-rejection
    path produces in the real pipeline)."""

    def __init__(self, n_error):
        self.n_error = n_error
        self.last_admission_stats = None
        self.last_effective_concurrency = None

    def run_batch(self, configs, **kwargs):
        out = []
        for i, c in enumerate(configs):
            if i < self.n_error:
                out.append(ProxyResult(
                    mixture_config=c, validation_loss=float("nan"),
                    per_task_accuracies=None, per_task_nlls=None,
                    metadata={"error": "STEM supply insufficient (test)"}))
            else:
                out.append(ProxyResult(
                    mixture_config=c, validation_loss=0.1 * (i + 1),
                    per_task_accuracies={"t1": 0.1 * (i + 1)},
                    per_task_nlls={"t1": 2.0},
                    metadata={}))
        return out


def make_bootstrapper(state_path=None):
    cfg = CLIMBConfig(
        val_tasks=["t1"],
        search=SearchConfig(configs_per_iter=[4, 4], num_iterations=2),
    )
    cluster_tokens = np.array([600, 600], dtype=np.int64)
    labels = np.array([0] * 4 + [1] * 4)
    bs = IterativeBootstrapper(cfg, cluster_tokens, labels,
                               state_path=state_path)
    return bs


# ── 1. below-floor iteration must RAISE (not warn-and-advance) ────────────
with tempfile.TemporaryDirectory() as td:
    state = os.path.join(td, "search_state.json")
    bs = make_bootstrapper(state)
    raised = None
    try:
        bs.run_iteration(1, 4, proxy_runner=FakeRunner(n_error=2))
    except RuntimeError as e:
        raised = str(e)
    check("floor: 2/4 valid raises RuntimeError", raised is not None)
    check("floor: message names the floor",
          raised is not None and "below the CONFIGS_PER_ITER floor" in raised)
    check("floor: message points at the guard families",
          raised is not None and "STEM supply" in raised
          and "single-pass" in raised)
    check("floor: message mentions resume semantics",
          raised is not None and "resume" in raised)

# ── 2. all-NaN iteration still raises the (more specific) all-failed ──────
with tempfile.TemporaryDirectory() as td:
    bs = make_bootstrapper(os.path.join(td, "search_state.json"))
    raised0 = None
    try:
        bs.run_iteration(1, 4, proxy_runner=FakeRunner(n_error=4))
    except RuntimeError as e:
        raised0 = str(e)
    check("floor: 0/4 valid raises (all-failed branch, fires first)",
          raised0 is not None
          and "All 4 proxy experiments of iteration 1 failed" in raised0)

# ── 3. healthy and at-floor iterations still advance ──────────────────────
with tempfile.TemporaryDirectory() as td:
    state = os.path.join(td, "search_state.json")
    bs = make_bootstrapper(state)
    try:
        res = bs.run_iteration(1, 4, proxy_runner=FakeRunner(n_error=0))
        check("floor: 4/4 valid advances, returns a result", res is not None)
    except Exception as e:  # noqa: BLE001
        check("floor: 4/4 valid advances, returns a result", False, repr(e))

with tempfile.TemporaryDirectory() as td:
    state = os.path.join(td, "search_state.json")
    bs = make_bootstrapper(state)
    try:
        res = bs.run_iteration(1, 4, proxy_runner=FakeRunner(n_error=0))
        # second iteration AT the floor (0 errors of 4) also advances
        res2 = bs.run_iteration(2, 4, proxy_runner=FakeRunner(n_error=0))
        check("floor: consecutive healthy iterations advance", res2 is not None)
    except Exception as e:  # noqa: BLE001
        check("floor: consecutive healthy iterations advance", False, repr(e))

# ── 4. raise leaves resumable state (pending persisted BEFORE scoring) ────
import json
with tempfile.TemporaryDirectory() as td:
    state = os.path.join(td, "search_state.json")
    bs = make_bootstrapper(state)
    try:
        bs.run_iteration(1, 4, proxy_runner=FakeRunner(n_error=2))
    except RuntimeError:
        pass
    # pending configs for iteration 1 were persisted at batch submission
    # (iterative_bootstrapper pending-state contract, :718) — resume replays
    # exactly them; the floor raise must not corrupt that.
    with open(state) as f:
        st = json.load(f)
    check("floor: pending state survives the raise (resume re-runs iter 1)",
          (st.get("pending") or {}).get("iteration") == 1
          and len(st["pending"]["configs"]) == 4)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {_failures}")
    sys.exit(1)
print(f"ALL PASS ({_n} checks)")
