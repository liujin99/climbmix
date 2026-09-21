#!/usr/bin/env python3
"""P1/A1+A2 final-selection mechanism verification (D19, 2026-09-21).

prod4 L1a: the paper-faithful design-space argmin (never measured) lost at
d28 to BOTH measured candidates of the same search (climb-cfg72 0.2142 /
climb-cfg25 0.2066 vs climb-终选 0.1972/0.1990). D19 changes the final
selection: the argmin must EARN the slot — its predicted advantage over the
best MEASURED config must exceed the no-claim margin (A1) — and the
selection model refits on ALL measured points with the early-stopping tree
count (A2). A3 exports top-k measured candidates for d28 arm promotion.

This file locks:
  1. refit_on_full (real LightGBM): tree count = best_iteration_, trained
     on every point, held-out metrics carried over (guard still honest);
  2. the bootstrapper's A1+A2 path with a REAL predictor (refit executed,
     claim report populated, deterministic no-claim under a huge margin);
  3. top-k greedy diversity + relaxed fill + NaN filtering;
  4. the direction mapping (target space <-> utility space);
  5. report rendering of the Final Selection section (incl. legacy states
     without one) and the labeled topk_mixture_candidates.json export.
"""

import os
import sys
import json
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from climbmix.core.iterative_bootstrapper import IterativeBootstrapper
from climbmix.core.predictor import LightGBMPredictor
from climbmix.core.types import (
    CLIMBConfig, ClusterInfo, MixtureConfig, MixtureWeights,
    PredictorConfig, SearchConfig,
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


def make_bootstrapper(k=3, margin=None, topk=3, div=0.15):
    cfg = CLIMBConfig(
        val_tasks=["t1"],
        search=SearchConfig(topk_arms=topk, topk_diversity_min_l1=div),
    )
    if margin is not None:
        cfg.predictor.final_claim_margin = margin
    tokens = np.full(k, 600, dtype=np.int64)
    labels = np.repeat(np.arange(k), 4)
    return IterativeBootstrapper(cfg, tokens, labels)


def mc(w):
    return MixtureConfig(mixture_weights=MixtureWeights(
        weights=np.asarray(w, dtype=np.float64)))


# ── 1. refit_on_full (real LightGBM) ───────────────────────────────────────
rng = np.random.default_rng(7)
N, K = 30, 5
W = rng.dirichlet(np.ones(K), size=N)
configs = [MixtureConfig.from_flattened(w, config_id=i) for i, w in enumerate(W)]
# Pure-noise targets: early stopping must fire well below the cap.
y = rng.normal(0.0, 1.0, size=N)

pcfg = PredictorConfig(n_estimators=200, early_stopping_rounds=10)
p_split = LightGBMPredictor(K, pcfg)
n_val = 6
p_split.fit(configs[:-n_val], y[:-n_val],
            val_configs=configs[-n_val:], val_losses=y[-n_val:])
best_iter = getattr(p_split._model, "best_iteration_", None)
check("refit: split fit early-stopped (best_iteration recorded)",
      best_iter is not None and 0 < best_iter < 200,
      f"best_iteration_={best_iter}")
check("refit: split fit saw N - n_val points", p_split.fit_n_ == N - n_val)
val_r2_split = p_split.val_r2_

p_full = p_split.refit_on_full(configs, y)
check("refit: returns a NEW fitted model on ALL points",
      p_full is not p_split and p_full._is_fitted and p_full.fit_n_ == N)
check("refit: tree count = the early-stopping choice",
      int(p_full._model.n_estimators) == int(best_iter),
      f"{p_full._model.n_estimators} vs {best_iter}")
check("refit: held-out metrics carried over (guard stays honest)",
      p_full.val_r2_ == val_r2_split and p_full.val_spearman_ == p_split.val_spearman_)
preds = p_full.predict(configs[:3])
check("refit: predict works, finite", np.isfinite(preds).all() and len(preds) == 3)

# non-finite targets are filtered inside refit (NaN scores in the fleet)
y_dirty = y.copy()
y_dirty[3] = float("nan")
p_dirty = p_split.refit_on_full(configs, y_dirty)
check("refit: non-finite targets filtered", p_dirty.fit_n_ == N - 1)

# ── 2. bootstrapper A1+A2 path with a REAL predictor ───────────────────────
# 12 valid configs, strongly structured utility (one dominant axis) so the
# fit carries signal (val R² > 0, no-signal guard stays closed).
rng2 = np.random.default_rng(11)
K2 = 3
W2 = rng2.dirichlet(np.ones(K2), size=12)
util = 3.0 * W2[:, 0] + 0.05 * rng2.normal(size=12)
configs2 = [MixtureConfig.from_flattened(w, config_id=i) for i, w in enumerate(W2)]
best_i2 = int(np.argmax(util))

p2 = LightGBMPredictor(K2, PredictorConfig(n_estimators=200,
                                           early_stopping_rounds=10))
# same split convention as _refit_predictor (rng 42, last 20% out)
perm = np.random.default_rng(42).permutation(12)
n_val2 = max(5, int(12 * 0.2))
tr, va = perm[:12 - n_val2], perm[12 - n_val2:]
p2.fit([configs2[i] for i in tr], -util[tr],
       val_configs=[configs2[i] for i in va], val_losses=-util[va])
check("A1+A2: real predictor carries signal (val R² > 0)",
      p2.val_r2_ is not None and p2.val_r2_ > 0, f"val_r2_={p2.val_r2_}")

# Huge margin -> guaranteed NO-CLAIM: best measured wins, refit executed.
bs = make_bootstrapper(k=K2, margin=10.0)
bs._accumulated_configs = configs2
bs._accumulated_scores = util.tolist()
bs._predictor = p2
cand = mc([0.05, 0.05, 0.90])
bs._search_full_design_space = lambda: cand
sel = bs._select_final_mixture()
check("A1+A2: huge margin -> best_measured_no_claim",
      bs._selection_mode == "best_measured_no_claim"
      and np.allclose(sel.mixture_weights.weights, W2[best_i2], atol=1e-6))
check("A1+A2: refit_on_full executed on the selection model",
      bs.selection_claim["refit_on_full"] is True
      and bs._predictor is not p2 and bs._predictor.fit_n_ == 12)
check("A1+A2: claim report populated (gain/margin/radius/neighborhood)",
      bs.selection_claim["margin"] == 10.0
      and bs.selection_claim["margin_source"] == "config_override"
      and bs.selection_claim["argmin_candidate"]["l1_radius_to_measured"] is not None
      and len(bs.selection_claim["argmin_candidate"]["nearest_measured"]) == 5)
check("A1+A2: guard reason names the margin",
      any("margin" in r for r in bs._selection_guard_reasons))

# Zero margin, real predictor: mechanics only (mode is data-dependent).
bs0 = make_bootstrapper(k=K2, margin=0.0)
bs0._accumulated_configs = configs2
bs0._accumulated_scores = util.tolist()
bs0._predictor = LightGBMPredictor(K2, PredictorConfig(
    n_estimators=200, early_stopping_rounds=10))
bs0._predictor.fit([configs2[i] for i in tr], -util[tr],
                   val_configs=[configs2[i] for i in va], val_losses=-util[va])
bs0._search_full_design_space = lambda: cand
sel0 = bs0._select_final_mixture()
check("A1+A2: zero margin -> a valid selection either way, claim recorded",
      bs0._selection_mode in ("best_measured_no_claim",
                              "predictor_design_space_claimed")
      and bs0.selection_claim is not None
      and bs0._predictor.fit_n_ == 12)

# Stub predictors (no refit_on_full attr) skip A2 gracefully.
class _Stub:
    val_r2_ = 0.5
    train_r2_ = 0.5

    def predict(self, cs):
        return np.full(len(cs), -99.0)

bs_stub = make_bootstrapper(k=K2)
bs_stub._accumulated_configs = configs2[:4]
bs_stub._accumulated_scores = util[:4].tolist()
bs_stub._predictor = _Stub()
bs_stub._search_full_design_space = lambda: cand
sel_stub = bs_stub._select_final_mixture()
check("A1+A2: stub predictor skips refit, still claims (big pred)",
      bs_stub._selection_mode == "predictor_design_space_claimed"
      and bs_stub.selection_claim["refit_on_full"] is False)

# ── 3. top-k greedy diversity + relaxed fill + NaN filtering ────────────────
A = mc([0.50, 0.30, 0.20])
A2d = mc([0.55, 0.27, 0.18])   # L1 = 0.10 < 0.15 -> near-duplicate of A
B = mc([0.10, 0.80, 0.10])
C = mc([0.80, 0.10, 0.10])
D = mc([0.20, 0.20, 0.60])
E = mc([0.33, 0.34, 0.33])
tk_configs = [A, A2d, B, C, D, E]
for i, c in enumerate(tk_configs):
    c.config_id = 100 + i
tk_scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

bs_t = make_bootstrapper(k=3, topk=3, div=0.15)
bs_t._accumulated_configs = tk_configs
bs_t._accumulated_scores = tk_scores
exp = bs_t._select_topk_candidates()
check("topk: near-duplicate skipped, diversity order kept",
      [c["config_id"] for c in exp["candidates"]] == [100, 102, 103]
      and exp["relaxed"] is False,
      f"got {[c['config_id'] for c in exp['candidates']]}")
check("topk: ranks and scores recorded",
      [c["rank"] for c in exp["candidates"]] == [1, 2, 3]
      and [c["score"] for c in exp["candidates"]] == [1.0, 0.8, 0.7])
check("topk: min_l1 recorded against the OTHER selected",
      exp["candidates"][0]["min_l1_to_other_selected"] is not None)

# k > diverse pool -> relaxed fill adds the near-duplicate
bs_t5 = make_bootstrapper(k=3, topk=7, div=0.15)
bs_t5._accumulated_configs = tk_configs
bs_t5._accumulated_scores = tk_scores
exp5 = bs_t5._select_topk_candidates()
check("topk: k > pool -> relaxed fill to k, flag set",
      len(exp5["candidates"]) == 6 and exp5["relaxed"] is True)

# NaN scores excluded from the ranking
bs_tn = make_bootstrapper(k=3, topk=3, div=0.15)
bs_tn._accumulated_configs = tk_configs
bs_tn._accumulated_scores = [float("nan"), 0.9, 0.8, 0.7, 0.6, 0.5]
expn = bs_tn._select_topk_candidates()
check("topk: NaN scores never selected",
      all(c["config_id"] != 100 for c in expn["candidates"])
      and len(expn["candidates"]) == 3)

# k = 0 disables
bs_t0 = make_bootstrapper(k=3, topk=0)
bs_t0._accumulated_configs = tk_configs
bs_t0._accumulated_scores = tk_scores
check("topk: k=0 disables the export",
      bs_t0._select_topk_candidates()["candidates"] == [])

# ── 4. direction mapping (target <-> utility) ──────────────────────────────
bs_max = make_bootstrapper(k=3)
check("direction: maximize -> utility = -prediction",
      bs_max._utility_of_prediction(-1.7) == 1.7
      and bs_max.metric_direction == "maximize")
cfg_min = CLIMBConfig(val_tasks=["t1"])
cfg_min.proxy.validation_metric = "loss"
bs_min = IterativeBootstrapper(cfg_min, np.full(3, 600, dtype=np.int64),
                               np.repeat(np.arange(3), 4))
check("direction: minimize -> utility = prediction (identity)",
      bs_min._utility_of_prediction(-1.7) == -1.7
      and bs_min.metric_direction == "minimize")

# ── 5. report rendering + topk_mixture_candidates.json export ───────────────
from climbmix.pipeline.report_generator import generate_markdown_report

cluster_info = [ClusterInfo(cluster_id=i, centroid=np.zeros(2), num_docs=10,
                            num_tokens=100, label=f"C{i}") for i in range(3)]
stats = {"original_distribution": [10, 10, 10],
         "selected_distribution": [5, 3, 2],
         "mixture_weights": [0.5, 0.3, 0.2]}

claim = {
    "best_measured": {"config_id": 25, "state_index": 0, "actual_utility": 1.255},
    "argmin_candidate": {"predicted_utility": 1.35, "claimed_gain": 0.095,
                         "l1_radius_to_measured": 0.65,
                         "nearest_measured": [
                             {"config_id": 72, "l1": 0.65, "actual_utility": 1.13},
                             {"config_id": 60, "l1": 0.71, "actual_utility": 1.05}]},
    "margin": 0.42, "margin_source": "auto_residual_sigma=0.412",
    "refit_on_full": True,
}
topk_export = {"k_requested": 3, "diversity_min_l1": 0.15, "relaxed": False,
               "candidates": [
                   {"rank": 1, "config_id": 25, "state_index": 0,
                    "score": 1.255, "weights": [0.5, 0.3, 0.2],
                    "min_l1_to_other_selected": 0.51},
                   {"rank": 2, "config_id": 72, "state_index": 1,
                    "score": 1.13, "weights": [0.1, 0.8, 0.1],
                    "min_l1_to_other_selected": 0.51}]}

def _render_report(td, search_state):
    path = generate_markdown_report(
        td, CLIMBConfig(val_tasks=["t1"]), cluster_info, mc([0.5, 0.3, 0.2]),
        [], stats, {}, 12.0, search_state=search_state,
    )
    with open(path) as f:
        return f.read()

with tempfile.TemporaryDirectory() as td:
    md = _render_report(td, {"selection": {
        "mode": "best_measured_no_claim",
        "guard_reasons": ["claimed gain ... margin"],
        "claim": claim, "topk": topk_export}})
    check("report: Final Selection section rendered",
          "## Final Selection" in md and "best_measured_no_claim" in md)
    check("report: claim numbers rendered",
          "1.255" in md and "0.095" in md and "auto_residual_sigma" in md)
    check("report: neighborhood + top-k tables rendered",
          "cfg#72" in md and "Top-2" in md)

with tempfile.TemporaryDirectory() as td:
    md_legacy = _render_report(td, {})
    check("report: legacy state (no selection) -> section skipped, no crash",
          "## Final Selection" not in md_legacy)

with tempfile.TemporaryDirectory() as td:
    md_guard = _render_report(td, {"selection": {
        "mode": "no_signal_best_measured",
        "guard_reasons": ["r2 <= 0"],
        "claim": None, "topk": topk_export}})
    check("report: guard path (no claim) still renders top-k",
          "no_signal_best_measured" in md_guard and "Top-2" in md_guard)

# labeled JSON export via the pipeline's _save_outputs
from climbmix.pipeline.climb_pipeline import CLIMBPipeline

with tempfile.TemporaryDirectory() as td:
    pl = CLIMBPipeline(CLIMBConfig(val_tasks=["t1"]))
    pl._save_outputs(
        td, mc([0.5, 0.3, 0.2]), [], cluster_info,
        np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64),
        np.full(3, 100, dtype=np.int64), None, None,
        stats, {}, 0.0,
        search_extras={"selection": {"mode": "best_measured_no_claim",
                                     "guard_reasons": [],
                                     "claim": claim, "topk": topk_export},
                       "_final_predictor": None},
    )
    topk_path = os.path.join(td, "topk_mixture_candidates.json")
    check("pipeline: topk_mixture_candidates.json written", os.path.exists(topk_path))
    payload = json.load(open(topk_path))
    check("pipeline: weights keyed by cluster label",
          payload["candidates"][0]["weights"] == {"C0": 0.5, "C1": 0.3, "C2": 0.2}
          and payload["selection_mode"] == "best_measured_no_claim"
          and payload["k_requested"] == 3)
    check("pipeline: optimal weights still written",
          os.path.exists(os.path.join(td, "optimal_mixture_weights.json")))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {_failures}")
    sys.exit(1)
print(f"ALL PASS ({_n} checks)")
