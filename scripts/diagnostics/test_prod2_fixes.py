#!/usr/bin/env python3
"""prod2 fix verification — balanced partition / structure gate / acc-only
fallback / no-signal guard / from_dict natural sort / f centered-unit noise
floor / cp4 NLL fallback / dispatch climbmix-ma bootstrap.

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_prod2_fixes.py
Exit 0 = all checks pass.
"""
import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))

from climbmix.core.cluster_merge import (  # noqa: E402
    balanced_macro_clusters, validate_cluster_structure)
from climbmix.core.types import (  # noqa: E402
    CLIMBConfig, ClusterInfo, MixtureWeights, MixtureConfig, SearchConfig)
from climbmix.core.iterative_bootstrapper import IterativeBootstrapper  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


# ── 1. balanced_macro_clusters: capacity + coverage + determinism ─────────
rng = np.random.default_rng(0)
F = 30
groups = [rng.normal(loc, 0.05, size=(10, 2)) for loc in ((0, 0), (5, 5), (10, 0))]
centroids = np.array([g.mean(0) for g in groups for _ in range(10)],
                     dtype=np.float32)  # 3 directions x 10 near-duplicates
# skewed fine-cluster token masses: fine cluster 0 holds 8x the small ones
# (kept <= cap so the capacity invariant is in scope; the overflow case is
# tested separately below)
fine_tok = np.ones(F, dtype=np.int64)
fine_tok[0] = 8
n_docs_per_fine = 4
labels = np.repeat(np.arange(F), n_docs_per_fine)
token_counts = np.repeat(fine_tok, n_docs_per_fine)

with tempfile.TemporaryDirectory() as td:
    prof = os.path.join(td, "balanced_profile.json")
    macro_labels, macro_centroids, fmap = balanced_macro_clusters(
        labels, centroids, token_counts=token_counts, K=3,
        profile_path=prof)
    total_tok = token_counts.sum()
    shares = np.sort([token_counts[macro_labels == k].sum() / total_tok
                      for k in np.unique(macro_labels)])[::-1]
    check("balanced: K_final == 3", len(macro_centroids) == 3)
    check("balanced: max share <= cap (1+slack)/K",
          shares[0] <= 1.15 / 3 + 1e-9, f"max={shares[0]:.3f}")
    check("balanced: no unlabeled docs (pruned -1 preserved only)",
          (macro_labels >= 0).all() and len(macro_labels) == len(labels))
    check("balanced: fine->macro map complete", len(fmap) == F)
    check("balanced: profile written", os.path.exists(prof))

    macro_labels2, macro_centroids2, _ = balanced_macro_clusters(
        labels, centroids, token_counts=token_counts, K=3)
    check("balanced: deterministic", np.array_equal(macro_labels, macro_labels2)
          and np.allclose(macro_centroids, macro_centroids2))

# K > F degrades to F
labels_k = np.repeat(np.arange(4), 5)
ml_k, mc_k, _ = balanced_macro_clusters(
    labels_k, centroids[:4],
    token_counts=np.ones(20, dtype=np.int64), K=6)
check("balanced: K > F degrades to F", len(mc_k) == 4)

# overflow: one fine cluster with 90% of tokens, K=2
big_labels = np.repeat(np.arange(3), 10)
big_tokens = np.repeat(np.array([90, 1, 1], dtype=np.int64), 10)
ml_o, mc_o, _ = balanced_macro_clusters(
    big_labels, centroids[:3], token_counts=big_tokens, K=2)
check("balanced: overflow tolerated, labels complete",
      (ml_o >= 0).all() and len(mc_o) == 2)

# token_counts=None -> doc-count weights
ml_n, mc_n, _ = balanced_macro_clusters(labels, centroids, K=3)
check("balanced: token_counts=None path", (ml_n >= 0).all() and len(mc_n) == 3)

# ── 2. validate_cluster_structure gate ────────────────────────────────────
def ci(i, docs, toks):
    return ClusterInfo(cluster_id=i, centroid=np.zeros(1), num_docs=docs,
                       num_tokens=toks, label=f"C{i}")

try:
    validate_cluster_structure([ci(0, 500, 500), ci(1, 500, 500)])
    check("gate: 50/50 passes", True)
except ValueError:
    check("gate: 50/50 passes", False)

try:
    validate_cluster_structure([ci(0, 990, 990), ci(1, 10, 10)])
    check("gate: 99% raises ValueError", False)
except ValueError as e:
    check("gate: 99% raises ValueError", "StructureGate" in str(e))

# ── 3. MixtureWeights.from_dict natural sort ──────────────────────────────
d = {"C10": 1.0, "C2": 2.0, "C1": 3.0}
mw = MixtureWeights.from_dict(d)
check("from_dict: natural order C1<C2<C10",
      np.allclose(mw.weights, [3.0, 2.0, 1.0]), f"got {mw.weights}")

# ── 4. _compute_scores: acc-only fallback (mmlu nll=nan) ──────────────────
cfg = CLIMBConfig(val_tasks=["mmlu_stem", "other_task"])
cluster_labels = np.array([0, 0, 0, 1])
cluster_tokens = np.array([600, 600], dtype=np.int64)
bs = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
# 4 configs: acc present everywhere; nll nan on mmlu_stem for ALL (prod1
# situation), finite on other_task. mmlu acc alternates strongly.
bs._accumulated_per_benchmark = [
    ({"mmlu_stem": 0.9, "other_task": 0.5}, {"mmlu_stem": float("nan"), "other_task": 1.0}),
    ({"mmlu_stem": 0.1, "other_task": 0.5}, {"mmlu_stem": float("nan"), "other_task": 1.0}),
    ({"mmlu_stem": 0.9, "other_task": 0.5}, {"mmlu_stem": float("nan"), "other_task": 1.0}),
    ({"mmlu_stem": 0.1, "other_task": 0.5}, {"mmlu_stem": float("nan"), "other_task": 1.0}),
]
bs._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([w, 1 - w])))
    for w in (0.9, 0.1, 0.8, 0.2)
]
scores = bs._compute_scores()
check("scores: all finite with benchmark-wide nan NLL",
      np.isfinite(scores).all(), f"scores={np.round(scores, 3)}")
check("scores: mmlu_stem (acc-only) actually contributes",
      scores[0] > scores[1] and abs(scores[0] - scores[2]) < 1e-9,
      f"scores={np.round(scores, 3)}")
check("scores: task f recorded", "mmlu_stem" in bs._task_f
      and "other_task" in bs._task_f)

# partial nan NLL (per-config): those configs scored acc-only, rest blended
bs2 = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
bs2._accumulated_per_benchmark = [
    ({"mmlu_stem": 0.9, "other_task": 0.5}, {"mmlu_stem": float("nan"), "other_task": 1.0}),
    ({"mmlu_stem": 0.9, "other_task": 0.5}, {"mmlu_stem": 0.5, "other_task": 1.0}),
]
bs2._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([0.5, 0.5])))
    for _ in range(2)
]
s2 = bs2._compute_scores()
check("scores: per-config nan NLL handled (all finite)",
      np.isfinite(s2).all(), f"scores={np.round(s2, 3)}")

# ── 5. no-signal guard + no-claim margin on final selection (D19) ──────────
class StubPredictor:
    """predict() returns TARGET-space values (lower = better, the
    _refit_predictor convention). Default pred = -10 -> utility +10: the
    argmin claims a huge unmeasured advantage."""

    def __init__(self, r2, train_r2=None, pred_target=-10.0):
        self.val_r2_ = r2
        self.train_r2_ = train_r2 if train_r2 is not None else r2
        self.pred_target = pred_target

    def predict(self, configs):
        return np.full(len(configs), self.pred_target, dtype=np.float64)


bs3 = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
bs3._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([w, 1 - w])))
    for w in (0.1, 0.9)
]
bs3._accumulated_scores = [1.0, 2.0]  # metric maximize -> index 1 best

bs3._predictor = None
sel = bs3._select_final_mixture()
check("selection: no predictor -> best measured",
      bs3._selection_mode == "no_predictor_best_measured"
      and np.allclose(sel.mixture_weights.weights, [0.9, 0.1]))

bs3._predictor = StubPredictor(-0.5)
called = {"design": False}
bs3._search_full_design_space = lambda: (called.__setitem__("design", True)
                                         or bs3._accumulated_configs[0])
sel = bs3._select_final_mixture()
check("selection: R2<=0 -> guard fires, best measured, design-space NOT called",
      bs3._selection_mode == "no_signal_best_measured"
      and not called["design"]
      and np.allclose(sel.mixture_weights.weights, [0.9, 0.1])
      and len(bs3._selection_guard_reasons) >= 1)

# R2>0, huge claimed gain (+8 over best measured) -> extrapolation CLAIMS
bs3._predictor = StubPredictor(0.5)  # pred -10 -> utility +10 vs best 2.0
sel = bs3._select_final_mixture()
check("selection: R2>0 + claimed gain > margin -> design-space candidate",
      bs3._selection_mode == "predictor_design_space_claimed" and called["design"]
      and np.allclose(sel.mixture_weights.weights, [0.1, 0.9]))
check("selection: claim report populated",
      (bs3._selection_claim or {}).get("argmin_candidate", {}).get("claimed_gain", 0) >= 7.9
      and bs3._selection_claim["best_measured"]["actual_utility"] == 2.0
      and len(bs3._selection_claim["argmin_candidate"]["nearest_measured"]) == 2)

# R2>0 but small claimed gain (+0.5) vs margin 1.0 -> NO-CLAIM, best measured
cfg_nc = CLIMBConfig(val_tasks=["mmlu_stem", "other_task"])
cfg_nc.predictor.final_claim_margin = 1.0
bs_nc = IterativeBootstrapper(cfg_nc, cluster_tokens, cluster_labels)
bs_nc._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([w, 1 - w])))
    for w in (0.1, 0.9)
]
bs_nc._accumulated_scores = [1.0, 2.0]
bs_nc._predictor = StubPredictor(0.5, pred_target=-2.5)  # utility 2.5
called_nc = {"design": False}
bs_nc._search_full_design_space = lambda: (called_nc.__setitem__("design", True)
                                           or bs_nc._accumulated_configs[0])
sel = bs_nc._select_final_mixture()
check("selection: claimed gain 0.5 <= margin 1.0 -> NO-CLAIM, best measured",
      bs_nc._selection_mode == "best_measured_no_claim" and called_nc["design"]
      and np.allclose(sel.mixture_weights.weights, [0.9, 0.1])
      and any("margin" in r for r in bs_nc._selection_guard_reasons))
check("selection: top-k exported on the no-claim path",
      [c["config_id"] for c in bs_nc.topk_export["candidates"]]
      == [bs_nc._accumulated_configs[1].config_id,
          bs_nc._accumulated_configs[0].config_id]
      and bs_nc.topk_export["relaxed"] is True)

# margin auto: residual sigma from predictor_eval pairs (floor 0.30)
bs_m = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
bs_m._predictor_eval = [{
    "iteration": 2, "n_val": 6,
    "val_preds": [0.0] * 6,
    "val_targets": [1.0, -1.0, 2.0, -2.0, 3.0, -3.0],
}]
m_sig, src_sig = bs_m._final_claim_margin()
sigma_expect = float(np.std(
    np.array([0.0] * 6) - np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])))
check("margin: auto = max(0.30, residual sigma)",
      abs(m_sig - sigma_expect) < 1e-9 and m_sig > 2.0
      and "auto_residual_sigma" in src_sig)
bs_m._predictor_eval = [{
    "iteration": 2, "n_val": 6,
    "val_preds": [0.001] * 6, "val_targets": [0.0] * 6,
}]
m_floor, src_floor = bs_m._final_claim_margin()
check("margin: residual sigma below floor -> 0.30",
      m_floor == 0.30 and "auto_residual_sigma" in src_floor)
m_cfg, src_cfg = bs_nc._final_claim_margin()
check("margin: config override wins", m_cfg == 1.0 and src_cfg == "config_override")
bs_nc2 = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
m_def, src_def = bs_nc2._final_claim_margin()
check("margin: no eval pairs -> conservative default",
      m_def == 0.30 and src_def == "conservative_default_no_eval_pairs")

# no val split (<10 configs): train_r2 fallback drives the guard
bs5 = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
bs5._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([w, 1 - w])))
    for w in (0.1, 0.9)
]
bs5._accumulated_scores = [1.0, 2.0]
bs5._task_f = {"mmlu_stem": 0.5}
called5 = {"design": False}
bs5._search_full_design_space = lambda: (called5.__setitem__("design", True)
                                         or bs5._accumulated_configs[0])
bs5._predictor = StubPredictor(None, train_r2=-0.2)
sel = bs5._select_final_mixture()
check("selection: no val split, train R2<=0 -> guard fires",
      bs5._selection_mode == "no_signal_best_measured" and not called5["design"])
bs5._predictor = StubPredictor(None, train_r2=0.9)
sel = bs5._select_final_mixture()
check("selection: no val split, train R2>0 + big claim -> design-space path",
      bs5._selection_mode == "predictor_design_space_claimed" and called5["design"])

# all-task f<0 guard fires even with healthy R2
bs4 = IterativeBootstrapper(cfg, cluster_tokens, cluster_labels)
bs4._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([w, 1 - w])))
    for w in (0.1, 0.9)
]
bs4._accumulated_scores = [1.0, 2.0]
bs4._task_f = {"mmlu_stem": -0.3, "other_task": -0.8}
bs4._predictor = StubPredictor(0.7)
called2 = {"design": False}
bs4._search_full_design_space = lambda: (called2.__setitem__("design", True)
                                         or bs4._accumulated_configs[0])
sel = bs4._select_final_mixture()
check("selection: all-task f<0 -> guard fires despite R2>0",
      bs4._selection_mode == "no_signal_best_measured" and not called2["design"])

# ── 6. f noise floor in centered units (prod2 2026-09-09 unit bug;
#        prod3 2026-09-15 chance-semantics + worst-case-p bug) ──────────────
# Centered = (raw - chance)/(1 - chance) per benchmark (BENCHMARK_CHANCE);
# the floor is p_hat*(1-p_hat)/K/(1-chance)^2 with p_hat the empirical mean
# in raw units. prod2 regression: 0.25/K not rescaled by /0.75^2 (MC tasks).
# prod3 regression: MC semantics (0.25, 0.75^2) applied to CoT tasks whose
# centered == raw — gsm8k_cot's noise was overstated ~20x, its w hit 0 and
# the objective silently flipped to NLL.
cfg6 = CLIMBConfig(val_tasks=["mmlu_stem"])
bs6 = IterativeBootstrapper(cfg6, cluster_tokens, cluster_labels)
bs6._accumulated_per_benchmark = [
    ({"mmlu_stem": v}, {}) for v in (0.0396, 0.0196, 0.0396, 0.0196)
]
bs6._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([0.5, 0.5])))
    for _ in range(4)
]
bs6._compute_scores()
_var6 = np.var([0.0396, 0.0196, 0.0396, 0.0196])
_p6 = 0.25 + 0.75 * float(np.mean([0.0396, 0.0196, 0.0396, 0.0196]))
_f6_expected = 1.0 - (_p6 * (1.0 - _p6) / 3545 / 0.75 ** 2) / (_var6 + 1e-12)
_f6 = bs6._task_f["mmlu_stem"]
check("f: MC noise floor = p_hat(1-p_hat)/K/0.75^2 (empirical p_hat)",
      abs(_f6 - _f6_expected) < 1e-9,
      f"f={_f6:.4f} expected={_f6_expected:.4f}")
check("f: mmlu-scale variance sits AT the noise bound — corrected floor "
      "claims no signal (old hardcoded formula lied +0.29)",
      _f6 < 0.05, f"f={_f6:.4f}")

# prod3 regression: gsm8k_cot (chance~0, centered==raw, p~0.02). Old floor
# 0.25/1319/0.75^2 = 3.37e-4 -> f=-1.97 -> w=0 (signal discarded); true floor
# ~1.7e-5 -> f~+0.85. w_floor pinned to 0.0 here to expose the RAW w
# difference (the 0.5 default would clamp both sides).
cfg7 = CLIMBConfig(val_tasks=["gsm8k_cot"], search=SearchConfig(w_floor=0.0))
bs7 = IterativeBootstrapper(cfg7, cluster_tokens, cluster_labels)
_g = [0.034, 0.012, 0.033, 0.013]  # mean 0.023, var ~1.105e-4 (prod3-like)
bs7._accumulated_per_benchmark = [({"gsm8k_cot": v}, {}) for v in _g]
bs7._accumulated_configs = [
    MixtureConfig(mixture_weights=MixtureWeights(weights=np.array([0.5, 0.5])))
    for _ in range(4)
]
bs7._compute_scores()
_var7 = np.var(_g)
_f7_old = 1.0 - (0.25 / 1319 / 0.75 ** 2) / (_var7 + 1e-12)
_f7_expected = 1.0 - (0.023 * 0.977 / 1319) / (_var7 + 1e-12)
_f7 = bs7._task_f["gsm8k_cot"]
check("f: CoT floor uses raw units (no 0.75^2), empirical p_hat",
      abs(_f7 - _f7_expected) < 1e-9,
      f"f={_f7:.4f} expected={_f7_expected:.4f}")
check("f: prod3's discarded gsm8k signal is recovered "
      f"(old-code f would be {_f7_old:.2f})",
      _f7 > 0.5 and _f7_old < 0.0, f"f_new={_f7:.4f} f_old={_f7_old:.4f}")

# w_floor default is 0.5 (acc-primary: NLL may at most halve a vote — the
# prod3 silent flip to a pure NLL objective must be structurally impossible)
check("w_floor default = 0.5 (prod3 silent NLL flip)",
      SearchConfig().w_floor == 0.5)

# ── 8. sampler weight floor (prod3 corner-seeking / sticky-zero insurance) ─
from climbmix.core.dirichlet_sampler import DirichletSampler  # noqa: E402
_tok8 = np.array([500, 300, 50, 20, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                 dtype=np.int64)  # tiny clusters -> alpha_i << 1 -> near-zero draws
_cfg8 = CLIMBConfig()  # weight_floor default 0.01
_s8 = DirichletSampler(15, _tok8, _cfg8, seed=7)
_b8 = _s8.sample_batch(300)
_W8 = np.array([c.mixture_weights.weights for c in _b8])
check("sampler floor: every cluster >= floor/(1+K*floor) in batch draws",
      _W8.min() >= 0.01 / 1.15 - 1e-12, f"min={_W8.min():.5f}")
check("sampler floor: rows renormalized to 1",
      np.allclose(_W8.sum(axis=1), 1.0))

# guided exploration around a corner base (C10=1.0, 14 dead) revives dead
# clusters while staying centered on the base — the sticky-zero fix.
# 50 copies of the corner as bases + m=50 keeps every draw on the guided
# path (m > len(bases) would fill the remainder from the base distribution)
_corner8 = MixtureConfig(mixture_weights=MixtureWeights(
    weights=np.array([0.0] * 9 + [1.0] + [0.0] * 5)))
_bases8 = [_corner8] * 50
_s8b = DirichletSampler(15, _tok8, _cfg8, seed=11)
_g8 = _s8b.sample_from_top_n(_bases8, m=50, exploration_concentration=5.0)
_G8 = np.array([c.mixture_weights.weights for c in _g8])
check("sampler floor: dead clusters revived in guided draws (sticky zero fixed)",
      _G8.min() >= 0.01 / 1.15 - 1e-12, f"min={_G8.min():.5f}")
check("sampler floor: insurance is bounded — corner base stays dominant",
      _G8[:, 9].mean() > 0.5, f"C10 mean={_G8[:, 9].mean():.3f}")

# weight_floor=0 disables (pre-fix behavior: sparse draws return)
_cfg8c = CLIMBConfig(search=SearchConfig(weight_floor=0.0))
_s8c = DirichletSampler(15, _tok8, _cfg8c, seed=7)
_b8c = _s8c.sample_batch(300)
_W8c = np.array([c.mixture_weights.weights for c in _b8c])
check("sampler floor=0: disabled, sparse draws return (pre-fix behavior)",
      (_W8c < 0.005).any(), f"min={_W8c.min():.5f}")

# ── 7. cp4_report: STEM NLL nan -> per-task N-weighted fallback ───────────
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "cp4_report_fixtest",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "cp4_report.py"))
cp4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp4)
with tempfile.TemporaryDirectory() as td:
    csv_path = os.path.join(td, "eval_x.csv")
    with open(csv_path, "w") as f:
        f.write("Task, Accuracy, Centered, NLL\n")
        f.write("STEM, , 0.175165, nan\n")
        f.write("arc_easy, 0.752525, 0.670034, 2.270158\n")
        f.write("arc_challenge, 0.458200, 0.277600, 3.100000\n")
        f.write("mmlu_stem, 0.292500, 0.056700, nan\n")
    parsed = cp4.parse_eval_csv(csv_path)
    _exp7 = (2.270158 * 2376 + 3.100000 * 1172) / (2376 + 1172)
    check("cp4: STEM NLL nan -> per-task N-weighted mean",
          parsed["stem_nll"] is not None
          and abs(parsed["stem_nll"] - _exp7) < 1e-9,
          f"got={parsed['stem_nll']} expected={_exp7:.4f}")
    check("cp4: stem centered parsed alongside", parsed["stem"] == 0.175165)
    csv2 = os.path.join(td, "eval_ok.csv")
    with open(csv2, "w") as f:
        f.write("STEM, , 0.100000, 2.500000\n")
        f.write("arc_easy, 0.700000, 0.600000, 2.200000\n")
    parsed2 = cp4.parse_eval_csv(csv2)
    check("cp4: finite STEM NLL passes through unchanged",
          parsed2["stem_nll"] == 2.5)
    csv3 = os.path.join(td, "eval_dead.csv")
    with open(csv3, "w") as f:
        f.write("STEM, , 0.100000, nan\n")
        f.write("arc_easy, 0.700000, 0.600000, nan\n")
    parsed3 = cp4.parse_eval_csv(csv3)
    check("cp4: no finite per-task NLL -> None (report skips)",
          parsed3["stem_nll"] is None)

# ── 8. dispatch scripts bootstrap vendored climbmix-ma ────────────────────
for _script in ("dispatch_target_arm.py", "dispatch_remote.py"):
    _src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", _script)).read()
    check(f"dispatch bootstrap: {_script} self-adds climbmix-ma to sys.path",
          '"climbmix-ma"' in _src and "sys.path" in _src)

# ── summary ───────────────────────────────────────────────────────────────
print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
