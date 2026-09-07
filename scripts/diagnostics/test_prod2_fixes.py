#!/usr/bin/env python3
"""prod2 fix verification — balanced partition / structure gate / acc-only
fallback / no-signal guard / from_dict natural sort.

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
    CLIMBConfig, ClusterInfo, MixtureWeights, MixtureConfig)
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

# ── 5. no-signal guard on final selection ─────────────────────────────────
class StubPredictor:
    def __init__(self, r2, train_r2=None):
        self.val_r2_ = r2
        self.train_r2_ = train_r2 if train_r2 is not None else r2


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

bs3._predictor = StubPredictor(0.5)
sel = bs3._select_final_mixture()
check("selection: R2>0 -> paper-faithful design-space path",
      bs3._selection_mode == "predictor_design_space" and called["design"])

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
check("selection: no val split, train R2>0 -> design-space path",
      bs5._selection_mode == "predictor_design_space" and called5["design"])

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

# ── summary ───────────────────────────────────────────────────────────────
print()
if FAILED:
    print(f"FAILED ({len(FAILED)}): {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASSED")
