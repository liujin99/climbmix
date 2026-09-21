#!/usr/bin/env python3
"""Post-hoc predictor audit (B2/B3/B4 of docs/algorithm_review.md §2.4).

Replays prod4's LightGBM predictor on the persisted search fleet (weights +
final rescored utility, search_state.json -> data/prod4_search_dump.csv)
with zero NPU cost and answers:

  B4  margin calibration — reproduce the final split fit, run the A2
      refit-on-full, and check the D19 no-claim counterfactual at the
      actual extrapolated winner: would the guard have fired?
  B3  feature-importance artifact — split counts vs gain vs mean|SHAP|
      (does the "C5/C12 divergence" survive honest re-scoring?)
  B2  ranking/pruning audit — where the final models re-rank the measured
      fleet relative to the recorded guided-round cutoffs; top-N predicted
      vs top-N actual overlap; top-k arm preview (k=3, L1>=0.15).
  A2  5-fold CV: split-fit protocol vs refit-on-full protocol, pooled
      out-of-fold Spearman / R2 (does the training-tax recovery actually
      predict better?).

Determinism: LightGBM random_state=42, split rng(42) — mirrors
_refit_predictor exactly; a faithful replay reproduces prod4's terminal
val R2=0.3204 / rho=0.5008 to rounding.

Usage:
  python3 scripts/diagnostics/predictor_audit.py \
    --csv scripts/diagnostics/data/prod4_search_dump.csv \
    --winner-weights "0.009,...,0.4737,..." \
    --margin 0.4522 \
    --cutoffs "-0.0388,-0.2989" \
    --round-bounds "54,92"
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np

from climbmix.core.types import MixtureConfig, MixtureWeights, PredictorConfig
from climbmix.core.predictor import LightGBMPredictor


def spearman(a, b):
    return LightGBMPredictor._spearman(np.asarray(a, float), np.asarray(b, float))


def load_fleet(path):
    rows = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)  # header
        for line in r:
            if not line:
                continue
            vals = [float(x) for x in line]
            rows.append((vals[:15], int(vals[15]), vals[16]))
    configs = [MixtureConfig(mixture_weights=MixtureWeights(weights=np.array(w)),
                             config_id=cid) for w, cid, _ in rows]
    scores = np.array([s for _, _, s in rows])
    ids = [cid for _, cid, _ in rows]
    return configs, scores, ids


def split_fit(configs, targets):
    """Mirror _refit_predictor: rng(42) permutation, last 20% (>=5) out."""
    n = len(configs)
    n_val = max(5, int(n * 0.2))
    perm = np.random.default_rng(42).permutation(n)
    tr, va = perm[:n - n_val], perm[n - n_val:]
    p = LightGBMPredictor(len(configs[0].flatten()), PredictorConfig())
    p.fit([configs[i] for i in tr], targets[tr],
          val_configs=[configs[i] for i in va], val_losses=targets[va])
    return p, tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--winner-weights", required=True,
                    help="comma-separated K weights of the extrapolated selection")
    ap.add_argument("--margin", type=float, required=True,
                    help="D19 auto margin from the run's last-iter predictor_eval")
    ap.add_argument("--cutoffs", default="",
                    help="per-guided-round top-N cutoffs (target space), comma-separated")
    ap.add_argument("--round-bounds", default="",
                    help="fleet index where each guided round starts, comma-separated"
                         " (e.g. '54,92' = rounds 2,3)")
    ap.add_argument("--cv", type=int, default=5, help="CV folds (0 disables)")
    args = ap.parse_args()

    configs, scores, ids = load_fleet(args.csv)
    targets = -scores  # metric_direction = maximize (accuracy)
    n = len(configs)
    print(f"fleet: {n} configs, K={len(configs[0].flatten())}, "
          f"best measured = cfg{ids[int(np.argmax(scores))]} ({scores.max():.4f})")

    # ── B4a: reproduce the terminal split fit ──────────────────────────
    print("\n=== B4a: split-fit reproduction (rng 42, 20% out) ===")
    p_split, tr, va = split_fit(configs, targets)
    print(f"train {len(tr)} / val {len(va)}; val R2={p_split.val_r2_:.4f} "
          f"rho={p_split.val_spearman_:.4f}  (prod4 terminal: 0.3204 / 0.5008)")
    best_iter = getattr(p_split._model, "best_iteration_", None)
    print(f"early stopping: best_iteration={best_iter}/500")

    # ── B4b: A2 refit-on-full + no-claim counterfactual ─────────────────
    print("\n=== B4b: A2 refit-on-full + no-claim counterfactual ===")
    p_full = p_split.refit_on_full(configs, targets)
    winner = MixtureConfig.from_flattened(np.array(
        [float(x) for x in args.winner_weights.split(",")]))
    for name, model in (("split", p_split), ("full", p_full)):
        pred = float(model.predict([winner])[0])
        print(f"winner predicted utility ({name} model): {-pred:+.4f}")
    pred_full = float(p_full.predict([winner])[0])
    pred_split = float(p_split.predict([winner])[0])
    best_actual = float(scores.max())
    claimed_full = -pred_full - best_actual
    claimed_split = -pred_split - best_actual
    print(f"best measured actual: {best_actual:+.4f}")
    print(f"claimed gain vs margin: split {claimed_split:+.4f} / "
          f"full {claimed_full:+.4f} vs {args.margin:.4f} -> "
          f"guard fires: split={claimed_split <= args.margin}, "
          f"full={claimed_full <= args.margin}")
    # Context: what the same models predict at the top MEASURED points
    top2 = list(np.argsort(-scores)[:2])
    for i in top2:
        ps = float(p_split.predict([configs[i]])[0])
        pf = float(p_full.predict([configs[i]])[0])
        print(f"cfg{ids[i]} (actual {scores[i]:+.4f}): predicted utility "
              f"split {-ps:+.4f} / full {-pf:+.4f}")

    # ── B3: importances — split counts vs gain vs mean|SHAP| ───────────
    print("\n=== B3: feature importance, three scorings ===")
    imp_split = np.asarray(p_split._model.feature_importances_, float)
    gain = np.asarray(p_full._model.booster_.feature_importance(
        importance_type="gain"), float)
    X_full = np.array([c.flatten() for c in configs])
    shap = np.abs(p_full._model.predict(X_full, pred_contrib=True)[:, :-1])
    shap_mean = shap.mean(axis=0)
    labels = [f"C{i}" for i in range(15)]

    def share_table(v, title):
        s = 100.0 * v / max(v.sum(), 1e-12)
        order = np.argsort(-s)
        top = " ".join(f"{labels[i]} {s[i]:.1f}%" for i in order[:6])
        print(f"{title}: {top}")
        return s

    s_split = share_table(imp_split, "split counts (split model)")
    s_gain = share_table(gain, "gain          (full model)")
    s_shap = share_table(shap_mean, "mean|SHAP|     (full model)")
    for c in ("C5", "C12", "C10"):
        i = labels.index(c)
        print(f"  {c}: split {s_split[i]:.1f}% / gain {s_gain[i]:.1f}% / "
              f"shap {s_shap[i]:.1f}%")

    # ── B2: ranking / pruning audit ─────────────────────────────────────
    print("\n=== B2: ranking / pruning audit ===")
    preds_split = p_split.predict(configs)
    preds_full = p_full.predict(configs)
    rho_models = spearman(preds_split, preds_full)
    print(f"model agreement (split vs full on the fleet): rho={rho_models:.4f}")
    for name, preds in (("split", preds_split), ("full", preds_full)):
        top_pred = {ids[i] for i in np.argsort(preds)[:10]}
        top_act = {ids[i] for i in np.argsort(-scores)[:10]}
        print(f"top-10 predicted vs top-10 actual overlap ({name}): "
              f"{len(top_pred & top_act)}/10 -> {sorted(top_pred)}")
    if args.cutoffs:
        cutoffs = [float(x) for x in args.cutoffs.split(",")]
        starts = [int(x) for x in args.round_bounds.split(",")] \
            if args.round_bounds else []
        # starts[i] = fleet row where guided round i+2 begins (round 1 is
        # the history injection, never cutoff-filtered)
        for si, lo in enumerate(starts):
            if si >= len(cutoffs):
                break
            hi = starts[si + 1] if si + 1 < len(starts) else n
            cut = cutoffs[si]
            # target space: lower = better; the band is pred <= cutoff
            pruned = [ids[i] for i in range(lo, hi) if preds_full[i] > cut]
            print(f"round {si + 2}: fleet rows [{lo},{hi}) cutoff {cut:+.4f}; "
                  f"final-model re-rank: {len(pruned)}/{hi - lo} measured "
                  f"points now OUTSIDE the band (pred > cutoff), "
                  f"e.g. {pruned[:8]}")
    # top-k arm preview (actual scores, greedy L1 >= 0.15, k = 3)
    order = list(np.argsort(-scores))
    sel = []
    for i in order:
        if len(sel) >= 3:
            break
        if all(np.abs(configs[i].flatten() - configs[s].flatten()).sum() >= 0.15
               for s in sel):
            sel.append(int(i))
    print("top-k=3 arm preview (actual scores, L1>=0.15): "
          + ", ".join(f"cfg{ids[i]}({scores[i]:+.4f})" for i in sel))

    # ── A2 CV: split-fit protocol vs refit-on-full protocol ─────────────
    if args.cv:
        print(f"\n=== A2 CV: {args.cv}-fold, split-fit vs refit-on-full ===")
        rng = np.random.default_rng(7)
        fold_idx = rng.permutation(n)
        folds = np.array_split(fold_idx, args.cv)
        out_a, out_b, truth = [], [], []
        for k, te in enumerate(folds):
            trn = np.setdiff1d(fold_idx, te)
            cfg_tr = [configs[i] for i in trn]
            # protocol A: internal split for early stopping (mirror prod)
            pa, _, _ = split_fit(cfg_tr, targets[trn])
            out_a.extend(pa.predict([configs[i] for i in te]).tolist())
            # protocol B: A2-style — same tree count, refit on ALL train
            pb = pa.refit_on_full(cfg_tr, targets[trn])
            out_b.extend(pb.predict([configs[i] for i in te]).tolist())
            truth.extend(targets[te].tolist())
            print(f"  fold {k + 1}: n_train={len(trn)} n_test={len(te)}")
        for name, out in (("split-fit", out_a), ("refit-on-full", out_b)):
            o = np.asarray(out, dtype=np.float64)
            t = np.asarray(truth, dtype=np.float64)
            r2 = 1.0 - float(((o - t) ** 2).sum()) / float(
                ((t - t.mean()) ** 2).sum())
            print(f"{name}: pooled out-of-fold rho={spearman(out, truth):.4f} "
                  f"R2={r2:.4f}")


if __name__ == "__main__":
    main()
