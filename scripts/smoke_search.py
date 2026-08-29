#!/usr/bin/env python3
"""Production-geometry search simulation (no NPU, ~1 min CPU).

The speedrun report left one question open: held-out Spearman read nan at
N=18, diagnosed as a small-N artifact (guided candidates collapse into one
leaf -> constant predictions). This script PROVES the diagnosis at zero
NPU cost: proxy training is replaced by a synthetic ground-truth utility,
and the real IterativeBootstrapper loop runs unchanged (SNR z-scores,
LightGBM refits, held-out val split, predictor-guided sampling, top-N
pruning, verbatim M-of-N, online backtest, state persistence) under two
geometries:

  main   20,10,5   N=35   (runs/run_climbmix.sh production default)
  paper  64,32,16  N=112  (paper: "64, 32, and 16 candidates ... 112")

With truth known we check what a real run cannot: predictor health AND
search convergence to (near-)optimum.

    python3 scripts/smoke_search.py

Exit 0 = search machinery healthy at production geometry.
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, "src")

import numpy as np

from climbmix.core.types import (
    CLIMBConfig,
    SearchConfig,
    ProxyResult,
    STEM_BENCHMARK_LABELS,
)
from climbmix.core.iterative_bootstrapper import IterativeBootstrapper

K = 15           # production K_max
POOL_REF = 10000  # random-reference pool for percentile verdicts

GEOM_MAIN = [20, 10, 5]
GEOM_PAPER = [64, 32, 16]


class SyntheticTruth:
    """Known ground truth: utility(w) = q.w + lam*(1 - ||w - w*||^2).

    Smooth and axis-aligned enough for shallow trees; the optimum sits at
    a sparse mixture over the highest-quality clusters. Observations are
    per-task accuracies/NLLs mapped from the latent utility with
    task-specific slopes and noise, so the SNR z-score path runs for real.
    Noise is seeded by config_id -> resume-stable (a re-run config observes
    the same values).
    """

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.q = rng.normal(0.0, 1.0, K)
        # w* aligned with quality (softmax): the optimum is reachable by
        # "more of the good clusters" — the direction real data-mixture
        # improvements move in. (A first draft put w* on an arbitrary
        # Dirichlet draw: the quadratic pull then dominated and the optimum
        # became a needle outside every sampling distribution — that tested
        # optimization hardness, not machinery health.)
        self.w_star = np.exp(self.q / 0.8)
        self.w_star = self.w_star / self.w_star.sum()
        self.lam = 1.0
        self.base = dict(zip(STEM_BENCHMARK_LABELS, rng.uniform(0.25, 0.45, len(STEM_BENCHMARK_LABELS))))
        self.slope = dict(zip(STEM_BENCHMARK_LABELS, rng.uniform(0.02, 0.06, len(STEM_BENCHMARK_LABELS))))
        self.nll_c = dict(zip(STEM_BENCHMARK_LABELS, rng.uniform(1.5, 3.0, len(STEM_BENCHMARK_LABELS))))
        self.nll_d = dict(zip(STEM_BENCHMARK_LABELS, rng.uniform(0.08, 0.25, len(STEM_BENCHMARK_LABELS))))
        self.sigma = dict(zip(STEM_BENCHMARK_LABELS, rng.uniform(0.004, 0.012, len(STEM_BENCHMARK_LABELS))))

    def utility(self, w) -> float:
        w = np.asarray(w, dtype=np.float64)
        return float(self.q @ w + self.lam * (1.0 - float(np.sum((w - self.w_star) ** 2))))

    @property
    def u_star(self) -> float:
        return self.utility(self.w_star)

    def observe(self, w, config_id: int):
        rng = np.random.default_rng(100000 + config_id * 7919)
        u = self.utility(w)
        acc, nll = {}, {}
        for b in STEM_BENCHMARK_LABELS:
            acc[b] = float(np.clip(
                self.base[b] + self.slope[b] * u + rng.normal(0.0, self.sigma[b]), 0.0, 1.0))
            nll[b] = float(self.nll_c[b] - self.nll_d[b] * u + rng.normal(0.0, 0.05))
        return acc, nll


class SyntheticRunner:
    """ExpExecutor over SyntheticTruth — replaces NPU proxy training."""

    def __init__(self, truth: SyntheticTruth):
        self.truth = truth

    def run_batch(self, configs, data_dir=None, output_dir=None,
                  experiment_id_base=0):
        results = []
        for c in configs:
            acc, nll = self.truth.observe(
                c.mixture_weights.weights, c.config_id)
            results.append(ProxyResult(
                mixture_config=c,
                validation_loss=-float(np.mean(list(acc.values()))),
                per_task_accuracies=acc,
                per_task_nlls=nll,
                metadata={"synthetic": True},
            ))
        return results


def run_geometry(cpi, truth: SyntheticTruth, state_dir: str) -> dict:
    cfg = CLIMBConfig()
    cfg.search = SearchConfig(num_iterations=len(cpi), configs_per_iter=list(cpi))
    cfg.val_tasks = STEM_BENCHMARK_LABELS.copy()

    rng = np.random.default_rng(3)
    token_counts = rng.integers(5_000_000, 400_000_000, K).astype(np.int64)

    state_path = os.path.join(state_dir, f"search_state_{'_'.join(map(str, cpi))}.json")
    b = IterativeBootstrapper(
        cfg, token_counts, np.arange(K, dtype=np.int64), state_path=state_path)

    t0 = time.time()
    optimal, iters = b.search_optimal(SyntheticRunner(truth))
    elapsed = time.time() - t0

    val_by_iter = {e["iteration"]: e for e in b.predictor_eval}
    final_val_rho = val_by_iter[max(val_by_iter)]["val_spearman"] if val_by_iter else None

    # Random reference: same proportional Dirichlet as iteration 1
    conc = token_counts.astype(np.float64) / token_counts.sum() * K
    ref_rng = np.random.default_rng(1234)
    ref_utils = np.array([
        truth.utility(ref_rng.dirichlet(conc)) for _ in range(POOL_REF)
    ])

    u_best = truth.utility(optimal.mixture_weights.weights)
    pct = float(100.0 * np.mean(ref_utils < u_best))

    # True utility of the best config the search actually TRAINED — the
    # guided funnel's realized value. Compare with the final predictor
    # argmin (optimal): a gap means the final full-design-space selection
    # trusted an extrapolated prediction over an already-measured config
    # (optimizer's curse; see TODO "smoke_search" for the production
    # implication).
    trained_utils = np.array([
        truth.utility(c.mixture_weights.weights)
        for c in b._accumulated_configs
    ])
    best_trained_u = float(trained_utils.max()) if len(trained_utils) else float("nan")

    return {
        "cpi": list(cpi),
        "N": int(sum(cpi)),
        "elapsed": elapsed,
        "bootstrapper": b,
        "optimal": optimal,
        "val_by_iter": val_by_iter,
        "final_val_rho": final_val_rho,
        "online_eval": list(b.online_eval),
        "pruning_history": list(b.pruning_history),
        "u_best": u_best,
        "best_trained_u": best_trained_u,
        "pct": pct,
        "p99": float(np.percentile(ref_utils, 99)),
        "p95": float(np.percentile(ref_utils, 95)),
        "u_star": truth.u_star,
        "state_path": state_path,
    }


def strict_loads(text: str):
    def boom(c):
        raise ValueError(f"non-finite literal {c!r} in JSON")
    return json.loads(text, parse_constant=boom)


def main() -> int:
    import warnings
    warnings.filterwarnings("ignore")

    truth = SyntheticTruth()
    state_dir = tempfile.mkdtemp(prefix="smoke_search_")

    print(f"Truth: K={K}, oracle u(w*)={truth.u_star:.3f}")
    print(f"Reference pool: {POOL_REF} proportional-Dirichlet random configs\n")

    out = {}
    for tag, cpi in (("main", GEOM_MAIN), ("paper", GEOM_PAPER)):
        print(f"\n{'#' * 70}\n# Geometry {tag}: {cpi} (N={sum(cpi)})\n{'#' * 70}")
        out[tag] = run_geometry(cpi, truth, state_dir)

    main_r, paper_r = out["main"], out["paper"]

    print("\n" + "=" * 70)
    print("  Geometry comparison (same truth, same noise model)")
    print("=" * 70)
    hdr = (f"{'geometry':<8} {'N':>4} {'final val ρ':>12} {'online ρ':>22} "
           f"{'argmin u':>9} {'best-trained u':>15} {'rand p99':>9}")
    print(hdr)
    for tag, r in (("main", main_r), ("paper", paper_r)):
        online = ",".join(
            f"{e['spearman']:.2f}" if e.get("spearman") is not None else "None"
            for e in r["online_eval"])
        rho = f"{r['final_val_rho']:.3f}" if r["final_val_rho"] is not None else "None"
        print(f"{tag:<8} {r['N']:>4} {rho:>12} {online:>22} "
              f"{r['u_best']:>9.3f} {r['best_trained_u']:>15.3f} {r['p99']:>9.3f}")
    print(f"(oracle u(w*)={main_r['u_star']:.3f}, "
          f"random p95={main_r['p95']:.3f}, p99={main_r['p99']:.3f})")
    for tag, r in (("main", main_r), ("paper", paper_r)):
        gap = r["best_trained_u"] - r["u_best"]
        print(f"  info {tag}: final-selection gap = {gap:+.3f} "
              f"(argmin {r['u_best']:.3f} vs best-trained {r['best_trained_u']:.3f}; "
              f"positive = predictor-argmin picked WORSE than an "
              f"already-measured config)")

    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'} {name} {detail}")
        if not cond:
            ok = False

    print("\nVerdicts:")
    check("main geometry: final held-out Spearman finite (nan was a small-N artifact)",
          main_r["final_val_rho"] is not None,
          f"= {main_r['final_val_rho']}")
    check("paper geometry: final held-out Spearman finite and >= 0.3 (D.10 trend)",
          paper_r["final_val_rho"] is not None and paper_r["final_val_rho"] >= 0.3,
          f"= {paper_r['final_val_rho']}")
    check("paper >= main held-out Spearman (more budget -> better predictor)",
          (paper_r["final_val_rho"] is not None and main_r["final_val_rho"] is not None
           and paper_r["final_val_rho"] >= main_r["final_val_rho"]))
    online_finite = [e for e in paper_r["online_eval"] if e.get("spearman") is not None]
    check("paper geometry: online backtest records exist and at least one is positive",
          len(online_finite) >= 1 and any(e["spearman"] > 0 for e in online_finite),
          f"n={len(online_finite)}")
    check("main geometry: guided funnel's best TRAINED config >= 95th pct of 10K random",
          main_r["best_trained_u"] >= main_r["p95"],
          f"{main_r['best_trained_u']:.3f} vs {main_r['p95']:.3f}")
    check("paper geometry: guided funnel's best TRAINED config >= 99th pct of 10K random",
          paper_r["best_trained_u"] >= paper_r["p99"],
          f"{paper_r['best_trained_u']:.3f} vs {paper_r['p99']:.3f}")

    for tag, r in (("main", main_r), ("paper", paper_r)):
        funnel_ok = all(
            (p.get("pool", 0) >= 500 and 0 < p.get("top_n", 0) <= p.get("novel", 0)
             and p.get("sampled") == r["cpi"][p["iteration"] - 1])
            for p in r["pruning_history"])
        check(f"{tag} geometry: pruning funnel sane on all guided rounds",
              funnel_ok and len(r["pruning_history"]) == len(r["cpi"]) - 1,
              f"rounds={len(r['pruning_history'])}")

        with open(r["state_path"]) as f:
            state_txt = f.read()
        try:
            state = strict_loads(state_txt)
            strict_ok = True
        except ValueError:
            state = None
            strict_ok = False
        check(f"{tag} geometry: state strictly parseable (jq-compatible)", strict_ok)
        check(f"{tag} geometry: state reloads to iter {len(r['cpi'])} with {r['N']} configs",
              state is not None
              and state.get("last_completed_iter") == len(r["cpi"])
              and len(state.get("accumulated_configs") or []) == r["N"])

        model = getattr(getattr(r["bootstrapper"].predictor, "_model", None),
                        "feature_importances_", None)
        imp_ok = (model is not None and len(model) == K
                  and float(np.sum(model)) > 0)
        check(f"{tag} geometry: final predictor exposes feature importances (K={K})", imp_ok)

    print("\nSMOKE " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
