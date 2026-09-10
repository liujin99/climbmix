#!/usr/bin/env python3
"""rescore — recompute derived scores from a search_state's RAW measurements.

docs/reuse_design.md §4.1. The search state stores RAW per-benchmark
measurements (accumulated_per_benchmark) as immutable facts and scores as
DERIVED values (raw x scoring-formula version). When the scoring formula
changes (bug fix — e.g. the 2026-09-10 centered-unit f noise floor — or a
design change), the derived layer can be recomputed without touching any
NPU: this tool rebuilds the scoring path from the current code and reports
what the ranking WOULD be.

Read-only w.r.t. the source state: results go to a sidecar
`<state>.rescored.json` (formula version = git SHA of this checkout).

    python3 scripts/rescore_search.py --state result/<run>/search_state.json \
        [--top 10] [--w-floor 0.05] [--benchmarks arc_easy,...]

Exit 0 = rescored (sidecar written); 1 = unusable state.
"""
import argparse
import json
import os
import subprocess
import sys
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import numpy as np  # noqa: E402

from climbmix.core.types import (  # noqa: E402
    CLIMBConfig, SearchConfig, STEM_BENCHMARK_LABELS)
from climbmix.core.iterative_bootstrapper import IterativeBootstrapper  # noqa: E402
from climbmix.utils.io_utils import atomic_write_json  # noqa: E402


def _scoring_commit() -> str:
    repo = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    try:
        sha = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        return sha or "unknown"
    except Exception:
        return "unknown"


def _restore_pb(v):
    """State-JSON -> in-memory form (None inside a dict was a non-finite
    float before the write sanitized it — back to nan, matching
    IterativeBootstrapper._load_state)."""
    if isinstance(v, dict):
        return {k: (float("nan") if x is None else float(x))
                for k, x in v.items()}
    return v


def load_state_points(state_path: str):
    """(K, points) where each point is a dict with weights / config_id /
    acc / nll / old_score. Points appear in the state's original order."""
    with open(state_path) as f:
        state = json.load(f)

    configs = state.get("accumulated_configs") or []
    per_bench = state.get("accumulated_per_benchmark") or []
    old_scores = state.get("accumulated_scores") or []
    if not configs or len(configs) != len(per_bench):
        raise SystemExit(
            f"✗ {state_path}: no accumulated measurements "
            f"({len(configs)} configs vs {len(per_bench)} per-benchmark rows)")

    points = []
    for i, (c, d) in enumerate(zip(configs, per_bench)):
        weights = [float(x) for x in c["weights"]]
        points.append({
            "config_id": c.get("config_id", i),
            "weights": weights,
            "acc": _restore_pb(d.get("acc")),
            "nll": _restore_pb(d.get("nll")),
            "old_score": (float(old_scores[i])
                          if i < len(old_scores) and old_scores[i] is not None
                          else None),
        })
    k = len(points[0]["weights"])
    bad = [i for i, p in enumerate(points) if len(p["weights"]) != k]
    if bad:
        raise SystemExit(f"✗ {state_path}: config weight vectors disagree on K "
                         f"(first offender index {bad[0]})")
    declared = state.get("n_clusters")
    if declared is not None and int(declared) != k:
        raise SystemExit(
            f"✗ {state_path}: n_clusters={declared} but weights have K={k} "
            f"— state is internally inconsistent")
    return k, points


def benchmark_union(points):
    """Measured-benchmark union in stable order: STEM canonical order first,
    then extras sorted (so a re-eval extension appends predictably)."""
    seen = set()
    for p in points:
        acc = p["acc"]
        if isinstance(acc, dict):
            seen.update(acc.keys())
    ordered = [b for b in STEM_BENCHMARK_LABELS if b in seen]
    ordered += sorted(seen - set(ordered))
    return ordered


def rescore(points, benchmarks, w_floor):
    """Recompute scores with the CURRENT formula. Returns (scores, task_f)."""
    cfg = CLIMBConfig(
        val_tasks=list(benchmarks),
        search=SearchConfig(w_floor=w_floor),
    )
    k = len(points[0]["weights"])
    bs = IterativeBootstrapper(cfg, np.ones(k, dtype=np.int64), np.arange(k))
    bs._accumulated_per_benchmark = [(p["acc"], p["nll"]) for p in points]
    scores = bs._compute_scores()
    return scores, dict(bs._task_f)


def w_from_f(f: float, w_floor: float) -> float:
    """The _compute_scores clamp, recomputed for reporting."""
    return max(w_floor, min(1.0, max(0.0, (1.0 + f) / 2.0)))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--state", required=True, help="search_state.json to rescore")
    p.add_argument("--top", type=int, default=10,
                   help="how many ranked configs to print (default 10)")
    p.add_argument("--w-floor", type=float,
                   default=SearchConfig().w_floor,
                   help=f"SNR weight floor (default {SearchConfig().w_floor})")
    p.add_argument("--benchmarks", default="",
                   help="comma list overriding the measured-benchmark union")
    p.add_argument("--sidecar-suffix", default=".rescored.json",
                   help="written next to the state (default .rescored.json)")
    args = p.parse_args()

    k, points = load_state_points(args.state)
    benchmarks = ([b.strip() for b in args.benchmarks.split(",") if b.strip()]
                  or benchmark_union(points))
    if not benchmarks:
        print("✗ no benchmark has any measurement in this state")
        return 1

    scores, task_f = rescore(points, benchmarks, args.w_floor)

    n_measured = int(np.isfinite(scores).sum())
    if n_measured == 0:
        print("✗ no point has a finite score under the current formula")
        return 1

    commit = _scoring_commit()
    print(f"\n── rescore: {args.state}")
    print(f"   formula: commit {commit}, w_floor={args.w_floor}, "
          f"K={k}, points={len(points)} "
          f"({n_measured} scored, {len(points) - n_measured} unmeasured)")
    print(f"   benchmarks: {', '.join(benchmarks)}")
    print("   per-benchmark SNR (current formula):")
    for b in benchmarks:
        f = task_f.get(b)
        if f is None:
            print(f"     {b:16s} no valid measurements — skipped")
        else:
            print(f"     {b:16s} f={f:+.3f}  w={w_from_f(f, args.w_floor):.3f}")

    # Ranking: default metric_direction is maximize (accuracy) — higher
    # score is better. Finite only; unmeasured points sink to the bottom.
    order = sorted(
        range(len(points)),
        key=lambda i: (np.isfinite(scores[i]),
                       float(scores[i]) if np.isfinite(scores[i]) else 0.0),
        reverse=True,
    )
    top = order[:max(1, args.top)]
    print(f"\n   top-{len(top)} by current formula "
          f"(old → new; Δ vs old):")
    for rank, i in enumerate(top, 1):
        old = points[i]["old_score"]
        old_s = f"{old:+.4f}" if old is not None else "  n/a "
        new = float(scores[i])
        d = f"{new - old:+.4f}" if old is not None else "  n/a "
        print(f"     {rank:2d}. config {points[i]['config_id']:>4}  "
              f"{old_s} → {new:+.4f}  (Δ {d})")

    sidecar = {
        "source_state": os.path.abspath(args.state),
        "scoring_commit": commit,
        "rescored_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_points": len(points),
        "n_scored": n_measured,
        "n_clusters": k,
        "benchmarks": benchmarks,
        "w_floor": args.w_floor,
        "task_f": {b: float(f) for b, f in task_f.items()},
        "task_w": {b: w_from_f(f, args.w_floor) for b, f in task_f.items()},
        "points": [
            {
                "config_id": pt["config_id"],
                "old_score": pt["old_score"],
                "new_score": (float(scores[i])
                              if np.isfinite(scores[i]) else None),
            }
            for i, pt in enumerate(points)
        ],
        "ranking": [points[i]["config_id"] for i in order],
    }
    out_path = args.state + args.sidecar_suffix
    atomic_write_json(out_path, sidecar, indent=2)
    print(f"\n   sidecar → {out_path} (source state untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
