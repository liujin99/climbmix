#!/usr/bin/env python3
"""inject_history — warm-start a new run with points measured by old runs.

docs/reuse_design.md §4.2. Cross-run d20 reuse: the new run's
search_state.json is seeded with the old runs' RAW measurements
(per-benchmark acc/NLL + weights), placed as the initial observation pool
("iteration 1"). The new run then:

  - refits its predictor over history + new points at resume
    (run_climbmix.sh always passes --resume-search),
  - samples iteration 2+ predictor-guided, deduped against history
    (run_iteration's existing_flats),
  - allocates exp ids from len(accumulated) onward — no collisions,
  - attributes history to iteration 1 in reports and crash-resumes
    (realized_configs_per_iter = [N_history]).

Credentials checked HERE (the immutable layer, docs §2): K self-consistency
per source, K agreement across sources, and (with --pool) the new run's
cluster_cache.npz must have the same K. The pool itself must be COPIED from
the old run, never regenerated. Training/eval protocol equality remains the
operator's checklist (docs §8) — the new run's fingerprint machinery guards
it from the other side.

NEVER overwrites real progress: refuses an existing search_state.json unless
--force. Original source states are read-only.

    python3 scripts/inject_history.py \
        --source result/prod2_.../search_state.json \
        --target-dir result/prod3_current \
        [--pool result/prod3_current/cluster_cache.npz] \
        [--dry-run] [--top 10]

CONFIGS_PER_ITER of the new launch counts the history as slot 1:
history 30 + new 60 => CONFIGS_PER_ITER="30,20,10".
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from climbmix.core.types import SearchConfig  # noqa: E402
from climbmix.utils.io_utils import atomic_write_json  # noqa: E402

import rescore_search as rs  # noqa: E402  (sibling script, shares the path)


def _measured(point) -> bool:
    """A point carries information iff at least one finite acc measurement
    exists (failed experiments stored {} / None — zero information, and they
    would waste exp-id space and realized bookkeeping in the new run)."""
    acc = point["acc"]
    return isinstance(acc, dict) and any(
        v is not None and np.isfinite(v) for v in acc.values())


def _pool_k_and_sha(pool_path: str):
    try:
        labels = np.load(pool_path, allow_pickle=False)["final_labels"]
    except KeyError:
        raise SystemExit(
            f"✗ {pool_path}: no 'final_labels' key — this is not a "
            f"cluster_cache.npz from the search stage")
    k = int(np.unique(labels[labels >= 0]).size)
    h = hashlib.sha256()
    with open(pool_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return k, h.hexdigest()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", action="append", required=True,
                   help="source run's search_state.json (repeat to merge)")
    p.add_argument("--target-dir", required=True,
                   help="new run's output dir (created if missing)")
    p.add_argument("--pool", default="",
                   help="cluster_cache.npz shared by source and target runs "
                        "(the copied, NOT regenerated, cache) — K is "
                        "validated and its sha256 recorded")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing target search_state.json")
    p.add_argument("--dry-run", action="store_true",
                   help="report and validate without writing")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--w-floor", type=float, default=SearchConfig().w_floor)
    args = p.parse_args()

    # ── load sources ──
    per_source = []
    k_expected = None
    for src in args.source:
        k, points = rs.load_state_points(src)
        if k_expected is None:
            k_expected = k
        elif k != k_expected:
            raise SystemExit(
                f"✗ K mismatch across sources: {args.source[0]} has K="
                f"{k_expected}, {src} has K={k} — different search spaces, "
                f"not mergeable")
        per_source.append((src, points))
        print(f"[source] {src}: K={k}, {len(points)} points")

    # ── pool credential ──
    pool_k = pool_sha = None
    if args.pool:
        pool_k, pool_sha = _pool_k_and_sha(args.pool)
        print(f"[pool] {args.pool}: K={pool_k}, sha256={pool_sha[:16]}…")
        if pool_k != k_expected:
            raise SystemExit(
                f"✗ pool K={pool_k} != source-state K={k_expected} — the "
                f"target run's cluster space differs from the runs that "
                f"produced these measurements; injection would be invalid")

    # ── select + dedup (in CLI order; first occurrence wins) ──
    selected = []
    seen = {}
    n_unmeasured = n_dup = 0
    for src, points in per_source:
        kept_from_src = 0
        for pt in points:
            if not _measured(pt):
                n_unmeasured += 1
                continue
            key = tuple(np.round(np.asarray(pt["weights"]), 4))
            if key in seen:
                n_dup += 1
                continue
            seen[key] = True
            selected.append({**pt, "source": src})
            kept_from_src += 1
        print(f"[select] {src}: kept {kept_from_src}/{len(points)} "
              f"(dropped unmeasured/uninformative, dups across sources)")
    if not selected:
        print("✗ no measured points left after selection — nothing to inject")
        return 1
    n = len(selected)
    print(f"[select] total: {n} history points "
          f"({n_unmeasured} unmeasured dropped, {n_dup} duplicates dropped)")

    # ── rescore with the current formula (same path as rescore_search) ──
    benchmarks = rs.benchmark_union(selected)
    if not benchmarks:
        print("✗ union of measured benchmarks is empty")
        return 1
    scores, task_f = rs.rescore(selected, benchmarks, args.w_floor)
    commit = rs._scoring_commit()
    print(f"\n── rescore under current formula (commit {commit}, "
          f"w_floor={args.w_floor})")
    for b in benchmarks:
        f = task_f.get(b)
        if f is not None:
            print(f"   {b:16s} f={f:+.3f}  w={rs.w_from_f(f, args.w_floor):.3f}")
    order = sorted(
        range(n),
        key=lambda i: (np.isfinite(scores[i]),
                       float(scores[i]) if np.isfinite(scores[i]) else 0.0),
        reverse=True,
    )
    print(f"   top-{min(args.top, n)} history points:")
    for rank, i in enumerate(order[:max(1, args.top)], 1):
        print(f"     {rank:2d}. config {selected[i]['config_id']:>4}  "
              f"score {float(scores[i]):+.4f}")

    # ── target safety ──
    target_state = os.path.join(args.target_dir, "search_state.json")
    if os.path.isfile(target_state) and not args.force:
        raise SystemExit(
            f"✗ {target_state} already exists — refusing to clobber "
            f"(move it away, or pass --force if you really mean it)")
    if os.path.isdir(args.target_dir):
        others = [f for f in (".fingerprint_search", "search.log",
                              "optimal_mixture_weights.json")
                  if os.path.exists(os.path.join(args.target_dir, f))]
        if others:
            print(f"⚠ target dir already holds run products ({', '.join(others)}) "
                  f"— history seed assumes a FRESH run dir")

    # ── build + write the seed ──
    seed = {
        "last_completed_iter": 1,
        "n_clusters": k_expected,
        "accumulated_scores": [
            float(s) if np.isfinite(s) else None for s in scores],
        "accumulated_configs": [
            {"weights": pt["weights"], "config_id": pt["config_id"]}
            for pt in selected
        ],
        "accumulated_per_benchmark": [
            {"acc": pt["acc"], "nll": pt["nll"]} for pt in selected
        ],
        # Old-run diagnostics are deliberately DROPPED (docs §4.2): their
        # iteration numbering belongs to the source run's sampling path and
        # would mislead the new run's reports. Provenance lives below.
        "predictor_eval": [],
        "online_eval": [],
        "pruning_history": [],
        "pending": None,
        "realized_configs_per_iter": [n],
        "last_c_eff": None,
        "history_seed": {
            "source_runs": [os.path.basename(os.path.dirname(s))
                            for s in args.source],
            "source_states": [os.path.abspath(s) for s in args.source],
            "n_points": n,
            "n_dropped_unmeasured": n_unmeasured,
            "n_dropped_duplicates": n_dup,
            "n_clusters": k_expected,
            "benchmarks": benchmarks,
            "pool_sha256": pool_sha,
            "pool_k": pool_k,
            "scoring_commit": commit,
            "w_floor": args.w_floor,
            "injected_at": datetime.datetime.now().isoformat(
                timespec="seconds"),
        },
    }

    if args.dry_run:
        print(f"\n[dry-run] would write {target_state} "
              f"({n} points as iteration 1, K={k_expected})")
        return 0

    os.makedirs(args.target_dir, exist_ok=True)
    atomic_write_json(target_state, seed, indent=2)
    print(f"\n✓ seed written → {target_state}")
    print(f"  iteration 1 = {n} history points; the new run starts at "
          f"iteration 2 with predictor-guided sampling")
    print(f"  launch with CONFIGS_PER_ITER counting history in slot 1 "
          f"(e.g. \"{n},20,10\" for {n}+30 total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
