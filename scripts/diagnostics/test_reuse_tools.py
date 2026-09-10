#!/usr/bin/env python3
"""Reuse-tools verification (docs/reuse_design.md): rescore_search,
inject_history, and the bootstrapper's warm-start semantics
(history_seed round-trip, guided iteration 2, exp-id continuation,
realized bookkeeping, K-mismatch discard).

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_reuse_tools.py
Exit 0 = all checks pass.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import numpy as np  # noqa: E402

from climbmix.core.types import CLIMBConfig, SearchConfig  # noqa: E402
from climbmix.core.iterative_bootstrapper import (  # noqa: E402
    IterativeBootstrapper)
import rescore_search as rs  # noqa: E402
import inject_history as ih  # noqa: E402
import prepare_random_baseline as prb  # noqa: E402

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


BENCHES = ["arc_easy", "arc_challenge"]
K = 4


def make_source_state(path, n_points=12, seed=42, stale_scores=True):
    """A synthetic but shape-faithful search_state.json: K=4, 2 benches,
    config 0 is the clear winner, one point carries a None (non-finite)
    measurement, stale old_scores."""
    rng = np.random.default_rng(seed)
    configs, per_bench, scores = [], [], []
    for i in range(n_points):
        if i == 0:
            w = np.array([0.70, 0.10, 0.10, 0.10])
            acc = {"arc_easy": 0.300, "arc_challenge": 0.280}
            nll = {"arc_easy": 1.80, "arc_challenge": 1.90}
        else:
            w = rng.dirichlet(np.ones(K))
            ae = 0.20 + 0.04 * rng.random()
            ach = 0.18 + 0.04 * rng.random()
            acc = {"arc_easy": round(ae, 4), "arc_challenge": round(ach, 4)}
            # one mid-list point: non-finite arc_challenge acc (JSON null),
            # one point with no NLL at all (acc-only path)
            if i == 5:
                acc["arc_challenge"] = None
            nll = None if i == 6 else {
                "arc_easy": round(2.4 - 0.8 * (ae - 0.20), 4),
                "arc_challenge": round(2.5 - 0.8 * (ach - 0.18), 4),
            }
        configs.append({"weights": [round(float(x), 6) for x in w],
                        "config_id": i})
        per_bench.append({"acc": acc, "nll": nll})
        scores.append(0.5 if stale_scores else None)
    state = {
        "last_completed_iter": 3,
        "n_clusters": K,
        "accumulated_scores": scores,
        "accumulated_configs": configs,
        "accumulated_per_benchmark": per_bench,
        "predictor_eval": [{"iteration": 1}],
        "online_eval": [],
        "pruning_history": [],
        "pending": None,
        "realized_configs_per_iter": [8, 2, 2],
        "last_c_eff": 7,
    }
    with open(path, "w") as f:
        json.dump(state, f)
    return state


def run_main(mod, argv):
    old = sys.argv
    sys.argv = [mod.__name__] + argv
    try:
        return mod.main()
    finally:
        sys.argv = old


def expect_exit(name, fn, needle=""):
    try:
        fn()
    except SystemExit as e:
        ok = needle in str(e)
        check(name, ok, f"exit: {str(e)[:90]}")
        return
    check(name, False, "no SystemExit raised")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="reuse_tools_")
    try:
        _rescore_tests(tmp)
        _inject_tests(tmp)
        _warmstart_tests(tmp)
        _custom_arm_tests(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print()
    if FAILED:
        print(f"✗ {len(FAILED)} check(s) failed: {FAILED}")
        return 1
    print("✓ all reuse-tools checks passed")
    return 0


def _rescore_tests(tmp):
    print("\n── rescore_search ──")
    state_path = os.path.join(tmp, "search_state.json")
    make_source_state(state_path)
    before = open(state_path, "rb").read()

    rc = run_main(rs, ["--state", state_path, "--top", "5"])
    check("rescore: exit 0", rc == 0, f"rc={rc}")
    check("rescore: source untouched", open(state_path, "rb").read() == before)

    sidecar_path = state_path + ".rescored.json"
    check("rescore: sidecar written", os.path.isfile(sidecar_path))
    sidecar = json.load(open(sidecar_path))
    check("rescore: sidecar provenance",
          bool(sidecar["scoring_commit"]) and sidecar["n_points"] == 12
          and sidecar["n_clusters"] == K)
    check("rescore: ranking best-first",
          sidecar["ranking"][0] == 0 and sidecar["ranking"][1] != 0,
          f"top of ranking: {sidecar['ranking'][:3]}")
    best = next(p for p in sidecar["points"] if p["config_id"] == 0)
    check("rescore: winner beats its stale old score",
          best["new_score"] > best["old_score"],
          f"old={best['old_score']:.3f} new={best['new_score']:.3f}")
    check("rescore: task_f covers both benches",
          set(sidecar["task_f"]) == set(BENCHES))
    f_easy = sidecar["task_f"]["arc_easy"]
    check("rescore: strong between-variance -> f > 0", f_easy > 0.0,
          f"f={f_easy:.3f}")

    # rank stability under the library API (same code path as the injector)
    _, points = rs.load_state_points(state_path)
    scores, task_f = rs.rescore(points, BENCHES, 0.0)
    check("rescore: library API agrees with CLI sidecar",
          abs(float(scores[0]) - best["new_score"]) < 1e-9)
    check("rescore: acc-only fallback keeps partially-measured point finite",
          np.isfinite(scores).all(), "all 12 scored (incl. null-acc + nll-less)")

    # unusable state -> hard exit (message + exit code 1 via SystemExit)
    bad = os.path.join(tmp, "bad.json")
    with open(bad, "w") as f:
        json.dump({"accumulated_configs": [], "accumulated_per_benchmark": []},
                  f)
    expect_exit("rescore: empty state -> hard exit",
                lambda: run_main(rs, ["--state", bad]),
                "no accumulated measurements")

    # K inconsistency -> hard error
    bad2 = os.path.join(tmp, "bad2.json")
    make_source_state(bad2)
    st = json.load(open(bad2))
    st["n_clusters"] = K + 1
    json.dump(st, open(bad2, "w"))
    expect_exit("rescore: n_clusters/weights mismatch -> SystemExit",
                lambda: rs.load_state_points(bad2), "internally inconsistent")


def _inject_tests(tmp):
    print("\n── inject_history ──")
    src1 = os.path.join(tmp, "runA", "search_state.json")
    os.makedirs(os.path.dirname(src1))
    make_source_state(src1, n_points=12, seed=42)
    src2 = os.path.join(tmp, "runB", "search_state.json")
    os.makedirs(os.path.dirname(src2))
    make_source_state(src2, n_points=6, seed=7)
    st2 = json.load(open(src2))
    st2["accumulated_configs"][0]["weights"] = json.load(open(src1))[
        "accumulated_configs"][0]["weights"]          # dup of runA winner
    st2["accumulated_per_benchmark"][1]["acc"] = {}   # unmeasured
    json.dump(st2, open(src2, "w"))

    target = os.path.join(tmp, "prod3_current")

    # pool mismatch first (K=3 pool vs K=4 states)
    pool = os.path.join(tmp, "cluster_cache.npz")
    np.savez(pool, final_labels=np.array([0, 1, 2, 2, 1]))
    expect_exit("inject: pool K mismatch -> SystemExit",
                lambda: run_main(ih, ["--source", src1, "--target-dir", target,
                                      "--pool", pool]),
                "cluster space differs")

    np.savez(pool, final_labels=np.array([0, 1, 2, 3, 3, 2, 1, 0]))
    rc = run_main(ih, ["--source", src1, "--source", src2,
                       "--target-dir", target, "--pool", pool])
    check("inject: merged sources -> exit 0", rc == 0, f"rc={rc}")

    seed_path = os.path.join(target, "search_state.json")
    seed = json.load(open(seed_path))
    # 12 + 6 - 1 dup - 1 unmeasured = 16
    check("inject: 16 points kept (1 dup + 1 unmeasured dropped)",
          len(seed["accumulated_configs"]) == 16,
          f"n={len(seed['accumulated_configs'])}")
    check("inject: history occupies iteration 1",
          seed["last_completed_iter"] == 1
          and seed["realized_configs_per_iter"] == [16]
          and seed["pending"] is None)
    check("inject: old-run diagnostics dropped",
          seed["predictor_eval"] == [] and seed["online_eval"] == []
          and seed["pruning_history"] == []
          and seed["last_c_eff"] is None)
    hs = seed["history_seed"]
    check("inject: history_seed provenance",
          hs["n_points"] == 16 and hs["n_dropped_duplicates"] == 1
          and hs["n_dropped_unmeasured"] == 1
          and hs["n_clusters"] == K and hs["pool_k"] == K
          and isinstance(hs["pool_sha256"], str) and len(hs["pool_sha256"]) == 64
          and hs["source_runs"] == ["runA", "runB"])
    check("inject: scores recomputed (not the stale 0.5)",
          any(s != 0.5 for s in seed["accumulated_scores"]))
    check("inject: strict JSON (jq-clean)",
          "NaN" not in open(seed_path).read()
          and "Infinity" not in open(seed_path).read())

    # target-exists guard + force
    expect_exit("inject: existing target state -> SystemExit",
                lambda: run_main(ih, ["--source", src1,
                                      "--target-dir", target]),
                "refusing to clobber")
    rc = run_main(ih, ["--source", src1, "--target-dir", target, "--force"])
    check("inject: --force overwrites", rc == 0
          and len(json.load(open(seed_path))["accumulated_configs"]) == 12)

    # dry-run writes nothing
    t2 = os.path.join(tmp, "dryrun_target")
    rc = run_main(ih, ["--source", src1, "--target-dir", t2, "--dry-run"])
    check("inject: dry-run writes nothing",
          rc == 0 and not os.path.exists(os.path.join(t2, "search_state.json")))

    # cross-source K mismatch
    src3 = os.path.join(tmp, "runC", "search_state.json")
    os.makedirs(os.path.dirname(src3))
    make_source_state(src3, n_points=4, seed=9)
    st3 = json.load(open(src3))
    for c in st3["accumulated_configs"]:
        c["weights"] = c["weights"] + [0.0]
    st3["n_clusters"] = 5
    json.dump(st3, open(src3, "w"))
    expect_exit("inject: cross-source K mismatch -> SystemExit",
                lambda: run_main(ih, ["--source", src1, "--source", src3,
                                      "--target-dir", os.path.join(tmp, "t3")]),
                "not mergeable")


def _warmstart_tests(tmp):
    print("\n── bootstrapper warm-start (history_seed) ──")
    src = os.path.join(tmp, "runA", "search_state.json")
    target = os.path.join(tmp, "warm_current")
    rc = run_main(ih, ["--source", src, "--target-dir", target])
    check("warmstart: seed injected", rc == 0)
    seed_path = os.path.join(target, "search_state.json")
    n_hist = len(json.load(open(seed_path))["accumulated_configs"])
    n_new = 3

    cfg = CLIMBConfig(
        val_tasks=list(BENCHES),
        search=SearchConfig(num_iterations=2,
                            configs_per_iter=[n_hist, n_new]),
    )
    bs = IterativeBootstrapper(cfg, np.ones(K, dtype=np.int64),
                               np.arange(K), state_path=seed_path)
    hist_flats = {tuple(np.round(c.mixture_weights.weights, 4))
                  for c in bs._accumulated_configs}
    optimal, iter_results = bs.search_optimal(proxy_runner=None)

    check("warmstart: accumulated = history + new",
          len(bs._accumulated_configs) == n_hist + n_new,
          f"{len(bs._accumulated_configs)} vs {n_hist}+{n_new}")
    check("warmstart: realized bookkeeping [hist, new]",
          bs._realized_configs == [n_hist, n_new],
          f"{bs._realized_configs}")
    check("warmstart: exp ids continue after history",
          [c.config_id for c in bs._accumulated_configs[n_hist:]]
          == list(range(n_hist, n_hist + n_new)))
    new_flats = {tuple(np.round(c.mixture_weights.weights, 4))
                 for c in bs._accumulated_configs[n_hist:]}
    check("warmstart: new configs deduped against history",
          not (new_flats & hist_flats))
    check("warmstart: iteration 2 was predictor-guided",
          len(bs._pruning_history) == 1
          and bs._pruning_history[0]["iteration"] == 2,
          "pruning record exists only on the guided path")
    check("warmstart: iteration_results attribution",
          len(iter_results) == 2
          and iter_results[0].n_trained == n_hist
          and iter_results[1].n_trained == n_new,
          f"{[r.n_trained for r in iter_results]}")

    # state re-saved with history_seed intact + correct iter bookkeeping
    final_state = json.load(open(seed_path))
    check("warmstart: history_seed survives re-save",
          (final_state.get("history_seed") or {}).get("n_points") == n_hist)
    check("warmstart: final state iter/realized",
          final_state["last_completed_iter"] == 2
          and final_state["realized_configs_per_iter"] == [n_hist, n_new])

    # a SECOND resume on the completed state must not double-run iterations
    bs2 = IterativeBootstrapper(cfg, np.ones(K, dtype=np.int64),
                                np.arange(K), state_path=seed_path)
    bs2.search_optimal(proxy_runner=None)
    check("warmstart: completed seed re-resumes idempotently",
          len(bs2._accumulated_configs) == n_hist + n_new
          and bs2._realized_configs == [n_hist, n_new],
          f"n={len(bs2._accumulated_configs)} realized={bs2._realized_configs}")

    # K mismatch: bootstrapper discards the seed (existing guard)
    bad_seed = os.path.join(tmp, "badseed", "search_state.json")
    os.makedirs(os.path.dirname(bad_seed))
    shutil.copy(seed_path, bad_seed)
    st = json.load(open(bad_seed))
    st["n_clusters"] = K + 1
    json.dump(st, open(bad_seed, "w"))
    bs3 = IterativeBootstrapper(cfg, np.ones(K, dtype=np.int64),
                                np.arange(K), state_path=bad_seed)
    check("warmstart: K-mismatch seed discarded by loader",
          bs3._load_state() == 0 and not bs3._accumulated_configs)


def _custom_arm_tests(tmp):
    print("\n── custom arm (prepare_random_baseline --weights + dispatch) ──")

    # ── _parse_weights unit ──
    w = prb._parse_weights("0.5,0.25,0.25", 3)
    check("weights: comma list", np.allclose(w, [0.5, 0.25, 0.25]))
    w = prb._parse_weights("2,1,1", 3)
    check("weights: normalized", np.allclose(w, [0.5, 0.25, 0.25]))
    arr_path = os.path.join(tmp, "arr.json")
    json.dump([0.2, 0.3, 0.5], open(arr_path, "w"))
    check("weights: JSON array file",
          np.allclose(prb._parse_weights(arr_path, 3), [0.2, 0.3, 0.5]))
    opt_path = os.path.join(tmp, "optimal_mixture_weights.json")
    json.dump({"C0": 0.1, "C1": 0.6, "C2": 0.3}, open(opt_path, "w"))
    check("weights: optimal_mixture_weights.json dict",
          np.allclose(prb._parse_weights(opt_path, 3), [0.1, 0.6, 0.3]))
    expect_exit("weights: K mismatch -> SystemExit",
                lambda: prb._parse_weights("0.5,0.5", 3), "dimension mismatch")
    expect_exit("weights: negative -> SystemExit",
                lambda: prb._parse_weights("1.5,-0.5,0", 3), "sum > 0")

    # ── --weights e2e: skewed plan over a real parquet pool ──
    import pyarrow as pa
    import pyarrow.parquet as pq
    import subprocess

    scripts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    repo = os.path.dirname(scripts_dir)
    prep = os.path.join(scripts_dir, "prepare_random_baseline.py")
    schema = os.path.join(repo, "config", "schema_stem.yaml")

    data_dir = os.path.join(tmp, "pool")
    os.makedirs(data_dir)
    qual_cols = ["stem_relevance", "knowledge_value", "notation_fidelity",
                 "rigor_coherence", "noise_level"]
    texts, labels = [], []
    for k, n in enumerate([300, 200, 150]):
        for i in range(n):
            prefix = f"c{k}d{i}:"
            texts.append(prefix + "x" * (400 - len(prefix)))
            labels.append(k)
    labels = np.array(labels, dtype=np.int64)
    for start in range(0, len(texts), 250):
        chunk = texts[start:start + 250]
        n = len(chunk)
        pq.write_table(pa.table({
            "category_name": ["数学"] * n,
            **{qc: pa.array([4.0] * n) for qc in qual_cols},
            "text": chunk,
            "char_count_col": pa.array([400] * n, type=pa.int64()),
            "source_record_idx": pa.array(range(n), type=pa.int64()),
        }), os.path.join(data_dir, f"part-{start // 250:05d}.parquet"))
    cache_path = os.path.join(tmp, "cluster_cache.npz")
    np.savez(cache_path, final_labels=labels)

    def run_prep(out_dir, weights):
        return subprocess.run(
            [sys.executable, prep, "--data-dir", data_dir,
             "--output-dir", out_dir, "--cluster-cache", cache_path,
             "--schema", schema, "--target-tokens", "30000",
             "--num-npu", "2", "--weights", weights],
            capture_output=True, text=True, cwd=repo)

    out_dir = os.path.join(tmp, "fixratio_shards")
    p = run_prep(out_dir, "0.6,0.2,0.2")
    check("weights: e2e run rc=0", p.returncode == 0,
          p.stderr[-300:] if p.returncode else "")
    done = json.load(open(os.path.join(out_dir, ".done")))
    check("weights: e2e doc split follows the ratio",
          done["cluster_docs"] == [180, 60, 60],
          str(done["cluster_docs"]))
    check("weights: .done records the plan",
          np.allclose(done["planned_weights"], [0.6, 0.2, 0.2]))

    # dict-form input (the winner-retrain path's natural input); 0.6 quota
    # on a 150-doc cluster (180 needed) exercises the shortfall policy
    json.dump({"C0": 0.2, "C1": 0.2, "C2": 0.6}, open(opt_path, "w"))
    out_dir2 = os.path.join(tmp, "winner_shards")
    p2 = run_prep(out_dir2, opt_path)
    check("weights: optimal-weights file e2e rc=0", p2.returncode == 0,
          p2.stderr[-300:] if p2.returncode else "")
    done2 = json.load(open(os.path.join(out_dir2, ".done")))
    check("weights: dict plan honored (with shortfall policy)",
          done2["cluster_docs"] == [60, 60, 150]
          and done2["shortfall_clusters"] == [2],
          f"{done2['cluster_docs']} shortfall={done2['shortfall_clusters']}")

    # ── dispatch --arm name validation ──
    dt = os.path.join(scripts_dir, "dispatch_target_arm.py")
    p3 = subprocess.run([sys.executable, dt, "--arm", "bad name!"],
                        capture_output=True, text=True, cwd=repo)
    check("dispatch: unsafe custom arm name rejected",
          p3.returncode != 0 and "must match" in p3.stderr + p3.stdout,
          f"rc={p3.returncode}")
    p4 = subprocess.run([sys.executable, dt, "--arm", "fixratio_v1",
                         "--data-dir", "/nonexistent"],
                        capture_output=True, text=True, cwd=repo)
    check("dispatch: path-safe custom arm name passes the gate",
          "must match" not in p4.stderr + p4.stdout, p4.stderr[-200:])


if __name__ == "__main__":
    sys.exit(main())
