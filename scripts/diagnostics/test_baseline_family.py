#!/usr/bin/env python3
"""Baseline-family verification — gen_natural_weights (pool-proportional
natural baseline) + prepare_random_baseline --label-source domain
(quadmix-style fixed four-domain ratio) + .done identity label_source
guard + _parse_weights domain-name keys.

Standalone (repo convention: no pytest infra). Run:
    python3 scripts/diagnostics/test_baseline_family.py
Exit 0 = all checks pass.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(REPO, "scripts", f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


prb = load_script("prepare_random_baseline")
DOMAINS = ["数学", "化学", "生物学", "物理"]  # schema_stem.yaml order
QUADMIX = {"数学": 0.60, "物理": 0.15, "化学": 0.125, "生物学": 0.125}

# ── 1. _parse_weights: domain-name keys ─────────────────────────────
with tempfile.TemporaryDirectory() as td:
    wf = os.path.join(td, "domainfix_weights.json")
    with open(wf, "w") as f:
        json.dump(QUADMIX, f)
    vec = prb._parse_weights(wf, 4, label_names=DOMAINS)
    check("domain keys ordered by schema domain_names",
          np.allclose(vec, [0.60, 0.125, 0.125, 0.15]), f"{list(vec)}")

    sparse = os.path.join(td, "sparse.json")
    with open(sparse, "w") as f:
        json.dump({"数学": 2.0}, f)
    vec = prb._parse_weights(sparse, 4, label_names=DOMAINS)
    check("sparse domain dict -> absent domains are 0",
          np.allclose(vec, [1.0, 0.0, 0.0, 0.0]), f"{list(vec)}")

    bad = os.path.join(td, "bad.json")
    with open(bad, "w") as f:
        json.dump({"数学": 0.5, "foo": 0.5}, f)
    try:
        prb._parse_weights(bad, 4, label_names=DOMAINS)
        check("unknown domain key fails loud", False)
    except SystemExit as e:
        check("unknown domain key fails loud", "ERROR" in str(e), str(e)[:60])

    # C0..C{k-1} keys still parse when label_names is offered (fallback;
    # C-style dicts require all K keys — sparseness is a domain-name path
    # semantic only)
    cf = os.path.join(td, "cstyle.json")
    with open(cf, "w") as f:
        json.dump({"C0": 0.4, "C1": 0.3, "C2": 0.2, "C3": 0.1}, f)
    vec = prb._parse_weights(cf, 4, label_names=DOMAINS)
    check("C-style keys still parse with label_names",
          np.allclose(vec, [0.4, 0.3, 0.2, 0.1]), f"{list(vec)}")

# regression: no label_names, C-style (prod4 winner format)
with tempfile.TemporaryDirectory() as td:
    cf = os.path.join(td, "w.json")
    with open(cf, "w") as f:
        json.dump({"C0": 0.2, "C1": 0.8}, f)
    vec = prb._parse_weights(cf, 2)
    check("C-style keys without label_names (regression)",
          np.allclose(vec, [0.2, 0.8]), f"{list(vec)}")

# ── 2. .done identity: label_source dimension ───────────────────────
base = {"seed": 42, "requested_target_tokens": 3000000000,
        "weights_id": "file:abc123"}
check("legacy .done (no label_source) matches cluster mode",
      prb._done_matches(dict(base), seed=42, requested_tokens=3000000000,
                        weights_id="file:abc123", label_source="cluster"))
check("legacy .done does NOT match domain mode",
      not prb._done_matches(dict(base), seed=42, requested_tokens=3000000000,
                            weights_id="file:abc123", label_source="domain"))
check("domain .done matches domain mode",
      prb._done_matches(dict(base, label_source="domain"), seed=42,
                        requested_tokens=3000000000,
                        weights_id="file:abc123", label_source="domain"))
check("domain .done does NOT match cluster mode",
      not prb._done_matches(dict(base, label_source="domain"), seed=42,
                            requested_tokens=3000000000,
                            weights_id="file:abc123", label_source="cluster"))

# ── 3. gen_natural_weights: pool-proportional shares ────────────────
with tempfile.TemporaryDirectory() as td:
    pool = os.path.join(td, "pool")
    os.makedirs(pool)
    # 6 docs: C0 x2 (40+80 chars), C1 x1 (400 chars), C2 x3 (40 each)
    chars = np.array([40, 80, 400, 40, 40, 40], dtype=np.int64)
    labels = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
    np.savez(os.path.join(pool, "metadata_cache.npz"),
             doc_char_counts=chars)
    np.savez(os.path.join(td, "cluster_cache.npz"), final_labels=labels)
    out = os.path.join(td, "natural_weights.json")
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "gen_natural_weights.py"),
         "--pool-dir", pool, "--cluster-cache",
         os.path.join(td, "cluster_cache.npz"), "--output", out],
        capture_output=True, text=True)
    check("gen_natural_weights runs", r.returncode == 0,
          r.stderr.strip()[-120:] if r.returncode else "")
    if r.returncode == 0:
        w = json.load(open(out))
        tok = np.array([10.0, 20.0, 100.0, 10.0, 10.0, 10.0])
        tot = tok.sum()
        exp = np.bincount(labels, weights=tok) / tot
        check("natural weights = est-token pool shares",
              np.allclose([w["C0"], w["C1"], w["C2"]], exp, atol=1e-12)
              and abs(sum(w.values()) - 1.0) < 1e-12, f"{w}")

    # mismatched cache lengths fail loud
    np.savez(os.path.join(td, "cc_short.npz"),
             final_labels=np.array([0, 0], dtype=np.int64))
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "gen_natural_weights.py"),
         "--pool-dir", pool, "--cluster-cache",
         os.path.join(td, "cc_short.npz"), "--output", out],
        capture_output=True, text=True)
    check("cache length mismatch fails loud",
          r.returncode != 0 and "mismatch" in r.stderr)

# ── 4. --label-source domain end-to-end on a tiny parquet pool ──────
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    pool = os.path.join(td, "pool")
    os.makedirs(pool)
    # 4 docs per domain, 40 chars each -> 10 est tokens each; quadmix
    # quotas at T=48: 数学 28, 化学 6, 生物 6, 物理 7 -> 3+1+1+1 docs.
    rows = []
    ri = 0
    for dom in DOMAINS:
        for _ in range(4):
            rows.append({
                "text": "x" * 40,
                "category_name": dom,
                "char_count_col": 40,
                "source_record_idx": ri,
                "stem_relevance": 5, "knowledge_value": 5,
                "notation_fidelity": 5, "rigor_coherence": 5,
                "noise_level": 5,
            })
            ri += 1
    for s in range(2):
        df = pd.DataFrame(rows[s * 8:(s + 1) * 8])
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False),
                       os.path.join(pool, f"shard_{s:05d}.parquet"))

    wf = os.path.join(td, "domainfix_weights.json")
    with open(wf, "w") as f:
        json.dump(QUADMIX, f)
    out = os.path.join(td, "domainfix_shards")

    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "prepare_random_baseline.py"),
         "--data-dir", pool, "--output-dir", out,
         "--schema", os.path.join(REPO, "config", "schema_stem.yaml"),
         "--label-source", "domain", "--weights", wf,
         "--target-tokens", "48", "--num-npu", "1"],
        capture_output=True, text=True)
    check("domain-mode prepare runs", r.returncode == 0,
          (r.stderr or r.stdout).strip()[-160:] if r.returncode else "")
    if r.returncode == 0:
        done = json.load(open(os.path.join(out, ".done")))
        check(".done records label_source=domain",
              done.get("label_source") == "domain")
        check("K=4 domains", done.get("K") == 4)
        check("planned weights in schema domain order",
              np.allclose(done.get("planned_weights"),
                          [0.60, 0.125, 0.125, 0.15]),
              f"{done.get('planned_weights')}")
        # 数学 quota 28 tok -> 3 docs of 10; others 1 doc each
        cd = done.get("cluster_docs")
        check("per-domain doc counts match quotas",
              cd == [3, 1, 1, 1], f"{cd}")
        check("domain names printed in plan",
              "[数学]" in r.stdout, "plan line uses schema names")

    # domain mode without --schema fails loud
    r2 = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "prepare_random_baseline.py"),
         "--data-dir", pool, "--output-dir", os.path.join(td, "x"),
         "--label-source", "domain", "--weights", wf,
         "--target-tokens", "48", "--num-npu", "1"],
        capture_output=True, text=True)
    check("domain mode without --schema fails loud",
          r2.returncode != 0 and "schema" in r2.stderr)

    # cluster mode regression on the same pool (fake search cache)
    cc = os.path.join(td, "cluster_cache.npz")
    np.savez(cc, final_labels=np.array(
        [i % 2 for i in range(16)], dtype=np.int64))
    cf = os.path.join(td, "cw.json")
    with open(cf, "w") as f:
        json.dump({"C0": 0.5, "C1": 0.5}, f)
    out2 = os.path.join(td, "cluster_shards")
    r3 = subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts",
                                      "prepare_random_baseline.py"),
         "--data-dir", pool, "--output-dir", out2,
         "--cluster-cache", cc,
         "--schema", os.path.join(REPO, "config", "schema_stem.yaml"),
         "--weights", cf, "--target-tokens", "48", "--num-npu", "1"],
        capture_output=True, text=True)
    check("cluster mode regression still runs",
          r3.returncode == 0 and "[label source: cluster]" in r3.stdout,
          (r3.stderr or r3.stdout).strip()[-120:] if r3.returncode else "")
    if r3.returncode == 0:
        done2 = json.load(open(os.path.join(out2, ".done")))
        check("cluster .done keeps label_source=cluster",
              done2.get("label_source") == "cluster" and done2.get("K") == 2)

print()
if FAILED:
    print(f"FAILED: {len(FAILED)} -> {FAILED}")
    sys.exit(1)
print("ALL CHECKS PASS")
