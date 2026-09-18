"""Prepare a baseline-arm dataset for CLIMB validation.

Default = the paper's Random baseline (Appendix C.1): "randomly select data
for language model training, where each cluster is assigned an equal and
uniform weight."

So the baseline is NOT a uniform draw over documents — that would weight
clusters by their natural size. It is the SAME mixture-weighted selection
machinery as the CLIMB arm (sampling.data_selector.select_data_by_mixture)
with the weights pinned to uniform alpha_k = 1/K:

  - per-cluster token quota = (1/K) * target_tokens
  - cluster smaller than its quota: take ALL its docs — no duplication, no
    redistribution; the identical shortfall policy the CLIMB arm's selector
    applies to overweight clusters, so both arms degrade the same way and
    stay comparable. The paper documents no small-cluster policy: its pool
    (800B tokens / 21 clusters, 40B budget) makes shortfalls unlikely
    (~1.9B-token quota per cluster); our smaller pools can hit them, and
    mirroring the CLIMB arm is the deviation-free choice (loudly logged).
  - same token budget cap as the CLIMB arm (--target-tokens)

--weights (docs/reuse_design.md §4.4) generalizes this to ANY fixed ratio
(comma list / JSON array / optimal_mixture_weights.json) — e.g. a
hand-designed heuristic baseline, or re-prepping a previous run's winner
mixture for a target-model retrain with changed training params.

Cluster labels come from the search stage's cluster_cache.npz (final_labels,
pool doc order == ShardMetadataManager order). A length mismatch fails
loudly — the pool changed after the cache was written (the fingerprint gate
normally prevents this).

Last shard is the val split (real docs, nanochat convention: last file=val).

Memory: only parquet METADATA plus the precomputed char-count column are
read for the pool scan (no full text load); texts are read only for the
selected docs. Peak memory ~ selected sample.

Crash safety: shards are written to temp names and renamed into place; a
.done marker is written only after everything succeeded. Shards without
.done are treated as a crashed partial run and wiped before redoing.

DDP row-group safety: nanochat's dataloader shards row groups round-robin
across ranks, so every shard must contain at least num_npu row groups
(see prepare_shards.py).
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from climbmix.core.types import MixtureWeights
from climbmix.data.column_schema import DatasetSchema
from climbmix.data.metadata_manager import ShardMetadataManager
from climbmix.sampling.data_selector import select_data_by_mixture
from climbmix.utils.token_estimate import parse_token_count


def _parse_weights(spec: str, K: int, label_names=None) -> np.ndarray:
    """Custom mixture weights for a baseline arm (docs/reuse_design.md §4.4).

    Accepts a comma list ("0.25,0.25,0.25,0.25"), a JSON array file, or a
    dict file. Dict keys are either C0..C{k-1} (optimal_mixture_weights.json
    format, cluster labels sorted numerically) or — when label_names is
    given (--label-source domain) — the schema's domain names, ordered by
    that list (e.g. 数学/化学/生物学/物理 per config/schema_stem.yaml);
    a domain absent from the dict means weight 0 (sparse dicts OK).
    Returns the normalized K-vector; uniform = the paper's random
    baseline (default when --weights is absent).
    """
    if os.path.isfile(spec):
        with open(spec) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if label_names is not None and data and all(
                    str(k) in label_names for k in data.keys()):
                vec = [float(data.get(name, 0.0)) for name in label_names]
            else:
                try:
                    items = sorted(
                        data.items(),
                        key=lambda kv: int(str(kv[0]).lstrip("Cc")))
                except ValueError:
                    hint = (" or one of " + "/".join(label_names)
                            if label_names is not None else "")
                    raise SystemExit(
                        "ERROR: --weights dict keys must be C0..C{k-1} "
                        "(optimal_mixture_weights.json format)" + hint)
                vec = [float(v) for _, v in items]
        else:
            vec = [float(v) for v in data]
    else:
        vec = [float(x) for x in spec.split(",") if x.strip() != ""]
    if len(vec) != K:
        raise SystemExit(
            f"ERROR: --weights has {len(vec)} entries but the cluster cache "
            f"has K={K} — dimension mismatch")
    arr = np.array(vec, dtype=np.float64)
    if not np.all(np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
        raise SystemExit("ERROR: --weights must be finite, >= 0, sum > 0")
    return arr / arr.sum()


def _weights_identity(spec: str) -> str:
    """Cheap, honest identity of the --weights input.

    A comma list is hashed as the string; a file path (JSON array /
    optimal_mixture_weights.json) is hashed by CONTENT — the prod4 winner
    weights file is passed by path and its content is what selects the
    docs. "" = uniform baseline. Deliberately computed without loading
    the cluster cache: the .done skip path must stay cheap.
    """
    if not spec:
        return "uniform"
    if os.path.isfile(spec):
        with open(spec, "rb") as f:
            return "file:" + hashlib.sha256(f.read()).hexdigest()[:16]
    return "str:" + hashlib.sha256(spec.encode("utf-8")).hexdigest()[:16]


def _done_matches(done: dict, *, seed: int, requested_tokens: int,
                  weights_id: str, label_source: str = "cluster") -> bool:
    """.done identity guard: same seed, same requested budget, same weights,
    same label source.

    Legacy .done files (pre-2026-09-16) lack weights_id — unverifiable,
    treated as a mismatch (fail loud; remove the dir or use a new
    ARM_NAME). Legacy files without label_source predate --label-source
    and were all cluster mode, so the "cluster" default is safe.
    requested_target_tokens is compared raw; the legacy resolved
    target_tokens field is accepted only when it equals the raw request
    (nonzero budgets — the only mode arm_engine uses).
    """
    if done.get("seed") != seed:
        return False
    if done.get("label_source", "cluster") != label_source:
        return False
    rec_req = done.get("requested_target_tokens")
    if rec_req is not None:
        if rec_req != requested_tokens:
            return False
    else:
        rec_tok = done.get("target_tokens")
        if requested_tokens and rec_tok != requested_tokens:
            return False
    return done.get("weights_id") == weights_id


def _write_shards(train_texts, val_texts, output_dir, n_shards, shard_size,
                  rg_size, workers=None):
    """Write the selection shards (+ the tail val shard) in a thread pool.

    pyarrow write+compression releases the GIL, so workers overlap the
    per-shard write cost that dominated the old serial loop (prod4
    2026-09-16: 398 shards single-core on a 192-vCPU host). Content per
    shard is index-sliced — parallel vs serial output is byte-identical
    (asserted in test_prod4_fixes.py)."""
    from multiprocessing.pool import ThreadPool

    def _atomic_write(name, table, rg):
        shard_path = os.path.join(output_dir, name)
        tmp_path = shard_path + ".tmp.parquet"
        pq.write_table(table, tmp_path, row_group_size=rg)
        os.replace(tmp_path, shard_path)

    def _write_train(i):
        start = i * shard_size
        end = min(start + shard_size, len(train_texts))
        _atomic_write(f"shard_{i:05d}.parquet",
                      pa.table({"text": train_texts[start:end]}), rg_size)

    w = workers or max(1, int(os.environ.get("CLIMB_MIX_WRITE_WORKERS") or "16"))
    with ThreadPool(w) as pool:
        pool.map(_write_train, range(n_shards))
    _atomic_write(f"shard_{n_shards:05d}.parquet",
                  pa.table({"text": val_texts}), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Baseline-arm STEM shards from cluster weights "
                    "(default: equal 1/K, paper App. C.1 random baseline; "
                    "--weights: any fixed ratio, docs/reuse_design.md §4.4)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-cache", default=None,
                        help="cluster_cache.npz from the search stage "
                             "(final_labels, pool doc order); required for "
                             "--label-source cluster, unused for domain")
    parser.add_argument("--label-source", choices=["cluster", "domain"],
                        default="cluster",
                        help="cluster = search-stage K-means labels "
                             "(cluster_cache.npz final_labels, default); "
                             "domain = the parquet domain column cached in "
                             "metadata_cache.npz (category_name mapped via "
                             "schema domain_names, e.g. 数学/化学/生物学/物理)")
    parser.add_argument("--schema", default=None,
                        help="column schema YAML (default: manager's default)")
    parser.add_argument("--target-tokens", type=parse_token_count, default=0,
                        help="Token budget, same cap as the CLIMB arm's "
                             "--target-tokens (0 = all available; suffixes "
                             "2B/10M/500K supported)")
    parser.add_argument("--weights", default="",
                        help="comma list, JSON array file, or "
                             "optimal_mixture_weights.json (dict C0..C{k-1}); "
                             "default = uniform 1/K (paper random baseline)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-npu", type=int, default=8)
    args = parser.parse_args()

    done_marker = os.path.join(args.output_dir, ".done")
    if os.path.exists(done_marker):
        # Identity guard (2026-09-16): an unconditional skip here was the
        # LOCAL twin of the OBS stale-mixture landmine — same output dir,
        # different seed/budget/weights -> silently reuse the old shards.
        try:
            with open(done_marker) as f:
                done_info = json.load(f)
        except (OSError, ValueError):
            done_info = {}
        weights_id = _weights_identity(args.weights)
        if _done_matches(done_info, seed=args.seed,
                         requested_tokens=args.target_tokens,
                         weights_id=weights_id,
                         label_source=args.label_source):
            print(f"  Random baseline already complete (.done), skipping")
            return
        raise SystemExit(
            "ERROR: .done in the output dir was built for a DIFFERENT config "
            f"(recorded seed={done_info.get('seed')}, "
            f"tokens={done_info.get('requested_target_tokens', done_info.get('target_tokens'))}, "
            f"weights_id={done_info.get('weights_id', 'legacy')} vs requested "
            f"seed={args.seed}, tokens={args.target_tokens}, "
            f"weights_id={weights_id}). Arm dirs are name-addressed: use a "
            "new ARM_NAME for a new config, or remove the output dir to "
            "rebuild.")

    random.seed(args.seed)

    schema = DatasetSchema.from_yaml(args.schema) if args.schema else None
    mm = ShardMetadataManager(args.data_dir, schema=schema,
                              cache_dir=args.data_dir)

    if args.label_source == "cluster":
        if not args.cluster_cache:
            raise SystemExit(
                "ERROR: --label-source cluster requires --cluster-cache "
                "(cluster_cache.npz from the search stage)")
        labels = np.load(args.cluster_cache,
                         allow_pickle=False)["final_labels"].astype(np.int64)
        if len(labels) != mm.num_docs:
            raise SystemExit(
                f"ERROR: cluster cache has {len(labels):,} labels but the pool has "
                f"{mm.num_docs:,} docs. The data pool changed after the cluster "
                f"cache was written — rerun the search stage (the fingerprint gate "
                f"normally prevents this).")
        label_names = None
    else:
        if schema is None or not getattr(schema, "domain_names", None):
            raise SystemExit(
                "ERROR: --label-source domain requires --schema with "
                "domain_names (e.g. config/schema_stem.yaml: "
                "数学/化学/生物学/物理)")
        # The metadata cache's label array IS the parquet domain column
        # (category_name -> int via schema domain_names order) — the 15
        # K-means labels live in the search stage's cluster_cache.npz.
        # Same array the cache was built from, so alignment is by
        # construction (no second file to drift).
        labels = np.asarray(mm.cluster_labels, dtype=np.int64)
        label_names = list(schema.domain_names)

    token_counts = mm.estimate_token_counts()
    K = len(np.unique(labels[labels >= 0]))
    if args.label_source == "domain" and K != len(label_names):
        raise SystemExit(
            f"ERROR: domain labels have {K} distinct values but the schema "
            f"defines {len(label_names)} domains ({'/'.join(label_names)}) — "
            f"pool/schema mismatch")
    target_tokens = args.target_tokens or int(token_counts.sum())

    if args.weights:
        weights_vec = _parse_weights(args.weights, K, label_names=label_names)
        mode = "custom fixed-ratio baseline (docs/reuse_design.md §4.4)"
    else:
        weights_vec = np.full(K, 1.0 / K, dtype=np.float64)
        mode = "equal-weight baseline (paper App. C.1 random)"

    axis = "domains" if args.label_source == "domain" else "clusters"
    print(f"\n[Random] {mode} [label source: {args.label_source}]: "
          f"K={K} {axis}, target_tokens={target_tokens:,}")
    print(f"[Random] planned weights: "
          + ", ".join(f"{w:.4f}" for w in weights_vec))

    mixture = MixtureWeights(weights=weights_vec)
    selected, _ = select_data_by_mixture(
        labels, mixture, token_counts=token_counts,
        target_tokens=target_tokens, seed=args.seed,
    )

    sel_labels = labels[selected]
    sel_tokens = token_counts[selected]
    cluster_docs = np.bincount(sel_labels, minlength=K).tolist()
    cluster_tokens = np.bincount(sel_labels, weights=sel_tokens,
                                 minlength=K).tolist()
    avail_docs = np.bincount(labels[labels >= 0], minlength=K).tolist()
    quota_tokens = (weights_vec * target_tokens).astype(np.int64).tolist()

    print(f"[Random] Per-cluster plan (quota = w_k x target_tokens):")
    shortfall = []
    for k in range(K):
        short = cluster_tokens[k] < quota_tokens[k] * 0.999
        if short:
            shortfall.append(k)
        marker = "  <- SHORTFALL (took all docs, no duplication)" if short else ""
        kname = label_names[k] if label_names is not None else f"{k:>2d}"
        print(f"  [{kname}] avail {avail_docs[k]:>9,} docs "
              f"({cluster_tokens[k] if short else quota_tokens[k]:>12,} tok) "
              f"-> took {cluster_docs[k]:>9,} docs{marker}")
    if shortfall:
        print(f"[Random] NOTE: {len(shortfall)}/{K} clusters cannot fill their "
              f"planned quota (same policy as the CLIMB arm: take all, no "
              f"duplication, no redistribution) — effective weights deviate "
              f"from plan for those clusters")
    n = len(selected)
    print(f"[Random] Selected {n:,} docs, "
          f"{int(sum(cluster_tokens)):,} tokens "
          f"(planned {target_tokens:,})")

    if n < 4 * args.num_npu:
        raise SystemExit(
            f"ERROR: only {n} docs < 4*num_npu ({4 * args.num_npu}). Need >= 2*num_npu docs each "
            f"for train and val so that every rank owns >= 2 row groups in both splits; "
            f"the DDP dataloader assigns row groups round-robin per rank and ranks with "
            f"no row group hang forever before the first all_reduce."
        )

    texts = mm.read_texts(selected)
    random.shuffle(texts)

    # Real val split (tail of the sampled data), same policy as prepare_shards.py
    val_n = min(256, max(2 * args.num_npu, n // 100))
    train_texts = texts[:n - val_n]
    val_texts = texts[n - val_n:]

    shard_size = 10000
    n_train = len(train_texts)
    n_shards = max(1, math.ceil(n_train / shard_size))
    # Absorb a tiny remainder into the previous shard: a last shard with
    # < 2*num_npu docs cannot provide one row group per rank -> DDP starvation.
    while n_shards > 1 and n_train - (n_shards - 1) * shard_size < 2 * args.num_npu:
        n_shards -= 1
    # Row groups sized from the ACTUAL doc count of the last (smallest) shard
    last_shard_docs = n_train - (n_shards - 1) * shard_size
    rg_size = max(1, last_shard_docs // (args.num_npu * 2))

    os.makedirs(args.output_dir, exist_ok=True)

    # Shards without .done = crashed partial run: wipe and redo.
    leftovers = [f for f in os.listdir(args.output_dir)
                 if f.startswith("shard_") or f.endswith(".tmp.parquet")]
    if leftovers:
        print(f"  Cleaning {len(leftovers)} partial files from a crashed run (no .done)")
        for f in leftovers:
            os.remove(os.path.join(args.output_dir, f))

    _write_shards(train_texts, val_texts, args.output_dir, n_shards,
                  shard_size, rg_size)

    with open(done_marker, "w") as f:
        json.dump({
            "n_train_shards": n_shards, "val_docs": val_n,
            "rg_size": rg_size, "num_npu": args.num_npu, "seed": args.seed,
            "K": K, "target_tokens": target_tokens,
            "requested_target_tokens": args.target_tokens,
            "label_source": args.label_source,
            "weights_id": _weights_identity(args.weights),
            "planned_weights": [float(w) for w in weights_vec],
            "effective_doc_shares": [d / n for d in cluster_docs],
            "cluster_docs": cluster_docs,
            "cluster_tokens": cluster_tokens,
            "shortfall_clusters": shortfall,
        }, f, indent=2)

    print(f"[Random] Baseline: {n} docs -> {n_shards} shards (rg_size={rg_size}) "
          f"+ 1 val shard ({val_n} docs, rg_size=1) -> {args.output_dir}")


if __name__ == "__main__":
    main()
