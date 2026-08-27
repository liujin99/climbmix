"""Prepare random baseline dataset for CLIMB validation.

Paper (Appendix C.1): "Random: randomly select data for language model
training, where each cluster is assigned an equal and uniform weight."

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


def main():
    parser = argparse.ArgumentParser(
        description="Equal-cluster-weight random baseline (paper App. C.1)")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cluster-cache", required=True,
                        help="cluster_cache.npz from the search stage "
                             "(final_labels, pool doc order)")
    parser.add_argument("--schema", default=None,
                        help="column schema YAML (default: manager's default)")
    parser.add_argument("--target-tokens", type=parse_token_count, default=0,
                        help="Token budget, same cap as the CLIMB arm's "
                             "--target-tokens (0 = all available; suffixes "
                             "2B/10M/500K supported)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-npu", type=int, default=8)
    args = parser.parse_args()

    done_marker = os.path.join(args.output_dir, ".done")
    if os.path.exists(done_marker):
        print(f"  Random baseline already complete (.done), skipping")
        return

    random.seed(args.seed)

    schema = DatasetSchema.from_yaml(args.schema) if args.schema else None
    mm = ShardMetadataManager(args.data_dir, schema=schema,
                              cache_dir=args.data_dir)

    labels = np.load(args.cluster_cache,
                     allow_pickle=False)["final_labels"].astype(np.int64)
    if len(labels) != mm.num_docs:
        raise SystemExit(
            f"ERROR: cluster cache has {len(labels):,} labels but the pool has "
            f"{mm.num_docs:,} docs. The data pool changed after the cluster "
            f"cache was written — rerun the search stage (the fingerprint gate "
            f"normally prevents this).")

    token_counts = mm.estimate_token_counts()
    K = len(np.unique(labels[labels >= 0]))
    target_tokens = args.target_tokens or int(token_counts.sum())

    print(f"\n[Random] Equal-weight baseline: K={K} clusters, "
          f"alpha_k=1/{K}, target_tokens={target_tokens:,}")

    uniform = MixtureWeights(weights=np.full(K, 1.0 / K, dtype=np.float64))
    selected, _ = select_data_by_mixture(
        labels, uniform, token_counts=token_counts,
        target_tokens=target_tokens, seed=args.seed,
    )

    sel_labels = labels[selected]
    sel_tokens = token_counts[selected]
    cluster_docs = np.bincount(sel_labels, minlength=K).tolist()
    cluster_tokens = np.bincount(sel_labels, weights=sel_tokens,
                                 minlength=K).tolist()
    avail_docs = np.bincount(labels[labels >= 0], minlength=K).tolist()
    quota_tokens = target_tokens // K

    print(f"[Random] Per-cluster plan (quota={quota_tokens:,} tokens each):")
    shortfall = []
    for k in range(K):
        short = cluster_tokens[k] < quota_tokens * 0.999
        if short:
            shortfall.append(k)
        marker = "  <- SHORTFALL (took all docs, no duplication)" if short else ""
        print(f"  [{k:>2d}] avail {avail_docs[k]:>9,} docs "
              f"({cluster_tokens[k] if short else quota_tokens:>12,} tok) "
              f"-> took {cluster_docs[k]:>9,} docs{marker}")
    if shortfall:
        print(f"[Random] NOTE: {len(shortfall)}/{K} clusters cannot fill their "
              f"1/K quota (same policy as the CLIMB arm: take all, no "
              f"duplication, no redistribution) — effective weights deviate "
              f"from uniform for those clusters")
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

    def _atomic_write(name, table, rg):
        shard_path = os.path.join(args.output_dir, name)
        tmp_path = shard_path + ".tmp.parquet"
        pq.write_table(table, tmp_path, row_group_size=rg)
        os.replace(tmp_path, shard_path)

    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, n_train)
        shard_texts = train_texts[start:end]
        _atomic_write(f"shard_{i:05d}.parquet", pa.table({"text": shard_texts}), rg_size)

    _atomic_write(f"shard_{n_shards:05d}.parquet", pa.table({"text": val_texts}), 1)

    with open(done_marker, "w") as f:
        json.dump({
            "n_train_shards": n_shards, "val_docs": val_n,
            "rg_size": rg_size, "num_npu": args.num_npu, "seed": args.seed,
            "K": K, "target_tokens": target_tokens,
            "planned_weights": [1.0 / K] * K,
            "effective_doc_shares": [d / n for d in cluster_docs],
            "cluster_docs": cluster_docs,
            "cluster_tokens": cluster_tokens,
            "shortfall_clusters": shortfall,
        }, f, indent=2)

    print(f"[Random] Baseline: {n} docs -> {n_shards} shards (rg_size={rg_size}) "
          f"+ 1 val shard ({val_n} docs, rg_size=1) -> {args.output_dir}")


if __name__ == "__main__":
    main()
