#!/usr/bin/env python3
"""Prune-profile report: cluster-quality analysis on the REAL pool, discovery-only.

Runs Step 1's sampled path (embed sample → K-means → prune diagnostics →
prune → merge + merge diagnostics) WITHOUT the training pipeline, so the
user can inspect what PRUNE_THRESHOLD would do before committing to a
production run.

What you get in --output-dir:
  prune_profile.json   threshold sweep / per-column means / domain breakdown
                       + advice (printed to console too)
  merge_profile.json   real-pool K structure (dendrogram cut, natural_K(tau))
  clusters.csv         per-K_init-cluster table sorted by avg quality
                       (weakest first; the clusters pruning would remove)

Embeddings + K-means are cached in the SAME pool-keyed directory the
pipeline uses (--embedding-cache-dir root + content key), so a re-run is
instant and a later run with the same sample size reuses them.

Typical (server, full 100B pool, first run):
  python3 scripts/prune_report.py --data-dir /home/ma-user/work/100B_stem_parquet_filtered
  # Analyzes a seed-42 subset of 100 shards (~12M docs; metadata scan and
  # text reads touch only those shards): ~2-4 min metadata + ~18 min
  # single-NPU embedding of the 100K doc-level sample + minutes of K-means.
  # --sample-shards 0 switches to the full 1000-shard pool (~20-40 min scan,
  # same 100K sample); --sample-size 200000 doubles precision for ~35 min
  # embedding. Smaller samples are fast but leave fewer docs per K_init=1000
  # cluster (20K → ~20/cluster, cluster-mean SE ~0.11-0.22 — same order as
  # the 0.25 threshold steps).
Re-run: seconds (all caches hit).

No NPU? --embedding-device cpu works (20K docs ≈ tens of minutes).
"""

import argparse
import csv
import os
import sys
import time


def _select_shard_dir(data_dir: str, n_shards: int, work_root: str) -> str:
    """Directory of symlinks to n_shards seed-42-random shards of data_dir.

    Stratified-ish subsampling: N shards × (sample_size/N docs each on
    average) instead of scanning all 1000 shards. Dilutes shard-level skew
    across N shards (vs picking a few whole shards) while cutting the
    metadata scan / text I/O / memory by ~1000/N. The symlink names keep the
    original basenames so pool-cache keys stay content-addressed.

    Deterministic: same data_dir + n_shards → same selection.
    """
    import glob
    import numpy as np
    shards = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if len(shards) <= n_shards:
        return data_dir  # nothing to subsample
    rng = np.random.default_rng(42)
    selected = sorted(rng.choice(len(shards), size=n_shards, replace=False))
    tag = f"shards{n_shards}_of_{len(shards)}"
    out_dir = os.path.join(work_root, tag)
    os.makedirs(out_dir, exist_ok=True)
    n_linked = 0
    for i in selected:
        src = shards[int(i)]
        dst = os.path.join(out_dir, os.path.basename(src))
        if not os.path.islink(dst) and not os.path.exists(dst):
            os.symlink(src, dst)
            n_linked += 1
    print(f"[Report] Shard subsample: {n_shards}/{len(shards)} shards "
          f"({n_linked} new symlinks) → {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Discovery-only prune/merge profile report for a data pool")
    parser.add_argument("--data-dir", required=True,
                        help="STEM parquet directory (the pool to analyze)")
    parser.add_argument("--schema", default=None,
                        help="DatasetSchema YAML (default: config/schema_stem.yaml "
                             "next to this repo)")
    parser.add_argument("--sample-shards", type=int, default=100,
                        help="analyze a seed-42 random subset of N shards "
                             "(default 100 of ~1000: metadata scan/I/O/memory "
                             "cut ~10x, shard-level skew diluted across 100 "
                             "shards; 0 = all shards). Each selected shard "
                             "contributes ~sample_size/N docs to the doc-level "
                             "sample, so doc-level randomness is preserved")
    parser.add_argument("--sample-size", type=int, default=100000,
                        help="docs to embed+cluster, sampled with the pipeline's "
                             "seed-42 scheme (default 100000: ~100 docs per "
                             "K_init=1000 cluster, cluster-mean SE ~0.05-0.10 "
                             "on the 1-5 scale; 20K would leave ~20 docs/cluster "
                             "and SE comparable to the 0.25 threshold steps)")
    parser.add_argument("--K-init", type=int, default=1000,
                        help="K-means clusters (default 1000 = production; "
                             "speedrun used 100)")
    parser.add_argument("--K-enhanced", type=int, default=3)
    parser.add_argument("--K-max", type=int, default=15)
    parser.add_argument("--merge-distance", type=float, default=0.9)
    parser.add_argument("--prune-threshold", type=float, default=None,
                        help="default: schema's prune_threshold (3.0)")
    parser.add_argument("--embedding-model", default="NovaSearch/stella_en_400M_v5")
    parser.add_argument("--embedding-device", default="npu")
    parser.add_argument("--embedding-truncate-len", type=int, default=512)
    parser.add_argument("--embedding-cache-dir", default=None,
                        help="pool-cache root (default: <repo>/cache/embeddings, "
                             "same as run_climbmix.sh)")
    parser.add_argument("--output-dir", default=None,
                        help="default: <repo>/result/prune_report")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "src"))

    schema_path = args.schema or os.path.join(repo_root, "config", "schema_stem.yaml")
    embedding_cache_root = args.embedding_cache_dir or os.path.join(
        repo_root, "cache", "embeddings")
    output_dir = args.output_dir or os.path.join(repo_root, "result", "prune_report")
    os.makedirs(output_dir, exist_ok=True)

    from climbmix.data.column_schema import DatasetSchema
    from climbmix.data.metadata_manager import ShardMetadataManager
    from climbmix.core.types import CLIMBConfig, ClusterDiscoveryConfig
    from climbmix.pipeline.climb_pipeline import CLIMBPipeline
    from climbmix.core.discovery import get_discovery

    schema = DatasetSchema.from_yaml(schema_path)
    prune_threshold = (args.prune_threshold if args.prune_threshold is not None
                       else schema.prune_threshold)

    print("=" * 70)
    print("  Prune-profile report (discovery-only)")
    print(f"  pool: {args.data_dir}")
    print(f"  sample_size={args.sample_size:,}  K_init={args.K_init}  "
          f"prune_threshold={prune_threshold}")
    print("=" * 70)

    # ── Stage 0: shard subsample + metadata (domain + quality + char_count).
    # With --sample-shards>0, only the selected shards are scanned (symlinked
    # into a work dir; cache lands next to the symlinks, keyed by the subset's
    # manifest, and is reused across re-runs with the same N).
    t0 = time.time()
    analyze_dir = args.data_dir
    if args.sample_shards and args.sample_shards > 0:
        analyze_dir = _select_shard_dir(args.data_dir, args.sample_shards,
                                        os.path.join(output_dir, ".work"))
    mm = ShardMetadataManager(analyze_dir, schema=schema, cache_dir=analyze_dir)
    n_total = mm.num_docs
    print(f"[Report] Pool: {n_total:,} docs in {analyze_dir} "
          f"(metadata {'cached' if time.time() - t0 < 5 else 'scanned fresh'}, "
          f"{time.time() - t0:.0f}s)")

    if n_total <= args.sample_size:
        print(f"[Report] Pool ({n_total:,}) <= sample_size ({args.sample_size:,}) — "
              f"nothing to subsample; lower --sample-size or check --data-dir")
        return 1

    # ── Stage 1: discovery config + pool-keyed cache dir via the PIPELINE's
    # own key logic (shard manifest + model + truncate len + sample size) so
    # caches are shared with real runs.
    disc_cfg = ClusterDiscoveryConfig(
        method="embedding_cluster",
        K_init=args.K_init,
        K_enhanced=args.K_enhanced,
        K_max=args.K_max,
        embedding_model=args.embedding_model,
        embedding_truncate_len=args.embedding_truncate_len,
        embedding_device=args.embedding_device,
        embedding_sample_size=args.sample_size,
        prune_threshold=prune_threshold,
        merge_distance=args.merge_distance,
    )
    config = CLIMBConfig(
        discovery=disc_cfg,
        data_dir=analyze_dir,
        embedding_cache_dir=embedding_cache_root,
        schema_path=schema_path,
    )
    pool_cache_dir = CLIMBPipeline(config)._pool_embedding_cache_dir(analyze_dir)
    if pool_cache_dir:
        print(f"[Report] Pool-level embedding/kmeans cache: {pool_cache_dir}")

    quality_scores = mm.quality_scores
    token_counts = mm.estimate_token_counts()

    # ── Stage 2: sampled discovery — embed (cached) → K-means (cached) →
    # PRUNE DIAGNOSTICS (prune_profile.json) → prune → merge + merge
    # diagnostics (merge_profile.json).
    discovery = get_discovery("embedding_cluster", disc_cfg)
    cluster_info, final_labels = discovery.discover(
        texts=None,
        cluster_labels=mm.cluster_labels,
        quality_scores=quality_scores,
        token_counts=token_counts,
        metadata_manager=mm,
        cache_dir=output_dir,
        embedding_cache_dir=pool_cache_dir,
    )
    print(f"[Report] Final clusters: {len(cluster_info)} "
          f"(docs {sum(c.num_docs for c in cluster_info):,}, "
          f"tokens {sum(c.num_tokens for c in cluster_info):,})")

    # ── Stage 3: per-K_init-cluster table (weakest first) from the K-means
    # cache + the same seed-42 sample the discovery used.
    import numpy as np
    kmeans_cache = (os.path.join(pool_cache_dir, f"kmeans_K{args.K_init}.npz")
                    if pool_cache_dir else None)
    if kmeans_cache and os.path.exists(kmeans_cache):
        data = np.load(kmeans_cache)
        sample_labels = data["labels"]
        rng = np.random.default_rng(42)
        sample_indices = np.sort(rng.choice(n_total, size=args.sample_size,
                                            replace=False))
        if sample_labels.shape[0] == len(sample_indices):
            from climbmix.core.cluster_merge import _cluster_quality_matrix
            q_sample = quality_scores[sample_indices]
            col_means, counts, token_sums, n_excl = _cluster_quality_matrix(
                sample_labels, q_sample, token_counts=token_counts[sample_indices])
            col_names = list(schema.quality_cols)
            rows = []
            for cid in range(len(counts)):
                if counts[cid] == 0:
                    continue
                row = {
                    "cluster_id": cid,
                    "n_docs": int(counts[cid]),
                    "tokens": int(token_sums[cid]) if token_sums is not None else "",
                    "avg_quality": round(float(col_means[cid].mean()), 4),
                    **{f"mean_{n}": round(float(col_means[cid, j]), 4)
                       for j, n in enumerate(col_names)},
                    "pruned": int(col_means[cid].mean() < prune_threshold),
                }
                rows.append(row)
            rows.sort(key=lambda r: r["avg_quality"])
            csv_path = os.path.join(output_dir, "clusters.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            n_pruned_rows = sum(r["pruned"] for r in rows)
            print(f"[Report] Per-cluster table → {csv_path} "
                  f"({len(rows)} clusters, {n_pruned_rows} below "
                  f"threshold={prune_threshold}; weakest 10:")
            for r in rows[:10]:
                cols = " ".join(f"{r[f'mean_{n}']:.2f}" for n in col_names)
                print(f"    C{r['cluster_id']:<4} avg={r['avg_quality']:.2f} "
                      f"docs={r['n_docs']:<6} tokens={r['tokens']:<10} [{cols}]")
        else:
            print(f"[Report] kmeans cache row count mismatch "
                  f"({sample_labels.shape[0]} vs {len(sample_indices)}) — "
                  f"skipping clusters.csv")
    else:
        print("[Report] kmeans cache not found — skipping clusters.csv")

    print("\n[Report] Artifacts:")
    for name in ("prune_profile.json", "merge_profile.json", "clusters.csv"):
        p = os.path.join(output_dir, name)
        if os.path.exists(p):
            print(f"  {p}")
    print("\n[Report] Re-tune knobs (--prune-threshold / --merge-distance / "
          "--K-max) and re-run: seconds (embeddings are pool-cached).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
