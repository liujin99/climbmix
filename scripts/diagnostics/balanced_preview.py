#!/usr/bin/env python3
"""Offline preview of MERGE_STRATEGY=balanced on the real pool — the
production function, real K-means cache, real quality/tokens, zero NPU.

The pre-launch check for prod2-style runs: balanced_macro_clusters on the
cached K_init labels + pruned centroids, plus a per-macro semantic probe
(sample texts matched against format/topic regexes). Accept before
launching the proxy fleet (see paper_deviations.md D14):

  - max token share <= ~(1+slack)/K and min share > 0
  - fine->anchor cosine mean >= ~0.5 (slices are tight)
  - per-macro sample rows readable/distinguishable by eye

Usage (server):
  python3 scripts/diagnostics/balanced_preview.py \
      --data-dir /home/ma-user/work/100B_stem_parquet_filtered \
      --kmeans /l00916525/prod/climbmix/cache/embeddings/*/kmeans_K1000.npz \
      --K 15 --samples 4
"""

import argparse
import glob
import re
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/src")

from climbmix.core.cluster_merge import (  # noqa: E402
    balanced_macro_clusters,
    compute_cluster_column_mins,
    compute_cluster_quality,
    prune_clusters,
)

PATTERNS = {
    "code": r"\bdef \w+\(|\bimport \w+|\bclass \w+[:(]|#include",
    "math": r"theorem|lemma|\\frac|prove that|equation",
    "paper": r"abstract|arxiv|references|\[1\]|et al\.|doi:",
    "qa": r"User:|Assistant:|Question:|Answer:",
    "lesson": r"Lesson|Teaching Objectives|Learning Objectives",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", required=True, help="pool parquet dir")
    ap.add_argument("--kmeans", required=True,
                    help="kmeans_K*.npz path (glob ok, newest match wins)")
    ap.add_argument("--K", type=int, default=15)
    ap.add_argument("--slack", type=float, default=0.15)
    ap.add_argument("--prune-threshold", type=float, default=3.0)
    ap.add_argument("--prune-column-floor", type=float, default=2.0)
    ap.add_argument("--samples", type=int, default=4,
                    help="sample texts per macro for the semantic probe")
    ap.add_argument("--no-semantics", action="store_true",
                    help="skip reading texts (shares/cos only, no parquet IO)")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.kmeans))
    if not matches:
        sys.exit(f"no kmeans cache matches {args.kmeans}")
    kmeans_path = matches[-1]

    from climbmix.data.metadata_manager import ShardMetadataManager
    mm = ShardMetadataManager(args.data_dir, cache_dir=args.data_dir)

    z = np.load(kmeans_path)
    lab, cen = z["labels"], z["centroids"]
    if len(lab) != mm.num_docs:
        sys.exit(f"kmeans cache holds {len(lab):,} labels but pool has "
                 f"{mm.num_docs:,} docs — stale cache, aborting")
    print(f"kmeans: {kmeans_path} (labels={len(lab):,}, K_init={len(cen)})")

    tok = mm.estimate_token_counts()
    q = mm.quality_scores

    cq = compute_cluster_quality(lab, q, prune_threshold=args.prune_threshold)
    cm = compute_cluster_column_mins(lab, q)
    plab, pcen, _ = prune_clusters(
        lab, cen, cq, threshold=args.prune_threshold,
        column_mins=cm if args.prune_column_floor > 0 else None,
        column_floor=args.prune_column_floor)

    macro_labels, macro_centroids, _ = balanced_macro_clusters(
        plab, pcen, token_counts=tok, K=args.K, slack=args.slack)

    total = tok[macro_labels >= 0].sum()
    print()
    print("per-macro semantic probe (first sample truncated):")
    rng = np.random.default_rng(0)
    for k in range(len(macro_centroids)):
        idx = np.flatnonzero(macro_labels == k)
        n_tok = int(tok[idx].sum())
        if args.no_semantics or args.samples <= 0:
            print(f"M{k:02d} {n_tok/1e9:6.2f}B docs={len(idx):>11,}")
            continue
        take = np.sort(rng.choice(idx, min(args.samples, len(idx)),
                                  replace=False))
        texts = mm.read_texts(take, verbose=False)
        fr = {name: sum(100 if re.search(p, t[:2000], re.I | re.M) else 0
                        for t in texts) // len(texts)
              for name, p in PATTERNS.items()}
        tag = " ".join(f"{n}:{fr[n]:>3}%" for n in PATTERNS)
        print(f"M{k:02d} {n_tok/1e9:6.2f}B docs={len(idx):>11,} | {tag} "
              f"| {texts[0][:80]!r}")

    print()
    print(f"pool kept by prune: {total/1e9:.2f}B tokens; "
          f"accept if max share <= ~{100*(1+args.slack)/args.K:.1f}% "
          f"and cosine mean >= ~0.5 (see balanced_profile block above)")


if __name__ == "__main__":
    main()
