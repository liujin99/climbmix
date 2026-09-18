#!/usr/bin/env python3
"""Generate the natural-distribution baseline weights (pool-proportional).

The natural baseline answers "what would uniform sampling over DOCUMENTS
give?": docs drawn at random from the whole pool make the final mixture's
token composition equal the pool's token composition (in expectation), so
weights_k = pool_k_tokens / pool_total. Contrast with the default random
arm (uniform alpha_k = 1/K per cluster regardless of pool size — the
paper's baseline) and with the search winner (learned w). Emitting it as
a WEIGHTS file lets the natural arm ride the standard custom-arm path:

    python3 scripts/gen_natural_weights.py \
        --pool-dir /home/ma-user/work/100B_stem_parquet_filtered \
        --cluster-cache result/prod4_current/cluster_cache.npz \
        --output /home/ma-user/work/tmp/natural_weights.json

    ARM_NAME=natural WEIGHTS=/home/ma-user/work/tmp/natural_weights.json \
        RUN_DIR=result/prod4_current TARGET_TOKENS=3B TARGET_ARM_NODES=8 \
        MID_DEVICE_BATCH_SIZE=1 AUTO_EVAL=0 LAUNCH=1 \
        nohup bash runs/lib/arm_engine.sh > .../prod4_natural_arm.log 2>&1 &

Also prints the per-cluster est-token pool table — the exact numbers the
20B feasibility math needs (C10/C5 binding checks; char/4 estimates, see
the CHAR_TO_TOKEN_EST legacy debt).

Standalone: numpy only.
"""
import argparse
import json
import os

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="pool-proportional natural weights")
    p.add_argument("--pool-dir", required=True,
                   help="stem pool dir holding metadata_cache.npz")
    p.add_argument("--cluster-cache", required=True,
                   help="cluster_cache.npz from the search stage (final_labels)")
    p.add_argument("--output", required=True,
                   help="output weights JSON (C0..C{k-1} dict)")
    args = p.parse_args()

    cache = os.path.join(args.pool_dir, "metadata_cache.npz")
    if not os.path.exists(cache):
        raise SystemExit(f"ERROR: metadata cache not found: {cache}")
    if not os.path.exists(args.cluster_cache):
        raise SystemExit(f"ERROR: cluster cache not found: {args.cluster_cache}")

    meta = np.load(cache, allow_pickle=False)
    labels = np.load(args.cluster_cache,
                     allow_pickle=False)["final_labels"].astype(np.int64)
    char = meta["doc_char_counts"].astype(np.float64)
    if len(labels) != len(char):
        raise SystemExit(
            f"ERROR: cluster cache has {len(labels):,} labels but the metadata "
            f"cache has {len(char):,} docs — pool/cache mismatch (rerun the "
            f"search stage, or point --pool-dir at the pool the cluster cache "
            f"was built from)")

    tok = char / 4.0  # CHAR_TO_TOKEN_EST: same char/4 heuristic as selection
    valid = labels >= 0
    pool = np.bincount(labels[valid], weights=tok[valid])
    total = pool.sum()
    if total <= 0:
        raise SystemExit("ERROR: empty pool (all char counts zero?)")
    weights = {f"C{k}": float(v / total) for k, v in enumerate(pool)}

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(weights, f, indent=2)

    print("[natural] per-cluster est-token pool (char/4) and natural weights:")
    for k, v in enumerate(pool):
        print(f"  C{k:<2d} pool {v / 1e9:7.2f}B  w={weights[f'C{k}']:.4f}")
    print(f"[natural] total est pool: {total / 1e9:.2f}B tokens")
    print(f"[natural] weights -> {args.output} "
          f"(sum={sum(weights.values()):.6f})")


if __name__ == "__main__":
    main()
