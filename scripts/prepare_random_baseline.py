"""Prepare random baseline dataset for CLIMB validation.

Sample N documents randomly from preprocessed shards, matching the
token count of the CLIMB selected dataset. Last shard is the val split
(real docs, nanochat convention: last file = val).

DDP row-group safety: nanochat's dataloader shards row groups round-robin
across ranks, so every shard must contain at least num_npu row groups
(see prepare_shards.py).
"""

import argparse
import os
import math
import random
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-docs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-npu", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)

    shard_files = sorted(
        f for f in os.listdir(args.data_dir) if f.startswith("preprocessed_") and f.endswith(".parquet")
    )
    if not shard_files:
        raise FileNotFoundError(f"No preprocessed_*.parquet in {args.data_dir}")

    all_texts = []
    for sf in shard_files:
        path = os.path.join(args.data_dir, sf)
        try:
            table = pq.read_table(path, columns=["text"])
            all_texts.extend(table["text"].to_pylist())
        except Exception:
            continue

    sampled = random.sample(all_texts, min(args.num_docs, len(all_texts)))
    n = len(sampled)

    if n < 4 * args.num_npu:
        raise SystemExit(
            f"ERROR: only {n} docs < 4*num_npu ({4 * args.num_npu}). Need >= 2*num_npu docs each "
            f"for train and val so that every rank owns >= 2 row groups in both splits; "
            f"the DDP dataloader assigns row groups round-robin per rank and ranks with "
            f"no row group hang forever before the first all_reduce."
        )

    # Real val split (tail of the sampled data), same policy as prepare_shards.py
    val_n = min(256, max(2 * args.num_npu, n // 100))
    train_texts = sampled[:n - val_n]
    val_texts = sampled[n - val_n:]

    shard_size = 10000
    n_train = len(train_texts)
    n_shards = max(1, math.ceil(n_train / shard_size))
    # Row groups sized from the ACTUAL doc count of the last (smallest) shard
    last_shard_docs = n_train - (n_shards - 1) * shard_size
    rg_size = max(1, last_shard_docs // (args.num_npu * 2))

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, n_train)
        shard_texts = train_texts[start:end]
        shard_table = pa.table({"text": shard_texts})
        pq.write_table(
            shard_table,
            os.path.join(args.output_dir, f"shard_{i:05d}.parquet"),
            row_group_size=rg_size,
        )

    pq.write_table(
        pa.table({"text": val_texts}),
        os.path.join(args.output_dir, f"shard_{n_shards:05d}.parquet"),
        row_group_size=1,
    )

    print(f"Random baseline: {n} docs -> {n_shards} shards (rg_size={rg_size}) "
          f"+ 1 val shard ({val_n} docs, rg_size=1) -> {args.output_dir}")


if __name__ == "__main__":
    main()
