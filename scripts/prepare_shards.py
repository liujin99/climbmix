#!/usr/bin/env python3
"""
Convert a parquet file (sampled_dataset.parquet) into nanochat shard format.
Last shard is the val split (nanochat convention: last file = val, real docs).

DDP row-group safety: nanochat's dataloader shards row groups round-robin
across ranks (rank r reads rg r, r+W, r+2W, ... per file). Every shard must
therefore contain at least num_npu row groups, or ranks with no assigned
row group spin forever inside the dataloader and HCCL times out.
"""
import argparse
import os
import math
import pyarrow.parquet as pq
import pyarrow as pa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output shard directory")
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    existing = [f for f in os.listdir(args.output_dir) if f.startswith("shard_")]
    if existing:
        print(f"  Shards already exist ({len(existing)}), skipping")
        return

    table = pq.read_table(args.input, columns=["text"])
    texts = table["text"].to_pylist()
    n = len(texts)

    if n < 4 * args.num_npu:
        raise SystemExit(
            f"ERROR: only {n} docs < 4*num_npu ({4 * args.num_npu}). Need >= 2*num_npu docs each "
            f"for train and val so that every rank owns >= 2 row groups in both splits; "
            f"the DDP dataloader assigns row groups round-robin per rank and ranks with "
            f"no row group hang forever before the first all_reduce."
        )

    # Hold out a real val split (tail of the data): ~1% of docs, capped at 256,
    # at least 2 docs per NPU so every rank gets >= 2 val row groups (rg_size=1).
    val_n = min(256, max(2 * args.num_npu, n // 100))
    train_texts = texts[:n - val_n]
    val_texts = texts[n - val_n:]

    n_train = len(train_texts)
    n_shards = max(1, math.ceil(n_train / args.shard_size))
    # Row groups sized from the ACTUAL doc count of the last (smallest) shard,
    # guaranteeing every shard has >= num_npu*2 row groups.
    last_shard_docs = n_train - (n_shards - 1) * args.shard_size
    rg_size = max(1, last_shard_docs // (args.num_npu * 2))

    for i in range(n_shards):
        start = i * args.shard_size
        end = min(start + args.shard_size, n_train)
        shard_texts = train_texts[start:end]
        shard_table = pa.table({"text": shard_texts})
        pq.write_table(shard_table, os.path.join(args.output_dir, f"shard_{i:05d}.parquet"), row_group_size=rg_size)

    pq.write_table(pa.table({"text": val_texts}),
                   os.path.join(args.output_dir, f"shard_{n_shards:05d}.parquet"),
                   row_group_size=1)

    print(f"  {n} docs -> {n_shards} train shards (rg_size={rg_size}) "
          f"+ 1 val shard ({val_n} docs, rg_size=1) -> {args.output_dir}")


if __name__ == "__main__":
    main()
