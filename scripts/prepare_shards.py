#!/usr/bin/env python3
"""
Convert a parquet file (sampled_dataset.parquet) into nanochat shard format.
Last shard is a dummy val split (nanochat convention).
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
    n_shards = max(1, math.ceil(n / args.shard_size))
    rg_size = max(1, args.shard_size // (args.num_npu * 2))

    for i in range(n_shards):
        start = i * args.shard_size
        end = min(start + args.shard_size, n)
        shard_texts = texts[start:end]
        shard_table = pa.table({"text": shard_texts})
        pq.write_table(shard_table, os.path.join(args.output_dir, f"shard_{i:05d}.parquet"), row_group_size=rg_size)

    dummy = pa.table({"text": ["dummy"]})
    pq.write_table(dummy, os.path.join(args.output_dir, f"shard_{n_shards:05d}.parquet"), row_group_size=1)

    print(f"  {n} docs -> {n_shards} train shards + 1 dummy val -> {args.output_dir}")


if __name__ == "__main__":
    main()
