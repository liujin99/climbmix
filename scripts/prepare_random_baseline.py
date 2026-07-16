"""Prepare random baseline dataset for CLIMB validation.

Sample N documents randomly from preprocessed shards, matching the
token count of the CLIMB selected dataset.
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

    shard_size = 10000
    n_shards = max(1, math.ceil(len(sampled) / shard_size))
    rg_size = max(1, shard_size // (args.num_npu * 2))

    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(n_shards):
        start = i * shard_size
        end = min(start + shard_size, len(sampled))
        shard_texts = sampled[start:end]
        shard_table = pa.table({"text": shard_texts})
        pq.write_table(
            shard_table,
            os.path.join(args.output_dir, f"shard_{i:05d}.parquet"),
            row_group_size=rg_size,
        )

    dummy = pa.table({"text": ["dummy"]})
    pq.write_table(
        dummy,
        os.path.join(args.output_dir, f"shard_{n_shards:05d}.parquet"),
        row_group_size=1,
    )

    print(f"Random baseline: {len(sampled)} docs -> {n_shards} shards -> {args.output_dir}")


if __name__ == "__main__":
    main()
