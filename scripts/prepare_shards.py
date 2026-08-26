#!/usr/bin/env python3
"""
Convert a parquet file (sampled_dataset.parquet) into nanochat shard format.
Last shard is the val split (nanochat convention: last file = val, real docs).

Crash safety: shards are written to temp names and renamed into place, and a
.done marker is only written after everything succeeded. A directory with
shards but no .done is treated as a crashed partial run: it is wiped and
redone (never fed to the dataloader half-written).

DDP row-group safety: nanochat's dataloader shards row groups round-robin
across ranks (rank r reads rg r, r+W, r+2W, ... per file). Every shard must
therefore contain at least num_npu row groups, or ranks with no assigned
row group spin forever inside the dataloader and HCCL times out.
"""
import argparse
import json
import os
import math
import pyarrow.parquet as pq
import pyarrow as pa


def _atomic_write(args_output_dir, name, table, rg_size):
    shard_path = os.path.join(args_output_dir, name)
    tmp_path = shard_path + ".tmp.parquet"
    pq.write_table(table, tmp_path, row_group_size=rg_size)
    os.replace(tmp_path, shard_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output-dir", required=True, help="Output shard directory")
    parser.add_argument("--num-npu", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    done_marker = os.path.join(args.output_dir, ".done")
    if os.path.exists(done_marker):
        print(f"  Shards already complete (.done), skipping")
        return

    # Shards without .done = crashed partial run: wipe and redo.
    leftovers = [f for f in os.listdir(args.output_dir)
                 if f.startswith("shard_") or f.endswith(".tmp.parquet")]
    if leftovers:
        print(f"  Cleaning {len(leftovers)} partial files from a crashed run (no .done)")
        for f in leftovers:
            os.remove(os.path.join(args.output_dir, f))

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
    # Absorb a tiny remainder into the previous shard: a last shard with
    # < 2*num_npu docs cannot provide one row group per rank -> DDP starvation.
    while n_shards > 1 and n_train - (n_shards - 1) * args.shard_size < 2 * args.num_npu:
        n_shards -= 1
    # Row groups sized from the ACTUAL doc count of the last (smallest) shard,
    # guaranteeing every shard has >= num_npu*2 row groups.
    last_shard_docs = n_train - (n_shards - 1) * args.shard_size
    rg_size = max(1, last_shard_docs // (args.num_npu * 2))

    for i in range(n_shards):
        start = i * args.shard_size
        end = min(start + args.shard_size, n_train)
        shard_texts = train_texts[start:end]
        _atomic_write(args.output_dir, f"shard_{i:05d}.parquet",
                      pa.table({"text": shard_texts}), rg_size)

    _atomic_write(args.output_dir, f"shard_{n_shards:05d}.parquet",
                  pa.table({"text": val_texts}), 1)

    with open(done_marker, "w") as f:
        json.dump({"n_train_shards": n_shards, "val_docs": val_n,
                   "rg_size": rg_size, "num_npu": args.num_npu}, f)

    print(f"  {n} docs -> {n_shards} train shards (rg_size={rg_size}) "
          f"+ 1 val shard ({val_n} docs, rg_size=1) -> {args.output_dir}")


if __name__ == "__main__":
    main()
