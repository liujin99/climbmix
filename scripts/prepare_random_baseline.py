"""Prepare random baseline dataset for CLIMB validation.

Sample N documents randomly from the data pool, matching the doc count
of the CLIMB selected dataset. Last shard is the val split (real docs,
nanochat convention: last file = val).

Streaming-safe: only parquet METADATA is read in pass 1 (row counts);
pass 2 reads each file once and keeps only its sampled share, so peak
memory is ~1 file + the sample (~GBs on the 116M-doc pool) instead of
the whole pool (~1TB of Python strings).

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


def sample_documents(data_dir: str, num_docs: int) -> list:
    """Uniformly sample num_docs texts from all *.parquet files in data_dir.

    Pass 1 reads row counts from metadata only. The sample is allocated
    across files proportionally (largest remainder), then each file is read
    once and its share is drawn with random.sample. The result is shuffled
    so the val tail (taken later) is unbiased w.r.t. file order.
    """
    shard_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".parquet")
    )
    if not shard_files:
        raise FileNotFoundError(f"No *.parquet files in {data_dir}")

    file_rows = []
    for sf in shard_files:
        path = os.path.join(data_dir, sf)
        try:
            file_rows.append((path, pq.ParquetFile(path).metadata.num_rows))
        except Exception:
            print(f"  WARNING: could not read metadata for {sf}, skipping")
    total_docs = sum(r for _, r in file_rows)
    if total_docs == 0:
        raise FileNotFoundError(f"No readable parquet data in {data_dir}")

    n_want = min(num_docs, total_docs)

    # Proportional allocation with largest-remainder rounding
    alloc = [n_want * rows // total_docs for _, rows in file_rows]
    rem = n_want - sum(alloc)
    if rem > 0:
        frac_order = sorted(
            range(len(file_rows)),
            key=lambda i: -(n_want * file_rows[i][1] % total_docs),
        )
        for i in range(rem):
            alloc[frac_order[i % len(frac_order)]] += 1

    sampled = []
    for (path, rows), k in zip(file_rows, alloc):
        if k <= 0 or rows == 0:
            continue
        try:
            table = pq.read_table(path, columns=["text"])
        except Exception:
            print(f"  WARNING: could not read {path}, skipping")
            continue
        idx = random.sample(range(rows), min(k, rows))
        sampled.extend(table.take(idx)["text"].to_pylist())
        del table

    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-docs", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-npu", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)

    sampled = sample_documents(args.data_dir, args.num_docs)
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
    # Absorb a tiny remainder into the previous shard: a last shard with
    # < 2*num_npu docs cannot provide one row group per rank -> DDP starvation.
    while n_shards > 1 and n_train - (n_shards - 1) * shard_size < 2 * args.num_npu:
        n_shards -= 1
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
