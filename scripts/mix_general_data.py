#!/usr/bin/env python3
"""
Mix STEM data with ClimbMix general data for anti-forgetting during mid-training.

Document-level mixing: 70% STEM + 30% ClimbMix.
Downloads ClimbMix shards from the end (6541 backwards) to avoid overlap with pretrain data (shards 0-999).
Adaptive shard count: downloads only as many ClimbMix shards as needed based on STEM data size.

References:
  - MAI-Thinking-1: 10% General in pretrain mixture
  - Apple Intelligence: "some fraction of bulk pre-train data" in continued pre-training
  - DeepSeek V3: weight_decay=0.1, warmup=2K steps
  - Kimi K2: weight_decay=0.1, warmup=500 steps
"""
import math
import os
import sys
import argparse
import random
import shutil
from pathlib import Path
from multiprocessing.pool import ThreadPool

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    from nanochat.dataset import (
        download_single_file,
        stream_texts_uniform,
        index_to_filename,
        MAX_SHARD,
    )
except ImportError:
    # Direct CLI invocation (not imported by proxy/target runner): nanochat is
    # not on sys.path yet. NANOCHAT_REPO must point at the nanochat-npu checkout
    # (run scripts set it to $NANOCHAT_DIR on the NPU server).
    NANOCHAT_REPO = os.environ.get("NANOCHAT_REPO", "/home/liujin99/nanochat-npu")
    sys.path.insert(0, NANOCHAT_REPO)
    from nanochat.dataset import (
        download_single_file,
        stream_texts_uniform,
        index_to_filename,
        MAX_SHARD,
    )

STEM_RATIO = 0.7
BATCH_PER_FILE = 10000
MIN_CLIMBMIX_SHARDS = 3
MAX_CLIMBMIX_SHARDS = 50
CLIMBMIX_DOCS_PER_SHARD = 500000


def count_stem_docs(stem_train_files):
    """Read parquet metadata to count total documents (metadata only, no data read)."""
    total = 0
    for f in stem_train_files:
        try:
            total += pq.ParquetFile(f).num_rows
        except Exception:
            print(f"  WARNING: could not read metadata for {f}")
    return total


def detect_shard_size(stem_train_files):
    """Read the actual docs-per-shard from the first STEM parquet (metadata only)."""
    if not stem_train_files:
        return BATCH_PER_FILE
    try:
        return pq.ParquetFile(stem_train_files[0]).num_rows
    except Exception:
        return BATCH_PER_FILE


def calc_climbmix_count(stem_docs, stem_ratio):
    """Calculate how many ClimbMix shards are needed for the given STEM doc count."""
    needed_climb = stem_docs * (1 - stem_ratio) / stem_ratio
    n = math.ceil(needed_climb / CLIMBMIX_DOCS_PER_SHARD)
    return max(MIN_CLIMBMIX_SHARDS, min(MAX_CLIMBMIX_SHARDS, n))


def download_climbmix(data_dir, num_shards, num_workers=16):
    """Download last N ClimbMix shards from the end to avoid overlap with pretrain (shards 0-999)."""
    os.makedirs(data_dir, exist_ok=True)
    climb_start = max(0, MAX_SHARD - num_shards + 1)
    climb_ids = list(range(climb_start, MAX_SHARD + 1))

    print(f"  Downloading ClimbMix shards {climb_ids[0]}-{climb_ids[-1]} ({len(climb_ids)} files)")

    remaining = list(climb_ids)
    for round_idx in range(1, 4):
        if not remaining:
            break
        round_workers = max(4, num_workers // round_idx)
        with ThreadPool(round_workers) as pool:
            results = pool.map(lambda i: download_single_file(i, data_dir, "climb"), remaining)
        failed = [remaining[i] for i, r in enumerate(results) if not r]
        if not failed:
            print(f"  Round {round_idx}: all {len(remaining)} files downloaded")
            break
        print(f"  Round {round_idx}: {len(failed)} files still failed")
        remaining = failed

    if remaining:
        print(f"  WARNING: {len(remaining)} ClimbMix files permanently failed: shards {remaining}")

    climb_files = [os.path.join(data_dir, index_to_filename(i)) for i in climb_ids]
    climb_files = [f for f in climb_files if os.path.exists(f)]
    print(f"  Downloaded {len(climb_files)} ClimbMix files")
    return climb_files


def endless_generator(gen_func, files):
    """Cycle through a generator infinitely."""
    while True:
        gen = gen_func(files)
        yield from gen


def mix_data(stem_dir, climb_files, output_dir, num_output_files, batch_per_file=BATCH_PER_FILE, num_npu=8):
    """Mix 70% STEM + 30% ClimbMix at document level."""
    if not climb_files:
        raise ValueError("No ClimbMix files available. Download failed?")

    os.makedirs(output_dir, exist_ok=True)

    all_files = sorted(Path(stem_dir).glob("shard_*.parquet"))
    if not all_files:
        raise ValueError(f"No STEM parquet files found in {stem_dir}")

    stem_files = [str(f) for f in all_files[:-1]]
    val_file = str(all_files[-1]) if len(all_files) >= 1 else None

    if not stem_files:
        raise ValueError(f"No train shards found in {stem_dir} (only val?)")

    # DDP row-group safety: every output shard must contain at least num_npu
    # row groups (dataloader assigns row groups round-robin per rank).
    rg_size = max(1, batch_per_file // (num_npu * 2))

    print(f"  STEM: {len(stem_files)} train shards from {stem_dir}")
    print(f"  ClimbMix: {len(climb_files)} shards")
    print(f"  Output: {num_output_files} files x {batch_per_file} docs each (rg_size={rg_size})")
    print(f"  Ratio: {STEM_RATIO*100:.0f}% STEM + {(1-STEM_RATIO)*100:.0f}% ClimbMix")

    stem_gen = endless_generator(stream_texts_uniform, stem_files)
    climb_gen = endless_generator(stream_texts_uniform, climb_files)

    random.seed(42)
    current = []
    file_idx = 0
    total = num_output_files * batch_per_file
    pbar = tqdm(desc=f"  Mixing {Path(stem_dir).name}", total=total)

    try:
        while file_idx < num_output_files:
            if random.random() < STEM_RATIO:
                txt = next(stem_gen)
            else:
                txt = next(climb_gen)

            current.append(txt)
            pbar.update(1)

            if len(current) >= batch_per_file:
                out_path = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
                pq.write_table(pa.table({"text": current}), out_path, row_group_size=rg_size)
                current = []
                file_idx += 1
    finally:
        if current and file_idx < num_output_files:
            out_path = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
            pq.write_table(pa.table({"text": current}), out_path, row_group_size=rg_size)
            file_idx += 1

        del stem_gen
        del climb_gen
        pbar.close()

    if val_file:
        val_dst = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
        shutil.copy2(val_file, val_dst)
        print(f"  Copied val shard: {Path(val_file).name} -> shard_{file_idx:05d}.parquet")

    print(f"  Done: {file_idx} train + 1 val shard -> {output_dir}")
    return file_idx


def main():
    parser = argparse.ArgumentParser(
        description="Mix STEM data with ClimbMix for anti-forgetting mid-training")
    parser.add_argument("--stem-dir", required=True,
                        help="Directory containing STEM parquet shards (from prepare_data.py)")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for mixed data")
    parser.add_argument("--climbmix-dir", required=True,
                        help="Directory to store/download ClimbMix shards")
    parser.add_argument("--num-output-files", type=int, default=None,
                        help="Number of output mixed shards (default: same as STEM train shards)")
    parser.add_argument("--stem-ratio", type=float, default=0.7,
                        help="STEM ratio (default 0.7 = 70%%)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="Download workers (default 16)")
    parser.add_argument("--num-npu", type=int, default=8,
                        help="NPUs used for training the mixed data (row-group sizing, default 8)")
    args = parser.parse_args()

    global STEM_RATIO
    STEM_RATIO = args.stem_ratio

    all_stem_files = sorted(Path(args.stem_dir).glob("shard_*.parquet"))
    stem_train_files = [str(f) for f in all_stem_files[:-1]] if all_stem_files else []
    if not stem_train_files:
        print("ERROR: No STEM train shards found. Run prepare_data.py first.")
        sys.exit(1)

    stem_train_count = len(stem_train_files)
    num_output_files = args.num_output_files or stem_train_count
    batch_per_file = detect_shard_size(stem_train_files)

    stem_docs = count_stem_docs(stem_train_files)
    needed_shards = calc_climbmix_count(stem_docs, STEM_RATIO)
    needed_climb_docs = int(stem_docs * (1 - STEM_RATIO) / STEM_RATIO)

    print(f"  STEM: {stem_train_count} train shards, {stem_docs:,} docs "
          f"({batch_per_file} docs/shard)")
    print(f"  Need ~{needed_climb_docs:,} ClimbMix docs -> {needed_shards} shards "
          f"(capped at {MIN_CLIMBMIX_SHARDS}-{MAX_CLIMBMIX_SHARDS})")

    existing_climb = []
    climb_start = max(0, MAX_SHARD - needed_shards + 1)
    climb_ids = list(range(climb_start, MAX_SHARD + 1))
    for i in climb_ids:
        fpath = os.path.join(args.climbmix_dir, index_to_filename(i))
        if os.path.exists(fpath):
            existing_climb.append(fpath)

    if len(existing_climb) < needed_shards:
        print(f"  ClimbMix shards incomplete ({len(existing_climb)}/{needed_shards}), downloading...")
        climb_files = download_climbmix(args.climbmix_dir, needed_shards, args.num_workers)
    else:
        climb_files = existing_climb
        print(f"  ClimbMix already downloaded: {len(climb_files)} files")

    mix_data(args.stem_dir, climb_files, args.output_dir, num_output_files, batch_per_file,
             num_npu=args.num_npu)


if __name__ == "__main__":
    main()
