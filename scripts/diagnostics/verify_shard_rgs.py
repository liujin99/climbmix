#!/usr/bin/env python3
"""Verify prepare_shards.py DDP row-group safety (no NPU needed).

Simulates nanochat dataloader sharding arithmetic: for a sorted shard dir,
train files = all but last, val = last; rank r reads row groups
r, r+W, r+2W, ... per file. A rank must own >=1 row group somewhere,
else it would hang (the HCCL bug).
"""
import math
import subprocess
import sys
import tempfile
import shutil
import os
import glob

import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "prepare_shards.py")
NUM_NPU = 8


def run_prepare(n_docs, workdir, shard_size=10000):
    outdir = os.path.join(workdir, "shards")
    os.makedirs(outdir, exist_ok=True)
    inp = os.path.join(workdir, "input.parquet")
    pq.write_table(pa.table({"text": [f"doc {i} content" for i in range(n_docs)]}), inp)
    r = subprocess.run([sys.executable, SCRIPT, "--input", inp, "--output-dir", outdir,
                        "--num-npu", str(NUM_NPU), "--shard-size", str(shard_size)],
                       capture_output=True, text=True)
    return r, outdir


def simulate_dataloader(outdir, world_size):
    """Return {rank: (n_train_rgs, n_val_rgs)} and per-file rg counts."""
    files = sorted(glob.glob(os.path.join(outdir, "shard_*.parquet")))
    train_files, val_file = files[:-1], files[-1]
    per_file = []
    for f in train_files:
        per_file.append(pq.ParquetFile(f).num_row_groups)
    val_rgs = pq.ParquetFile(val_file).num_row_groups

    ranks = {}
    for r in range(world_size):
        n_train = sum(1 for nrg in per_file for rg in range(nrg) if rg % world_size == r)
        n_val = sum(1 for rg in range(val_rgs) if rg % world_size == r)
        ranks[r] = (n_train, n_val)
    return per_file, val_rgs, ranks, os.path.basename(val_file)


def case(name, n_docs, expect_pass=True, min_train_rgs_per_rank=1, shard_size=10000):
    workdir = tempfile.mkdtemp(prefix=f"rgv_{name}_")
    try:
        r, outdir = run_prepare(n_docs, workdir, shard_size)
        if not expect_pass:
            ok = r.returncode != 0 and "num_npu" in r.stderr
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: rejected (rc={r.returncode})")
            print(f"        stderr: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else '(none)'}")
            return ok
        if r.returncode != 0:
            print(f"[FAIL] {name}: prepare_shards exited {r.returncode}\n{r.stderr}")
            return False

        per_file, val_rgs, ranks, val_name = simulate_dataloader(outdir, NUM_NPU)
        starved = [r for r, (t, v) in ranks.items() if t < min_train_rgs_per_rank or v < 1]
        val_rows = pq.ParquetFile(os.path.join(outdir, val_name)).metadata.num_rows
        # val must be real docs (not "dummy")
        val_texts = pq.read_table(os.path.join(outdir, val_name), columns=["text"])["text"].to_pylist()
        real_val = all(t != "dummy" for t in val_texts)
        ok = not starved and real_val and val_rgs >= NUM_NPU
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: n={n_docs} train_rgs/file={per_file} "
              f"val={val_rows} docs/{val_rgs} rgs, real_val={real_val}, "
              f"min/rank={min(t for t, _ in ranks.values())}t/{min(v for _, v in ranks.values())}v"
              + (f" STARVED={starved}" if starved else ""))
        return ok
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    results = []
    # speedrun shape: ~4K docs (10M tokens cap) -> was 7 rg (bug), must be >= 16
    results.append(case("speedrun_4k", 4000, min_train_rgs_per_rank=2))
    # full-run shape: 10K docs -> single full shard
    results.append(case("fullrun_10k", 10000, min_train_rgs_per_rank=2))
    # multi-shard full: 25000 -> 2 full shards + 1 partial (5000 docs) + real val
    results.append(case("fullrun_25k", 25000, min_train_rgs_per_rank=2))
    # boundary pass: n=32=4*num_npu -> train 16 docs/16 rgs + val 16 docs/16 rgs, 2 rgs/rank
    results.append(case("boundary_32", 32, min_train_rgs_per_rank=2))
    # boundary rejects: 15 and 16 (< 4*num_npu=32) must fail fast
    results.append(case("reject_15", 15, expect_pass=False))
    results.append(case("reject_16", 16, expect_pass=False))
    print()
    print(f"{'ALL PASS' if all(results) else 'SOME FAILED'} ({sum(results)}/{len(results)})")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
