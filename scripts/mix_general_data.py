#!/usr/bin/env python3
"""
Mix STEM data with ClimbMix general data for anti-forgetting during mid-training.

Document-level mixing: STEM + ClimbMix general data (default 70/30, configurable
via --stem-ratio / mix_data(stem_ratio=...)).
Downloads ClimbMix shards from the end (6541 backwards) to avoid overlap with pretrain data (shards 0-999).
Adaptive shard count: downloads only as many ClimbMix shards as needed based on STEM data size.

References:
  - MAI-Thinking-1: 10% General in pretrain mixture
  - Apple Intelligence: "some fraction of bulk pre-train data" in continued pre-training
  - DeepSeek V3: weight_decay=0.1, warmup=2K steps
  - Kimi K2: weight_decay=0.1, warmup=500 steps
"""
import json
import math
import os
import sys
import time
import argparse
import random
import shutil
import threading
from pathlib import Path
from multiprocessing.pool import ThreadPool

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from climbmix.utils.io_utils import shard_content_key  # noqa: E402

# HF reachability (server findings 2026-09-12/15): the egress proxy
# CONNECT-tunnels to huggingface.co return 503 bursts (90+ consecutive); only
# hf-mirror.com is reliable. run_climbmix.sh exports HF_ENDPOINT, but the
# pre-launched random-arm dispatch (spawned by run_search.sh BEFORE exec'ing
# run_climbmix.sh) and direct CLI invocations do not inherit it. This module is
# the ONLY download entry point, and dataset.py bakes BASE_URL from the env at
# import time — so default it here, before that import. Override with an
# explicit HF_ENDPOINT to use the origin.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[mix] WARNING: {name}={raw!r} is not an int — using {default}")
        return default


# General-side shard cap. Default 50 keeps every historical run byte-identical;
# scale-ups that need more general tokens (e.g. a 10B-consume target arm needs
# ~64 shards for a clean 70/30 — without the cap it would silently clamp to 50
# and drift the ratio to ~76/24) override via CLIMBMIX_MAX_SHARDS.
MAX_CLIMBMIX_SHARDS = max(MIN_CLIMBMIX_SHARDS, _env_int("CLIMBMIX_MAX_SHARDS", 50))
# Measured 2026-09-11 on the real HF ClimbMix shards (server probe): 84,992-86,016
# docs per file (index_to_filename range shard_06540-06542, ~2,940 chars/doc).
# The historical 500,000 was wrong by ~6x — it made calc_climbmix_count request
# 6x too FEW shards, so the general draw cycled (repeated docs) at real scale.
CLIMBMIX_DOCS_PER_SHARD = 85000


def count_stem_docs(stem_train_files):
    """Read parquet metadata to count total documents (metadata only, no data read)."""
    total = 0
    for f in stem_train_files:
        try:
            total += pq.ParquetFile(f).metadata.num_rows
        except Exception as e:
            raise RuntimeError(f"could not read parquet metadata for {f}: {e}") from e
    return total


def detect_shard_size(stem_train_files):
    """Read the actual docs-per-shard from the first STEM parquet (metadata only)."""
    if not stem_train_files:
        return BATCH_PER_FILE
    try:
        return pq.ParquetFile(stem_train_files[0]).metadata.num_rows
    except Exception as e:
        raise RuntimeError(
            f"could not read parquet metadata for {stem_train_files[0]}: {e}"
        ) from e


def calc_climbmix_count(stem_docs, stem_ratio, max_shards=MAX_CLIMBMIX_SHARDS):
    """Calculate how many ClimbMix shards are needed for the given STEM doc count.

    +1 safety shard: CLIMBMIX_DOCS_PER_SHARD (85K) is an ESTIMATE — real
    shards average ~84.6K docs, so a bare ceil() can land 1-2K docs short of
    the draw + binomial margin and the fail-loud general-supply guard refuses
    to mix (prod4 cfg11 2026-09-16: 44 shards = 3,724,288 docs vs 3,726,000
    + 8,075 needed — short by 0.046%, entire arm blocked). The extra shard
    is ~250MB of already-cached download and turns the estimate error into
    slack instead of a hard stop.
    """
    needed_climb = stem_docs * (1 - stem_ratio) / stem_ratio
    n = math.ceil(needed_climb / CLIMBMIX_DOCS_PER_SHARD) + 1
    return max(MIN_CLIMBMIX_SHARDS, min(max_shards, n))


def binomial_margin(total_docs, stem_ratio):
    """5-sigma binomial safety margin on the STEM draw — the exact quantity
    the supply preflight demands on top of the nominal draw."""
    return 5 * math.sqrt(total_docs * stem_ratio * (1 - stem_ratio)) + 1


def calc_output_files(stem_docs, batch_per_file, stem_ratio):
    """Largest output-file count whose STEM draw + binomial margin fits the
    available STEM docs — the supply preflight passes BY CONSTRUCTION.

    The historical floor(stem_docs / (batch*ratio)) left only the remainder
    r = stem_docs mod (batch*ratio) in [0, batch*ratio) docs of headroom,
    while the preflight demands a ~5-sigma binomial margin (~0.25% of the
    draw). Whenever r < margin the config died on a sub-0.25% near-miss
    (prod3 iter-1: 10/16 configs killed this way, shortest by 205 docs out
    of 1.22M). Shrinking the output by at most one file absorbs the margin.
    """
    if batch_per_file <= 0:
        raise ValueError(f"batch_per_file must be positive, got {batch_per_file}")
    if not (0.0 < stem_ratio < 1.0):
        raise ValueError(f"stem_ratio must be in (0, 1), got {stem_ratio}")
    n = max(1, int(stem_docs // (batch_per_file * stem_ratio)))
    while n > 1:
        total = n * batch_per_file
        if stem_ratio * total + binomial_margin(total, stem_ratio) <= stem_docs:
            break
        n -= 1
    return n


def download_climbmix(data_dir, num_shards, num_workers=16):
    """Download last N ClimbMix shards from the end to avoid overlap with pretrain (shards 0-999).

    Cross-process safe: the climb-arm mix (main script, Step 5) and the
    random-arm mix (pre-launched dispatch) can both arrive here for the same
    missing shards — they serialize on <data_dir>/.download.lock and the
    second arriver re-checks what the first left behind. Without the lock,
    two writers append to the SAME .tmp with interleaved Range resumes and
    corrupt the parquet (validate_parquet then fails, masking the cause).
    """
    import fcntl

    os.makedirs(data_dir, exist_ok=True)
    climb_start = max(0, MAX_SHARD - num_shards + 1)
    climb_ids = list(range(climb_start, MAX_SHARD + 1))

    lock_path = os.path.join(data_dir, ".download.lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return _download_climbmix_locked(
                data_dir, climb_ids, num_workers)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _download_climbmix_locked(data_dir, climb_ids, num_workers):
    fname = lambda i: os.path.join(data_dir, index_to_filename(i))
    missing = [i for i in climb_ids if not os.path.exists(fname(i))]
    if not missing:
        print(f"  ClimbMix shards complete ({len(climb_ids)} files — "
              f"lock peer downloaded them)", flush=True)
        return [fname(i) for i in climb_ids if os.path.exists(fname(i))]

    print(f"  Downloading ClimbMix shards {index_to_filename(climb_ids[0])}-"
          f"{index_to_filename(climb_ids[-1])} ({len(missing)}/{len(climb_ids)} "
          f"missing; endpoint {os.environ.get('HF_ENDPOINT')})", flush=True)

    remaining = list(missing)
    for round_idx in range(1, 4):
        if not remaining:
            break
        if round_idx > 1:
            # Transient proxy/network failures (e.g. 503 bursts lasting tens of
            # minutes) deserve a cooldown before the next round of retries.
            print(f"  Cooling down 60s before round {round_idx}...", flush=True)
            time.sleep(60)
        round_workers = max(4, num_workers // round_idx)

        def _dl(i):
            return i, download_single_file(i, data_dir, "climb")

        t0 = time.time()
        succeeded = set()
        with ThreadPool(round_workers) as pool:
            for i, ok in pool.imap_unordered(_dl, remaining):
                if ok:
                    succeeded.add(i)
                print(f"    {index_to_filename(i)}: {'ok' if ok else 'FAILED'} "
                      f"({len(succeeded)}/{len(remaining)} ok, "
                      f"{time.time() - t0:.0f}s)", flush=True)
        failed = [i for i in remaining if i not in succeeded]
        if not failed:
            print(f"  Round {round_idx}: all {len(remaining)} files downloaded",
                  flush=True)
            remaining = []
            break
        print(f"  Round {round_idx}: {len(failed)} files still failed", flush=True)
        remaining = failed

    if remaining:
        raise RuntimeError(
            f"ClimbMix download failed for shards "
            f"{[index_to_filename(i) for i in remaining]} after 3 rounds. "
            f"Check proxy/network, or pre-download manually into {data_dir} "
            f"(HF_ENDPOINT={os.environ.get('HF_ENDPOINT')} is the endpoint in use)."
        )

    climb_files = [fname(i) for i in climb_ids if os.path.exists(fname(i))]
    print(f"  Downloaded {len(climb_files)} ClimbMix files", flush=True)
    return climb_files


def endless_generator(gen_func, files):
    """Cycle through a generator infinitely."""
    while True:
        gen = gen_func(files)
        yield from gen


_MIX_LOCK = threading.Lock()


def mix_data(stem_dir, climb_files, output_dir, num_output_files, batch_per_file=BATCH_PER_FILE, num_npu=8, stem_ratio=None, allow_general_repeat=False):
    """Mix STEM + ClimbMix general data at document level.

    stem_ratio: fraction of output docs drawn from STEM (default: module
    STEM_RATIO, i.e. 0.7). Callers that loaded this module (proxy_runner /
    target_runner) MUST pass their own ratio — the module default silently
    diverged from their shard-count calculation when it differed from 0.7.

    allow_general_repeat: the general-side draw uses an endless (cycling)
    generator — when the available general docs fall short of the quota,
    it REPEATS them silently. The supply preflight turns that into a loud
    error; pass True (CLI: --allow-general-repeat) only as a deliberate,
    documented choice (e.g. large-scale sampling where the general pool is
    the binding constraint).

    Crash safety: shards are written to temp names and renamed into place; a
    .done marker is written only after everything (incl. the val shard copy)
    succeeded. Pre-existing shards without .done are treated as a crashed
    partial run and wiped before redoing.

    Thread safety: mixes serialize on a module lock. Both the ratio draws
    here (random.seed(42) below) and stream_texts_uniform's per-generator
    reseed use the GLOBAL random module, so two concurrent mixes in one
    process (the local parallel search runs sibling experiments in threads)
    interleave their streams — the output shards were statistically fine
    but not reproducible run to run (live: speedrun exp_0000's train shards
    could not be regenerated; only the pre-mix val shard matched). Mixing
    costs ~1s, so serializing is free.
    """
    with _MIX_LOCK:
        return _mix_data_locked(stem_dir, climb_files, output_dir,
                                num_output_files, batch_per_file,
                                num_npu, stem_ratio, allow_general_repeat)


def _mix_done_stale(done: dict, *, ratio: float, stem_key: str,
                    general_key: str) -> bool:
    """.done staleness guard for a mixed output dir.

    True = stale (remix). Identity = stem_ratio + the stem shard set +
    the general (ClimbMix) shard set — both sides of the mix. Legacy
    .done files (pre-2026-09-16, no keys) are unverifiable → stale.
    Catches the local twin of the OBS landmine: same output dir, new
    budget upstream (fresh {arm}_shards) or a changed general supply
    (e.g. the +1 safety shard from calc_climbmix_count) must not
    silently reuse the old mixture.
    """
    if done.get("stem_ratio") is None or float(done["stem_ratio"]) != ratio:
        return True
    if done.get("stem_key") != stem_key:
        return True
    return done.get("general_key") != general_key


def _mix_data_locked(stem_dir, climb_files, output_dir, num_output_files, batch_per_file=BATCH_PER_FILE, num_npu=8, stem_ratio=None, allow_general_repeat=False):
    if not climb_files:
        raise ValueError("No ClimbMix files available. Download failed?")
    ratio = STEM_RATIO if stem_ratio is None else stem_ratio
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"stem_ratio must be in (0, 1), got {ratio}")

    os.makedirs(output_dir, exist_ok=True)

    done_marker = os.path.join(output_dir, ".done")
    if os.path.exists(done_marker):
        with open(done_marker) as f:
            done_info = json.load(f)
        # A .done from a different ratio / stem set / general set is stale
        # output (the pipeline fingerprint normally archives the dir before
        # we get here; this guard covers direct CLI reuse of an output dir).
        stem_key = shard_content_key(stem_dir)
        general_key = shard_content_key(os.path.dirname(climb_files[0]))
        if _mix_done_stale(done_info, ratio=ratio, stem_key=stem_key,
                           general_key=general_key):
            print(f"  .done identity mismatch (ratio={done_info.get('stem_ratio')}"
                  f" vs {ratio}, stem_key={done_info.get('stem_key', 'legacy')}"
                  f" vs {stem_key}, general_key={done_info.get('general_key', 'legacy')}"
                  f" vs {general_key}) -> remixin (stale output)")
        else:
            print(f"  Mix already complete (.done), skipping: {output_dir}")
            return done_info.get("n_train_shards", 0)

    leftovers = [f for f in os.listdir(output_dir)
                 if f.startswith("shard_") or f.endswith(".tmp.parquet")]
    if leftovers:
        print(f"  Cleaning {len(leftovers)} partial files from a crashed run (no .done)")
        for f in leftovers:
            os.remove(os.path.join(output_dir, f))

    all_files = sorted(Path(stem_dir).glob("shard_*.parquet"))
    if not all_files:
        raise ValueError(f"No STEM parquet files found in {stem_dir}")

    stem_files = [str(f) for f in all_files[:-1]]
    val_file = str(all_files[-1]) if len(all_files) >= 1 else None

    if not stem_files:
        raise ValueError(f"No train shards found in {stem_dir} (only val?)")

    # ── supply preflight: both draws cycle (endless_generator) — a short
    # supply would REPEAT docs silently, ratio intact, zero signal. Count
    # actual docs (parquet metadata only, fast) and fail loud instead.
    total_docs = num_output_files * batch_per_file
    margin = binomial_margin(total_docs, ratio)
    stem_need = total_docs * ratio
    climb_need = total_docs * (1 - ratio)
    stem_have = count_stem_docs(stem_files)
    climb_have = count_stem_docs(climb_files)
    if stem_have < stem_need + margin:
        raise ValueError(
            f"STEM supply insufficient: {stem_have:,} docs available but the mix "
            f"draws ~{int(stem_need):,} (+{int(margin)} binomial margin) — the "
            f"cycling draw would REPEAT STEM docs silently. Check the selection "
            f"output (shortfall?) or num_output_files/batch_per_file alignment.")
    if climb_have < climb_need + margin:
        msg = (f"general data insufficient: {climb_have:,} docs available but "
               f"the mix draws ~{int(climb_need):,} (+{int(margin)} binomial "
               f"margin) — the cycling draw would REPEAT general docs silently "
               f"(ratio stays perfect, model sees duplicates). Raise "
               f"--max-climbmix-shards (check general-pool availability), lower "
               f"the STEM budget, or pass --allow-general-repeat to accept "
               f"repetition deliberately.")
        if not allow_general_repeat:
            raise ValueError("✗ " + msg)
        print(f"  ⚠ ALLOWED general repetition: {msg}")

    # DDP row-group safety: every output shard must contain at least num_npu
    # row groups (dataloader assigns row groups round-robin per rank).
    rg_size = max(1, batch_per_file // (num_npu * 2))

    def _atomic_write(name, table):
        out_path = os.path.join(output_dir, name)
        tmp_path = out_path + ".tmp.parquet"
        pq.write_table(table, tmp_path, row_group_size=rg_size)
        os.replace(tmp_path, out_path)

    print(f"  STEM: {len(stem_files)} train shards from {stem_dir}")
    print(f"  ClimbMix: {len(climb_files)} shards")
    print(f"  Output: {num_output_files} files x {batch_per_file} docs each (rg_size={rg_size})")
    print(f"  Ratio: {ratio*100:.0f}% STEM + {(1-ratio)*100:.0f}% ClimbMix")

    stem_gen = endless_generator(stream_texts_uniform, stem_files)
    climb_gen = endless_generator(stream_texts_uniform, climb_files)

    random.seed(42)
    current = []
    file_idx = 0
    # Parallel parquet writes (2026-09-16): the interleave loop below is
    # UNCHANGED and stays sequential — random.random() call order and the
    # generator draw order are the determinism contract. Each finished
    # batch is handed to a worker; pyarrow write+compression releases the
    # GIL, so workers overlap the ~0.35s/shard write cost that dominated
    # the old single-core profile (28K docs/s on a 192-vCPU host, ~1% CPU).
    # Byte-identical to the serial writer, asserted in test_prod4_fixes.py.
    write_workers = max(1, int(os.environ.get("CLIMB_MIX_WRITE_WORKERS") or "16"))
    pbar = tqdm(desc=f"  Mixing {Path(stem_dir).name}",
                total=num_output_files, unit="shard")
    pool = ThreadPool(write_workers)
    pending = []

    def _write_batch(idx, texts):
        _atomic_write(f"shard_{idx:05d}.parquet", pa.table({"text": texts}))

    try:
        while file_idx < num_output_files:
            if random.random() < ratio:
                txt = next(stem_gen)
            else:
                txt = next(climb_gen)

            current.append(txt)

            if len(current) >= batch_per_file:
                pending.append(pool.apply_async(_write_batch,
                                                (file_idx, current)))
                pbar.update(1)
                current = []
                file_idx += 1
    finally:
        if current and file_idx < num_output_files:
            pending.append(pool.apply_async(_write_batch, (file_idx, current)))
            pbar.update(1)
            file_idx += 1
        for fut in pending:
            fut.get()  # write failures propagate BEFORE any .done marker
        pool.close()
        pool.join()
        del stem_gen
        del climb_gen
        pbar.close()

    if val_file:
        val_dst = os.path.join(output_dir, f"shard_{file_idx:05d}.parquet")
        tmp_dst = val_dst + ".tmp.parquet"
        shutil.copy2(val_file, tmp_dst)
        os.replace(tmp_dst, val_dst)
        print(f"  Copied val shard: {Path(val_file).name} -> shard_{file_idx:05d}.parquet")

    with open(done_marker, "w") as f:
        json.dump({"n_train_shards": file_idx, "has_val": bool(val_file),
                    "batch_per_file": batch_per_file, "rg_size": rg_size,
                    "stem_ratio": ratio,
                    "stem_key": shard_content_key(stem_dir),
                    "general_key": shard_content_key(
                        os.path.dirname(climb_files[0]))}, f)

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
    parser.add_argument("--max-climbmix-shards", type=int, default=MAX_CLIMBMIX_SHARDS,
                        help="cap on general-data shards (default 50 = historical cap). "
                             "When the cap binds, the mix draws FEWER general shards than "
                             "the quota needs — the supply preflight then fails loud "
                             "(raise this flag or accept repetition via "
                             "--allow-general-repeat).")
    parser.add_argument("--allow-general-repeat", action="store_true",
                        help="deliberately accept general-doc repetition when the pool "
                             "falls short of the quota (loud warning still printed). "
                             "Default: hard error (the cycling draw would otherwise "
                             "repeat docs silently).")
    args = parser.parse_args()

    # Progress visibility: when piped (nohup redirect / dispatch run_logged)
    # stdout is block-buffered and phase prints sit invisible for minutes —
    # a 30-min silent shard download is indistinguishable from a hang.
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(line_buffering=True)
        except Exception:
            pass

    global STEM_RATIO
    STEM_RATIO = args.stem_ratio

    all_stem_files = sorted(Path(args.stem_dir).glob("shard_*.parquet"))
    stem_train_files = [str(f) for f in all_stem_files[:-1]] if all_stem_files else []
    if not stem_train_files:
        print("ERROR: No STEM train shards found. Run prepare_data.py first.")
        sys.exit(1)

    stem_train_count = len(stem_train_files)
    stem_docs = count_stem_docs(stem_train_files)
    batch_per_file = detect_shard_size(stem_train_files)
    # Pool sizing: mix total docs = STEM docs / stem_ratio (minus the binomial
    # margin) so the pool carries the FULL selection plus its general
    # complement AND the supply preflight (draw + 5-sigma margin) passes by
    # construction — see calc_output_files.
    default_output_files = calc_output_files(stem_docs, batch_per_file, STEM_RATIO)
    num_output_files = args.num_output_files or default_output_files

    stem_docs = count_stem_docs(stem_train_files)
    needed_shards = calc_climbmix_count(stem_docs, STEM_RATIO, args.max_climbmix_shards)
    needed_climb_docs = int(stem_docs * (1 - STEM_RATIO) / STEM_RATIO)

    print(f"  STEM: {stem_train_count} train shards, {stem_docs:,} docs "
          f"({batch_per_file} docs/shard)")
    print(f"  Need ~{needed_climb_docs:,} ClimbMix docs -> {needed_shards} shards "
          f"(cap {args.max_climbmix_shards})")

    existing_climb = []
    climb_start = max(0, MAX_SHARD - needed_shards + 1)
    climb_ids = list(range(climb_start, MAX_SHARD + 1))
    for i in climb_ids:
        fpath = os.path.join(args.climbmix_dir, index_to_filename(i))
        if os.path.exists(fpath):
            existing_climb.append(fpath)

    if len(existing_climb) < needed_shards:
        print(f"  ClimbMix shards incomplete ({len(existing_climb)}/{needed_shards}), downloading...", flush=True)
        climb_files = download_climbmix(args.climbmix_dir, needed_shards, args.num_workers)
    else:
        climb_files = existing_climb
        print(f"  ClimbMix already downloaded: {len(climb_files)} files")

    mix_data(args.stem_dir, climb_files, args.output_dir, num_output_files, batch_per_file,
             num_npu=args.num_npu, stem_ratio=args.stem_ratio,
             allow_general_repeat=args.allow_general_repeat)


if __name__ == "__main__":
    main()
