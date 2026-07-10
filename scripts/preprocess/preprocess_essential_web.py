#!/usr/bin/env python3
"""
Preprocess Essential-Web v1.0: extract FDC domain labels and quality signals.

Self-contained version for climbmix — no dependency on quadmix package.
FDC mapping and quality column names are defined inline.

Usage:
  python scripts/preprocess/preprocess_essential_web.py \
      --input-dir data/essential-web-v1 \
      --output-dir data/essential-web-v1-preprocessed
"""

import argparse, json, os, sys, time, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

FDC_PREFIX_TO_DOMAIN = {
    "00": 0,
    "01": 1, "02": 1, "03": 1, "04": 1, "05": 1,
    "06": 1, "07": 1, "08": 1, "09": 1,
    "10": 2, "11": 2, "12": 2, "13": 2, "14": 2,
    "15": 2, "16": 2, "17": 2, "18": 2, "19": 2,
    "20": 3, "21": 3, "22": 3, "23": 3, "24": 3,
    "25": 3, "26": 3, "27": 3, "28": 3, "29": 3,
    "32": 4, "34": 4,
    "33": 5,
    "37": 6,
    "30": 7, "31": 7, "35": 7, "36": 7, "38": 7, "39": 7, "92": 7,
    "40": 8, "41": 8, "42": 8,
    "51": 9,
    "53": 10, "54": 10,
    "50": 11, "52": 11, "55": 11, "56": 11, "57": 11, "58": 11, "59": 11,
    "61": 12,
    "65": 13,
    "60": 14, "62": 14, "66": 14, "67": 14, "68": 14, "69": 14,
    "63": 15,
    "70": 16, "71": 16, "72": 16, "73": 16, "74": 16,
    "75": 16, "76": 16, "77": 16, "78": 16,
    "79": 17,
    "80": 18, "81": 18, "82": 18, "83": 18, "84": 18,
    "85": 18, "86": 18, "87": 18, "88": 18, "89": 18,
    "90": 19, "93": 19, "94": 19, "95": 19, "96": 19,
    "97": 19, "98": 19, "99": 19,
    "91": 20,
    "64": 21,
}

NUM_DOMAINS = 22

DOMAIN_NAMES = [
    "Computers_and_Electronics", "News_and_General_Works",
    "Philosophy_and_Psychology", "Religion", "Law_and_Government",
    "Economics_and_Finance", "Education", "People_and_Society",
    "English_Language", "Mathematics", "Physics_and_Chemistry",
    "Earth_and_Life_Sciences", "Medicine_and_Health",
    "Business_and_Management", "Engineering", "Agriculture",
    "Arts_and_Entertainment", "Sports_and_Recreation",
    "Books_and_Literature", "History", "Geography_and_Travel",
    "Home_Economics",
]

FASTTEXT_FIELDS = [
    "dclm", "fineweb_edu_approx", "english",
    "eai_general_math", "eai_open_web_math",
]

QUALITY_COLUMNS = [
    "qs_dclm", "qs_fineweb_edu_approx", "qs_english",
    "qs_eai_general_math", "qs_eai_open_web_math",
]


def extract_domain_level_2(eai_taxonomy):
    if isinstance(eai_taxonomy, str):
        try:
            eai_taxonomy = json.loads(eai_taxonomy)
        except (json.JSONDecodeError, ValueError):
            return -1
    if not isinstance(eai_taxonomy, dict):
        return -1
    fdc = eai_taxonomy.get("free_decimal_correspondence", {})
    if not isinstance(fdc, dict):
        return -1
    primary = fdc.get("primary", {})
    if not isinstance(primary, dict):
        return -1
    code = primary.get("code", "")
    if not isinstance(code, str) or len(code) < 2:
        return -1
    prefix = code[:2]
    return FDC_PREFIX_TO_DOMAIN.get(prefix, -1)


def extract_quality_signals(quality_signals):
    if not isinstance(quality_signals, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    fasttext = quality_signals.get("fasttext", {})
    if not isinstance(fasttext, dict):
        return np.zeros(len(FASTTEXT_FIELDS), dtype=np.float32)
    return np.array([
        (fasttext.get(field, 0.0) or 0.0)
        for field in FASTTEXT_FIELDS
    ], dtype=np.float32)


def process_shard(shard_path: str, shard_idx: int, output_dir: str) -> dict:
    t0 = time.time()
    try:
        df = pd.read_parquet(shard_path,
                             columns=["text", "eai_taxonomy", "quality_signals"])
    except Exception as e:
        print(f"  [{shard_idx:05d}] ERROR: Failed to read {shard_path}: {e}")
        return None
    n = len(df)

    domains = df["eai_taxonomy"].apply(extract_domain_level_2)
    quality = df["quality_signals"].apply(extract_quality_signals)
    quality_matrix = np.stack(quality.to_numpy())

    valid_mask = domains.values >= 0
    n_discarded = n - valid_mask.sum()
    if n_discarded > 0:
        df = df[valid_mask]
        domains = domains[valid_mask]
        quality_matrix = quality_matrix[valid_mask]
        n = len(df)

    output = pd.DataFrame({
        "text": df["text"],
        "doc_char_count": df["text"].str.len().to_numpy(dtype=np.int64),
        "domain": domains,
        "shard_idx": np.full(n, shard_idx, dtype=np.int64),
        "row_in_shard": np.arange(n, dtype=np.int64),
        QUALITY_COLUMNS[0]: quality_matrix[:, 0],
        QUALITY_COLUMNS[1]: quality_matrix[:, 1],
        QUALITY_COLUMNS[2]: quality_matrix[:, 2],
        QUALITY_COLUMNS[3]: quality_matrix[:, 3],
        QUALITY_COLUMNS[4]: quality_matrix[:, 4],
    })

    out_name = f"preprocessed_{shard_idx:05d}.parquet"
    out_path = os.path.join(output_dir, out_name)
    output.to_parquet(out_path, index=False, row_group_size=1000)

    elapsed = time.time() - t0
    print(f"  [{shard_idx:05d}] {out_name}: {n:,} docs"
          f" ({n_discarded} discarded), {elapsed:.1f}s")

    return {
        "shard_idx": int(shard_idx),
        "file": out_name,
        "path": out_path,
        "num_docs": int(n),
        "valid_domains": int(n),
        "num_discarded": int(n_discarded),
        "elapsed_seconds": float(elapsed),
    }


def parse_shard_idx_from_path(shard_path: str) -> int:
    basename = os.path.basename(shard_path)
    name = basename.replace(".parquet", "")
    if name.startswith("train-") and "-of-" in name:
        return int(name.split("-")[1])
    return -1


def main():
    p = argparse.ArgumentParser(
        description="Preprocess Essential-Web v1.0 for CLIMB")
    p.add_argument("--input-dir",
                   default="/home/liujin99/data/essential-web-v1",
                   help="Directory containing raw parquet shards")
    p.add_argument("--output-dir",
                   default="/home/liujin99/data/essential-web-v1-preprocessed",
                   help="Output directory for preprocessed shards")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of shards to process (for testing)")
    p.add_argument("--force", action="store_true",
                   help="Force reprocess even if output already exists")
    p.add_argument("--workers", type=int, default=64,
                   help="Number of parallel workers (default: 64)")
    args = p.parse_args()

    if args.force and os.path.isdir(args.output_dir):
        import shutil
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    shard_paths = sorted(glob.glob(os.path.join(args.input_dir, "*.parquet")))
    if not shard_paths:
        print(f"Error: no parquet files found in {args.input_dir}")
        return 1

    if args.limit:
        shard_paths = shard_paths[:args.limit]

    print(f"Found {len(shard_paths)} shards in {args.input_dir}")

    existing_preprocessed = set()
    if not args.force:
        for fname in os.listdir(args.output_dir):
            if fname.startswith("preprocessed_") and fname.endswith(".parquet"):
                idx_str = fname.replace("preprocessed_", "").replace(".parquet", "")
                try:
                    existing_preprocessed.add(int(idx_str))
                except ValueError:
                    pass

    if existing_preprocessed and not args.force:
        print(f"  Already preprocessed: {len(existing_preprocessed)} shards")

    needed_shard_indices = set()
    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx >= 0:
            needed_shard_indices.add(shard_idx)

    missing_shards = needed_shard_indices - existing_preprocessed
    if missing_shards and not args.force:
        print(f"  Missing {len(missing_shards)} shards (will reprocess)")

    t_start = time.time()
    shard_index = []
    skipped = 0
    to_process = []

    for sp in shard_paths:
        shard_idx = parse_shard_idx_from_path(sp)
        if shard_idx < 0:
            shard_idx = len(shard_index)

        if shard_idx in existing_preprocessed and shard_idx not in missing_shards and not args.force:
            out_name = f"preprocessed_{shard_idx:05d}.parquet"
            out_path = os.path.join(args.output_dir, out_name)
            try:
                df = pd.read_parquet(out_path, columns=["domain"])
                skipped += 1
                n = len(df)
                shard_index.append({
                    "shard_idx": int(shard_idx),
                    "file": out_name,
                    "path": out_path,
                    "num_docs": int(n),
                    "valid_domains": int(n),
                    "num_discarded": 0,
                    "elapsed_seconds": 0.0,
                })
                continue
            except Exception:
                print(f"  [WARN] Corrupted: {out_name}, will reprocess")

        to_process.append((sp, shard_idx, args.output_dir))

    workers = min(args.workers, len(to_process)) if to_process else 1
    failed_shards = []
    if len(to_process) > 1:
        print(f"  Processing {len(to_process)} shards with {workers} workers...")
        t_proc_start = time.time()
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_shard, sp, si, odir): (sp, si)
                       for sp, si, odir in to_process}
            for future in as_completed(futures):
                stats = future.result()
                if stats is None:
                    sp, si = futures[future]
                    failed_shards.append((si, sp))
                else:
                    shard_index.append(stats)
                completed += 1
                if completed % 10 == 0 or completed == len(to_process):
                    elapsed = time.time() - t_proc_start
                    speed = completed / elapsed if elapsed > 0 else 0
                    eta = (len(to_process) - completed) / speed if speed > 0 else 0
                    print(f"  [Progress] {completed}/{len(to_process)} "
                          f"({completed*100//len(to_process)}%), "
                          f"{speed:.1f} shards/s, ETA {eta:.0f}s")
    else:
        for sp, shard_idx, odir in to_process:
            stats = process_shard(sp, shard_idx, odir)
            if stats is None:
                failed_shards.append((shard_idx, sp))
            else:
                shard_index.append(stats)

    shard_index.sort(key=lambda s: s["shard_idx"])

    if skipped > 0:
        print(f"  Skipped {skipped} already-preprocessed shards")

    if failed_shards:
        print(f"  Failed {len(failed_shards)} shards")

    index_path = os.path.join(args.output_dir, "shard_index.json")
    total_docs = sum(s["num_docs"] for s in shard_index)
    total_valid = sum(s.get("valid_domains", s["num_docs"]) for s in shard_index)
    total_discarded = sum(s.get("num_discarded", 0) for s in shard_index)
    index_data = {
        "num_shards": len(shard_index),
        "total_docs": total_docs,
        "total_valid_domains": total_valid,
        "total_discarded": total_discarded,
        "shards": shard_index,
    }
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Preprocessing complete!")
    print(f"  Shards:     {len(shard_index)}")
    print(f"  Total docs: {total_docs:,}")
    print(f"  Valid:      {total_valid:,}")
    print(f"  Discarded:  {total_discarded:,}")
    print(f"  Index:      {index_path}")
    print(f"  Duration:   {elapsed:.1f}s")
    print(f"  Output:     {args.output_dir}/")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
