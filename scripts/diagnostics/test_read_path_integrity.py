#!/usr/bin/env python3
"""read_texts integrity verification (prod1/2/3 root cause, 2026-09-14).

Root cause being locked here: DatasetSchema.metadata_read_columns() did NOT
include row_in_shard_col, so _read_shard_metadata never read the real
'source_record_idx' column and silently fell back to np.arange positions
when building metadata caches. Poisoned caches then made read_texts pass
ROW POSITIONS as sri VALUES into parquet filters: ~82% of docs came back
wrong (a permutation scramble of the shard) and ~18% matched nothing and
silently returned "" — every STEM mixture to date was same-shard random
docs instead of the selected docs (avg 2,765 vs 9,055 chars).

Checks:
  1.  metadata_read_columns() includes row_in_shard_col (writer-side fix);
  2.  fresh build on a gapped-sri pool stores REAL column values (not arange)
      and read_texts returns exactly the positional-truth texts;
  3.  cache roundtrip preserves real values;
  4.  POISONED cache (arange in npz, the exact server state) is detected and
      self-healed on load — arrays + on-disk cache + reads all correct;
  5.  old-format cache (no row arrays) heals too;
  6.  configured-but-missing column fails loud (no silent arange);
  7.  duplicate sri values fail loud;
  8.  missing requested sri at read time raises (no silent "");
  9.  genuinely-sequential pools keep the fast path and stay correct;
  10. row_in_shard_col: null mode (positional reads) unchanged;
  11. spawn multiprocessing path (read + heal) end-to-end.
"""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import pandas as pd

from climbmix.data.column_schema import DatasetSchema
from climbmix.data.metadata_manager import (
    ShardMetadataManager, _read_one_shard_texts, _read_shard_metadata,
)

_failures = []
_n = 0


def check(name, cond, extra=""):
    global _n
    _n += 1
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f" — {extra}" if extra and not cond else ""))
    if not cond:
        _failures.append(name)


N_SHARDS = 3
N_DOCS = 50
SRI_WINDOW = 300  # 50 unique values drawn from a 300-wide window → real gaps


def build_pool(base, shard_idx, sri_mode="gapped"):
    """Write one parquet shard; returns (path, sri_values, texts)."""
    os.makedirs(base, exist_ok=True)
    rng = np.random.default_rng(1000 + shard_idx)
    if sri_mode == "gapped":
        sri = np.sort(rng.choice(SRI_WINDOW, size=N_DOCS, replace=False))
    elif sri_mode == "dup":
        sri = np.sort(rng.choice(SRI_WINDOW, size=N_DOCS, replace=True))
    else:  # sequential
        sri = np.arange(N_DOCS, dtype=np.int64)
    rows = []
    for i in range(N_DOCS):
        text = f"shard{shard_idx}_sri{sri[i]}_doc{i}_" + "x" * (i % 97)
        rows.append({
            "text": text,
            "char_count": len(text),
            "domain": int(i % 3),
            "q_relevance": float(i) / N_DOCS,
            "q_quality": float((i * 7) % 13) / 13.0,
            "source_record_idx": int(sri[i]),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(base, f"shard_{shard_idx:03d}.parquet")
    df.to_parquet(path, index=False)
    return path, np.asarray(sri, dtype=np.int64), df["text"].tolist()


def write_schema_yaml(base, row_col="source_record_idx"):
    path = os.path.join(base, "schema.yaml")
    with open(path, "w") as f:
        f.write(
            "domain_col: domain\n"
            "quality_cols: [q_relevance, q_quality]\n"
            "text_col: text\n"
            "char_count_col: char_count\n"
            + (f"row_in_shard_col: {row_col}\n" if row_col else "row_in_shard_col: null\n")
            + "preprocessed_pattern: shard_*.parquet\n"
        )
    return path


def positional_truth(base):
    """texts + char_counts in global (shard, row) order."""
    texts, chars = [], []
    for s in range(N_SHARDS):
        df = pd.read_parquet(os.path.join(base, f"shard_{s:03d}.parquet"))
        texts.extend(df["text"].tolist())
        chars.extend(df["char_count"].tolist())
    return texts, np.asarray(chars, dtype=np.int64)


def read_pool_npz(cache_path):
    with np.load(cache_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def main():
    tmp = tempfile.mkdtemp(prefix="read_path_integrity_")
    pool = os.path.join(tmp, "pool")
    sri_arrays = []
    for s in range(N_SHARDS):
        _, sri, _ = build_pool(pool, s)
        sri_arrays.append(sri)
    truth_texts, truth_chars = positional_truth(pool)
    yaml_path = write_schema_yaml(tmp)
    schema = DatasetSchema.from_yaml(yaml_path)
    cache_path = os.path.join(pool, "metadata_cache.npz")

    # 1. writer-side fix: the column is requested during metadata reads
    check("metadata_read_columns includes row_in_shard_col",
          "source_record_idx" in schema.metadata_read_columns())

    # 2. fresh build stores REAL gapped values, reads match positional truth
    mm = ShardMetadataManager(pool, schema=schema, max_workers=1)
    real_ok = all(
        np.array_equal(mm._row_in_shard_cols[s], sri_arrays[s])
        for s in range(N_SHARDS)
    )
    check("fresh build caches real sri values (not arange)", real_ok)
    check("gapped pool is not flagged sequential",
          mm._is_row_col_sequential is False)
    rng = np.random.default_rng(7)
    gids = rng.choice(mm.num_docs, size=40, replace=False)
    got = mm.read_texts(np.sort(gids), verbose=False)
    exp = [truth_texts[g] for g in np.sort(gids)]
    check("read_texts == positional truth (fresh, serial)", got == exp)
    lens_ok = all(len(t) == mm.doc_char_counts[g] for t, g in zip(got, np.sort(gids)))
    check("read lengths match char_count metadata", lens_ok)

    # 3. cache roundtrip keeps real values
    mm2 = ShardMetadataManager(pool, schema=schema, max_workers=1)
    check("cache roundtrip preserves real sri",
          all(np.array_equal(mm2._row_in_shard_cols[s], sri_arrays[s])
              for s in range(N_SHARDS)))

    # 4. POISONED cache — the exact server state (arange in the npz)
    z = read_pool_npz(cache_path)
    bounds = z["row_in_shard_boundaries"]
    poison = np.concatenate([np.arange(N_DOCS, dtype=np.int64)
                             for _ in range(N_SHARDS)])
    tmp_npz = cache_path + ".poison.npz"
    np.savez(tmp_npz,
             cluster_labels=z["cluster_labels"],
             quality_scores=z["quality_scores"],
             doc_char_counts=z["doc_char_counts"],
             row_in_shard_cols_concat=poison,
             row_in_shard_boundaries=bounds)
    os.replace(tmp_npz, cache_path)
    mm3 = ShardMetadataManager(pool, schema=schema, max_workers=1)
    check("poisoned cache self-heals (in-memory)",
          all(np.array_equal(mm3._row_in_shard_cols[s], sri_arrays[s])
              for s in range(N_SHARDS)))
    healed = read_pool_npz(cache_path)
    check("poisoned cache self-heals (on-disk npz rewritten)",
          np.array_equal(healed["row_in_shard_cols_concat"],
                         np.concatenate(sri_arrays)))
    got3 = mm3.read_texts(np.sort(gids), verbose=False)
    check("read_texts correct after heal", got3 == exp)

    # 5. old-format cache (row arrays missing entirely) heals too
    z = read_pool_npz(cache_path)
    tmp_npz = cache_path + ".oldfmt.npz"
    np.savez(tmp_npz,
             cluster_labels=z["cluster_labels"],
             quality_scores=z["quality_scores"],
             doc_char_counts=z["doc_char_counts"])
    os.replace(tmp_npz, cache_path)
    mm4 = ShardMetadataManager(pool, schema=schema, max_workers=1)
    check("old-format cache heals",
          all(np.array_equal(mm4._row_in_shard_cols[s], sri_arrays[s])
              for s in range(N_SHARDS)))

    # 6. configured-but-unreadable column fails loud (no silent arange)
    path0 = os.path.join(pool, "shard_000.parquet")
    try:
        _read_shard_metadata(path0, {
            "domain_col": "domain", "domain_names": None,
            "quality_cols": ["q_relevance", "q_quality"],
            "text_col": "text", "char_count_col": "char_count",
            "row_in_shard_col": "source_record_idx",
            "metadata_read_columns": ["domain", "q_relevance", "q_quality",
                                      "char_count"],
        })
        check("missing row col in read set fails loud", False, "no raise")
    except ValueError as e:
        check("missing row col in read set fails loud",
              "source_record_idx" in str(e))

    # pool without the column at all → manager init raises (pandas loud path)
    nofree = os.path.join(tmp, "pool_nocol")
    os.makedirs(nofree, exist_ok=True)
    for s in range(N_SHARDS):
        df = pd.read_parquet(os.path.join(pool, f"shard_{s:03d}.parquet"))
        df.drop(columns=["source_record_idx"]).to_parquet(
            os.path.join(nofree, f"shard_{s:03d}.parquet"), index=False)
    try:
        ShardMetadataManager(nofree, schema=schema, max_workers=1)
        check("pool without sri column fails loud", False, "no raise")
    except Exception as e:
        check("pool without sri column fails loud",
              "source_record_idx" in str(e))

    # 7. duplicate sri values fail loud
    dupdir = os.path.join(tmp, "pool_dup")
    os.makedirs(dupdir, exist_ok=True)
    for s in range(N_SHARDS):
        build_pool(dupdir, s, sri_mode="dup")
    try:
        ShardMetadataManager(dupdir, schema=schema, max_workers=1)
        check("duplicate sri fails loud", False, "no raise")
    except ValueError as e:
        check("duplicate sri fails loud", "duplicate" in str(e))

    # 8. missing requested sri at read time raises (both sri-keyed branches)
    shard0_path = os.path.join(pool, "shard_000.parquet")
    good_rcv = sri_arrays[0][:5]
    bogus = np.array([SRI_WINDOW + 999], dtype=np.int64)
    for branch, rcv, n_req in (("filters", np.concatenate([good_rcv, bogus]), 6),
                               ("chunk_map", bogus, N_DOCS)):  # ratio 1.0 > 0.3
        try:
            _read_one_shard_texts(
                shard0_path, "text", "source_record_idx", rcv,
                np.arange(len(rcv), dtype=np.int64), N_DOCS, False)
            check(f"missing sri raises ({branch})", False, "no raise")
        except RuntimeError as e:
            check(f"missing sri raises ({branch})", "not found" in str(e))

    # 9. genuinely-sequential pool keeps the fast path and stays correct
    seqdir = os.path.join(tmp, "pool_seq")
    for s in range(N_SHARDS):
        build_pool(seqdir, s, sri_mode="seq")
    mm5 = ShardMetadataManager(seqdir, schema=schema, max_workers=1)
    check("sequential pool flagged sequential", mm5._is_row_col_sequential is True)
    seq_truth, _ = positional_truth(seqdir)
    dense = np.arange(0, N_SHARDS * N_DOCS, 2)  # 50% of every shard > 0.3
    got5 = mm5.read_texts(dense, verbose=False)
    exp5 = [seq_truth[g] for g in dense]
    check("sequential pool dense read (fast path) correct", got5 == exp5)

    # 10. row_in_shard_col: null mode — positional reads, unchanged behavior
    yaml_null = write_schema_yaml(tmp, row_col=None)
    schema_null = DatasetSchema.from_yaml(yaml_null)
    mm6 = ShardMetadataManager(pool, schema=schema_null, max_workers=1)
    got6 = mm6.read_texts(np.sort(gids), verbose=False)
    check("null-mode positional read correct", got6 == exp)

    # 11. spawn multiprocessing path (default workers) end-to-end
    mm7 = ShardMetadataManager(pool, schema=schema)
    got7 = mm7.read_texts(np.sort(gids), verbose=False)
    check("spawn-path read correct", got7 == exp)

    print()
    if _failures:
        print(f"FAILED {_n - len(_failures)}/{_n}: " + ", ".join(_failures))
        sys.exit(1)
    print(f"ALL PASS ({_n} checks)")


if __name__ == "__main__":
    main()
