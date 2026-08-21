"""
ShardMetadataManager — loads metadata (cluster/domain + quality + char_count)
from preprocessed multi-shard parquet files into memory, with npz cache.

Uses DatasetSchema for column mapping (YAML-driven).
Text is loaded on-demand via read_texts().

Simplified vs quadmix: no cross-shard domain consistency checks (K-means
replaces domain labels anyway), no domain_cat_map. Just per-shord astype
conversion when domain column is string.
"""

import json
import os
import re
import glob
import time
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import numpy.typing as npt
import pandas as pd

from climbmix.data.column_schema import DatasetSchema
from climbmix.utils.token_estimate import estimate_token_counts_array


_CACHE_FILENAME = "metadata_cache.npz"
_CACHE_META = "metadata_shard_info.json"


def _parse_shard_idx(basename: str) -> Optional[int]:
    m = re.search(r'(\d+)', basename)
    if m:
        return int(m.group(1))
    return None


def _read_shard_metadata(shard_path: str, schema_dict: dict) -> dict:
    """Worker function for parallel shard metadata loading.

    schema_dict is a plain dict version of DatasetSchema (for pickling).
    """
    pf = pd.read_parquet(shard_path, columns=schema_dict["metadata_read_columns"])
    n = len(pf)
    basename = os.path.basename(shard_path)
    parsed_idx = _parse_shard_idx(basename)

    # Domain/cluster column — convert string to int if needed
    domain_col = schema_dict["domain_col"]
    domain_data = pf[domain_col]
    if pd.api.types.is_numeric_dtype(domain_data):
        domain_arr = domain_data.to_numpy(dtype=np.int64)
    else:
        domain_arr = pd.Categorical(domain_data).codes.astype(np.int64)

    # Quality columns
    quality_cols = schema_dict["quality_cols"]
    quality_arr = np.column_stack([
        pf[c].to_numpy(dtype=np.float64) for c in quality_cols
    ])
    nan_count = np.isnan(quality_arr).sum()
    if nan_count > 0:
        pct = nan_count / quality_arr.size * 100
        print(f"[ShardMetadata] WARNING: {basename} has {nan_count} NaN "
              f"({pct:.1f}%), filling with 0.0")
        quality_arr = np.nan_to_num(quality_arr, nan=0.0)

    # Char count
    char_count_col = schema_dict.get("char_count_col")
    if char_count_col is not None and char_count_col in pf.columns:
        char_count_arr = pf[char_count_col].to_numpy(dtype=np.int64)
    elif schema_dict.get("text_col") in pf.columns:
        text_series = pf[schema_dict["text_col"]]
        char_count_arr = text_series.apply(
            lambda t: len(str(t)) if t is not None else 0
        ).to_numpy(dtype=np.int64)
    else:
        char_count_arr = np.zeros(n, dtype=np.int64)

    # Row in shard
    row_col = schema_dict.get("row_in_shard_col")
    if row_col is not None and row_col in pf.columns:
        row_arr = pf[row_col].to_numpy(dtype=np.int64)
    else:
        row_arr = np.arange(n, dtype=np.int64)

    return {
        "shard_idx": parsed_idx,
        "path": shard_path,
        "num_docs": n,
        "domain": domain_arr,
        "quality": quality_arr,
        "char_count": char_count_arr,
        "row_in_shard_col": row_arr,
        "computed_char_count": char_count_col is None,
    }


class ShardMetadataManager:
    """
    Manages metadata from a directory of preprocessed parquet shards.

    Lazy loading:
      - __init__ reads only metadata columns from all shards
      - text is loaded on-demand via read_texts()

    Schema-driven:
      - Accepts DatasetSchema for column mapping
      - String domain columns auto-converted to int per-shard
      - char_count computed from text if column not specified
      - npz cache for fast reload
    """

    def __init__(
        self,
        data_dir: str,
        schema: Optional[DatasetSchema] = None,
        max_workers: Optional[int] = None,
    ):
        self._dir = data_dir
        self._schema = schema or DatasetSchema.from_yaml("config/schema_stem.yaml")
        self._max_workers = max_workers

        self._shard_files: List[str] = sorted(
            glob.glob(os.path.join(data_dir, self._schema.preprocessed_pattern))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No {self._schema.preprocessed_pattern} files in {data_dir}"
            )

        self._cache_path = os.path.join(data_dir, _CACHE_FILENAME)
        self._shard_info_path = os.path.join(data_dir, _CACHE_META)

        if not self._try_load_cache():
            self._load_from_shards()
            self._write_cache()

    # ── Cache ──

    def _try_load_cache(self) -> bool:
        """Check cache validity and load if valid. Returns True if loaded."""
        cache_path = self._cache_path
        shard_info_path = self._shard_info_path

        current_stats = {
            os.path.basename(f): {"size": os.path.getsize(f), "mtime": os.path.getmtime(f)}
            for f in self._shard_files
        }
        current_basenames = sorted(current_stats.keys())

        cache_valid = False
        if os.path.exists(cache_path) and os.path.exists(shard_info_path):
            try:
                with open(shard_info_path) as f:
                    cached_info = json.load(f)
                cached_basenames = cached_info.get("shard_basenames", [])
                cached_stats = cached_info.get("shard_stats", {})
                cached_schema_key = cached_info.get("schema_key", None)
                current_schema_key = self._schema_key()

                if (cached_basenames == current_basenames and
                    cached_schema_key == current_schema_key):
                    mismatches = []
                    for bn in current_basenames:
                        cs = cached_stats.get(bn, {})
                        cr = current_stats[bn]
                        if cs.get("size") != cr["size"] or cs.get("mtime") != cr["mtime"]:
                            mismatches.append(bn)
                    if not mismatches:
                        cache_valid = True
                    else:
                        print(f"[ShardMetadataManager] Cache invalid: {len(mismatches)} shard(s) changed")
                else:
                    print(f"[ShardMetadataManager] Cache invalid: schema or shard list changed")
            except Exception as e:
                print(f"[ShardMetadataManager] Cache read error: {e}")

        if not cache_valid:
            return False

        print(f"[ShardMetadataManager] Cache valid, loading from: {cache_path}")
        cached = np.load(cache_path, allow_pickle=False)
        self._cluster_labels = cached["cluster_labels"]
        self._quality_scores = cached["quality_scores"]
        self._doc_char_counts = cached["doc_char_counts"]
        self._num_docs = len(self._cluster_labels)
        self._num_shards = len(self._shard_files)
        self._per_shard_info = cached_info["per_shard_info"]
        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        if "row_in_shard_cols_concat" in cached:
            concat = cached["row_in_shard_cols_concat"]
            boundaries = cached["row_in_shard_boundaries"]
            self._row_in_shard_cols = {}
            for i in range(len(self._per_shard_info)):
                start = int(boundaries[i])
                end = int(boundaries[i + 1]) if i + 1 < len(boundaries) else len(concat)
                self._row_in_shard_cols[i] = concat[start:end]
        else:
            self._row_in_shard_cols = {}

        self._is_row_col_sequential = self._check_row_col_sequential()

        valid_labels = self._cluster_labels[self._cluster_labels >= 0]
        self._num_clusters = int(len(np.unique(valid_labels)))
        self._num_quality_criteria = len(self._schema.quality_cols)

        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards) from cache")
        print(f"[ShardMetadataManager] Quality scores: {self._quality_scores.shape}")
        return True

    def _load_from_shards(self) -> None:
        """Load metadata from all shards in parallel."""
        total_shards = len(self._shard_files)
        load_t0 = time.time()

        n_workers = self._max_workers if self._max_workers is not None else min(8, total_shards)
        print(f"[ShardMetadataManager] Discovered {total_shards} shards, "
              f"loading metadata with {n_workers} workers")

        schema_dict = {
            "domain_col": self._schema.domain_col,
            "quality_cols": list(self._schema.quality_cols),
            "text_col": self._schema.text_col,
            "char_count_col": self._schema.char_count_col,
            "row_in_shard_col": self._schema.row_in_shard_col,
            "metadata_read_columns": self._schema.metadata_read_columns(),
        }

        shard_data: List[Optional[dict]] = [None] * total_shards
        done = 0
        log_interval = max(1, total_shards // 20)

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            future_to_idx = {
                pool.submit(_read_shard_metadata, sf, schema_dict): i
                for i, sf in enumerate(self._shard_files)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                shard_data[i] = future.result()
                done += 1
                if done % log_interval == 0 or done == total_shards:
                    elapsed = time.time() - load_t0
                    pct = done / total_shards * 100
                    docs_so_far = sum(r["num_docs"] for r in shard_data if r is not None)
                    eta = elapsed / done * (total_shards - done)
                    print(f"[ShardMetadataManager] {done}/{total_shards} "
                          f"({pct:.0f}%) — {docs_so_far:,} docs, "
                          f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        computed_char_count = any(r["computed_char_count"] for r in shard_data)
        if computed_char_count:
            total_docs = sum(r["num_docs"] for r in shard_data)
            print(f"[ShardMetadataManager] WARNING: 从 text 列计算 "
                  f"char_count ({total_docs:,} docs)")

        global_start = 0
        cluster_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []
        row_col_list: List[np.ndarray] = []
        row_col_boundaries: List[int] = [0]
        self._per_shard_info: List[dict] = []

        for i, data in enumerate(shard_data):
            cluster_list.append(data["domain"])
            quality_list.append(data["quality"])
            char_count_list.append(data["char_count"])
            row_col_list.append(data["row_in_shard_col"])

            self._per_shard_info.append({
                "shard_idx": data["shard_idx"] if data["shard_idx"] is not None else i,
                "path": data["path"],
                "num_docs": data["num_docs"],
                "start_idx": global_start,
                "end_idx": global_start + data["num_docs"],
            })
            row_col_boundaries.append(row_col_boundaries[-1] + data["num_docs"])
            global_start += data["num_docs"]

        self._cluster_labels = np.concatenate(cluster_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = total_shards
        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        self._row_in_shard_cols: Dict[int, np.ndarray] = {}
        for i, data in enumerate(shard_data):
            self._row_in_shard_cols[i] = data["row_in_shard_col"]
        self._is_row_col_sequential = self._check_row_col_sequential()

        valid_labels = self._cluster_labels[self._cluster_labels >= 0]
        self._num_clusters = int(len(np.unique(valid_labels)))
        self._num_quality_criteria = len(self._schema.quality_cols)

        total_time = time.time() - load_t0
        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards, {self._num_clusters} initial clusters, "
              f"{self._num_quality_criteria} quality criteria) "
              f"in {total_time:.1f}s")
        print(f"[ShardMetadataManager] Cluster labels: {self._cluster_labels.shape}, "
              f"Quality scores: {self._quality_scores.shape}")

    def _write_cache(self) -> None:
        """Save metadata cache to disk."""
        try:
            row_col_list = [
                self._row_in_shard_cols[i] for i in range(len(self._per_shard_info))
            ]
            row_in_shard_boundaries = [0]
            for arr in row_col_list:
                row_in_shard_boundaries.append(row_in_shard_boundaries[-1] + len(arr))

            row_in_shard_cols_concat = np.concatenate(row_col_list)
            row_in_shard_boundaries = np.array(row_in_shard_boundaries, dtype=np.int64)

            tmp_npz = self._cache_path + ".tmp"
            np.savez(tmp_npz,
                     cluster_labels=self._cluster_labels,
                     quality_scores=self._quality_scores,
                     doc_char_counts=self._doc_char_counts,
                     row_in_shard_cols_concat=row_in_shard_cols_concat,
                     row_in_shard_boundaries=row_in_shard_boundaries)
            os.replace(tmp_npz, self._cache_path)

            cache_meta = {
                "shard_basenames": sorted(os.path.basename(f) for f in self._shard_files),
                "shard_stats": {
                    os.path.basename(f): {"size": os.path.getsize(f), "mtime": os.path.getmtime(f)}
                    for f in self._shard_files
                },
                "per_shard_info": self._per_shard_info,
                "schema_key": self._schema_key(),
            }
            tmp_json = self._shard_info_path + ".tmp"
            with open(tmp_json, "w") as f:
                json.dump(cache_meta, f)
            os.replace(tmp_json, self._shard_info_path)

            cache_size = os.path.getsize(self._cache_path) / (1024 ** 3)
            print(f"[ShardMetadataManager] Saved cache: {self._cache_path} "
                  f"({cache_size:.2f} GB)")
        except Exception as e:
            print(f"[ShardMetadataManager] Failed to save cache: {e}")

    def _schema_key(self) -> str:
        return (
            f"{self._schema.domain_col}"
            f":{','.join(self._schema.quality_cols)}"
            f":{self._schema.text_col}"
            f":{self._schema.char_count_col}"
            f":{self._schema.row_in_shard_col}"
        )

    def _check_row_col_sequential(self) -> bool:
        if not self._row_in_shard_cols:
            return True
        for sid, arr in self._row_in_shard_cols.items():
            expected = np.arange(len(arr), dtype=np.int64)
            if not np.array_equal(arr, expected):
                return False
        return True

    # ── Properties ──

    @property
    def cluster_labels(self) -> npt.NDArray[np.int64]:
        return self._cluster_labels

    @property
    def quality_scores(self) -> npt.NDArray[np.float64]:
        return self._quality_scores

    @property
    def doc_char_counts(self) -> npt.NDArray[np.int64]:
        return self._doc_char_counts

    @property
    def num_docs(self) -> int:
        return self._num_docs

    @property
    def num_shards(self) -> int:
        return self._num_shards

    @property
    def shard_info(self) -> List[dict]:
        return list(self._per_shard_info)

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    def estimate_token_counts(self) -> npt.NDArray[np.int64]:
        return estimate_token_counts_array(self._doc_char_counts)

    # ── Index resolution ──

    def global_to_shard_rows(
        self, global_indices: npt.NDArray[np.int64]
    ) -> Dict[int, Tuple[str, npt.NDArray[np.int64]]]:
        shard_ids = np.searchsorted(
            self._shard_starts, global_indices, side="right"
        ) - 1
        shard_ids = np.clip(shard_ids, 0, self._num_shards - 1)

        order = np.argsort(shard_ids)
        sorted_shard_ids = shard_ids[order]
        sorted_global_idx = global_indices[order]

        result: Dict[int, Tuple[str, npt.NDArray[np.int64]]] = {}
        unique_ids, starts, counts = np.unique(
            sorted_shard_ids, return_index=True, return_counts=True
        )

        for sid, start, cnt in zip(unique_ids, starts, counts):
            group_global = sorted_global_idx[start:start + cnt]
            local_rows = group_global - self._shard_starts[int(sid)]
            shard_path = self._per_shard_info[int(sid)]["path"]
            result[int(sid)] = (shard_path, local_rows)

        return result

    def local_to_row_col(self, sid: int, local_positions: np.ndarray) -> np.ndarray:
        if self._is_row_col_sequential or sid not in self._row_in_shard_cols:
            return local_positions
        return self._row_in_shard_cols[sid][local_positions]

    # ── Text loading ──

    def read_texts(
        self, global_indices: npt.NDArray[np.int64], verbose: bool = True
    ) -> List[str]:
        if len(global_indices) == 0:
            return []

        shard_groups = self.global_to_shard_rows(global_indices)

        text_col = self._schema.text_col
        row_col = self._schema.row_in_shard_col if self._schema.row_in_shard_col else None
        is_row_col_sequential = self._is_row_col_sequential

        n_shards = len(shard_groups)
        n_workers = min(8, n_shards)
        if n_shards <= 1:
            n_workers = 1

        if verbose:
            print(f"[read_texts] {len(global_indices):,} texts from {n_shards} shards, "
                  f"{n_workers} I/O processes")

        t0 = time.time()

        shard_task_args: Dict[int, Tuple] = {}
        for sid, (shard_path, local_rows) in shard_groups.items():
            shard_total_rows = self._per_shard_info[sid]["num_docs"]
            if row_col is not None:
                rcv = self.local_to_row_col(sid, local_rows)
            else:
                rcv = None
            shard_task_args[sid] = (
                shard_path, text_col, row_col, rcv,
                local_rows, shard_total_rows, is_row_col_sequential,
            )

        shard_results: Dict[int, List[str]] = {}

        if n_workers <= 1:
            for sid in shard_groups:
                shard_results[sid] = _read_one_shard_texts(*shard_task_args[sid])
        else:
            log_interval = max(1, n_shards // 20)
            done = 0
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
                future_map = {
                    pool.submit(_read_one_shard_texts, *shard_task_args[sid]): sid
                    for sid in shard_groups
                }
                for future in as_completed(future_map):
                    sid = future_map[future]
                    shard_results[sid] = future.result()
                    done += 1
                    if verbose and (done % log_interval == 0 or done == n_shards):
                        elapsed = time.time() - t0
                        pct = done / n_shards * 100
                        eta = elapsed / done * (n_shards - done)
                        print(f"[read_texts] {done}/{n_shards} shards "
                              f"({pct:.0f}%) — elapsed {elapsed:.0f}s, ETA {eta:.0f}s")

        result = _assemble_texts(
            global_indices, shard_groups, shard_results, self._shard_starts
        )

        elapsed = time.time() - t0
        if verbose:
            print(f"[read_texts] Done: {len(global_indices):,} texts in {elapsed:.1f}s "
                  f"({n_shards} shards, {n_workers} processes)")

        return result

    # ── Token estimation ──

    def get_total_tokens_estimate(self, chars_per_token: float = 4.0) -> int:
        return int(np.sum(self._doc_char_counts) / chars_per_token)

    def save_shard_index(self, output_path: str):
        index = {
            "num_shards": self._num_shards,
            "total_docs": self._num_docs,
            "shards": self._per_shard_info,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(index, f, indent=2)


# ── Worker functions for multiprocessing (spawn-safe) ──

def _read_one_shard_texts(
    shard_path: str,
    text_col: str,
    row_col: Optional[str],
    row_col_values: Optional[np.ndarray],
    local_rows: np.ndarray,
    shard_total_rows: int,
    is_row_col_sequential: bool,
) -> List[str]:
    """Read texts from a single shard using pyarrow."""
    import pyarrow.parquet as pq

    if row_col is None:
        table = pq.read_table(shard_path, columns=[text_col], use_threads=False)
        text_arr = table.column(text_col).to_numpy(zero_copy_only=False)
        result = []
        for i in local_rows:
            if 0 <= int(i) < len(text_arr):
                val = text_arr[int(i)]
                result.append(str(val) if val is not None else "")
            else:
                result.append("")
        return result

    n_requested = len(row_col_values)
    select_ratio = n_requested / max(shard_total_rows, 1)

    if is_row_col_sequential:
        table = pq.read_table(shard_path, columns=[text_col], use_threads=False)
        text_arr = table.column(text_col).to_numpy(zero_copy_only=False)
        texts = []
        for rv in row_col_values:
            idx = int(rv)
            if 0 <= idx < len(text_arr):
                val = text_arr[idx]
                texts.append(str(val) if val is not None else "")
            else:
                texts.append("")
        return texts

    if select_ratio > 0.3:
        table = pq.read_table(shard_path, columns=[row_col, text_col], use_threads=False)
        row_arr = table.column(row_col).to_numpy(zero_copy_only=False)
        text_arr = table.column(text_col).to_numpy(zero_copy_only=False)
        chunk_map: dict = {}
        for k, v in zip(row_arr, text_arr):
            chunk_map[int(k)] = str(v) if v is not None else ""
        return [chunk_map.get(int(rv), "") for rv in row_col_values]

    table = pq.read_table(
        shard_path,
        columns=[row_col, text_col],
        filters=[(row_col, "in", row_col_values.tolist())],
        use_threads=False,
    )
    row_arr = table.column(row_col).to_numpy(zero_copy_only=False)
    text_arr = table.column(text_col).to_numpy(zero_copy_only=False)
    sort_idx = np.argsort(row_arr)
    texts = [str(text_arr[i]) if text_arr[i] is not None else "" for i in sort_idx]
    text_map = dict(zip(row_arr[sort_idx].tolist(), texts))
    return [text_map.get(int(rv), "") for rv in row_col_values]


def _assemble_texts(
    global_indices: npt.NDArray[np.int64],
    shard_groups: Dict[int, Tuple[str, npt.NDArray[np.int64]]],
    shard_results: Dict[int, List[str]],
    shard_starts: npt.NDArray[np.int64],
) -> List[str]:
    """Assemble texts from shard results into global_indices order."""
    n = len(global_indices)
    if n == 0:
        return []
    gids = np.asarray(global_indices, dtype=np.int64)
    order = np.argsort(gids, kind="stable")
    sorted_gids = gids[order]
    result = np.full(n, "", dtype=object)
    for sid, (_shard_path, local_rows) in shard_groups.items():
        texts = shard_results[sid]
        shard_gids = shard_starts[sid] + np.asarray(local_rows, dtype=np.int64)
        left = np.searchsorted(sorted_gids, shard_gids, side="left")
        right = np.searchsorted(sorted_gids, shard_gids, side="right")
        if np.array_equal(left + 1, right):
            result[order[left]] = texts
        else:
            for i in range(len(texts)):
                lo = int(left[i])
                hi = int(right[i])
                for pos in order[lo:hi]:
                    result[pos] = texts[i]
    return result.tolist()
