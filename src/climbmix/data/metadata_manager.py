"""
Shard metadata manager.

Loads metadata from all preprocessed parquet files in a data directory.
Uses column_schema for column name resolution.
Text is loaded on-demand via read_texts().
"""

import json
import os
import glob
import numpy as np
import numpy.typing as npt
import pandas as pd
from typing import Dict, List, Optional, Tuple

from climbmix.data.column_schema import DEFAULT_SCHEMA
from climbmix.utils.token_estimate import estimate_token_counts_array


class ShardMetadataManager:
    def __init__(self, data_dir: str, schema: Optional[object] = None):
        self._dir = data_dir
        self._schema = schema or DEFAULT_SCHEMA
        self._shard_files: List[str] = sorted(
            glob.glob(os.path.join(data_dir, self._schema.preprocessed_pattern))
        )
        if not self._shard_files:
            raise FileNotFoundError(
                f"No {self._schema.preprocessed_pattern} files in {data_dir}"
            )

        print(f"[ShardMetadataManager] Discovered {len(self._shard_files)} shards")

        cluster_list: List[np.ndarray] = []
        quality_list: List[np.ndarray] = []
        char_count_list: List[np.ndarray] = []
        self._per_shard_info: List[dict] = []

        global_start = 0
        for sf in self._shard_files:
            pf = pd.read_parquet(sf)
            available_cols = list(pf.columns)
            cluster_col = self._schema.resolve_cluster_col(available_cols)
            quality_cols = self._schema.resolve_quality_cols(available_cols)
            load_cols = [cluster_col, self._schema.char_count_col, *quality_cols]
            actual_load = [c for c in load_cols if c in available_cols]

            df_meta = pf[actual_load]

            n = len(df_meta)
            cluster_list.append(df_meta[cluster_col].to_numpy(dtype=np.int64))

            if quality_cols:
                avail_q = [c for c in quality_cols if c in df_meta.columns]
                if avail_q:
                    quality_list.append(df_meta[avail_q].to_numpy(dtype=np.float64))
                else:
                    quality_list.append(np.zeros((n, max(len(quality_cols), 1)), dtype=np.float64))
            else:
                quality_list.append(np.zeros((n, 1), dtype=np.float64))

            if self._schema.char_count_col in df_meta.columns:
                char_count_list.append(df_meta[self._schema.char_count_col].to_numpy(dtype=np.int64))
            else:
                char_count_list.append(np.ones(n, dtype=np.int64) * 500)

            basename = os.path.basename(sf)
            idx_str = basename.replace("preprocessed_", "").replace(".parquet", "")
            parsed_idx = int(idx_str)

            self._per_shard_info.append({
                "shard_idx": parsed_idx,
                "path": sf,
                "num_docs": n,
                "start_idx": global_start,
                "end_idx": global_start + n,
            })
            global_start += n

        self._cluster_labels = np.concatenate(cluster_list)
        self._quality_scores = np.concatenate(quality_list)
        self._doc_char_counts = np.concatenate(char_count_list)
        self._num_docs = global_start
        self._num_shards = len(self._shard_files)
        self._shard_starts = np.array(
            [s["start_idx"] for s in self._per_shard_info], dtype=np.int64
        )

        print(f"[ShardMetadataManager] Loaded {self._num_docs:,} docs "
              f"({self._num_shards} shards)")

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

    def estimate_token_counts(self) -> npt.NDArray[np.int64]:
        return estimate_token_counts_array(self._doc_char_counts)

    def global_to_shard_rows(
        self, global_indices: npt.NDArray[np.int64]
    ) -> Dict[int, Tuple[str, npt.NDArray[np.int64]]]:
        shard_ids = np.searchsorted(self._shard_starts, global_indices, side="right") - 1
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

    def read_texts(self, global_indices: npt.NDArray[np.int64]) -> List[str]:
        if len(global_indices) == 0:
            return []

        shard_groups = self.global_to_shard_rows(global_indices)
        pos_map: Dict[int, List[int]] = {}
        for p, idx in enumerate(global_indices):
            pos_map.setdefault(int(idx), []).append(p)

        result = [""] * len(global_indices)

        for sid, (shard_path, local_rows) in shard_groups.items():
            df_chunk = pd.read_parquet(
                shard_path,
                columns=[self._schema.row_in_shard_col, self._schema.text_col],
                filters=[(self._schema.row_in_shard_col, "in", local_rows.tolist())],
            )
            chunk_map = dict(zip(df_chunk[self._schema.row_in_shard_col], df_chunk[self._schema.text_col]))
            for local_row in local_rows:
                text = chunk_map.get(local_row, "")
                global_idx = self._shard_starts[sid] + local_row
                for pos in pos_map.get(int(global_idx), []):
                    result[pos] = text

        return result

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
