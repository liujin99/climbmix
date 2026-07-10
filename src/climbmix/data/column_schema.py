"""
Centralized column schema for parquet data files.

All column-name knowledge is defined here. Other modules reference
this instead of hardcoding column names.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ColumnSchema:
    cluster_col: str = "cluster"
    cluster_col_alt: str = "domain"
    text_col: str = "text"
    char_count_col: str = "doc_char_count"
    row_in_shard_col: str = "row_in_shard"
    shard_idx_col: str = "shard_idx"

    quality_columns: tuple = (
        "qs_dclm",
        "qs_fineweb_edu_approx",
        "qs_english",
        "qs_eai_general_math",
        "qs_eai_open_web_math",
    )

    nemotron_quality_columns: tuple = (
        "qs_quality",
        "qs_educational",
        "qs_informational",
        "qs_advertisement",
    )

    fdc_domain_col: str = "domain"
    fdc_prefix_col: str = "eai_taxonomy"

    preprocessed_pattern: str = "preprocessed_*.parquet"

    def resolve_cluster_col(self, available_columns: List[str]) -> str:
        if self.cluster_col in available_columns:
            return self.cluster_col
        if self.cluster_col_alt in available_columns:
            return self.cluster_col_alt
        raise ValueError(
            f"Neither '{self.cluster_col}' nor '{self.cluster_col_alt}' "
            f"found in columns: {available_columns}"
        )

    def resolve_quality_cols(self, available_columns: List[str]) -> List[str]:
        cols = [c for c in self.quality_columns if c in available_columns]
        if not cols:
            cols = [c for c in self.nemotron_quality_columns if c in available_columns]
        return cols


DEFAULT_SCHEMA = ColumnSchema()
