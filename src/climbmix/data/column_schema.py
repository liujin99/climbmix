"""
Centralized column schema for parquet data files.

All column-name knowledge is defined here. Other modules reference
this instead of hardcoding column names.

Quality columns are configurable via YAML:
  config/quality_columns.yaml

Supported quality label sets (checked in order):
  1. Custom YAML config (if quality_config_path is set)
  2. STEM labels: stem_relevance, knowledge_value, notation_fidelity, rigor_coherence, noise_level
  3. FineWeb labels: qs_dclm, qs_fineweb_edu_approx, qs_english, ...
  4. Nemotron labels: qs_quality, qs_educational, qs_informational, qs_advertisement

All quality scores are 1-5 discrete, higher = better (including noise_level).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


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

    stem_quality_columns: tuple = (
        "stem_relevance",
        "knowledge_value",
        "notation_fidelity",
        "rigor_coherence",
        "noise_level",
    )

    preprocessed_pattern: str = "preprocessed_*.parquet"
    quality_config_path: Optional[str] = None

    def resolve_cluster_col(self, available_columns: List[str]) -> str:
        if self.cluster_col in available_columns:
            return self.cluster_col
        if self.cluster_col_alt in available_columns:
            return self.cluster_col_alt
        raise ValueError(
            f"Neither '{self.cluster_col}' nor '{self.cluster_col_alt}' "
            f"found in columns: {available_columns}"
        )

    def _load_custom_quality_cols(self) -> List[str]:
        if not self.quality_config_path or not os.path.exists(self.quality_config_path):
            return []
        import yaml
        with open(self.quality_config_path) as f:
            config = yaml.safe_load(f)
        return list(config.get("quality_columns", []))

    def load_prune_threshold(self, default: float = 3.0) -> float:
        if not self.quality_config_path or not os.path.exists(self.quality_config_path):
            return default
        import yaml
        with open(self.quality_config_path) as f:
            config = yaml.safe_load(f)
        return float(config.get("prune_threshold", default))

    def resolve_quality_cols(self, available_columns: List[str]) -> List[str]:
        if self.quality_config_path:
            custom_cols = self._load_custom_quality_cols()
            if custom_cols:
                found = [c for c in custom_cols if c in available_columns]
                if found:
                    print(f"[Schema] Using custom quality columns from {self.quality_config_path}: {found}")
                    return found

        for label, cols in [
            ("STEM", self.stem_quality_columns),
            ("FineWeb", self.quality_columns),
            ("Nemotron", self.nemotron_quality_columns),
        ]:
            found = [c for c in cols if c in available_columns]
            if found:
                print(f"[Schema] Using {label} quality columns: {found}")
                return found
        print("[Schema] No quality columns found in data")
        return []


DEFAULT_SCHEMA = ColumnSchema()
