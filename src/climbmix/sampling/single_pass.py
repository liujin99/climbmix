"""Single-pass (epoch <= 1) guard for target-arm annealing consumption.

The problem this module closes (TODO "epoch>1 repetition guard"): the
mixture pool and the training consumption are set by TWO independent
knobs —

  - pool:      ``--target-tokens`` drives STEM selection, general data
               fills to ``1 / stem_ratio``  (TARGET_TOKENS=2B -> ~2.86B)
  - consume:   ``TARGET_STEPS x total_batch_size``  (2000 x 1,048,576 ~ 2.1B)

nanochat's dataloader wraps back to the first shard when the mixture runs
out — SILENTLY. A config where consumption exceeds the pool (e.g. steps
doubled, tokens not) therefore re-samples documents with no error, no
warning and no log line, which (a) deviates from the paper's single-pass
annealing semantics and (b) hits the two arms asymmetrically (cluster
size distributions differ, so the wrap point differs).

The guard measures the ACTUAL mixture pool (train shards' text chars via
the same chars-per-token estimator the selection budget uses) and raises
loudly when planned consumption exceeds it. It is observability-only:
it never touches data products, so it is fingerprint-excluded
(utils/fingerprint.py GLOBAL_EXCLUDE).

Precision notes:
  - the pool estimate uses the same ``chars/4`` heuristic as
    ``ShardMetadataManager.estimate_token_counts`` (the budget currency),
    while consumption is real tokenizer tokens (total_batch_size). The
    mismatch is the same one the budget itself carries; ``max_epochs``
    defaults to 1.02 to absorb measurement noise, NOT semantics.
  - the last ``shard_*.parquet`` is the val split (nanochat convention,
    held by every writer in this repo) and is excluded from the pool.
"""

import glob
import json
import math
import os
from typing import Dict, Optional

import pyarrow.compute as pc
import pyarrow.parquet as pq

from climbmix.utils.token_estimate import DEFAULT_CHARS_PER_TOKEN


def measure_train_tokens(
    data_dir: str,
    chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
) -> int:
    """Estimated token count of the TRAIN shards in a mixed-data dir.

    Sums text char lengths over all ``shard_*.parquet`` except the LAST
    (val split). Crashed partial writes (``*.tmp.parquet``) are ignored.
    """
    names = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("shard_") and f.endswith(".parquet")
        and not f.endswith(".tmp.parquet")
    )
    if not names:
        raise FileNotFoundError(
            f"no shard_*.parquet in {data_dir} — not a mixture dir?")
    train_names = names[:-1]  # last shard = val split (nanochat convention)
    if not train_names:
        raise ValueError(
            f"{data_dir} contains only one shard (the val split?) — "
            f"no train data to measure")
    total_chars = 0
    for name in train_names:
        table = pq.read_table(os.path.join(data_dir, name), columns=["text"])
        lengths = pc.utf8_length(table["text"])
        total_chars += pc.sum(lengths).as_py() or 0
    return int(total_chars / chars_per_token)


def read_total_batch_size(ckpt_dir: Optional[str]) -> Optional[int]:
    """total_batch_size from the newest meta_*.json under ckpt_dir.

    Returns None when the dir is missing, empty or no meta carries the
    key — callers must fail loudly on None (a guessed batch size would
    silently weaken the guard by the guess factor).
    """
    if not ckpt_dir:
        return None
    for path in reversed(sorted(glob.glob(os.path.join(ckpt_dir, "meta_*.json")))):
        try:
            with open(path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        tbs = meta.get("total_batch_size")
        if tbs:
            return int(tbs)
    return None


def derive_num_iterations(target_tokens: int, total_batch_size: int) -> int:
    """steps = floor(target_tokens / total_batch_size) — single-knob form.

    TARGET_TOKENS is the single source of truth for the annealing budget:
    the pool is selected at TARGET_TOKENS (STEM) diluted 1/stem_ratio with
    general data, and consumption = steps x total_batch_size ~= the same
    budget — so the run is single-pass (epoch ~= stem_ratio = 0.7) BY
    CONSTRUCTION and the guard always passes on derived values. An
    explicit TARGET_STEPS override (same-data/different-steps reuse
    experiments) bypasses the derivation and is checked by
    check_single_pass instead.
    """
    if target_tokens <= 0:
        raise ValueError(
            f"target_tokens must be positive to derive steps, got {target_tokens}")
    if total_batch_size <= 0:
        raise ValueError(
            f"total_batch_size must be positive, got {total_batch_size}")
    return max(1, target_tokens // total_batch_size)


def check_single_pass(
    num_iterations: int,
    total_batch_size: int,
    pool_tokens: int,
    stem_ratio: float = 0.7,
    max_epochs: float = 1.02,
    context: str = "target arm",
) -> Dict[str, int]:
    """Validate steps x total_batch_size <= pool_tokens.

    Returns {"consume_tokens", "pool_tokens", "epochs_x100"} on success
    (epochs as a fixed-point int for log-friendly formatting by callers).
    Raises ValueError with the two concrete fixes when epochs > max_epochs.
    """
    consume = int(num_iterations) * int(total_batch_size)
    if pool_tokens <= 0:
        raise ValueError(
            f"[{context}] single-pass guard: measured pool is {pool_tokens} "
            f"tokens — the mixture dir is empty or unreadable")
    epochs = consume / pool_tokens
    if epochs <= max_epochs:
        return {
            "consume_tokens": consume,
            "pool_tokens": pool_tokens,
            "epochs_x100": round(epochs * 100),
        }

    max_steps = pool_tokens // total_batch_size
    stem_needed = math.ceil(consume * stem_ratio) if stem_ratio > 0 else consume
    raise ValueError(
        f"[{context}] single-pass violation: planned consumption "
        f"{consume:,} tokens ({num_iterations} steps x "
        f"{total_batch_size:,} tokens/step) exceeds the measured mixture "
        f"pool {pool_tokens:,} tokens ({epochs:.2f} epochs > {max_epochs}). "
        f"The nanochat loader would silently wrap and re-sample documents, "
        f"breaking the paper's single-pass annealing semantics "
        f"asymmetrically across arms. Fix one of: "
        f"(a) raise the selection budget to >= {stem_needed:,} STEM tokens "
        f"(--target-tokens, stem_ratio={stem_ratio}) and rebuild the "
        f"mixture, or (b) lower --target-steps to <= {max_steps}."
    )
