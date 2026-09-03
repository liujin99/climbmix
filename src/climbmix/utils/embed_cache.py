"""Shared content key for pool-level embedding caches.

The key decides WHERE the (expensive, ~40h) full-pool embedding cache
lives: `<EMBEDDING_CACHE_DIR>/<key>/embedding_cache.npz`. It is computed
by CLIMBPipeline._pool_embedding_cache_dir at cache-hit time (Step 1)
and MUST be reproduced bit-exactly by the offline merge tool
(scripts/embed_merge.py) that assembles the multi-node unit partials
into that same cache. Two independent copies of the formula would drift
silently — a drifted key is a silent cache miss that re-embeds the pool
— so the formula lives here, once.

Inputs (all must match the RUN CONFIG, not the merge-time mood):
  - shard names + file sizes, sorted by name (the pool identity)
  - embedding model name
  - truncate length
  - sample size (hashed only when > 0, so full-pool keys stay stable)

NOT keyed by K_enhanced/K_max/merge_distance/prune_threshold: those
change the merge stage only, which must reuse the embeddings.
"""

import hashlib
from typing import Iterable, Tuple


def pool_embedding_cache_key(
    shard_name_sizes: Iterable[Tuple[str, int]],
    embedding_model: str,
    truncate_len: int,
    sample_size: int = 0,
) -> str:
    """12-hex content key; see the module docstring for the contract.

    shard_name_sizes: iterable of (parquet basename, size in bytes);
    order-insensitive (sorted by name internally — the same order the
    pipeline's os.listdir+sorted produces).
    """
    hasher = hashlib.sha256()
    for name, size in sorted(shard_name_sizes, key=lambda ns: ns[0]):
        hasher.update(name.encode())
        hasher.update(str(int(size)).encode())
    hasher.update(embedding_model.encode())
    hasher.update(str(int(truncate_len)).encode())
    if sample_size and sample_size > 0:
        hasher.update(f"sample={int(sample_size)}".encode())
    return hasher.hexdigest()[:12]
