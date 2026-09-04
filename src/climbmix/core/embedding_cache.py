"""Sharded pool-embedding cache (format sharded-v1).

Why sharded: a full-pool cache is ~443 GB (116M x 1024 fp32). A single
.npy of that size hits FUSE single-file limits (live finding 2026-09-04:
the OBS mount refused the write at ~191 GiB with EFBIG), is awkward to
move/verify/back up, and must restart end-to-end after any interruption.
The industrial shape is many moderate files + a manifest: one block per
embed unit (~7.5 GB), blocks globally consecutive (unit k covers exactly
the pool rows its shards span), and manifest.json is the publish gate —
a cache exists iff a complete, validated manifest points at complete
blocks.

Layout at <EMBEDDING_CACHE_DIR>/<key>/:
  manifest.json    {"format": "sharded-v1", row_count, emb_dim, model,
                    truncate_len, blocks: [{file, unit_id, rows,
                    global_start, shards}]}
  block_<unit>.npy one mmap-able block per unit

The reader mirrors the single-file semantics it replaces: slicing stays
chunk-friendly (cluster_assign's prescan/assign loops consume [lo:hi]
views unchanged), full scans iterate blocks in order, and faiss K-means
trains on a deterministic gathered sample (faiss subsamples to
<= max_points_per_centroid points internally anyway — the gather just
picks that sample with sequential per-block I/O instead of random reads
over a 443 GB memmap). Single-file .npy/.npz caches keep working
alongside (discovery prefers whichever exists); the single-machine
embed path still writes .npy.
"""

import json
import os
import threading

MANIFEST_NAME = "manifest.json"
FORMAT = "sharded-v1"


def manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, MANIFEST_NAME)


def is_sharded_cache(cache_dir: str) -> bool:
    """True when cache_dir holds a published sharded manifest."""
    return os.path.isfile(manifest_path(cache_dir))


def load_manifest(cache_dir: str) -> dict:
    """Parsed + structurally validated manifest.

    Raises ValueError on any inconsistency — a half-written or foreign
    manifest must never reach the reader as a cache hit (the merge
    publishes via tmp+rename, so a torn manifest simply does not exist).
    """
    with open(manifest_path(cache_dir)) as f:
        man = json.load(f)
    if man.get("format") != FORMAT:
        raise ValueError(f"{manifest_path(cache_dir)}: format "
                         f"{man.get('format')!r} != {FORMAT!r}")
    blocks = man.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(f"{manifest_path(cache_dir)}: no blocks")
    row_count = int(man["row_count"])
    emb_dim = int(man["emb_dim"])
    expect_start = 0
    for b in blocks:
        rows = int(b["rows"])
        start = int(b["global_start"])
        if start != expect_start or rows <= 0:
            raise ValueError(f"{manifest_path(cache_dir)}: block "
                             f"{b.get('unit_id')!r} breaks the global "
                             f"row chain (start {start}, rows {rows}, "
                             f"expected start {expect_start})")
        if int(b.get("emb_dim", emb_dim)) != emb_dim:
            raise ValueError(f"{manifest_path(cache_dir)}: block "
                             f"{b.get('unit_id')!r} emb_dim mismatch")
        expect_start += rows
    if expect_start != row_count:
        raise ValueError(f"{manifest_path(cache_dir)}: blocks cover "
                         f"{expect_start:,} rows, manifest says "
                         f"{row_count:,}")
    return man


class ShardedEmbeddingCache:
    """Manifest-driven read view over block_<unit>.npy files.

    Quacks like the ndarray/memmap it replaces for every consumer in
    the cluster path: .shape/.ndim/.dtype, contiguous [lo:hi] slicing
    (a view inside one block, a concatenate across block borders —
    cluster_assign's chunk loops never notice), and block-order
    iteration for full scans. Fancy indexing is deliberately NOT
    supported — use gather_train_sample for row sampling.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest = load_manifest(cache_dir)
        self._blocks = self.manifest["blocks"]
        self._starts = [int(b["global_start"]) for b in self._blocks]
        self._rows = [int(b["rows"]) for b in self._blocks]
        self.n_rows = int(self.manifest["row_count"])
        self.dim = int(self.manifest["emb_dim"])
        self._maps = [None] * len(self._blocks)
        self._lock = threading.Lock()

    @property
    def shape(self):
        return (self.n_rows, self.dim)

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        import numpy as np
        return np.dtype(np.float32)

    @property
    def block_count(self):
        return len(self._blocks)

    def _map(self, i: int):
        """Lazily mmap block i, verifying its shape against the manifest
        (a block edited behind the manifest's back fails loudly, never
        mis-aligns rows)."""
        m = self._maps[i]
        if m is None:
            import numpy as np
            path = os.path.join(self.cache_dir, self._blocks[i]["file"])
            m = np.load(path, mmap_mode="r")
            want = (self._rows[i], self.dim)
            if tuple(m.shape) != want:
                raise ValueError(f"{path}: shape {tuple(m.shape)} != "
                                 f"manifest {want}")
            with self._lock:
                self._maps[i] = m
        return m

    def _block_of(self, row: int) -> int:
        import bisect
        i = bisect.bisect_right(self._starts, row) - 1
        if i < 0 or row >= self._starts[i] + self._rows[i]:
            raise IndexError(f"row {row} out of range for "
                             f"{self.n_rows}-row sharded cache")
        return i

    def __getitem__(self, key):
        import numpy as np
        if isinstance(key, tuple) and len(key) == 2:
            rk, ck = key
            if isinstance(rk, int) and isinstance(ck, (int, slice)):
                return self[rk][ck]
            if isinstance(rk, slice) and isinstance(ck, (int, slice)):
                return self[rk][:, ck]
            raise NotImplementedError(
                "only basic (row, col) indexing is supported on sharded "
                "caches")
        if isinstance(key, int):
            lo = slice(key, key + 1).indices(self.n_rows)[0]
            i = self._block_of(lo)
            return self._map(i)[lo - self._starts[i]]
        if not isinstance(key, slice):
            raise NotImplementedError(
                "fancy indexing unsupported on sharded caches — use "
                "gather_train_sample / iter_blocks")
        lo, hi, step = key.indices(self.n_rows)
        if step != 1:
            raise NotImplementedError("strided slicing unsupported")
        if lo >= hi:
            return np.empty((0, self.dim), dtype=np.float32)
        i0 = self._block_of(lo)
        i1 = self._block_of(hi - 1)
        if i0 == i1:
            return self._map(i0)[lo - self._starts[i0]:
                                  hi - self._starts[i0]]
        parts = []
        for i in range(i0, i1 + 1):
            a = max(lo, self._starts[i]) - self._starts[i]
            b = min(hi, self._starts[i] + self._rows[i]) - self._starts[i]
            parts.append(self._map(i)[a:b])
        return np.concatenate(parts, axis=0)

    def iter_blocks(self):
        """(global_start, memmap) for every block, in global row order —
        the full-scan form (validation, sequential assignment passes)."""
        for i in range(len(self._blocks)):
            yield self._starts[i], self._map(i)

    def gather_train_sample(self, n: int, seed: int):
        """Deterministic (n, dim) fp32 training sample.

        Per-block counts by largest remainder, per-block permutation
        from a seeded generator: identical for a given (n, seed), and
        the I/O stays block-sequential instead of scattering random
        reads across a pool-sized memmap. n >= n_rows materializes the
        (small) pool exactly — matching what an in-RAM path would hand
        faiss.
        """
        import numpy as np
        if n >= self.n_rows:
            return np.ascontiguousarray(
                np.concatenate([np.asarray(v) for _, v in
                                self.iter_blocks()], axis=0),
                dtype=np.float32)
        rng = np.random.default_rng(seed)
        weights = self._rows
        total_w = sum(weights)
        raw = [n * w / total_w for w in weights]
        counts = [int(x) for x in raw]
        short = n - sum(counts)
        order = sorted(range(len(raw)),
                       key=lambda i: raw[i] - counts[i], reverse=True)
        for k in range(short):
            counts[order[k % len(order)]] += 1
        parts = []
        for (start, view), c in zip(self.iter_blocks(), counts):
            if c == 0:
                continue
            idx = np.sort(rng.permutation(view.shape[0])[:c])
            parts.append(np.asarray(view[idx]))
        return np.ascontiguousarray(np.concatenate(parts, axis=0),
                                    dtype=np.float32)
