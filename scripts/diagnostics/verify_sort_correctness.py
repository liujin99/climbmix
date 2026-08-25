#!/usr/bin/env python3
"""Verify length-sorted embedding produces identical results to unsorted.

Sorts texts by length, embeds in sorted order, unsorts, then compares
with direct (unsorted) embedding. Max diff should be 0.0 (same SDPA kernel,
same computation, just different batch composition).
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")

pf = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])[:2]
texts = []
for fname in pf:
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["text"])
    texts.extend([str(t) if t is not None else "" for t in table.column("text").to_pylist()])
    del table

rng = np.random.default_rng(42)
idx = rng.choice(len(texts), size=2000, replace=False)
idx.sort()
sample_texts = [texts[i] for i in idx]
print(f"Loaded {len(sample_texts)} sample texts")

from sentence_transformers import SentenceTransformer

print("\n=== Loading model (fp16, msl=512) ===")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512
print(f"  dtype: {next(m.parameters()).dtype}, msl: {m.max_seq_length}")

# ── 1. Direct embedding (unsorted) ──
print("\n--- 1. Direct embedding (unsorted, bs=512) ---")
emb_direct = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb_direct = np.array(emb_direct, dtype=np.float32)
n_nan = np.isnan(emb_direct).any(axis=1).sum()
print(f"  NaN: {n_nan}")

# ── 2. Length-sorted embedding ──
print("\n--- 2. Length-sorted embedding (bs=512) ---")
sort_order = sorted(range(len(sample_texts)), key=lambda i: len(sample_texts[i]))
sorted_texts = [sample_texts[i] for i in sort_order]

# Show length distribution
lens = [len(t) for t in sample_texts]
sorted_lens = [len(t) for t in sorted_texts]
print(f"  Original: min={min(lens)}, max={max(lens)}, median={sorted(lens)[len(lens)//2]}")
print(f"  Sorted batch 0: lens={sorted_lens[:5]}...{sorted_lens[507:512]}")
print(f"  Sorted batch 1: lens={sorted_lens[512:517]}...{sorted_lens[1019:1024]}")
print(f"  Sorted batch 2: lens={sorted_lens[1024:1029]}...{sorted_lens[1531:1536]}")
print(f"  Sorted batch 3: lens={sorted_lens[1536:1541]}...{sorted_lens[1995:2000]}")

emb_sorted = m.encode(sorted_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb_sorted = np.array(emb_sorted, dtype=np.float32)
n_nan_sorted = np.isnan(emb_sorted).any(axis=1).sum()
print(f"  NaN: {n_nan_sorted}")

# ── 3. Unsort and compare ──
print("\n--- 3. Unsort and compare ---")
emb_unsorted = np.empty_like(emb_sorted)
for orig_pos, new_pos in enumerate(
        [0] * len(sort_order)):
    pass
# Build inverse permutation
inv_order = [0] * len(sort_order)
for new_pos, orig_pos in enumerate(sort_order):
    inv_order[orig_pos] = new_pos
for orig_pos in range(len(sample_texts)):
    emb_unsorted[orig_pos] = emb_sorted[inv_order[orig_pos]]

diff = np.abs(emb_direct - emb_unsorted)
print(f"  Max abs diff:  {diff.max():.10f}")
print(f"  Mean abs diff: {diff.mean():.10f}")

cos = (emb_direct * emb_unsorted).sum(axis=1) / (
    np.linalg.norm(emb_direct, axis=1) * np.linalg.norm(emb_unsorted, axis=1) + 1e-12)
print(f"  Cosine sim:    min={cos.min():.10f}, mean={cos.mean():.10f}, max={cos.max():.10f}")

if diff.max() < 1e-6:
    print("  → IDENTICAL (sorted == unsorted)")
elif diff.max() < 1e-3:
    print("  → NEGLIGIBLE difference (< 1e-3, expected fp16 rounding)")
else:
    print("  → SIGNIFICANT difference (> 1e-3)")

# ── 4. Performance comparison ──
print("\n--- 4. Performance comparison ---")
import time

t0 = time.time()
for _ in range(3):
    _ = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
t_unsorted = (time.time() - t0) / 3

t0 = time.time()
for _ in range(3):
    _ = m.encode(sorted_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
t_sorted = (time.time() - t0) / 3

print(f"  Unsorted: {t_unsorted:.2f}s ({len(sample_texts)/t_unsorted:.0f} docs/s)")
print(f"  Sorted:   {t_sorted:.2f}s ({len(sample_texts)/t_sorted:.0f} docs/s)")
print(f"  Speedup:  {t_unsorted/t_sorted:.2f}x")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
