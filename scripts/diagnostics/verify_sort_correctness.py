#!/usr/bin/env python3
"""Verify length-sorted embedding via tokenize+model() path (streaming worker).

model.encode() already sorts by length internally, so testing through encode()
can't measure the benefit. The streaming worker uses model.tokenize() + model()
directly — no auto-sort. This script tests that path with sorted vs unsorted
batches, verifying both correctness and performance.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # noqa: F401 — installs fake xformers
import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq
import time

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")
BS = 512

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

device = torch.device("npu")

def _to_device(features):
    for key in features:
        if isinstance(features[key], torch.Tensor):
            features[key] = features[key].to(device)
        elif isinstance(features[key], dict):
            _to_device(features[key])
    return features

def embed_batch(model, texts_batch):
    """Embed a batch via tokenize+model() — same path as streaming worker."""
    features = model.tokenize(texts_batch)
    features = _to_device(features)
    with torch.no_grad():
        output = model(features)
    emb = output["sentence_embedding"].float()
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def embed_all(model, texts, batch_size=BS):
    """Embed all texts via tokenize+model() in batches."""
    all_emb = np.empty((len(texts), 1024), dtype=np.float32)
    for j in range(0, len(texts), batch_size):
        batch = texts[j:j + batch_size]
        emb = embed_batch(model, batch)
        all_emb[j:j + len(batch)] = emb
    return all_emb

# ── 1. Unsorted embedding (tokenize+model path) ──
print("\n--- 1. Unsorted embedding (tokenize+model, bs=512) ---")
emb_unsorted = embed_all(m, sample_texts)
n_nan = np.isnan(emb_unsorted).any(axis=1).sum()
print(f"  NaN: {n_nan}")

# ── 2. Length-sorted embedding ──
print("\n--- 2. Length-sorted embedding (tokenize+model, bs=512) ---")
sort_order = sorted(range(len(sample_texts)), key=lambda i: len(sample_texts[i]))
sorted_texts = [sample_texts[i] for i in sort_order]

lens = [len(t) for t in sample_texts]
sorted_lens = [len(t) for t in sorted_texts]
print(f"  Original: min={min(lens)}, max={max(lens)}, median={sorted(lens)[len(lens)//2]}")
print(f"  Sorted batch 0: lens={sorted_lens[:5]}...{sorted_lens[507:512]}")
print(f"  Sorted batch 1: lens={sorted_lens[512:517]}...{sorted_lens[1019:1024]}")
print(f"  Sorted batch 2: lens={sorted_lens[1024:1029]}...{sorted_lens[1531:1536]}")
print(f"  Sorted batch 3: lens={sorted_lens[1536:1541]}...{sorted_lens[1995:2000]}")

emb_sorted_raw = embed_all(m, sorted_texts)
n_nan_sorted = np.isnan(emb_sorted_raw).any(axis=1).sum()
print(f"  NaN: {n_nan_sorted}")

# ── 3. Unsort and compare ──
print("\n--- 3. Unsort and compare ---")
emb_sorted = np.empty_like(emb_sorted_raw)
for orig_pos, new_pos in enumerate(sort_order):
    emb_sorted[orig_pos] = emb_sorted_raw[new_pos]

diff = np.abs(emb_unsorted - emb_sorted)
print(f"  Max abs diff:  {diff.max():.10f}")
print(f"  Mean abs diff: {diff.mean():.10f}")

cos = (emb_unsorted * emb_sorted).sum(axis=1) / (
    np.linalg.norm(emb_unsorted, axis=1) * np.linalg.norm(emb_sorted, axis=1) + 1e-12)
print(f"  Cosine sim:    min={cos.min():.10f}, mean={cos.mean():.10f}, max={cos.max():.10f}")

if diff.max() < 1e-6:
    print("  → IDENTICAL (sorted == unsorted)")
elif diff.max() < 1e-3:
    print("  → NEGLIGIBLE difference (< 1e-3, expected fp16 rounding)")
else:
    print("  → SIGNIFICANT difference (> 1e-3)")

# ── 4. Performance comparison ──
print("\n--- 4. Performance comparison (tokenize+model path) ---")

t0 = time.time()
for _ in range(3):
    _ = embed_all(m, sample_texts)
t_unsorted = (time.time() - t0) / 3

t0 = time.time()
for _ in range(3):
    _ = embed_all(m, sorted_texts)
t_sorted = (time.time() - t0) / 3

print(f"  Unsorted: {t_unsorted:.2f}s ({len(sample_texts)/t_unsorted:.0f} docs/s)")
print(f"  Sorted:   {t_sorted:.2f}s ({len(sample_texts)/t_sorted:.0f} docs/s)")
print(f"  Speedup:  {t_unsorted/t_sorted:.2f}x")

# ── 5. Per-batch timing ──
print("\n--- 5. Per-batch timing ---")
for label, tlist in [("unsorted", sample_texts), ("sorted", sorted_texts)]:
    print(f"  {label}:")
    for j in range(0, len(tlist), BS):
        batch = tlist[j:j + BS]
        t0 = time.time()
        _ = embed_batch(m, batch)
        dt = time.time() - t0
        max_len = max(len(t) for t in batch)
        print(f"    batch {j//BS}: {dt:.2f}s, max_char_len={max_len}, n={len(batch)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
