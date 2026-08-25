#!/usr/bin/env python3
"""Verify fixed-512 padding produces batch-independent embeddings.

Root cause: model always unpads and uses BlockDiagonalMask. Our fallback
path re-pads to max(qs_list) which varies by batch (4 vs 512). SDPA kernel
uses different algorithms for different matrix sizes → fp16 results differ.

Fix: always pad to 512 (max_seq_length). All batches use [n, 16, 512, 512]
matrices → same kernel algorithm → batch-independent results.

This script verifies: sorted vs unsorted via tokenize+model() path
should now produce identical results (max diff = 0.0).
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
    features = model.tokenize(texts_batch)
    features = _to_device(features)
    with torch.no_grad():
        output = model(features)
    emb = output["sentence_embedding"].float()
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()

def embed_all(model, texts, batch_size=BS):
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
    print("  → IDENTICAL (sorted == unsorted) ✓")
elif diff.max() < 1e-3:
    print("  → NEGLIGIBLE difference (< 1e-3)")
else:
    print(f"  → SIGNIFICANT difference (> 1e-3) ✗ — padding fix did not resolve")

# ── 4. Also compare with model.encode() ──
print("\n--- 4. tokenize+model vs model.encode() ---")
emb_encode = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb_encode = np.array(emb_encode, dtype=np.float32)

diff2 = np.abs(emb_unsorted - emb_encode)
print(f"  Max abs diff:  {diff2.max():.10f}")
print(f"  Mean abs diff: {diff2.mean():.10f}")

if diff2.max() < 1e-6:
    print("  → IDENTICAL (tokenize+model == encode) ✓")
elif diff2.max() < 1e-3:
    print("  → NEGLIGIBLE difference (< 1e-3)")
else:
    print(f"  → SIGNIFICANT difference (> 1e-3)")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
