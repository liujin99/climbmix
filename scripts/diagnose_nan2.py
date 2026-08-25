#!/usr/bin/env python3
"""Targeted NaN diagnosis — isolate root cause.

Tests:
A. fp32 + TND path (current embed_documents config) → baseline NaN
B. fp32 + fallback SDPA path (disable TND) → is it our TND patch?
C. fp16 + TND path → is it fp32 precision?
D. fp16 + fallback SDPA → is it fp16 + fallback?
E. fp32 + max_seq_length=512 → is it long sequences?
F. fp16 + max_seq_length=512 + batch_size=512 → exact streaming config
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

# We'll import embedding_cluster first to set up fake xformers,
# but we need to be able to toggle TND on/off
import climbmix.core.embedding_cluster as ec
import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")

# Load texts
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
print(f"Loaded {len(sample_texts)} sample texts from {len(pf)} files")
print(f"Text lengths: min={min(len(t) for t in sample_texts)}, max={max(len(t) for t in sample_texts)}, mean={np.mean([len(t) for t in sample_texts]):.0f}")

from sentence_transformers import SentenceTransformer

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan_rows = np.isnan(emb).any(axis=1).sum()
    n_nan_total = np.isnan(emb).sum()
    n_inf = np.isinf(emb).sum()
    pct = n_nan_rows / emb.shape[0] * 100
    status = "PASS" if n_nan_rows == 0 else "FAIL"
    print(f"  {label:45s}: NaN_rows={n_nan_rows:4d}/{emb.shape[0]} ({pct:5.1f}%)  NaN_total={n_nan_total:6d}  Inf={n_inf}  [{status}]")
    return n_nan_rows

# Check default max_seq_length
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
default_msl = m.max_seq_length
print(f"\nDefault max_seq_length: {default_msl}")
print(f"Model dtype: {next(m.parameters()).dtype}")
del m
torch.npu.empty_cache()

# ── Test A: fp32 + TND (current embed_documents config) ──
print("\n--- Test A: fp32 + TND path (current embed_documents) ---")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
for bs in [8, 64, 256]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp32 TND bs={bs}")
del m
torch.npu.empty_cache()

# ── Test B: fp32 + fallback SDPA (disable TND) ──
print("\n--- Test B: fp32 + fallback SDPA (TND disabled) ---")
ec._HAS_NPU_FA = False  # Force fallback path
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
for bs in [8, 64, 256]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp32 fallback bs={bs}")
del m
torch.npu.empty_cache()

# ── Test C: fp16 + TND ──
print("\n--- Test C: fp16 + TND path ---")
ec._HAS_NPU_FA = True  # Re-enable TND
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
for bs in [8, 64, 256, 512]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp16 TND bs={bs}")
del m
torch.npu.empty_cache()

# ── Test D: fp16 + fallback SDPA ──
print("\n--- Test D: fp16 + fallback SDPA (TND disabled) ---")
ec._HAS_NPU_FA = False
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
for bs in [8, 64, 256, 512]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp16 fallback bs={bs}")
del m
torch.npu.empty_cache()

# ── Test E: fp32 + TND + max_seq_length=512 ──
print("\n--- Test E: fp32 + TND + max_seq_length=512 ---")
ec._HAS_NPU_FA = True
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.max_seq_length = 512
for bs in [8, 64, 256]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp32 TND msl=512 bs={bs}")
del m
torch.npu.empty_cache()

# ── Test F: fp16 + TND + max_seq_length=512 + bs=512 (streaming config) ──
print("\n--- Test F: fp16 + TND + msl=512 + bs=512 (streaming config) ---")
ec._HAS_NPU_FA = True
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.max_seq_length = 512
m.half()
emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "fp16 TND msl=512 bs=512")
del m
torch.npu.empty_cache()

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
