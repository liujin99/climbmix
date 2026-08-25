#!/usr/bin/env python3
"""Test whether fp32 warmup before fp16 prevents NaN on NPU."""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # sets up fake xformers
import torch, torch_npu, numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")
pf = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])[:1]
texts = []
for fname in pf:
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["text"])
    texts.extend([str(t) if t is not None else "" for t in table.column("text").to_pylist()[:300]])

rng = np.random.default_rng(42)
n = min(200, len(texts))
idx = rng.choice(len(texts), size=n, replace=False)
sample = [texts[i] for i in idx]
print(f"Loaded {n} sample texts")

from sentence_transformers import SentenceTransformer

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    pct = n_nan / len(emb) * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  {label:50s}: NaN={n_nan:4d}/{len(emb)} ({pct:5.1f}%) [{status}]")
    return n_nan

# ── Test 1: fp32 only (no half) ──
print("\n=== Test 1: fp32 only (msl=512, bs=512) ===")
m1 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m1.eval()
m1.max_seq_length = 512
print(f"  dtype: {next(m1.parameters()).dtype}")
emb = m1.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "fp32 msl=512 bs=512")

# ── Test 2: fp16 WITHOUT fp32 warmup (current code path) ──
print("\n=== Test 2: fp16 direct (no warmup) ===")
m2 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m2.eval()
m2.half()
m2.max_seq_length = 512
print(f"  dtype: {next(m2.parameters()).dtype}")
emb = m2.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "fp16 direct msl=512 bs=512")

# ── Test 3: fp16 WITH fp32 warmup (one fp32 encode first) ──
print("\n=== Test 3: fp16 with fp32 warmup ===")
m3 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m3.eval()
m3.max_seq_length = 512
print(f"  fp32 dtype: {next(m3.parameters()).dtype}")
# Warmup: one small fp32 encode
_ = m3.encode(sample[:10], batch_size=10, show_progress_bar=False, normalize_embeddings=True)
print("  fp32 warmup done")
m3.half()
print(f"  fp16 dtype: {next(m3.parameters()).dtype}")
emb = m3.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "fp16 warmup msl=512 bs=512")

# ── Test 4: fp16 with fp32 warmup, multiple fp16 runs ──
print("\n=== Test 4: fp16 warmup, repeat 3x ===")
for rep in range(3):
    emb = m3.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp16 warmup rep{rep}")

# ── Test 5: bf16 ──
print("\n=== Test 5: bf16 direct ===")
m4 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m4.eval()
m4.to(torch.bfloat16)
m4.max_seq_length = 512
print(f"  dtype: {next(m4.parameters()).dtype}")
emb = m4.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "bf16 direct msl=512 bs=512")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
