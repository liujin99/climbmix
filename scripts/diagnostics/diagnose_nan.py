#!/usr/bin/env python3
"""Clean NaN diagnosis — single model load, no reloads.

Key: load model ONCE, test all configs on same model instance.
Avoids NPU state corruption from multiple model reloads.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # sets up fake xformers
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
print(f"Loaded {len(sample_texts)} sample texts")
print(f"Text lengths: min={min(len(t) for t in sample_texts)}, max={max(len(t) for t in sample_texts)}")

from sentence_transformers import SentenceTransformer

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan_rows = np.isnan(emb).any(axis=1).sum()
    n_nan_total = np.isnan(emb).sum()
    pct = n_nan_rows / emb.shape[0] * 100
    status = "PASS" if n_nan_rows == 0 else "FAIL"
    print(f"  {label:50s}: NaN_rows={n_nan_rows:4d}/{emb.shape[0]} ({pct:5.1f}%)  [{status}]")
    return n_nan_rows

# ── Load model ONCE, never reload ──
print("\n=== Loading model (fp32, ONCE) ===")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
print(f"  max_seq_length: {m.max_seq_length}")
print(f"  dtype: {next(m.parameters()).dtype}")

# ── Phase 1: fp32, TND path, repeat same batch size 3x ──
print("\n--- Phase 1: fp32 + TND, repeat bs=256 three times ---")
for rep in range(3):
    emb = m.encode(sample_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp32 TND bs=256 rep{rep}")

# ── Phase 2: fp32, different batch sizes ──
print("\n--- Phase 2: fp32 + TND, different batch sizes ---")
for bs in [8, 32, 64, 128, 256]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp32 TND bs={bs}")

# ── Phase 3: switch to fp16 (model.half()), same model ──
print("\n--- Phase 3: fp16 + TND (model.half() on same model) ---")
m.half()
print(f"  dtype after half: {next(m.parameters()).dtype}")
for bs in [8, 64, 256, 512]:
    emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp16 TND bs={bs}")

# ── Phase 4: fp16, set max_seq_length=512 (streaming config) ──
print("\n--- Phase 4: fp16 + TND + msl=512 + bs=512 (streaming config) ---")
m.max_seq_length = 512
emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb, "fp16 TND msl=512 bs=512")

# ── Phase 5: fp16, repeat bs=512 three times (determinism check) ──
print("\n--- Phase 5: fp16 + TND + msl=512, repeat bs=512 three times ---")
for rep in range(3):
    emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"fp16 TND msl=512 bs=512 rep{rep}")

# ── Phase 6: Check which rows are NaN (if any) ──
print("\n--- Phase 6: Identify NaN rows (fp16, bs=512) ---")
emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb = np.array(emb, dtype=np.float32)
nan_rows = np.where(np.isnan(emb).any(axis=1))[0]
if len(nan_rows) > 0:
    print(f"  {len(nan_rows)} NaN rows. First 10:")
    for r in nan_rows[:10]:
        orig_idx = idx[r]
        t = sample_texts[r]
        tok_len = len(t.split())
        print(f"    row={r} orig_idx={orig_idx} char_len={len(t)} word_len={tok_len} text[:80]={repr(t[:80])}")
else:
    print("  No NaN rows!")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
