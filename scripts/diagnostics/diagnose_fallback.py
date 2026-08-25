#!/usr/bin/env python3
"""Test fallback path (pad+bool mask SDPA) vs TND path (npu_fusion_attention).

Single model load, no reloads. Goal: determine if fallback path
produces 0% NaN, so we can use it as a retry for TND NaN cases.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster as ec
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

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan_rows = np.isnan(emb).any(axis=1).sum()
    pct = n_nan_rows / emb.shape[0] * 100
    status = "PASS" if n_nan_rows == 0 else "FAIL"
    print(f"  {label:50s}: NaN_rows={n_nan_rows:4d}/{emb.shape[0]} ({pct:5.1f}%)  [{status}]")
    return n_nan_rows, np.where(np.isnan(emb).any(axis=1))[0]

# ── Load model ONCE ──
print("\n=== Loading model (fp16, msl=512) ===")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512
print(f"  dtype: {next(m.parameters()).dtype}, msl: {m.max_seq_length}")

# ── Test 1: TND path (current production) ──
print("\n--- Test 1: TND path (npu_fusion_attention) ---")
ec._HAS_NPU_FA = True
emb_tnd = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
n_tnd_nan, tnd_nan_rows = check_nan(emb_tnd, "TND bs=512")

# ── Test 2: Fallback path (pad + bool mask SDPA) ──
print("\n--- Test 2: Fallback path (pad + bool mask SDPA) ---")
ec._HAS_NPU_FA = False
emb_fb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
n_fb_nan, fb_nan_rows = check_nan(emb_fb, "Fallback bs=512")

# ── Test 3: Fallback with smaller batch sizes ──
if n_fb_nan > 0:
    print("\n--- Test 3: Fallback with different batch sizes ---")
    for bs in [256, 128, 64, 32, 8]:
        emb = m.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        check_nan(emb, f"Fallback bs={bs}")

# ── Test 4: If TND has NaN but fallback doesn't, verify embeddings match for non-NaN rows ──
if n_tnd_nan > 0 and n_fb_nan == 0:
    print("\n--- Test 4: Verify TND non-NaN rows match fallback ---")
    emb_t = np.array(emb_tnd, dtype=np.float32)
    emb_f = np.array(emb_fb, dtype=np.float32)
    valid = ~np.isnan(emb_t).any(axis=1)
    if valid.sum() > 0:
        diff = np.abs(emb_t[valid] - emb_f[valid]).max()
        print(f"  Max diff (TND vs fallback, non-NaN rows): {diff:.6f}")
        print(f"  Valid rows: {valid.sum()}/{len(emb_t)}")

    # ── Test 5: Retry TND NaN rows with fallback ──
    print("\n--- Test 5: Retry TND NaN rows with fallback ---")
    nan_texts = [sample_texts[i] for i in tnd_nan_rows]
    print(f"  Retrying {len(nan_texts)} NaN docs with fallback...")
    emb_retry = m.encode(nan_texts, batch_size=min(64, len(nan_texts)), show_progress_bar=False, normalize_embeddings=True)
    emb_retry = np.array(emb_retry, dtype=np.float32)
    n_retry_nan = np.isnan(emb_retry).any(axis=1).sum()
    print(f"  Retry result: {n_retry_nan}/{len(nan_texts)} still NaN")
    if n_retry_nan == 0:
        print("  SUCCESS: All NaN docs fixed by fallback path!")
    else:
        print(f"  PARTIAL: {n_retry_nan} docs still NaN, need individual retry")
        still_nan = np.where(np.isnan(emb_retry).any(axis=1))[0]
        for r in still_nan[:5]:
            t = nan_texts[r]
            print(f"    Still NaN: len={len(t)}, text[:80]={repr(t[:80])}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
