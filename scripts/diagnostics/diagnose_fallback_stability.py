#!/usr/bin/env python3
"""Test fallback path (pad+bool mask SDPA) stability across multiple runs.

If fallback is consistently 0% NaN, we can use it for ALL docs,
eliminating any bias from mixed TND/fallback computation.
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

print("\n=== Loading model (fp16, msl=512) ===")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512
print(f"  dtype: {next(m.parameters()).dtype}, msl: {m.max_seq_length}")

# ── Force fallback path for all runs ──
ec._HAS_NPU_FA = False
print(f"  _HAS_NPU_FA = {ec._HAS_NPU_FA} (forced fallback)")

# ── Run fallback 5 times ──
print("\n=== Fallback path: 5 repetitions (bs=512) ===")
all_embs = []
all_nan_free = True
for run in range(5):
    emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    n_nan = np.isnan(emb).any(axis=1).sum()
    pct = n_nan / emb.shape[0] * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  Run {run+1}: NaN_rows={n_nan:4d}/{emb.shape[0]} ({pct:5.1f}%)  [{status}]")
    all_embs.append(emb)
    if n_nan > 0:
        all_nan_free = False

# ── Check consistency across runs ──
if all_nan_free:
    print("\n=== Consistency check (fallback fp16 vs fallback fp16) ===")
    for i in range(1, 5):
        diff = np.abs(all_embs[0] - all_embs[i]).max()
        print(f"  Run 1 vs Run {i+1}: max_diff = {diff:.8f}")

    # ── Compare fallback fp16 vs TND fp16 (for reference) ──
    print("\n=== Cross-path comparison (fallback vs TND) ===")
    ec._HAS_NPU_FA = True
    emb_tnd = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    emb_tnd = np.array(emb_tnd, dtype=np.float32)
    n_tnd_nan = np.isnan(emb_tnd).any(axis=1).sum()
    print(f"  TND: NaN_rows={n_tnd_nan}")

    if n_tnd_nan == 0:
        diff = np.abs(all_embs[0] - emb_tnd).max()
        cos_sim = (all_embs[0] * emb_tnd).sum(axis=1) / (
            np.linalg.norm(all_embs[0], axis=1) * np.linalg.norm(emb_tnd, axis=1))
        print(f"  Max diff (fallback vs TND): {diff:.6f}")
        print(f"  Cosine similarity: min={cos_sim.min():.6f}, mean={cos_sim.mean():.6f}, max={cos_sim.max():.6f}")
        print(f"  → Difference {'is NEGLIGIBLE (<1e-4)' if diff < 1e-4 else 'is NOT negligible'}")

    # ── Test fallback fp32 ──
    print("\n=== Fallback path: fp32 (for comparison) ===")
    ec._HAS_NPU_FA = False
    m.float()
    emb_fp32 = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    emb_fp32 = np.array(emb_fp32, dtype=np.float32)
    n_fp32_nan = np.isnan(emb_fp32).any(axis=1).sum()
    print(f"  Fallback fp32: NaN_rows={n_fp32_nan}")
    if n_fp32_nan == 0:
        diff = np.abs(all_embs[0] - emb_fp32).max()
        cos_sim = (all_embs[0] * emb_fp32).sum(axis=1) / (
            np.linalg.norm(all_embs[0], axis=1) * np.linalg.norm(emb_fp32, axis=1))
        print(f"  Max diff (fp16 vs fp32, both fallback): {diff:.6f}")
        print(f"  Cosine similarity: min={cos_sim.min():.6f}, mean={cos_sim.mean():.6f}")

print("\n" + "=" * 70)
if all_nan_free:
    print("CONCLUSION: Fallback path is STABLE (0% NaN across 5 runs)")
    print("→ Use fallback for ALL docs: no bias, no data loss")
else:
    print("CONCLUSION: Fallback path also has NaN — need retry mechanism")
print("=" * 70)
