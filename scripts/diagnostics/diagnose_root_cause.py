#!/usr/bin/env python3
"""Root-cause diagnostic: TND vs fallback, stability, data-dependency, bias.

Tests:
1. Model inspection (attention class, unpad_inputs, memory_efficient_attention)
2. TND path (new convention: pse=None, atten_mask=None, tuples) — 5 runs
3. Fallback path (pad+bool mask SDPA) — 5 runs
4. Data-dependency: if NaN occurs, which texts?
5. Bias assessment: TND vs fallback embedding comparison
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings, json
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
    nan_rows = np.where(np.isnan(emb).any(axis=1))[0]
    n_nan = len(nan_rows)
    pct = n_nan / emb.shape[0] * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  {label:55s}: NaN={n_nan:4d}/{emb.shape[0]} ({pct:5.1f}%)  [{status}]")
    return n_nan, nan_rows

def cosine_sim(a, b):
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

# ── 1. Model inspection ──
print("\n" + "=" * 70)
print("1. MODEL INSPECTION")
print("=" * 70)
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

for name, mod in m.named_modules():
    if hasattr(mod, 'unpad_inputs') or hasattr(mod, 'use_memory_efficient_attention'):
        print(f"  {name}: {type(mod).__name__}")
        if hasattr(mod, 'unpad_inputs'):
            print(f"    unpad_inputs = {mod.unpad_inputs}")
        if hasattr(mod, 'use_memory_efficient_attention'):
            print(f"    use_memory_efficient_attention = {mod.use_memory_efficient_attention}")
        if hasattr(mod, 'memory_efficient_attention'):
            fn = mod.memory_efficient_attention
            print(f"    memory_efficient_attention = {fn}")
            print(f"    fn module = {getattr(fn, '__module__', 'N/A')}")
            print(f"    fn qualname = {getattr(fn, '__qualname__', 'N/A')}")
        for attr in ['num_attention_heads', 'attention_head_size', 'hidden_size',
                      'all_head_size', 'attention_dropout']:
            if hasattr(mod, attr):
                print(f"    {attr} = {getattr(mod, attr)}")

print(f"\n  Model dtype: {next(m.parameters()).dtype}")
print(f"  Model device: {next(m.parameters()).device}")
print(f"  max_seq_length: {m.max_seq_length}")
print(f"  _HAS_NPU_FA: {ec._HAS_NPU_FA}")

# ── 2. TND path: 5 runs ──
print("\n" + "=" * 70)
print("2. TND PATH (npu_fusion_attention, new convention: pse=None, tuples)")
print("=" * 70)
ec._HAS_NPU_FA = True
tnd_embs = []
tnd_nan_sets = []
for run in range(5):
    emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    n_nan, nan_rows = check_nan(emb, f"TND run {run+1}")
    tnd_embs.append(np.array(emb, dtype=np.float32))
    tnd_nan_sets.append(set(nan_rows.tolist()))

# ── 3. Fallback path: 5 runs ──
print("\n" + "=" * 70)
print("3. FALLBACK PATH (pad + bool mask SDPA)")
print("=" * 70)
ec._HAS_NPU_FA = False
fb_embs = []
fb_nan_sets = []
for run in range(5):
    emb = m.encode(sample_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    n_nan, nan_rows = check_nan(emb, f"Fallback run {run+1}")
    fb_embs.append(np.array(emb, dtype=np.float32))
    fb_nan_sets.append(set(nan_rows.tolist()))

# ── 4. Determinism & data-dependency analysis ──
print("\n" + "=" * 70)
print("4. DETERMINISM & DATA-DEPENDENCY ANALYSIS")
print("=" * 70)

# TND determinism
print("\n  TND: NaN row sets across runs:")
for i in range(5):
    for j in range(i+1, 5):
        if tnd_nan_sets[i] and tnd_nan_sets[j]:
            overlap = len(tnd_nan_sets[i] & tnd_nan_sets[j])
            print(f"    Run {i+1} vs Run {j+1}: "
                  f"|A|={len(tnd_nan_sets[i])}, |B|={len(tnd_nan_sets[j])}, overlap={overlap}")
        elif tnd_nan_sets[i] or tnd_nan_sets[j]:
            print(f"    Run {i+1} vs Run {j+1}: one has NaN, other doesn't")
        else:
            print(f"    Run {i+1} vs Run {j+1}: both 0 NaN")

# Fallback determinism
print("\n  Fallback: NaN row sets across runs:")
for i in range(5):
    for j in range(i+1, 5):
        if fb_nan_sets[i] and fb_nan_sets[j]:
            overlap = len(fb_nan_sets[i] & fb_nan_sets[j])
            print(f"    Run {i+1} vs Run {j+1}: "
                  f"|A|={len(fb_nan_sets[i])}, |B|={len(fb_nan_sets[j])}, overlap={overlap}")
        elif fb_nan_sets[i] or fb_nan_sets[j]:
            print(f"    Run {i+1} vs Run {j+1}: one has NaN, other doesn't")
        else:
            print(f"    Run {i+1} vs Run {j+1}: both 0 NaN")

# Data dependency
total_nan = set()
for s in tnd_nan_sets + fb_nan_sets:
    total_nan |= s
if total_nan:
    print(f"\n  Data dependency: {len(total_nan)} unique docs produced NaN across all runs")
    for doc_idx in sorted(total_nan)[:10]:
        t = sample_texts[doc_idx]
        in_tnd = [i+1 for i in range(5) if doc_idx in tnd_nan_sets[i]]
        in_fb = [i+1 for i in range(5) if doc_idx in fb_nan_sets[i]]
        print(f"    doc {doc_idx}: len={len(t)}, TND_nan_runs={in_tnd}, FB_nan_runs={in_fb}")
        print(f"      text[:100]={repr(t[:100])}")
else:
    print("\n  No NaN produced in any run — cannot assess data dependency")

# ── 5. Bias assessment: TND vs fallback ──
print("\n" + "=" * 70)
print("5. BIAS ASSESSMENT: TND vs FALLBACK (embedding comparison)")
print("=" * 70)

# Find a clean TND run and clean fallback run
tnd_clean = None
fb_clean = None
for i in range(5):
    if len(tnd_nan_sets[i]) == 0 and tnd_clean is None:
        tnd_clean = i
    if len(fb_nan_sets[i]) == 0 and fb_clean is None:
        fb_clean = i

if tnd_clean is not None and fb_clean is not None:
    e_tnd = tnd_embs[tnd_clean]
    e_fb = fb_embs[fb_clean]
    diff = np.abs(e_tnd - e_fb)
    cos = cosine_sim(e_tnd, e_fb)
    print(f"  TND run {tnd_clean+1} vs Fallback run {fb_clean+1}:")
    print(f"    Max abs diff:  {diff.max():.8f}")
    print(f"    Mean abs diff: {diff.mean():.8f}")
    print(f"    Cosine sim:    min={cos.min():.8f}, mean={cos.mean():.8f}, max={cos.max():.8f}")
    print(f"    → Bias {'is NEGLIGIBLE (<1e-4)' if diff.max() < 1e-4 else 'MAY matter (>1e-4)'}")
elif tnd_clean is None and fb_clean is not None:
    print("  TND had NaN in all runs — cannot compare")
    print("  → Fallback is the only stable option")
else:
    print("  No clean runs available for comparison")

# Fallback internal consistency
if fb_clean is not None:
    print("\n  Fallback internal consistency (run-to-run):")
    for i in range(5):
        if i == fb_clean:
            continue
        if len(fb_nan_sets[i]) == 0:
            diff = np.abs(fb_embs[fb_clean] - fb_embs[i])
            print(f"    Run {fb_clean+1} vs Run {i+1}: max_diff = {diff.max():.10f}")
        else:
            print(f"    Run {fb_clean+1} vs Run {i+1}: Run {i+1} has NaN, skipping")

# TND internal consistency
if tnd_clean is not None:
    print("\n  TND internal consistency (run-to-run):")
    for i in range(5):
        if i == tnd_clean:
            continue
        if len(tnd_nan_sets[i]) == 0:
            diff = np.abs(tnd_embs[tnd_clean] - tnd_embs[i])
            print(f"    Run {tnd_clean+1} vs Run {i+1}: max_diff = {diff.max():.10f}")
        else:
            print(f"    Run {tnd_clean+1} vs Run {i+1}: Run {i+1} has {len(tnd_nan_sets[i])} NaN, skipping")

# ── 6. fp32 fallback (if all fp16 paths had NaN) ──
if all(len(s) > 0 for s in tnd_nan_sets) and all(len(s) > 0 for s in fb_nan_sets):
    print("\n" + "=" * 70)
    print("6. FP32 FALLBACK (last resort)")
    print("=" * 70)
    m.float()
    ec._HAS_NPU_FA = False
    emb_fp32 = m.encode(sample_texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
    n_nan, _ = check_nan(emb_fp32, "Fallback fp32 bs=256")
    if n_nan == 0:
        print("  → fp32 fallback is STABLE")

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
