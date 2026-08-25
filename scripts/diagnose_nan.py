#!/usr/bin/env python3
"""Diagnose NaN in embeddings — is it small probability or systemic bug?

Tests:
1. Text statistics from parquet (empty, short, normal)
2. Embed with different batch sizes, check NaN per batch
3. Test with fp16 vs fp32
4. Test with npu_fusion_attention vs fallback (pad+bool-mask)
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

# Use production fake xformers
import climbmix.core.embedding_cluster  # sets up sys.modules["xformers"]

import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")
SMOKE_DATA = "/tmp/smoke_data"

# ── Step 1: Load texts and check statistics ──
print("=" * 70)
print("Step 1: Text statistics from parquet files")
print("=" * 70)

text_col = "text"
texts = []
for fname in sorted(os.listdir(SMOKE_DATA))[:2]:
    if not fname.endswith(".parquet"):
        continue
    fpath = os.path.join(SMOKE_DATA, fname)
    table = pq.read_table(fpath, columns=[text_col])
    col_texts = table.column(text_col).to_pylist()
    texts.extend([str(t) if t is not None else "" for t in col_texts])
    del table
    print(f"  {fname}: {len(col_texts)} texts")

n_total = len(texts)
char_lens = np.array([len(t) for t in texts])
n_empty = (char_lens == 0).sum()
n_very_short = ((char_lens > 0) & (char_lens < 10)).sum()
n_short = ((char_lens >= 10) & (char_lens < 50)).sum()
n_normal = (char_lens >= 50).sum()

print(f"\n  Total texts: {n_total}")
print(f"  Empty (0 chars):    {n_empty} ({n_empty/n_total*100:.2f}%)")
print(f"  Very short (1-9):   {n_very_short} ({n_very_short/n_total*100:.2f}%)")
print(f"  Short (10-49):      {n_short} ({n_short/n_total*100:.2f}%)")
print(f"  Normal (50+):       {n_normal} ({n_normal/n_total*100:.2f}%)")
print(f"  Char length: min={char_lens.min()}, max={char_lens.max()}, mean={char_lens.mean():.0f}, median={np.median(char_lens):.0f}")

# Sample 2000 docs (same as pipeline)
rng = np.random.default_rng(42)
sample_indices = rng.choice(n_total, size=min(2000, n_total), replace=False)
sample_indices.sort()
sample_texts = [texts[i] for i in sample_indices]
sample_lens = char_lens[sample_indices]

print(f"\n  Sampled {len(sample_texts)} docs:")
print(f"  Empty: {(sample_lens == 0).sum()}, Very short: {((sample_lens > 0) & (sample_lens < 10)).sum()}")
print(f"  Short: {((sample_lens >= 10) & (sample_lens < 50)).sum()}, Normal: {(sample_lens >= 50).sum()}")

# ── Step 2: Load model ──
print("\n" + "=" * 70)
print("Step 2: Load model and check config")
print("=" * 70)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()

auto_model = model[0].auto_model
attn = auto_model.encoder.layer[0].attention
print(f"  use_memory_efficient_attention: {attn.use_memory_efficient_attention}")
print(f"  memory_efficient_attention is None: {attn.memory_efficient_attention is None}")
print(f"  unpad_inputs: {attn.config.unpad_inputs}")
print(f"  Model dtype (before half): {next(model.parameters()).dtype}")

# ── Step 3: Test embedding with different batch sizes (fp32) ──
print("\n" + "=" * 70)
print("Step 3: Embed with different batch sizes (fp32, no model.half())")
print("=" * 70)

for bs in [8, 32, 64, 128, 256, 512]:
    emb = model.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    n_nan_rows = np.isnan(emb).any(axis=1).sum()
    n_nan_total = np.isnan(emb).sum()
    n_inf = np.isinf(emb).sum()
    print(f"  batch_size={bs:4d}: shape={emb.shape}, NaN_rows={n_nan_rows}/{emb.shape[0]} ({n_nan_rows/emb.shape[0]*100:.1f}%), NaN_total={n_nan_total}, Inf={n_inf}")
    if n_nan_rows > 0 and n_nan_rows <= 20:
        nan_indices = np.where(np.isnan(emb).any(axis=1))[0]
        for idx in nan_indices:
            orig_idx = sample_indices[idx]
            t = sample_texts[idx]
            print(f"    NaN doc [{orig_idx}]: len={len(t)}, text[:50]={repr(t[:50])}")

# ── Step 4: Test with fp16 (model.half()) ──
print("\n" + "=" * 70)
print("Step 4: Embed with model.half() (fp16)")
print("=" * 70)

model.half()
print(f"  Model dtype (after half): {next(model.parameters()).dtype}")

for bs in [8, 64, 256, 512]:
    emb = model.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    n_nan_rows = np.isnan(emb).any(axis=1).sum()
    print(f"  batch_size={bs:4d} (fp16): NaN_rows={n_nan_rows}/{emb.shape[0]} ({n_nan_rows/emb.shape[0]*100:.1f}%)")

# ── Step 5: Isolate — test with only empty/short texts ──
print("\n" + "=" * 70)
print("Step 5: Test with empty and very short texts specifically")
print("=" * 70)

model_f32 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model_f32.eval()

test_cases = {
    "all_empty": [""] * 64,
    "all_1char": ["a"] * 64,
    "all_5chars": ["hello"] * 64,
    "mixed_empty_normal": [""] * 8 + ["This is a normal sentence with enough words. " * 5] * 56,
    "all_normal": ["This is a normal sentence with enough words. " * 5] * 64,
}

for name, test_texts in test_cases.items():
    emb = model_f32.encode(test_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    emb = np.array(emb, dtype=np.float32)
    n_nan = np.isnan(emb).any(axis=1).sum()
    print(f"  {name:25s}: NaN_rows={n_nan}/{len(test_texts)}")

# ── Step 6: Test with fake xformers vs fallback (no xformers) ──
print("\n" + "=" * 70)
print("Step 6: Test WITHOUT fake xformers (standard SDPA path)")
print("=" * 70)

# Remove fake xformers
mods_to_remove = [k for k in sys.modules if k.startswith("xformers")]
for k in mods_to_remove:
    del sys.modules[k]

# Load model without xformers
try:
    model_noxops = SentenceTransformer(
        "NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True,
        model_kwargs={"attn_implementation": "sdpa"},
    )
    model_noxops.eval()

    for bs in [8, 64, 256]:
        emb = model_noxops.encode(sample_texts, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        emb = np.array(emb, dtype=np.float32)
        n_nan = np.isnan(emb).any(axis=1).sum()
        print(f"  no-xformers batch_size={bs:4d}: NaN_rows={n_nan}/{emb.shape[0]}")
except Exception as e:
    print(f"  Failed: {e}")

print("\n" + "=" * 70)
print("DIAGNOSIS COMPLETE")
print("=" * 70)
