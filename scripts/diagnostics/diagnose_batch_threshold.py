#!/usr/bin/env python3
"""Find the NaN threshold: test with increasing batch sizes of real data.

2 short sentences = 0 NaN. 200 real docs = 100% NaN.
Binary search for the threshold and trace where it breaks.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings, time
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # sets up fake xformers

import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")

# Load texts
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
print(f"Text char lengths: min={min(len(t) for t in sample)}, max={max(len(t) for t in sample)}, mean={np.mean([len(t) for t in sample]):.0f}")

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    pct = n_nan / len(emb) * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  {label:50s}: NaN={n_nan:4d}/{len(emb)} ({pct:5.1f}%) [{status}]")
    return n_nan

# ════════════════════════════════════════════════════════════════════════
# Part 1: Test with increasing batch sizes
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 1: Test with increasing batch sizes")
print(f"{'='*70}")

for bs in [2, 4, 8, 16, 32, 64, 128, 200]:
    t0 = time.time()
    try:
        emb = m.encode(sample[:bs], batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        check_nan(emb, f"bs={bs} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  bs={bs}: ERROR: {e}")
        break

# ════════════════════════════════════════════════════════════════════════
# Part 2: Test batch_size param vs number of docs
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 2: 200 docs with different batch_size param")
print(f"{'='*70}")

for bs in [2, 8, 32, 128, 512]:
    t0 = time.time()
    try:
        emb = m.encode(sample, batch_size=bs, show_progress_bar=False, normalize_embeddings=True)
        check_nan(emb, f"200 docs, bs={bs} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  200 docs, bs={bs}: ERROR: {e}")
        break

# ════════════════════════════════════════════════════════════════════════
# Part 3: Trace the batch that first produces NaN
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 3: Trace batch_size=8 (likely first NaN)")
print(f"{'='*70}")

# Tokenize 8 docs
tok = m.tokenizer
tok_res = tok(sample[:8], padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attention_mask = tok_res["attention_mask"].to("npu")
print(f"  Input shape: {input_ids.shape}")
print(f"  Attention mask sums (seq lens): {attention_mask.sum(dim=1).tolist()}")

tm = m[0].auto_model

# Trace transformer
print(f"\n  Transformer forward:")
with torch.no_grad():
    trans_out = tm(input_ids=input_ids, attention_mask=attention_mask)
    lhs = trans_out.last_hidden_state
    has_nan = torch.isnan(lhs).any().item()
    n_nan = int(torch.isnan(lhs).sum().item())
    print(f"    last_hidden_state: shape={list(lhs.shape)} NaN={n_nan}/{lhs.numel()}")

# Trace ST pipeline
print(f"\n  ST Transformer module:")
with torch.no_grad():
    features = {"input_ids": input_ids, "attention_mask": attention_mask}
    out = m[0](features)
    te = out.get('token_embeddings')
    if te is not None:
        has_nan = torch.isnan(te).any().item()
        print(f"    token_embeddings: shape={list(te.shape)} NaN={int(torch.isnan(te).sum())}/{te.numel()}")

    out = m[1](out)
    se = out.get('sentence_embedding')
    if se is not None:
        has_nan = torch.isnan(se).any().item()
        print(f"    sentence_embedding (after pooling): shape={list(se.shape)} NaN={int(torch.isnan(se).sum())}/{se.numel()}")

    out = m[2](out)
    se2 = out.get('sentence_embedding')
    if se2 is not None:
        has_nan = torch.isnan(se2).any().item()
        print(f"    sentence_embedding (after dense): shape={list(se2.shape)} NaN={int(torch.isnan(se2).sum())}/{se2.numel()}")

    norm = torch.nn.functional.normalize(se2, p=2, dim=1)
    has_nan = torch.isnan(norm).any().item()
    print(f"    after normalize: NaN={int(torch.isnan(norm).sum())}/{norm.numel()}")

# ════════════════════════════════════════════════════════════════════════
# Part 4: Check tokenization lengths
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 4: Tokenization stats for 200 docs")
print(f"{'='*70}")

tok_lens = []
for i in range(0, 200, 32):
    batch = sample[i:i+32]
    toks = tok(batch, padding=False, truncation=True, max_length=512)
    tok_lens.extend([len(x) for x in toks['input_ids']])

tok_lens = np.array(tok_lens)
print(f"  Token lengths: min={tok_lens.min()}, max={tok_lens.max()}, mean={tok_lens.mean():.0f}, median={np.median(tok_lens):.0f}")
print(f"  >256: {(tok_lens > 256).sum()}/{len(tok_lens)}")
print(f"  >400: {(tok_lens > 400).sum()}/{len(tok_lens)}")
print(f"  =512: {(tok_lens == 512).sum()}/{len(tok_lens)}")

# ════════════════════════════════════════════════════════════════════════
# Part 5: Test with short texts only vs long texts only
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 5: Short texts vs long texts")
print(f"{'='*70}")

short_texts = [t for t in sample if len(tok(t, truncation=True, max_length=512)['input_ids']) <= 50]
long_texts = [t for t in sample if len(tok(t, truncation=True, max_length=512)['input_ids']) >= 400]
print(f"  Short texts (<=50 tokens): {len(short_texts)}")
print(f"  Long texts (>=400 tokens): {len(long_texts)}")

if len(short_texts) >= 8:
    emb = m.encode(short_texts[:32], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"short texts (n={min(32, len(short_texts))})")

if len(long_texts) >= 8:
    emb = m.encode(long_texts[:32], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, f"long texts (n={min(32, len(long_texts))})")

# ════════════════════════════════════════════════════════════════════════
# Part 6: Test with unpad_inputs disabled
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 6: Disable unpad_inputs")
print(f"{'='*70}")

# Try to disable unpad_inputs
if hasattr(tm, 'config'):
    old_unpad = getattr(tm.config, 'unpad_inputs', None)
    old_mea = getattr(tm.config, 'use_memory_efficient_attention', None)
    print(f"  Original: unpad_inputs={old_unpad}, use_memory_efficient_attention={old_mea}")
    
    tm.config.unpad_inputs = False
    tm.config.use_memory_efficient_attention = False
    print(f"  Changed: unpad_inputs=False, use_memory_efficient_attention=False")
    
    emb = m.encode(sample[:32], batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    check_nan(emb, "unpad=False, 32 docs")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
