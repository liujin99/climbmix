#!/usr/bin/env python3
"""Trace NaN through the full SentenceTransformer.encode() pipeline.

Finding: transformer forward (NewModel) = 0 NaN.
NaN appears AFTER transformer — in Pooling, Dense, or normalize.

Pipeline: Tokenize → Transformer(0) → Pooling(1) → Dense(2) → normalize
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

def check_tensor(t, label):
    if not isinstance(t, torch.Tensor):
        print(f"    {label}: not a tensor ({type(t)})")
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    n_nan = int(torch.isnan(t).sum().item())
    n_elem = t.numel()
    print(f"    {label:50s}: shape={list(t.shape)} dtype={t.dtype} NaN={n_nan}/{n_elem} Inf={has_inf}")
    return has_nan

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

test_texts = ["This is a test sentence.", "Another sentence here."]
tok = m.tokenizer

# ════════════════════════════════════════════════════════════════════════
# Step-by-step pipeline trace
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Step-by-step pipeline trace")
print("=" * 70)

# Step 0: Tokenize
tok_res = tok(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attention_mask = tok_res["attention_mask"].to("npu")
print(f"\nStep 0: Tokenize")
check_tensor(input_ids.float(), "input_ids")
check_tensor(attention_mask.float(), "attention_mask")

# Step 1: Transformer (module 0)
print(f"\nStep 1: Transformer (module 0: {type(m[0]).__name__})")
mod0 = m[0]
tm = mod0.auto_model
print(f"  tm.unpad_inputs: {getattr(tm, 'unpad_inputs', 'N/A')}")
print(f"  tm.use_memory_efficient_attention: {getattr(tm, 'use_memory_efficient_attention', 'N/A')}")

with torch.no_grad():
    trans_out = tm(input_ids=input_ids, attention_mask=attention_mask)
    if hasattr(trans_out, 'last_hidden_state'):
        lhs = trans_out.last_hidden_state
        check_tensor(lhs, "transformer.last_hidden_state")
    if hasattr(trans_out, 'pooler_output'):
        check_tensor(trans_out.pooler_output, "transformer.pooler_output")

# Step 1b: Check what ST's Transformer module does
print(f"\nStep 1b: ST Transformer module forward")
with torch.no_grad():
    # The ST Transformer module calls auto_model and processes output
    st_trans_out = mod0(input_ids)
    if isinstance(st_trans_out, dict):
        for k, v in st_trans_out.items():
            if isinstance(v, torch.Tensor):
                check_tensor(v, f"mod0 output['{k}']")
    elif isinstance(st_trans_out, torch.Tensor):
        check_tensor(st_trans_out, "mod0 output")

# Step 2: Pooling (module 1)
print(f"\nStep 2: Pooling (module 1: {type(m[1]).__name__})")
pooling = m[1]
print(f"  Pooling config: {pooling}")
print(f"  Pooling attrs: {[a for a in dir(pooling) if not a.startswith('_') and not callable(getattr(pooling, a, None))][:20]}")

# Simulate what ST does: pass transformer output to pooling
with torch.no_grad():
    # ST passes features dict between modules
    features = {}
    if isinstance(st_trans_out, dict):
        features = st_trans_out
    else:
        features['token_embeddings'] = st_trans_out
    features['attention_mask'] = attention_mask
    
    print(f"  Input to pooling:")
    if 'token_embeddings' in features:
        check_tensor(features['token_embeddings'], "pooling input (token_embeddings)")
    
    pool_out = pooling(features)
    print(f"  Pooling output keys: {list(pool_out.keys()) if isinstance(pool_out, dict) else 'tensor'}")
    if isinstance(pool_out, dict):
        for k, v in pool_out.items():
            if isinstance(v, torch.Tensor):
                check_tensor(v, f"pooling output['{k}']")

# Step 3: Dense (module 2)
print(f"\nStep 3: Dense (module 2: {type(m[2]).__name__})")
dense = m[2]
print(f"  Dense config: {dense}")

with torch.no_grad():
    if isinstance(pool_out, dict):
        dense_in = pool_out
    else:
        dense_in = {'sentence_embedding': pool_out}
    
    if 'sentence_embedding' in dense_in:
        check_tensor(dense_in['sentence_embedding'], "dense input (sentence_embedding)")
    
    dense_out = dense(dense_in)
    print(f"  Dense output keys: {list(dense_out.keys()) if isinstance(dense_out, dict) else 'tensor'}")
    if isinstance(dense_out, dict):
        for k, v in dense_out.items():
            if isinstance(v, torch.Tensor):
                check_tensor(v, f"dense output['{k}']")

# Step 4: Normalize
print(f"\nStep 4: Normalize (F.normalize)")
with torch.no_grad():
    if isinstance(dense_out, dict) and 'sentence_embedding' in dense_out:
        emb = dense_out['sentence_embedding']
    elif isinstance(dense_out, torch.Tensor):
        emb = dense_out
    else:
        emb = None
        print(f"  Cannot find embedding in dense_out: {dense_out}")
    
    if emb is not None:
        check_tensor(emb, "pre-normalize embedding")
        norm_emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        check_tensor(norm_emb, "post-normalize embedding")
        final = norm_emb.cpu().float().numpy()
        n_nan = int(np.isnan(final).any(axis=1).sum())
        print(f"\n  FINAL: NaN={n_nan}/{len(final)}")

# ════════════════════════════════════════════════════════════════════════
# Full encode for comparison
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Full m.encode() for comparison")
print(f"{'='*70}")

emb_full = m.encode(test_texts, batch_size=2, show_progress_bar=False, normalize_embeddings=True)
emb_full = np.array(emb_full, dtype=np.float32)
n_nan = int(np.isnan(emb_full).any(axis=1).sum())
print(f"  m.encode() result: NaN={n_nan}/{len(emb_full)}")

# ════════════════════════════════════════════════════════════════════════
# Test with normalize_embeddings=False
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("m.encode() with normalize_embeddings=False")
print(f"{'='*70}")

emb_no_norm = m.encode(test_texts, batch_size=2, show_progress_bar=False, normalize_embeddings=False)
emb_no_norm = np.array(emb_no_norm, dtype=np.float32)
n_nan = int(np.isnan(emb_no_norm).any(axis=1).sum())
print(f"  Result: NaN={n_nan}/{len(emb_no_norm)}")
if n_nan == 0:
    print(f"  *** NaN comes from normalize_embeddings=True! ***")
    print(f"  Sample values (pre-norm): {emb_no_norm[0][:5]}")
    norms = np.linalg.norm(emb_no_norm, axis=1)
    print(f"  L2 norms: {norms}")
    print(f"  Any zero norms? {np.any(norms == 0.0)}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
