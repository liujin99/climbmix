#!/usr/bin/env python3
"""Narrow down: is NaN caused by padding, content, or sequence length?

Key findings so far:
- 2 short sentences (8 tokens, no padding) = 0 NaN
- 2+ real docs (512 tokens, with padding) = 100% NaN
- Transformer output itself has NaN (not post-processing)
- unpad_inputs=True, use_memory_efficient_attention=True

Tests:
1. 2 short sentences, forced padding to 512 → does padding cause NaN?
2. 2 real docs, no padding (same length) → does content cause NaN?
3. Check stella model's custom modeling.py code
4. Test with unpad_inputs path explicitly
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

def check_tensor(t, label):
    if not isinstance(t, torch.Tensor):
        print(f"    {label}: not a tensor ({type(t)})")
        return False
    n_nan = int(torch.isnan(t).sum().item())
    n_elem = t.numel()
    pct = n_nan / n_elem * 100 if n_elem > 0 else 0
    has_inf = torch.isinf(t).any().item()
    print(f"    {label:55s}: NaN={n_nan}/{n_elem} ({pct:.1f}%) Inf={has_inf} shape={list(t.shape)} dtype={t.dtype}")
    return n_nan > 0

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

tok = m.tokenizer
tm = m[0].auto_model

# ════════════════════════════════════════════════════════════════════════
# Test 1: 2 short sentences, NO padding (baseline — known 0 NaN)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 1: 2 short sentences, no padding (seq_len=8)")
print(f"{'='*70}")

texts_short = ["This is a test sentence.", "Another sentence here."]
tok_res = tok(texts_short, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attn_mask = tok_res["attention_mask"].to("npu")
print(f"  Input shape: {input_ids.shape}, attn_mask sums: {attn_mask.sum(dim=1).tolist()}")

with torch.no_grad():
    out = tm(input_ids=input_ids, attention_mask=attn_mask)
    check_tensor(out.last_hidden_state, "last_hidden_state")

# ════════════════════════════════════════════════════════════════════════
# Test 2: 2 short sentences, FORCED padding to 512
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 2: 2 short sentences, forced padding to 512")
print(f"{'='*70}")

tok_res = tok(texts_short, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attn_mask = tok_res["attention_mask"].to("npu")
print(f"  Input shape: {input_ids.shape}, attn_mask sums: {attn_mask.sum(dim=1).tolist()}")

with torch.no_grad():
    out = tm(input_ids=input_ids, attention_mask=attn_mask)
    check_tensor(out.last_hidden_state, "last_hidden_state")

# ════════════════════════════════════════════════════════════════════════
# Test 3: 2 real docs of SIMILAR length (no padding needed)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 3: 2 real docs of similar length (no padding)")
print(f"{'='*70}")

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")
pf = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])[:1]
texts = []
for fname in pf:
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["text"])
    texts.extend([str(t) if t is not None else "" for t in table.column("text").to_pylist()[:300]])

# Find 2 docs with same token length (~100 tokens)
tok_lens = []
for t in texts[:50]:
    tl = len(tok(t, truncation=True, max_length=512)['input_ids'])
    tok_lens.append((tl, t))

# Find pair with same length
same_len_pair = None
for i in range(len(tok_lens)):
    for j in range(i+1, len(tok_lens)):
        if tok_lens[i][0] == tok_lens[j][0] and tok_lens[i][0] > 20:
            same_len_pair = (tok_lens[i][1], tok_lens[j][1], tok_lens[i][0])
            break
    if same_len_pair:
        break

if same_len_pair:
    t1, t2, tl = same_len_pair
    print(f"  Found pair with {tl} tokens each")
    tok_res = tok([t1, t2], padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tok_res["input_ids"].to("npu")
    attn_mask = tok_res["attention_mask"].to("npu")
    print(f"  Input shape: {input_ids.shape}, attn_mask sums: {attn_mask.sum(dim=1).tolist()}")

    with torch.no_grad():
        out = tm(input_ids=input_ids, attention_mask=attn_mask)
        check_tensor(out.last_hidden_state, "last_hidden_state")
else:
    print("  No pair found with same token length")

# ════════════════════════════════════════════════════════════════════════
# Test 4: 1 real doc, no padding (batch_size=1)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 4: 1 real doc, batch_size=1 (no padding)")
print(f"{'='*70}")

# Find a doc with ~100 tokens
for tl, t in sorted(tok_lens, key=lambda x: abs(x[0] - 100)):
    if tl > 20:
        print(f"  Using doc with {tl} tokens (first 80 chars: {repr(t[:80])})")
        tok_res = tok([t], padding=False, truncation=True, max_length=512, return_tensors="pt")
        input_ids = tok_res["input_ids"].to("npu")
        attn_mask = tok_res["attention_mask"].to("npu")
        print(f"  Input shape: {input_ids.shape}, attn_mask sums: {attn_mask.sum(dim=1).tolist()}")

        with torch.no_grad():
            out = tm(input_ids=input_ids, attention_mask=attn_mask)
            check_tensor(out.last_hidden_state, "last_hidden_state")
        break

# ════════════════════════════════════════════════════════════════════════
# Test 5: 1 short sentence, forced padding to 512 (batch_size=1)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 5: 1 short sentence, forced padding to 512 (bs=1)")
print(f"{'='*70}")

tok_res = tok(["This is a test sentence."], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attn_mask = tok_res["attention_mask"].to("npu")
print(f"  Input shape: {input_ids.shape}, attn_mask sums: {attn_mask.sum(dim=1).tolist()}")

with torch.no_grad():
    out = tm(input_ids=input_ids, attention_mask=attn_mask)
    check_tensor(out.last_hidden_state, "last_hidden_state")

# ════════════════════════════════════════════════════════════════════════
# Test 6: Look at the stella model's custom code
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 6: Stella model's custom code")
print(f"{'='*70}")

# Find the modeling.py file
import glob
modeling_paths = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5/*/modeling.py"))
if modeling_paths:
    modeling_path = modeling_paths[0]
    print(f"  Found: {modeling_path}")

    # Read and look for key sections
    with open(modeling_path) as f:
        code = f.read()

    # Find the attention forward method
    lines = code.split('\n')
    in_attention = False
    for i, line in enumerate(lines):
        if 'class NewAttention' in line:
            in_attention = True
        if in_attention:
            if 'def forward' in line or 'memory_efficient_attention' in line or 'unpad' in line or 'attn_bias' in line:
                # Print surrounding context
                start = max(0, i - 2)
                end = min(len(lines), i + 15)
                for j in range(start, end):
                    print(f"  {j+1:4d}: {lines[j]}")
                print("  ...")
        if in_attention and 'class ' in line and 'NewAttention' not in line:
            in_attention = False
else:
    print("  modeling.py not found in cache")
    # Try to find it
    cache_dirs = [
        os.path.expanduser("~/.cache/huggingface"),
        "/home/ma-user/.cache/huggingface",
    ]
    for d in cache_dirs:
        if os.path.exists(d):
            for root, dirs, files in os.walk(d):
                if 'modeling.py' in files and 'stella' in root.lower():
                    print(f"  Found: {os.path.join(root, 'modeling.py')}")

# ════════════════════════════════════════════════════════════════════════
# Test 7: Monkey-patch xformers to trace calls during real data forward
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Test 7: Trace xformers calls with real data")
print(f"{'='*70}")

import xformers.ops as _xf_ops
_orig_mea = _xf_ops.memory_efficient_attention
_call_count = [0]

def _traced_mea(q, k, v, attn_bias=None, p=0.0, **kw):
    _call_count[0] += 1
    if _call_count[0] <= 5:
        print(f"\n  [CALL #{_call_count[0]}]")
        print(f"    q: shape={list(q.shape)} dtype={q.dtype}")
        print(f"    k: shape={list(k.shape)} dtype={k.dtype}")
        print(f"    v: shape={list(v.shape)} dtype={v.dtype}")
        print(f"    attn_bias: {type(attn_bias).__name__}")
        if hasattr(attn_bias, 'q_seqlen'):
            print(f"    q_seqlen: {attn_bias.q_seqlen[:10]}{'...' if len(attn_bias.q_seqlen)>10 else ''}")
            print(f"    kv_seqlen: {attn_bias.kv_seqlen[:10]}{'...' if len(attn_bias.kv_seqlen)>10 else ''}")
        # Check input for NaN
        q_nan = int(torch.isnan(q).sum().item())
        k_nan = int(torch.isnan(k).sum().item())
        v_nan = int(torch.isnan(v).sum().item())
        print(f"    input NaN: q={q_nan} k={k_nan} v={v_nan}")

    result = _orig_mea(q, k, v, attn_bias=attn_bias, p=p, **kw)

    if _call_count[0] <= 5:
        r_nan = int(torch.isnan(result).sum().item())
        r_elem = result.numel()
        print(f"    output: NaN={r_nan}/{r_elem} ({r_nan/r_elem*100:.1f}%)")

    return result

_xf_ops.memory_efficient_attention = _traced_mea
# Also patch the module-level reference
import xformers.ops.fmha as _xf_fmha
_xf_fmha.memory_efficient_attention = _traced_mea
# Also patch the sys.modules version
sys.modules["xformers.ops"].memory_efficient_attention = _traced_mea
sys.modules["xformers.ops.fmha"].memory_efficient_attention = _traced_mea

# Run with 2 real docs (padded to 512)
print("  Running with 2 real docs (padded to 512)...")
rng = np.random.default_rng(42)
idx = rng.choice(len(texts), size=2, replace=False)
real_texts = [texts[i] for i in idx]
tok_res = tok(real_texts, padding="max_length", max_length=512, truncation=True, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attn_mask = tok_res["attention_mask"].to("npu")
print(f"  Input shape: {input_ids.shape}")

_call_count[0] = 0
with torch.no_grad():
    out = tm(input_ids=input_ids, attention_mask=attn_mask)
    check_tensor(out.last_hidden_state, "last_hidden_state (real data)")

print(f"\n  Total xformers calls: {_call_count[0]}")

# Restore
_xf_ops.memory_efficient_attention = _orig_mea

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
