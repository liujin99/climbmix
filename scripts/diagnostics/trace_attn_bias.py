#!/usr/bin/env python3
"""Trace what attn_bias the stella model passes to fake xformers.

The main diagnostic showed sorted != unsorted (max diff 0.13) via
tokenize+model() path. This means batch composition affects results.
Need to understand what attn_bias the model passes.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster as ec  # installs fake xformers
import torch
import torch_npu
import numpy as np

# ── Monkey-patch _memory_efficient_attention to trace calls ──
import types as _types
_xformers_mod = sys_modules = __import__("sys").modules

# Find the fake xformers memory_efficient_attention
import xformers.ops.fmha as _fmha
_orig_fn = _fmha.memory_efficient_attention

_call_count = [0]
_traces = []

def _traced_fn(q, k, v, attn_bias=None, p=0.0, **kw):
    if _call_count[0] < 10:
        info = f"call #{_call_count[0]}: q.shape={q.shape}, q.dtype={q.dtype}"
        if attn_bias is None:
            info += ", attn_bias=None"
        elif isinstance(attn_bias, ec._BlockDiagonalMask):
            info += f", attn_bias=BlockDiagonalMask(q_seqlen={attn_bias.q_seqlen[:5]}..., n={len(attn_bias.q_seqlen)})"
        elif isinstance(attn_bias, ec._LowerTriangularMask):
            info += ", attn_bias=LowerTriangularMask"
        elif isinstance(attn_bias, torch.Tensor):
            info += f", attn_bias=Tensor(shape={attn_bias.shape}, dtype={attn_bias.dtype})"
        else:
            info += f", attn_bias={type(attn_bias).__name__}"
        _traces.append(info)
        _call_count[0] += 1
    return _orig_fn(q, k, v, attn_bias=attn_bias, p=p, **kw)

_fmha.memory_efficient_attention = _traced_fn

# Also patch the module where it was originally set
_xformers_mod["xformers.ops.fmha.attn_bias"]._memory_efficient_attention = _traced_fn
# And the xformers.ops module
import xformers.ops as _xops
_xops.memory_efficient_attention = _traced_fn

from sentence_transformers import SentenceTransformer

print("=== Loading model (fp16, msl=512) ===")
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

# Also check what unpad_inputs is on the model
print(f"\nmodel[0].unpad_inputs = {m[0].unpad_inputs}")
print(f"model[0].can_flatten_inputs = {m[0].can_flatten_inputs}")
print(f"model[0].config._attn_implementation = {m[0].config._attn_implementation}")

# Check the HuggingFace model's attention layers
hf_model = m[0].model
print(f"\nHF model class: {type(hf_model).__name__}")
print(f"HF model config attn_implementation: {hf_model.config._attn_implementation}")

# Check NewAttention attributes
for name, module in hf_model.named_modules():
    if "Attention" in type(module).__name__ or "attention" in name.lower():
        if hasattr(module, "use_memory_efficient_attention"):
            print(f"\nModule: {name}")
            print(f"  type: {type(module).__name__}")
            print(f"  use_memory_efficient_attention: {module.use_memory_efficient_attention}")
            print(f"  memory_efficient_attention: {module.memory_efficient_attention}")
            if hasattr(module, "unpad_inputs"):
                print(f"  unpad_inputs: {module.unpad_inputs}")
            break

# ── Test 1: Short batch (all < 100 chars) ──
print("\n--- Test 1: Short batch (all short texts) ---")
short_texts = ["hello world"] * 8
features = m.tokenize(short_texts)
print(f"  features keys: {list(features.keys())}")
if "attention_mask" in features:
    am = features["attention_mask"]
    print(f"  attention_mask shape: {am.shape}")
    print(f"  attention_mask sum per row: {am.sum(dim=1).tolist()}")
_call_count[0] = 0
_traces.clear()
device = torch.device("npu")
features_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
with torch.no_grad():
    out = m(features_dev)
for t in _traces:
    print(f"  {t}")

# ── Test 2: Mixed batch (short + long) ──
print("\n--- Test 2: Mixed batch (short + long) ---")
mixed_texts = ["hello world"] * 4 + ["This is a longer text. " * 200] * 4
features = m.tokenize(mixed_texts)
print(f"  features keys: {list(features.keys())}")
if "attention_mask" in features:
    am = features["attention_mask"]
    print(f"  attention_mask shape: {am.shape}")
    print(f"  attention_mask sum per row: {am.sum(dim=1).tolist()}")
_call_count[0] = 0
_traces.clear()
features_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in features.items()}
with torch.no_grad():
    out = m(features_dev)
for t in _traces:
    print(f"  {t}")

# ── Test 3: Compare embeddings for same text in different batches ──
print("\n--- Test 3: Same text, different batch context ---")
text = "This is a test sentence for embedding comparison."

# Alone (batch=1)
emb_alone = m.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=True)
emb_alone = np.array(emb_alone[0], dtype=np.float32)

# In a batch of short texts
emb_short_batch = m.encode([text] + ["hi"] * 7, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
emb_short = np.array(emb_short_batch[0], dtype=np.float32)

# In a batch of long texts
emb_long_batch = m.encode([text] + ["This is a very long text. " * 500] * 7, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
emb_long = np.array(emb_long_batch[0], dtype=np.float32)

diff_short = np.abs(emb_alone - emb_short).max()
diff_long = np.abs(emb_alone - emb_long).max()
diff_sl = np.abs(emb_short - emb_long).max()

print(f"  alone vs short_batch: max diff = {diff_short:.10f}")
print(f"  alone vs long_batch:  max diff = {diff_long:.10f}")
print(f"  short vs long batch:  max diff = {diff_sl:.10f}")

cos_short = np.dot(emb_alone, emb_short) / (np.linalg.norm(emb_alone) * np.linalg.norm(emb_short))
cos_long = np.dot(emb_alone, emb_long) / (np.linalg.norm(emb_alone) * np.linalg.norm(emb_long))
print(f"  cosine sim alone vs short: {cos_short:.10f}")
print(f"  cosine sim alone vs long:  {cos_long:.10f}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
