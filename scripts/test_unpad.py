#!/usr/bin/env python3
"""Test disabling unpad_inputs in stella model + read modeling.py unpad logic.

Usage:
    python scripts/test_unpad.py
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import time
import types
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu
import itertools

# ============================================================
# Fake xformers with npu_fusion_attention (production code)
# ============================================================
class _BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen=None, device=None):
        if isinstance(q_seqlen, int):
            q_seqlen = [q_seqlen]
        self.q_seqlen = list(q_seqlen)
        self.kv_seqlen = list(kv_seqlen) if kv_seqlen is not None else self.q_seqlen
        self.device = device

    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None, device=None, **kw):
        if (isinstance(q_seqlen, tuple) and len(q_seqlen) == 2
                and isinstance(q_seqlen[0], (list, tuple))):
            q_seqlen, kv_seqlen = q_seqlen
        return cls(q_seqlen, kv_seqlen, device)

class _LowerTriangularMask:
    def __init__(self, *a, **kw): pass

_attn_calls = []

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
    _attn_calls.append(type(attn_bias).__name__)
    need_transpose = q.dim() == 4 and q.shape[1] > q.shape[2]
    def _to_sdpa(t): return t.transpose(1, 2) if need_transpose else t
    def _from_sdpa(t): return t.transpose(1, 2).contiguous() if need_transpose else t

    if isinstance(attn_bias, _BlockDiagonalMask):
        qs_list = attn_bias.q_seqlen
        ks_list = attn_bias.kv_seqlen
        n = len(qs_list)
        _, _, H, D = q.shape

        if len(set(qs_list)) == 1 and len(set(ks_list)) == 1:
            s = qs_list[0]
            ks = ks_list[0]
            q_b = q.view(n, s, H, D).transpose(1, 2)
            k_b = k.view(n, ks, H, D).transpose(1, 2)
            v_b = v.view(n, ks, H, D).transpose(1, 2)
            out = _F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=p)
            return out.transpose(1, 2).reshape(1, -1, H, D).contiguous()
        else:
            cu_qlen = list(itertools.accumulate(qs_list))
            cu_kvlen = list(itertools.accumulate(ks_list))
            q_tnd = q.squeeze(0).contiguous() if q.dim() == 4 else q.contiguous()
            k_tnd = k.squeeze(0).contiguous() if k.dim() == 4 else k.contiguous()
            v_tnd = v.squeeze(0).contiguous() if v.dim() == 4 else v.contiguous()
            out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
                q_tnd, k_tnd, v_tnd,
                head_num=H, input_layout="TND",
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_kvlen,
                scale=1.0 / (D ** 0.5), keep_prob=1.0 - p,
            )
            if q.dim() == 4:
                out = out.unsqueeze(0)
            return out

    q_s, k_s, v_s = _to_sdpa(q), _to_sdpa(k), _to_sdpa(v)
    if isinstance(attn_bias, _LowerTriangularMask):
        out = _F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, dropout_p=p)
    elif attn_bias is not None:
        out = _F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_bias, dropout_p=p)
    else:
        out = _F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=p)
    return _from_sdpa(out)

_attn_bias_mod = types.ModuleType("xformers.ops.fmha.attn_bias")
_attn_bias_mod.BlockDiagonalMask = _BlockDiagonalMask
_attn_bias_mod.LowerTriangularMask = _LowerTriangularMask
_fmha_mod = types.ModuleType("xformers.ops.fmha")
_fmha_mod.attn_bias = _attn_bias_mod
_fmha_mod.memory_efficient_attention = _memory_efficient_attention
_ops = types.ModuleType("xformers.ops")
_ops.memory_efficient_attention = _memory_efficient_attention
_ops.fmha = _fmha_mod
_xfm = types.ModuleType("xformers")
_xfm.ops = _ops
_xfm.__version__ = "0.0.0"
sys.modules["xformers"] = _xfm
sys.modules["xformers.ops"] = _ops
sys.modules["xformers.ops.fmha"] = _fmha_mod
sys.modules["xformers.ops.fmha.attn_bias"] = _attn_bias_mod

# ============================================================
# Read stella modeling.py — find unpad_inputs logic
# ============================================================
print("=" * 60)
print("1. stella modeling.py — unpad_inputs logic")
print("=" * 60)
import pathlib
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    with open(p) as f:
        lines = f.readlines()
    print(f"File: {p} ({len(lines)} lines)")

    keywords = ["unpad_inputs", "unpad", "BlockDiagonalMask", "from_seqlens",
                "attention_mask_bool", "nonzero", "def forward"]
    seen = set()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                start = max(0, i - 3)
                end = min(len(lines), i + 10)
                key = (start, end)
                if key in seen:
                    break
                seen.add(key)
                print(f"\n--- line {i+1} ({kw}) ---")
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j].rstrip()}")
                break
    break

# ============================================================
# Load model + check config
# ============================================================
print("\n" + "=" * 60)
print("2. Model config check")
print("=" * 60)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

auto_model = model[0].auto_model
print(f"Model type: {type(auto_model).__name__}")
print(f"Config type: {type(auto_model.config).__name__}")

# Check unpad_inputs in config
config = auto_model.config
print(f"config.unpad_inputs: {getattr(config, 'unpad_inputs', 'NOT FOUND')}")
print(f"config.attn_implementation: {getattr(config, 'attn_implementation', 'NOT FOUND')}")

# Check if unpad_inputs is in forward signature
import inspect
sig = inspect.signature(auto_model.forward)
print(f"\nforward params: {list(sig.parameters.keys())}")

# Check encoder layer forward
encoder = auto_model.encoder if hasattr(auto_model, 'encoder') else None
if encoder:
    sig_enc = inspect.signature(encoder.forward)
    print(f"encoder.forward params: {list(sig_enc.parameters.keys())}")

    # Check first layer
    if hasattr(encoder, 'layer') and len(encoder.layer) > 0:
        layer = encoder.layer[0]
        sig_layer = inspect.signature(layer.forward)
        print(f"layer.forward params: {list(sig_layer.parameters.keys())}")

        # Check attention module
        attn = layer.attention if hasattr(layer, 'attention') else None
        if attn is None:
            attn = layer.attention.self if hasattr(layer, 'attention') else None
        if attn:
            sig_attn = inspect.signature(attn.forward) if hasattr(attn, 'forward') else None
            if sig_attn:
                print(f"attention.forward params: {list(sig_attn.parameters.keys())}")

# ============================================================
# Test 1: Baseline (current, with unpad)
# ============================================================
print("\n" + "=" * 60)
print("3. Baseline (with unpad_inputs)")
print("=" * 60)
texts = [" ".join([f"word{j}" for j in range(80 + (i*13)%320)]) for i in range(512)]
features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

_attn_calls.clear()
with torch.no_grad():
    for _ in range(3):
        _ = model(features)
torch.npu.synchronize()

_attn_calls.clear()
N = 10
torch.npu.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        output = model(features)
torch.npu.synchronize()
t_baseline = (time.time() - t0) / N * 1000
print(f"Baseline: {t_baseline:.1f} ms/batch ({512*1000/t_baseline:.0f} docs/s)")
print(f"Attention calls: {len(_attn_calls)}, types: {set(_attn_calls)}")

# ============================================================
# Test 2: Try unpad_inputs=False
# ============================================================
print("\n" + "=" * 60)
print("4. Try unpad_inputs=False")
print("=" * 60)

# Method A: Set config
try:
    config.unpad_inputs = False
    print(f"Set config.unpad_inputs = {config.unpad_inputs}")
except Exception as e:
    print(f"Cannot set config.unpad_inputs: {e}")

# Method B: Pass to forward
_attn_calls.clear()
try:
    with torch.no_grad():
        out_test = model(features)
    print(f"Forward with config.unpad_inputs=False: OK")
    print(f"Attention calls: {len(_attn_calls)}, types: {set(_attn_calls)}")
except Exception as e:
    print(f"Forward failed: {str(e)[:200]}")

# Benchmark if it works
if _attn_calls and "BlockDiagonalMask" not in set(_attn_calls):
    print(f"\nUnpadding disabled! Attention now uses: {set(_attn_calls)}")
    torch.npu.synchronize()
    for _ in range(3):
        with torch.no_grad():
            _ = model(features)
    torch.npu.synchronize()

    _attn_calls.clear()
    N = 10
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            output = model(features)
    torch.npu.synchronize()
    t_nounpad = (time.time() - t0) / N * 1000
    print(f"unpad=False: {t_nounpad:.1f} ms/batch ({512*1000/t_nounpad:.0f} docs/s)")
    print(f"Speedup: {t_baseline/t_nounpad:.2f}x")
    print(f"Attention calls: {len(_attn_calls)}, types: {set(_attn_calls)}")
else:
    print(f"\nunpad_inputs=False didn't change attention path: {set(_attn_calls)}")

    # Method C: Monkey-patch the forward to pass unpad_inputs=False
    print("\nTrying monkey-patch...")
    # Reload model with fresh config
    config2 = type(config).from_pretrained("NovaSearch/stella_en_400M_v5")
    if hasattr(config2, 'unpad_inputs'):
        config2.unpad_inputs = False
        print(f"Fresh config.unpad_inputs = {config2.unpad_inputs}")

    # Method D: Check if it's in the encoder
    if encoder:
        for attr in ['unpad_inputs', 'unpad']:
            if hasattr(encoder, attr):
                print(f"encoder.{attr} = {getattr(encoder, attr)}")
        for attr in ['unpad_inputs', 'unpad']:
            if hasattr(encoder.layer[0], attr):
                print(f"layer[0].{attr} = {getattr(encoder.layer[0], attr)}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline (npu_fusion_attention TND): {t_baseline:.1f} ms ({512*1000/t_baseline:.0f} docs/s)")
try:
    print(f"unpad_inputs=False:               {t_nounpad:.1f} ms ({512*1000/t_nounpad:.0f} docs/s)")
except:
    print("unpad_inputs=False: NOT WORKING")
