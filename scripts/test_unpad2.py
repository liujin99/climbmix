#!/usr/bin/env python3
"""Patch get_extended_attention_mask to accept device kwarg, then test unpad_inputs=False.

Usage:
    python scripts/test_unpad2.py
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import time
import types
import itertools
import functools
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu

# ============================================================
# Fake xformers with npu_fusion_attention (production)
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
            s = qs_list[0]; ks = ks_list[0]
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
                q_tnd, k_tnd, v_tnd, head_num=H, input_layout="TND",
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_kvlen,
                scale=1.0 / (D ** 0.5), keep_prob=1.0 - p,
            )
            if q.dim() == 4: out = out.unsqueeze(0)
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
# Patch get_extended_attention_mask to accept device kwarg
# ============================================================
import inspect
from transformers.modeling_utils import ModuleUtilsMixin

orig_get_mask = ModuleUtilsMixin.get_extended_attention_mask
sig = inspect.signature(orig_get_mask)
print(f"Original get_extended_attention_mask params: {list(sig.parameters.keys())}")

if "device" not in sig.parameters:
    print("Patching to add 'device' kwarg...")
    @functools.wraps(orig_get_mask)
    def patched_get_mask(self, attention_mask, input_shape, device=None, dtype=None, **kwargs):
        return orig_get_mask(self, attention_mask, input_shape, dtype=dtype)
    ModuleUtilsMixin.get_extended_attention_mask = patched_get_mask
    print("Patched OK")
else:
    print("'device' already in signature, no patch needed")

# ============================================================
# Load model
# ============================================================
import functools
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

texts = [" ".join([f"word{j}" for j in range(80 + (i*13)%320)]) for i in range(512)]
features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

# ============================================================
# Test A: Baseline (unpad_inputs=True, npu_fusion_attention TND)
# ============================================================
print("\n=== Baseline (unpad=True + npu_fusion_attention) ===")
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
t_base = (time.time() - t0) / N * 1000
calls_per_fwd = len(_attn_calls) // N
print(f"  {t_base:.1f} ms/batch ({512*1000/t_base:.0f} docs/s)")
print(f"  attention calls/fwd: {calls_per_fwd}, types: {set(_attn_calls)}")

# ============================================================
# Test B: unpad_inputs=False
# ============================================================
print("\n=== unpad_inputs=False ===")
model[0].auto_model.config.unpad_inputs = False
print(f"  config.unpad_inputs = {model[0].auto_model.config.unpad_inputs}")

_attn_calls.clear()
try:
    with torch.no_grad():
        output_test = model(features)
    print(f"  Forward OK!")
    print(f"  output shape: {output_test['sentence_embedding'].shape}")
    print(f"  attention calls: {len(_attn_calls)}, types: {set(_attn_calls)}")

    # Check output quality
    emb_base = output["sentence_embedding"]
    emb_test = output_test["sentence_embedding"]
    diff = (emb_base.float() - emb_test.float()).norm().item() / emb_base.float().norm().item()
    print(f"  output diff vs baseline: {diff:.6f} (relative L2)")

    # Benchmark
    with torch.no_grad():
        for _ in range(3):
            _ = model(features)
    torch.npu.synchronize()

    _attn_calls.clear()
    torch.npu.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            output2 = model(features)
    torch.npu.synchronize()
    t_nounpad = (time.time() - t0) / N * 1000
    calls_per_fwd2 = len(_attn_calls) // N
    print(f"  {t_nounpad:.1f} ms/batch ({512*1000/t_nounpad:.0f} docs/s)")
    print(f"  attention calls/fwd: {calls_per_fwd2}, types: {set(_attn_calls)}")
    print(f"  Speedup: {t_base/t_nounpad:.2f}x")
    print(f"  8-NPU projected: {512*1000/t_nounpad*8:.0f} docs/s")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline (unpad=True + npu_fa): {t_base:.1f} ms ({512*1000/t_base:.0f} docs/s)")
try:
    print(f"unpad_inputs=False:             {t_nounpad:.1f} ms ({512*1000/t_nounpad:.0f} docs/s)")
    print(f"Speedup: {t_base/t_nounpad:.2f}x")
except:
    print("unpad_inputs=False: FAILED")
