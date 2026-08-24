#!/usr/bin/env python3
"""Test npu_fusion_attention with TND layout + benchmark current approach.

Usage:
    python scripts/test_npu_fa2.py
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import time
import types
import itertools
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu

# ============================================================
# Fake xformers (production copy, no logging)
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

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
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
            max_s = max(qs_list)
            max_ks = max(ks_list)
            q_pad = torch.zeros(n, max_s, H, D, dtype=q.dtype, device=q.device)
            k_pad = torch.zeros(n, max_ks, H, D, dtype=k.dtype, device=k.device)
            v_pad = torch.zeros(n, max_ks, H, D, dtype=v.dtype, device=v.device)
            kmask = torch.zeros(n, 1, max_s, max_ks, dtype=torch.bool, device=q.device)
            q_off = k_off = 0
            for i, (qs_i, ks_i) in enumerate(zip(qs_list, ks_list)):
                q_pad[i, :qs_i] = q[0, q_off:q_off + qs_i]
                k_pad[i, :ks_i] = k[0, k_off:k_off + ks_i]
                v_pad[i, :ks_i] = v[0, k_off:k_off + ks_i]
                kmask[i, 0, :, :ks_i] = True
                q_off += qs_i
                k_off += ks_i
            out = _F.scaled_dot_product_attention(
                q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
                attn_mask=kmask, dropout_p=p
            )
            out = out.transpose(1, 2)
            chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
            return torch.cat(chunks, dim=1)

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
# Load model + capture seqlens
# ============================================================
print("Loading stella model...")
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

# Capture seqlens from a forward pass
_attn_calls = []
_orig_mea = _memory_efficient_attention
def _logging_mea(q, k, v, attn_bias=None, p=0.0, **kw):
    if isinstance(attn_bias, _BlockDiagonalMask):
        _attn_calls.append(list(attn_bias.q_seqlen))
    return _orig_mea(q, k, v, attn_bias=attn_bias, p=p, **kw)
_ops.memory_efficient_attention = _logging_mea
_fmha_mod.memory_efficient_attention = _logging_mea

with torch.no_grad():
    output = model(features)

_ops.memory_efficient_attention = _orig_mea
_fmha_mod.memory_efficient_attention = _orig_mea

seqlens = _attn_calls[0] if _attn_calls else []
T = sum(seqlens)
H, D = 16, 64
cu_qlen = list(itertools.accumulate(seqlens))
print(f"Layers: {len(_attn_calls)}, T={T}, n_seqs={len(seqlens)}")
print(f"seqlens range: {min(seqlens)}-{max(seqlens)}, avg={T/len(seqlens):.0f}")

# ============================================================
# Benchmark: npu_fusion_attention with TND layout
# ============================================================
print("\n=== npu_fusion_attention (TND) ===")
dtype = torch.float16
q_tnd = torch.randn(T, H, D, dtype=dtype, device="npu")
k_tnd = torch.randn(T, H, D, dtype=dtype, device="npu")
v_tnd = torch.randn(T, H, D, dtype=dtype, device="npu")

for layout in ["TND"]:
    try:
        out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
            q_tnd, k_tnd, v_tnd,
            head_num=H, input_layout=layout,
            actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_qlen,
            scale=1.0/(D**0.5), keep_prob=1.0,
        )
        print(f"  {layout}: OK, out shape={out.shape}")

        torch.npu.synchronize()
        for _ in range(3):
            torch_npu.npu_fusion_attention(
                q_tnd, k_tnd, v_tnd, head_num=H, input_layout=layout,
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_qlen,
                scale=1.0/(D**0.5), keep_prob=1.0,
            )
        torch.npu.synchronize()

        N = 20
        t0 = time.time()
        for _ in range(N):
            torch_npu.npu_fusion_attention(
                q_tnd, k_tnd, v_tnd, head_num=H, input_layout=layout,
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_qlen,
                scale=1.0/(D**0.5), keep_prob=1.0,
            )
        torch.npu.synchronize()
        t_fa = (time.time() - t0) / N * 1000
        print(f"  {layout}: {t_fa:.1f} ms/call × 24 layers = {t_fa*24:.1f} ms")
    except Exception as e:
        print(f"  {layout}: FAILED — {e}")

# Also try npu_prompt_flash_attention
print("\n=== npu_prompt_flash_attention ===")
try:
    help_fn = torch_npu.npu_prompt_flash_attention
    print(f"  signature: {help_fn.__doc__}")
except:
    print(f"  no doc")

for layout in ["BSND", "TND", "BNSD"]:
    try:
        if layout == "TND":
            q_in, k_in, v_in = q_tnd, k_tnd, v_tnd
        elif layout == "BSND":
            q_in = q_tnd.unsqueeze(0)
            k_in = k_tnd.unsqueeze(0)
            v_in = v_tnd.unsqueeze(0)
        else:
            q_in = q_tnd.unsqueeze(0).transpose(1, 2)
            k_in = k_tnd.unsqueeze(0).transpose(1, 2)
            v_in = v_tnd.unsqueeze(0).transpose(1, 2)

        out = torch_npu.npu_prompt_flash_attention(
            q_in, k_in, v_in,
            num_heads=H,
            actual_seq_qlen=cu_qlen,
            actual_seq_kvlen=cu_qlen,
            scale=1.0/(D**0.5),
        )
        print(f"  {layout}: OK, out shape={out.shape}")

        torch.npu.synchronize()
        for _ in range(3):
            torch_npu.npu_prompt_flash_attention(
                q_in, k_in, v_in, num_heads=H,
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_qlen,
                scale=1.0/(D**0.5),
            )
        torch.npu.synchronize()

        N = 20
        t0 = time.time()
        for _ in range(N):
            torch_npu.npu_prompt_flash_attention(
                q_in, k_in, v_in, num_heads=H,
                actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_qlen,
                scale=1.0/(D**0.5),
            )
        torch.npu.synchronize()
        t_pfa = (time.time() - t0) / N * 1000
        print(f"  {layout}: {t_pfa:.1f} ms/call × 24 layers = {t_pfa*24:.1f} ms")
    except Exception as e:
        print(f"  {layout}: FAILED — {str(e)[:200]}")

# ============================================================
# Benchmark: current approach (fake xformers pad+bool mask)
# ============================================================
print("\n=== Current: fake xformers (pad + bool mask SDPA) ===")
q_packed = torch.randn(1, T, H, D, dtype=dtype, device="npu")
k_packed = torch.randn(1, T, H, D, dtype=dtype, device="npu")
v_packed = torch.randn(1, T, H, D, dtype=dtype, device="npu")
bias = _BlockDiagonalMask(seqlens)

torch.npu.synchronize()
for _ in range(3):
    _ = _memory_efficient_attention(q_packed, k_packed, v_packed, attn_bias=bias)
torch.npu.synchronize()

N = 20
t0 = time.time()
for _ in range(N):
    _ = _memory_efficient_attention(q_packed, k_packed, v_packed, attn_bias=bias)
torch.npu.synchronize()
t_current = (time.time() - t0) / N * 1000
print(f"  fake xformers: {t_current:.1f} ms/call × 24 layers = {t_current*24:.1f} ms")

# ============================================================
# Benchmark: padded uniform (no unpad, all 512)
# ============================================================
print("\n=== Padded uniform (SDPA no mask, 512×512) ===")
B = 512
S = 512
q_uni = torch.randn(B, H, S, D, dtype=dtype, device="npu")
k_uni = torch.randn(B, H, S, D, dtype=dtype, device="npu")
v_uni = torch.randn(B, H, S, D, dtype=dtype, device="npu")

torch.npu.synchronize()
for _ in range(3):
    _ = _F.scaled_dot_product_attention(q_uni, k_uni, v_uni)
torch.npu.synchronize()

N = 20
t0 = time.time()
for _ in range(N):
    _ = _F.scaled_dot_product_attention(q_uni, k_uni, v_uni)
torch.npu.synchronize()
t_uni = (time.time() - t0) / N * 1000
print(f"  SDPA uniform: {t_uni:.1f} ms/call × 24 layers = {t_uni*24:.1f} ms")

# ============================================================
# torch.compile()
# ============================================================
print("\n=== torch.compile() ===")
print(f"dynamo backends: ", end="")
try:
    import torch._dynamo
    print(torch._dynamo.list_backends())
except:
    print("N/A")
print(f"torch_npu._compiler: {[x for x in dir(torch_npu._compiler) if not x.startswith('__')]}")

for mode in ["reduce-overhead", "default"]:
    print(f"\n--- mode={mode} ---")
    try:
        m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
        m.eval(); m.max_seq_length = 512; m.half()
        m[0].auto_model = torch.compile(m[0].auto_model, mode=mode)

        print("  warmup...")
        with torch.no_grad():
            for _ in range(5):
                _ = m(features)
        torch.npu.synchronize()

        N = 10
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                _ = m(features)
        torch.npu.synchronize()
        t = (time.time() - t0) / N * 1000
        print(f"  {t:.1f} ms/batch ({512*1000/t:.0f} docs/s)")
        del m
        torch.npu.empty_cache()
    except Exception as e:
        print(f"  FAILED: {str(e)[:300]}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY (per-call, 24 layers)")
print("=" * 60)
print(f"Current (pad+bool mask): {t_current:.1f} ms × 24 = {t_current*24:.1f} ms")
print(f"Uniform SDPA (no mask):  {t_uni:.1f} ms × 24 = {t_uni*24:.1f} ms")
try:
    print(f"npu_fusion_attention TND: {t_fa:.1f} ms × 24 = {t_fa*24:.1f} ms")
except:
    print("npu_fusion_attention TND: FAILED")
