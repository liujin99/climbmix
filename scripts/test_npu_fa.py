#!/usr/bin/env python3
"""Test npu_fusion_attention with variable-length packed sequences + read modeling.py.

Usage:
    python scripts/test_npu_fa.py
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

# ============================================================
# Fake xformers (production copy)
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

_attn_log = {"calls": []}

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
    _attn_log["calls"].append({
        "bias": type(attn_bias).__name__,
        "qshape": tuple(q.shape),
        "seqlens": list(attn_bias.q_seqlen) if isinstance(attn_bias, _BlockDiagonalMask) else None,
    })

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
# 1. Read stella modeling.py
# ============================================================
print("=" * 60)
print("1. stella modeling.py — attention/unpad code")
print("=" * 60)
import pathlib
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    print(f"File: {p}")
    with open(p) as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}\n")

    keywords = [
        "memory_efficient_attention", "BlockDiagonalMask", "from_seqlens",
        "unpad", "nonzero", "NonZero", "attention_mask_bool",
        "def forward", "class BertSelf", "class StellaSelf",
        "q_seqlen", "kv_seqlen", "cu_seqlens",
    ]
    seen = set()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                start = max(0, i - 5)
                end = min(len(lines), i + 15)
                key = (start, end)
                if key in seen:
                    break
                seen.add(key)
                print(f"--- line {i+1} (matched: '{kw}') ---")
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j].rstrip()}")
                print()
                break
    break

# ============================================================
# 2. Load model + get exact attention parameters
# ============================================================
print("=" * 60)
print("2. Load model + capture exact attention params")
print("=" * 60)
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

real_tokens = features["attention_mask"].sum().item()
print(f"Real tokens: {real_tokens}")

_attn_log["calls"] = []
with torch.no_grad():
    try:
        output = model(features)
    except Exception as e:
        print(f"Forward error: {e}")

calls = _attn_log["calls"]
print(f"Attention calls: {len(calls)}")
if calls:
    c = calls[0]
    print(f"  q_shape: {c['qshape']}")
    if c["seqlens"]:
        seqlens = c["seqlens"]
        print(f"  n_seqs: {len(seqlens)}")
        print(f"  seqlens (first 10): {seqlens[:10]}")
        print(f"  seqlens (last 10): {seqlens[-10:]}")
        print(f"  total: {sum(seqlens)}")
        print(f"  max: {max(seqlens)}")
        print(f"  min: {min(seqlens)}")
        print(f"  all same: {len(set(seqlens))==1}")

        # Compute cumulative seqlens for npu_fusion_attention
        import itertools
        cu_qlen = list(itertools.accumulate(seqlens))
        print(f"  cu_seqlens (first 5): {cu_qlen[:5]}")
        print(f"  cu_seqlens (last 5): {cu_qlen[-5:]}")

# ============================================================
# 3. Test npu_fusion_attention with packed variable-length
# ============================================================
print("\n" + "=" * 60)
print("3. Test npu_fusion_attention with packed var-len")
print("=" * 60)

if calls:
    c = calls[0]
    B, S, H, D = c["qshape"]
    seqlens = c["seqlens"]
    cu_qlen = list(itertools.accumulate(seqlens))
    cu_kvlen = cu_qlen  # self-attention

    # Create test tensors
    dtype = torch.float16
    q_test = torch.randn(B, S, H, D, dtype=dtype, device="npu")
    k_test = torch.randn(B, S, H, D, dtype=dtype, device="npu")
    v_test = torch.randn(B, S, H, D, dtype=dtype, device="npu")

    print(f"  q shape: {q_test.shape}, dtype: {dtype}")
    print(f"  head_num: {H}")
    print(f"  input_layout: BSND (B, S, N, D)")
    print(f"  actual_seq_qlen: {cu_qlen[:5]}...{cu_qlen[-3:]}")
    print(f"  actual_seq_kvlen: same")

    # Try npu_fusion_attention with BSND layout
    for layout in ["BSND", "BSH"]:
        print(f"\n  --- layout={layout} ---")
        try:
            if layout == "BSND":
                q_in, k_in, v_in = q_test, k_test, v_test
            else:
                q_in = q_test.reshape(B, S, H * D)
                k_in = k_test.reshape(B, S, H * D)
                v_in = v_test.reshape(B, S, H * D)

            out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
                q_in, k_in, v_in,
                head_num=H,
                input_layout=layout,
                actual_seq_qlen=cu_qlen,
                actual_seq_kvlen=cu_kvlen,
                scale=1.0 / (D ** 0.5),
                keep_prob=1.0,
            )
            print(f"  Success! output shape: {out.shape}")

            # Benchmark
            torch.npu.synchronize()
            for _ in range(3):
                out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
                    q_in, k_in, v_in,
                    head_num=H, input_layout=layout,
                    actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_kvlen,
                    scale=1.0/(D**0.5), keep_prob=1.0,
                )
            torch.npu.synchronize()

            N = 20
            t0 = time.time()
            for _ in range(N):
                out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
                    q_in, k_in, v_in,
                    head_num=H, input_layout=layout,
                    actual_seq_qlen=cu_qlen, actual_seq_kvlen=cu_kvlen,
                    scale=1.0/(D**0.5), keep_prob=1.0,
                )
            torch.npu.synchronize()
            t_fa = (time.time() - t0) / N * 1000
            print(f"  npu_fusion_attention: {t_fa:.1f} ms/call")
        except Exception as e:
            print(f"  Failed: {e}")

    # Compare with current approach (fake xformers)
    print(f"\n  --- current approach (fake xformers) ---")
    bias = _BlockDiagonalMask(seqlens)
    torch.npu.synchronize()
    for _ in range(3):
        _ = _memory_efficient_attention(q_test, k_test, v_test, attn_bias=bias)
    torch.npu.synchronize()

    t0 = time.time()
    for _ in range(N):
        _ = _memory_efficient_attention(q_test, k_test, v_test, attn_bias=bias)
    torch.npu.synchronize()
    t_current = (time.time() - t0) / N * 1000
    print(f"  fake xformers (pad+bool mask): {t_current:.1f} ms/call")

    # Also compare with uniform reshape (if all same length)
    if len(set(seqlens)) == 1:
        s = seqlens[0]
        n = len(seqlens)
        q_b = q_test.view(n, s, H, D).transpose(1, 2)
        k_b = k_test.view(n, s, H, D).transpose(1, 2)
        v_b = v_test.view(n, s, H, D).transpose(1, 2)
        torch.npu.synchronize()
        for _ in range(3):
            _ = _F.scaled_dot_product_attention(q_b, k_b, v_b)
        torch.npu.synchronize()
        t0 = time.time()
        for _ in range(N):
            _ = _F.scaled_dot_product_attention(q_b, k_b, v_b)
        torch.npu.synchronize()
        t_uniform = (time.time() - t0) / N * 1000
        print(f"  SDPA uniform (no mask):     {t_uniform:.1f} ms/call")

# ============================================================
# 4. Try torch.compile()
# ============================================================
print("\n" + "=" * 60)
print("4. torch.compile()")
print("=" * 60)

try:
    import torch._dynamo
    backends = torch._dynamo.list_backends()
    print(f"Backends: {backends}")
except:
    pass

print(f"torch_npu._compiler: {[x for x in dir(torch_npu._compiler) if not x.startswith('__')]}")

for mode in ["reduce-overhead", "default"]:
    print(f"\n--- mode={mode} ---")
    try:
        model2 = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
        model2.eval()
        model2.max_seq_length = 512
        model2.half()

        model2[0].auto_model = torch.compile(model2[0].auto_model, mode=mode)

        print("  Compiling + warmup (may take 1-2 min)...")
        with torch.no_grad():
            for _ in range(5):
                _ = model2(features)
        torch.npu.synchronize()

        N = 10
        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                _ = model2(features)
        torch.npu.synchronize()
        t = (time.time() - t0) / N * 1000
        print(f"  {t:.1f} ms/batch ({512*1000/t:.0f} docs/s)")

        del model2
        torch.npu.empty_cache()
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Model: stella_en_400M_v5, {len(calls)} attention layers")
if calls:
    print(f"Attention: packed (1, {calls[0]['qshape'][1]}, {H}, {D})")
    print(f"  {len(seqlens)} variable-length seqs, avg {sum(seqlens)/len(seqlens):.0f} tokens")
