#!/usr/bin/env python3
"""Test different attention implementations to find one that works on NPU.

The fake xformers currently uses SDPA with bool mask → 100% NaN on all dtypes.
This script tests alternatives:
  A. SDPA without any mask (sanity check — does SDPA work at all?)
  B. SDPA with float additive mask (-inf for masked)
  C. Manual attention (Q@K^T, softmax, @V) in fp32
  D. Manual attention in model dtype
  E. torch_npu fused attention (if available)
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, types, warnings, time
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
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

# ── Sanity check: does SDPA work at all on NPU? ──
print("\n=== Sanity Check: SDPA without mask ===")
for dtype in [torch.float32, torch.float16, torch.bfloat16]:
    q = torch.randn(1, 4, 8, 64, device="npu", dtype=dtype)
    k = torch.randn(1, 4, 8, 64, device="npu", dtype=dtype)
    v = torch.randn(1, 4, 8, 64, device="npu", dtype=dtype)
    try:
        out = _F.scaled_dot_product_attention(q, k, v)
        has_nan = torch.isnan(out).any().item()
        print(f"  SDPA {str(dtype):20s}: NaN={has_nan}  out.shape={out.shape}")
    except Exception as e:
        print(f"  SDPA {str(dtype):20s}: ERROR: {e}")

print("\n=== Sanity Check: SDPA with bool mask ===")
for dtype in [torch.float32, torch.float16]:
    q = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    k = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    v = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    mask = torch.ones(2, 1, 8, 8, dtype=torch.bool, device="npu")
    mask[1, 0, :, 4:] = False  # mask out half of seq for batch 1
    try:
        out = _F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        has_nan = torch.isnan(out).any().item()
        print(f"  SDPA+bool_mask {str(dtype):12s}: NaN={has_nan}")
    except Exception as e:
        print(f"  SDPA+bool_mask {str(dtype):12s}: ERROR: {e}")

print("\n=== Sanity Check: SDPA with float mask ===")
for dtype in [torch.float32, torch.float16]:
    q = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    k = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    v = torch.randn(2, 4, 8, 64, device="npu", dtype=dtype)
    mask = torch.zeros(2, 1, 8, 8, device="npu", dtype=dtype)
    mask[1, 0, :, 4:] = float("-inf")
    try:
        out = _F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        has_nan = torch.isnan(out).any().item()
        print(f"  SDPA+float_mask {str(dtype):10s}: NaN={has_nan}")
    except Exception as e:
        print(f"  SDPA+float_mask {str(dtype):10s}: ERROR: {e}")

# ── Now test with the actual model using different fake xformers ──
# We'll create different fake xformers and test each

def make_fake_xformers(attn_impl):
    """Create a fake xformers module with the specified attention implementation."""
    
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
        def __init__(self, *a, **kw):
            pass

    def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
        need_transpose = q.dim() == 4 and q.shape[1] > q.shape[2]

        def _to_sdpa(t):
            return t.transpose(1, 2) if need_transpose else t

        def _from_sdpa(t):
            return t.transpose(1, 2).contiguous() if need_transpose else t

        if isinstance(attn_bias, _BlockDiagonalMask):
            qs_list = attn_bias.q_seqlen
            ks_list = attn_bias.kv_seqlen
            n = len(qs_list)
            _, _, H, D = q.shape
            max_s = max(qs_list)
            max_ks = max(ks_list)

            q_pad = torch.zeros(n, max_s, H, D, dtype=q.dtype, device=q.device)
            k_pad = torch.zeros(n, max_ks, H, D, dtype=k.dtype, device=k.device)
            v_pad = torch.zeros(n, max_ks, H, D, dtype=v.dtype, device=v.device)
            q_off = k_off = 0
            for i, (qs_i, ks_i) in enumerate(zip(qs_list, ks_list)):
                q_pad[i, :qs_i] = q[0, q_off:q_off + qs_i]
                k_pad[i, :ks_i] = k[0, k_off:k_off + ks_i]
                v_pad[i, :ks_i] = v[0, k_off:k_off + ks_i]
                q_off += qs_i
                k_off += ks_i

            q_s = q_pad.transpose(1, 2)
            k_s = k_pad.transpose(1, 2)
            v_s = v_pad.transpose(1, 2)

            if attn_impl == "sdpa_bool":
                kmask = torch.zeros(n, 1, max_s, max_ks, dtype=torch.bool, device=q.device)
                k_off = 0
                for i, ks_i in enumerate(ks_list):
                    kmask[i, 0, :, :ks_i] = True
                    k_off += ks_i
                out = _F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=kmask, dropout_p=p)

            elif attn_impl == "sdpa_float":
                kmask = torch.zeros(n, 1, max_s, max_ks, dtype=q.dtype, device=q.device)
                kmask.fill_(float("-inf"))
                k_off = 0
                for i, ks_i in enumerate(ks_list):
                    kmask[i, 0, :, :ks_i] = 0.0
                    k_off += ks_i
                out = _F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=kmask, dropout_p=p)

            elif attn_impl == "manual_fp32":
                # Manual attention in fp32, regardless of model dtype
                q_f = q_s.float()
                k_f = k_s.float()
                v_f = v_s.float()
                scale = 1.0 / (q_f.shape[-1] ** 0.5)
                scores = torch.matmul(q_f, k_f.transpose(-2, -1)) * scale
                kmask = torch.zeros(n, 1, max_s, max_ks, dtype=torch.bool, device=q.device)
                k_off = 0
                for i, ks_i in enumerate(ks_list):
                    kmask[i, 0, :, :ks_i] = True
                    k_off += ks_i
                scores = scores.masked_fill(~kmask, float("-inf"))
                # Check for all-masked rows
                all_masked = scores.isinf().all(dim=-1, keepdim=True)
                if all_masked.any():
                    scores = scores.masked_fill(all_masked, 0.0)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v_f).to(q.dtype)

            elif attn_impl == "manual_native":
                # Manual attention in model dtype
                scale = 1.0 / (q_s.shape[-1] ** 0.5)
                scores = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale
                kmask = torch.zeros(n, 1, max_s, max_ks, dtype=torch.bool, device=q.device)
                k_off = 0
                for i, ks_i in enumerate(ks_list):
                    kmask[i, 0, :, :ks_i] = True
                    k_off += ks_i
                scores = scores.masked_fill(~kmask, float("-inf"))
                all_masked = scores.isinf().all(dim=-1, keepdim=True)
                if all_masked.any():
                    scores = scores.masked_fill(all_masked, 0.0)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, v_s)

            elif attn_impl == "npu_fused":
                # Try torch_npu fused attention
                # TND format: (T, H, D) where T = total tokens
                T = sum(qs_list)
                q_tnd = q[0]  # already (1, T, H, D) → (T, H, D)
                k_tnd = k[0]
                v_tnd = v[0]
                # npu_fusion_attention expects (B, S, H, D) or (T, H, D)
                # with actual_seq_qlen and actual_seq_kvlen
                try:
                    out, _ = torch_npu.npu_fusion_attention(
                        q_tnd, k_tnd, v_tnd,
                        actual_seq_qlen=qs_list,
                        actual_seq_kvlen=ks_list,
                        head_num=H,
                        input_layout="TND",
                        scale=1.0 / (D ** 0.5),
                    )
                    out = out.unsqueeze(0).transpose(1, 2)  # back to (B, H, S, D) format
                except Exception as e:
                    print(f"    npu_fused failed: {e}, falling back to manual_fp32")
                    return make_fake_xformers("manual_fp32")._memory_efficient_attention(
                        q, k, v, attn_bias=attn_bias, p=p, **kw)

            out = out.transpose(1, 2)  # back to (n, max_s, H, D)
            chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
            return torch.cat(chunks, dim=1)

        # Non-block-diagonal paths
        q_s, k_s, v_s = _to_sdpa(q), _to_sdpa(k), _to_sdpa(v)
        if isinstance(attn_bias, _LowerTriangularMask):
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, dropout_p=p)
        elif attn_bias is not None:
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_bias, dropout_p=p)
        else:
            out = _F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=p)
        return _from_sdpa(out)

    # Build module hierarchy
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
    
    # Clear any existing fake xformers
    for mod in ["xformers", "xformers.ops", "xformers.ops.fmha", "xformers.ops.fmha.attn_bias"]:
        if mod in sys.modules:
            del sys.modules[mod]
    
    sys.modules["xformers"] = _xfm
    sys.modules["xformers.ops"] = _ops
    sys.modules["xformers.ops.fmha"] = _fmha_mod
    sys.modules["xformers.ops.fmha.attn_bias"] = _attn_bias_mod
    
    return _xfm


def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    pct = n_nan / len(emb) * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  {label:50s}: NaN={n_nan:4d}/{len(emb)} ({pct:5.1f}%) [{status}]")
    return n_nan


# ── Test each implementation with the actual model ──
from sentence_transformers import SentenceTransformer

impls = ["sdpa_bool", "sdpa_float", "manual_fp32", "manual_native", "npu_fused"]

for impl in impls:
    print(f"\n{'='*70}")
    print(f"  Testing: {impl}")
    print(f"{'='*70}")
    
    make_fake_xformers(impl)
    
    try:
        m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
        m.eval()
        m.half()
        m.max_seq_length = 512
        t0 = time.time()
        emb = m.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
        elapsed = time.time() - t0
        check_nan(emb, f"{impl} (fp16, {elapsed:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
