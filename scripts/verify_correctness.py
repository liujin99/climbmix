#!/usr/bin/env python3
"""Correctness test: npu_fusion_attention vs fallback (pad+bool-mask SDPA).

Tests:
1. Standalone tensor test: compare TND path vs padded SDPA on random Q/K/V
2. Full model test: compare embeddings from both paths on real texts
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, time, types, warnings, itertools
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch_npu
import numpy as np

# ============================================================
# 1. Standalone tensor test
# ============================================================
print("=" * 70)
print("1. Standalone tensor test: npu_fusion_attention vs padded SDPA")
print("=" * 70)

def tnd_attention(q, k, v, qs_list, ks_list, H, D):
    """npu_fusion_attention TND path (from embedding_cluster.py)."""
    cu_qlen = list(itertools.accumulate(qs_list))
    cu_kvlen = list(itertools.accumulate(ks_list))
    q_tnd = q.squeeze(0).contiguous() if q.dim() == 4 else q.contiguous()
    k_tnd = k.squeeze(0).contiguous() if k.dim() == 4 else k.contiguous()
    v_tnd = v.squeeze(0).contiguous() if v.dim() == 4 else v.contiguous()
    out, _, _, _, _, _, _ = torch_npu.npu_fusion_attention(
        q_tnd, k_tnd, v_tnd,
        head_num=H,
        input_layout="TND",
        actual_seq_qlen=cu_qlen,
        actual_seq_kvlen=cu_kvlen,
        scale=1.0 / (D ** 0.5),
        keep_prob=1.0,
    )
    if q.dim() == 4:
        out = out.unsqueeze(0)
    return out

def padded_attention(q, k, v, qs_list, ks_list, H, D):
    """Fallback pad+bool-mask SDPA path (from embedding_cluster.py)."""
    n = len(qs_list)
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
    out = F.scaled_dot_product_attention(
        q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
        attn_mask=kmask, dropout_p=0.0,
    )
    out = out.transpose(1, 2)  # (n, max_s, H, D)
    chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
    return torch.cat(chunks, dim=1)  # (1, total_S, H, D)

# Test with variable-length sequences
seqlens = [128, 256, 64, 192, 320]
kslens = seqlens  # self-attention
total_S = sum(seqlens)
H, D = 16, 64

q = torch.randn(1, total_S, H, D, dtype=torch.float16, device="npu")
k = torch.randn(1, total_S, H, D, dtype=torch.float16, device="npu")
v = torch.randn(1, total_S, H, D, dtype=torch.float16, device="npu")

out_tnd = tnd_attention(q, k, v, seqlens, kslens, H, D)
out_pad = padded_attention(q, k, v, seqlens, kslens, H, D)
torch.npu.synchronize()

diff = (out_tnd.float() - out_pad.float()).abs()
print(f"  Seqlens: {seqlens}, total={total_S}")
print(f"  Output shape: TND={out_tnd.shape}, Padded={out_pad.shape}")
print(f"  Max abs diff:  {diff.max().item():.6f}")
print(f"  Mean abs diff: {diff.mean().item():.6f}")
print(f"  Relative diff: {(diff / (out_pad.float().abs() + 1e-6)).mean().item():.6f}")

# Check per-sequence correctness
offset = 0
for i, s in enumerate(seqlens):
    seg_tnd = out_tnd[0, offset:offset+s]
    seg_pad = out_pad[0, offset:offset+s]
    seg_diff = (seg_tnd.float() - seg_pad.float()).abs().max().item()
    print(f"  Seq {i} (len={s}): max diff = {seg_diff:.6f}")
    offset += s

# ============================================================
# 2. Full model test
# ============================================================
print("\n" + "=" * 70)
print("2. Full model test: npu_fusion_attention vs fallback")
print("=" * 70)

# Fake xformers with both paths
class _BlockDiagonalMask:
    def __init__(self, q_seqlen, kv_seqlen=None, device=None):
        if isinstance(q_seqlen, int): q_seqlen = [q_seqlen]
        self.q_seqlen = list(q_seqlen)
        self.kv_seqlen = list(kv_seqlen) if kv_seqlen is not None else self.q_seqlen
        self.device = device
    @classmethod
    def from_seqlens(cls, q_seqlen, kv_seqlen=None, device=None, **kw):
        if isinstance(q_seqlen, tuple) and len(q_seqlen) == 2 and isinstance(q_seqlen[0], (list, tuple)):
            q_seqlen, kv_seqlen = q_seqlen
        return cls(q_seqlen, kv_seqlen, device)

class _LowerTriangularMask:
    def __init__(self, *a, **kw): pass

_use_npu_fa = [True]  # mutable flag

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
    need_transpose = q.dim() == 4 and q.shape[1] > q.shape[2]
    def _to_sdpa(t): return t.transpose(1, 2) if need_transpose else t
    def _from_sdpa(t): return t.transpose(1, 2).contiguous() if need_transpose else t

    if isinstance(attn_bias, _BlockDiagonalMask):
        qs_list = attn_bias.q_seqlen; ks_list = attn_bias.kv_seqlen
        n = len(qs_list); _, _, H, D = q.shape

        if len(set(qs_list)) == 1 and len(set(ks_list)) == 1:
            s = qs_list[0]; ks = ks_list[0]
            q_b = q.view(n, s, H, D).transpose(1, 2)
            k_b = k.view(n, ks, H, D).transpose(1, 2)
            v_b = v.view(n, ks, H, D).transpose(1, 2)
            out = F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=p)
            return out.transpose(1, 2).reshape(1, -1, H, D).contiguous()
        elif _use_npu_fa[0]:
            # npu_fusion_attention TND path
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
        else:
            # Fallback: pad + bool-mask SDPA
            max_s = max(qs_list); max_ks = max(ks_list)
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
                q_off += qs_i; k_off += ks_i
            out = F.scaled_dot_product_attention(
                q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
                attn_mask=kmask, dropout_p=p,
            )
            out = out.transpose(1, 2)
            chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
            return torch.cat(chunks, dim=1)

    q_s, k_s, v_s = _to_sdpa(q), _to_sdpa(k), _to_sdpa(v)
    if isinstance(attn_bias, _LowerTriangularMask):
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=True, dropout_p=p)
    elif attn_bias is not None:
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=attn_bias, dropout_p=p)
    else:
        out = F.scaled_dot_product_attention(q_s, k_s, v_s, dropout_p=p)
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

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

# Small batch of variable-length texts
test_texts = [
    "Machine learning is a subset of artificial intelligence.",
    "The cat sat on the mat.",
    "In quantum mechanics, the Heisenberg uncertainty principle states that the position and momentum of a particle cannot be simultaneously measured with arbitrary precision. This fundamental limit arises from the wave-like nature of matter at microscopic scales.",
    "def hello_world():\n    print('Hello, World!')\n    return 42",
    "The Treaty of Westphalia, signed in 1648, ended the Thirty Years' War and established the modern system of sovereign states in Europe.",
    "Photosynthesis converts light energy into chemical energy stored in glucose molecules.",
    "The stock market crash of 1929 triggered the Great Depression.",
    "In topology, a manifold is a topological space that locally resembles Euclidean space near each point.",
]

features = model.tokenize(test_texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

attn_mask = features.get("attention_mask")
seqlens = attn_mask.sum(dim=1).cpu().numpy().tolist() if attn_mask is not None else [len(test_texts)]
print(f"  {len(test_texts)} texts, seqlens: {seqlens}")

# Get embeddings with npu_fusion_attention
_use_npu_fa[0] = True
emb_fa = model.encode(test_texts, batch_size=len(test_texts),
                      show_progress_bar=False, normalize_embeddings=True)
emb_fa = np.array(emb_fa, dtype=np.float32)

# Get embeddings with fallback (pad + bool-mask SDPA)
_use_npu_fa[0] = False
emb_pad = model.encode(test_texts, batch_size=len(test_texts),
                       show_progress_bar=False, normalize_embeddings=True)
emb_pad = np.array(emb_pad, dtype=np.float32)

# Compare
diff = np.abs(emb_fa - emb_pad)
print(f"\n  Embedding shape: {emb_fa.shape}")
print(f"  Max abs diff:    {diff.max():.6f}")
print(f"  Mean abs diff:   {diff.mean():.6f}")
print(f"  Median abs diff: {np.median(diff):.6f}")
print(f"  Cosine sim (normalized):")
cos_sim = np.sum(emb_fa * emb_pad, axis=1) / (np.linalg.norm(emb_fa, axis=1) * np.linalg.norm(emb_pad, axis=1))
print(f"    min={cos_sim.min():.6f}, max={cos_sim.max():.6f}, mean={cos_sim.mean():.6f}")

# Check if clustering would be affected
from sklearn.metrics.pairwise import cosine_similarity
sim_fa = cosine_similarity(emb_fa)
sim_pad = cosine_similarity(emb_pad)
sim_diff = np.abs(sim_fa - sim_pad)
print(f"\n  Pairwise similarity matrix diff:")
print(f"    Max diff:  {sim_diff.max():.6f}")
print(f"    Mean diff: {sim_diff.mean():.6f}")

# Would clustering change? Check nearest neighbor consistency
for i in range(len(test_texts)):
    nn_fa = np.argsort(-sim_fa[i])[1]  # nearest neighbor (excluding self)
    nn_pad = np.argsort(-sim_pad[i])[1]
    match = "OK" if nn_fa == nn_pad else "DIFF"
    print(f"  Text {i}: NN fa={nn_fa} vs pad={nn_pad} [{match}]")

print("\n" + "=" * 70)
print("CORRECTNESS VERDICT")
print("=" * 70)
max_emb_diff = diff.max()
mean_cos = cos_sim.mean()
if mean_cos > 0.999:
    print(f"  PASS: Embeddings are numerically equivalent (cos_sim={mean_cos:.6f})")
elif mean_cos > 0.99:
    print(f"  ACCEPTABLE: Minor numerical differences (cos_sim={mean_cos:.6f}, max_diff={max_emb_diff:.6f})")
    print(f"  Clustering results should be unaffected")
else:
    print(f"  WARNING: Significant differences (cos_sim={mean_cos:.6f}, max_diff={max_emb_diff:.6f})")
    print(f"  Need to investigate further")
