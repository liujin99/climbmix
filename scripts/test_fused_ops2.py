#!/usr/bin/env python3
"""Test torch_npu fused APIs (fixed signatures) + read RoPE code + benchmark components.

Key findings from previous run:
- npu_gelu_mul takes 1 input (GELU(x)*x), NOT GELU(a)*b — can't use for SwiGLU
- npu_fast_gelu = F.gelu in speed (no benefit)
- Need to fix: npu_add_layer_norm (epsilon not eps), npu_layer_norm_eval (normalized_shape as list)
- Need to fix: MLP weight shape (out_features, in_features) not (in, out)
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, time, types, warnings, inspect, pathlib
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch_npu
import numpy as np

T = 230000  # typical total tokens

# ============================================================
print("=" * 70)
print("1. Fix npu_add_layer_norm (epsilon, not eps)")
print("=" * 70)
H = 1024
x = torch.randn(T, H, dtype=torch.float16, device="npu")
gamma = torch.randn(H, dtype=torch.float16, device="npu")
beta = torch.randn(H, dtype=torch.float16, device="npu")
residual = torch.randn(T, H, dtype=torch.float16, device="npu")

ln = torch.nn.LayerNorm(H).half().npu()

# Current: LayerNorm + residual
for _ in range(3):
    _ = ln(x) + residual
torch.npu.synchronize()
t0 = time.time()
for _ in range(20):
    out_ref = ln(x) + residual
torch.npu.synchronize()
ms_sep = (time.time() - t0) / 20 * 1000
print(f"  LayerNorm(x) + residual:  {ms_sep:.2f} ms")

# npu_add_layer_norm — fixed: use epsilon=, returns (out, mean, rstd, additional)
try:
    result = torch_npu.npu_add_layer_norm(residual, x, gamma, beta, epsilon=1e-5)
    out_fused = result[0] if isinstance(result, tuple) else result
    torch.npu.synchronize()
    diff = (out_fused.float() - (ln(x.float()).float() + residual.float())).abs().max().item()
    print(f"  npu_add_layer_norm diff:  {diff:.6f}")

    for _ in range(3):
        _ = torch_npu.npu_add_layer_norm(residual, x, gamma, beta, epsilon=1e-5)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(20):
        _ = torch_npu.npu_add_layer_norm(residual, x, gamma, beta, epsilon=1e-5)
    torch.npu.synchronize()
    ms_fused = (time.time() - t0) / 20 * 1000
    print(f"  npu_add_layer_norm:       {ms_fused:.2f} ms  ({ms_sep/ms_fused:.2f}x faster)")
    print(f"  Savings per layer: {ms_sep - ms_fused:.2f} ms, ×24 = {(ms_sep - ms_fused)*24:.1f} ms")
except Exception as e:
    print(f"  npu_add_layer_norm FAILED: {e}")

# ============================================================
print("\n" + "=" * 70)
print("2. Fix npu_layer_norm_eval (normalized_shape as list)")
print("=" * 70)
try:
    out = torch_npu.npu_layer_norm_eval(x, [H], gamma, beta, 1e-5)
    torch.npu.synchronize()
    diff = (out.float() - ln(x.float())).abs().max().item()
    print(f"  npu_layer_norm_eval diff: {diff:.6f}")

    for _ in range(3):
        _ = torch_npu.npu_layer_norm_eval(x, [H], gamma, beta, 1e-5)
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(20):
        _ = torch_npu.npu_layer_norm_eval(x, [H], gamma, beta, 1e-5)
    torch.npu.synchronize()
    ms = (time.time() - t0) / 20 * 1000

    t0 = time.time()
    for _ in range(20):
        _ = ln(x)
    torch.npu.synchronize()
    ms_ref = (time.time() - t0) / 20 * 1000
    print(f"  npu_layer_norm_eval: {ms:.2f} ms")
    print(f"  nn.LayerNorm:        {ms_ref:.2f} ms  ({ms_ref/ms:.2f}x)")
except Exception as e:
    print(f"  npu_layer_norm_eval FAILED: {e}")

# ============================================================
print("\n" + "=" * 70)
print("3. Read RoPE code + benchmark")
print("=" * 70)
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    with open(p) as f:
        lines = f.readlines()

    # Find apply_rotary_pos_emb
    for i, line in enumerate(lines):
        if "def apply_rotary_pos_emb" in line:
            print(f"  Found apply_rotary_pos_emb at line {i+1}:")
            for j in range(i, min(i+40, len(lines))):
                print(f"    {j+1}: {lines[j].rstrip()}")
            break

    # Also find rotate_half
    for i, line in enumerate(lines):
        if "def rotate_half" in line:
            print(f"\n  Found rotate_half at line {i+1}:")
            for j in range(i, min(i+15, len(lines))):
                print(f"    {j+1}: {lines[j].rstrip()}")
            break

    # Print full attention forward (after the assert)
    for i, line in enumerate(lines):
        if "class NewAttention" in line:
            print(f"\n  Full NewAttention forward:")
            for j in range(i, min(i+80, len(lines))):
                print(f"    {j+1}: {lines[j].rstrip()}")
            break
    break

# ============================================================
# Benchmark RoPE on realistic tensors
# ============================================================
print("\n  --- RoPE benchmark ---")
H_heads, D_head = 16, 64
q = torch.randn(1, T, H_heads, D_head, dtype=torch.float16, device="npu")
k = torch.randn(1, T, H_heads, D_head, dtype=torch.float16, device="npu")

# Simulate RoPE: cos/sin tables
cos = torch.randn(1, T, 1, D_head, dtype=torch.float16, device="npu")
sin = torch.randn(1, T, 1, D_head, dtype=torch.float16, device="npu")

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

# Warm up
for _ in range(3):
    _ = apply_rope(q, k, cos, sin)
torch.npu.synchronize()

t0 = time.time()
for _ in range(20):
    q_rot, k_rot = apply_rope(q, k, cos, sin)
torch.npu.synchronize()
ms_rope = (time.time() - t0) / 20 * 1000
print(f"  apply_rotary_pos_emb (q+k): {ms_rope:.2f} ms")
print(f"  Per layer overhead estimate: {ms_rope:.2f} ms")
print(f"  Total RoPE (×24): {ms_rope * 24:.1f} ms")

# Also benchmark individual operations
# GELU
gate = torch.randn(T, 4096, dtype=torch.float16, device="npu")
up = torch.randn(T, 4096, dtype=torch.float16, device="npu")

for _ in range(3):
    _ = F.gelu(gate)
torch.npu.synchronize()
t0 = time.time()
for _ in range(20):
    _ = F.gelu(gate)
torch.npu.synchronize()
ms_gelu = (time.time() - t0) / 20 * 1000

# Mul
for _ in range(3):
    _ = gate * up
torch.npu.synchronize()
t0 = time.time()
for _ in range(20):
    _ = gate * up
torch.npu.synchronize()
ms_mul = (time.time() - t0) / 20 * 1000

# GELU + mul combined
for _ in range(3):
    _ = F.gelu(gate) * up
torch.npu.synchronize()
t0 = time.time()
for _ in range(20):
    _ = F.gelu(gate) * up
torch.npu.synchronize()
ms_gelu_mul = (time.time() - t0) / 20 * 1000

print(f"\n  F.gelu(gate):     {ms_gelu:.2f} ms")
print(f"  gate * up:        {ms_mul:.2f} ms")
print(f"  F.gelu(gate)*up:  {ms_gelu_mul:.2f} ms (separate: {ms_gelu+ms_mul:.2f} ms)")
print(f"  Per layer gelu/mul overhead (hook shows 30.9ms, standalone {ms_gelu_mul:.1f}ms)")
print(f"  Diff: {30.9 - ms_gelu_mul:.1f} ms (sync + split + dropout overhead)")

# ============================================================
print("\n" + "=" * 70)
print("4. Fix MLP benchmark (weight shape: out, in)")
print("=" * 70)
# F.linear(x, weight) does x @ weight.T, so weight = (out_features, in_features)
up_gate_w = torch.randn(8192, 1024, dtype=torch.float16, device="npu")
down_w = torch.randn(1024, 4096, dtype=torch.float16, device="npu")
hidden = torch.randn(T, 1024, dtype=torch.float16, device="npu")

def mlp_current(x, ug_w, d_w):
    up_gate = F.linear(x, ug_w)
    up_states, gate = torch.split(up_gate, 4096, dim=-1)
    gate = F.gelu(gate)
    gated = gate * up_states
    return F.linear(gated, d_w)

def mlp_fast_gelu(x, ug_w, d_w):
    up_gate = F.linear(x, ug_w)
    up_states, gate = torch.split(up_gate, 4096, dim=-1)
    gate = torch_npu.npu_fast_gelu(gate)
    gated = gate * up_states
    return F.linear(gated, d_w)

# Warm up
for _ in range(3):
    _ = mlp_current(hidden, up_gate_w, down_w)
torch.npu.synchronize()
for _ in range(3):
    _ = mlp_fast_gelu(hidden, up_gate_w, down_w)
torch.npu.synchronize()

# Benchmark current
t0 = time.time()
for _ in range(10):
    out_ref = mlp_current(hidden, up_gate_w, down_w)
torch.npu.synchronize()
ms_current = (time.time() - t0) / 10 * 1000

# Benchmark fast_gelu
t0 = time.time()
for _ in range(10):
    out_fast = mlp_fast_gelu(hidden, up_gate_w, down_w)
torch.npu.synchronize()
ms_fast = (time.time() - t0) / 10 * 1000

# Correctness
diff = (out_fast.float() - out_ref.float()).abs().max().item()
print(f"  MLP current (F.gelu):     {ms_current:.2f} ms")
print(f"  MLP fast_gelu:            {ms_fast:.2f} ms  ({ms_current/ms_fast:.2f}x)")
print(f"  Output diff: {diff:.6f}")

# Breakdown: time each component
print("\n  MLP component breakdown:")
for _ in range(3):
    ug = F.linear(hidden, up_gate_w)
torch.npu.synchronize()
t0 = time.time()
for _ in range(10):
    ug = F.linear(hidden, up_gate_w)
torch.npu.synchronize()
ms_ug = (time.time() - t0) / 10 * 1000

up_s, g = torch.split(ug, 4096, dim=-1)
for _ in range(3):
    _ = F.gelu(g) * up_s
torch.npu.synchronize()
t0 = time.time()
for _ in range(10):
    gated = F.gelu(g) * up_s
torch.npu.synchronize()
ms_gm = (time.time() - t0) / 10 * 1000

for _ in range(3):
    _ = F.linear(gated, down_w)
torch.npu.synchronize()
t0 = time.time()
for _ in range(10):
    _ = F.linear(gated, down_w)
torch.npu.synchronize()
ms_dp = (time.time() - t0) / 10 * 1000

print(f"    up_gate_proj:  {ms_ug:.2f} ms")
print(f"    gelu+mul:      {ms_gm:.2f} ms  (F.gelu={ms_gelu:.2f} + mul={ms_mul:.2f} = {ms_gelu+ms_mul:.2f})")
print(f"    down_proj:     {ms_dp:.2f} ms")
print(f"    total:         {ms_ug + ms_gm + ms_dp:.2f} ms (measured: {ms_current:.2f} ms)")

# ============================================================
print("\n" + "=" * 70)
print("5. Full model end-to-end with npu_add_layer_norm patch")
print("=" * 70)

# Load model with fake xformers
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

import itertools
_attn_calls = []
def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
    _attn_calls.append(type(attn_bias).__name__)
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
auto_model = model[0].auto_model
encoder = auto_model.encoder

# Generate variable-length texts (production-like)
var_texts = []
for i in range(512):
    length = 50 + (i * 37) % 450
    var_texts.append(" ".join([f"word{(i*7+j*13)%500}" for j in range(length)]))
features = model.tokenize(var_texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

# Baseline
for _ in range(3):
    with torch.no_grad():
        _ = model(features)
torch.npu.synchronize()

_attn_calls.clear()
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(features)
torch.npu.synchronize()
ms_base = (time.time() - t0) / 10 * 1000
print(f"  Baseline: {ms_base:.1f} ms ({512*1000/ms_base:.0f} docs/s)")

# Patch: npu_add_layer_norm for all LayerNorm layers
print("\n  Patching LayerNorm modules with npu_add_layer_norm...")
patched_count = 0

for layer in encoder.layer:
    # Patch attn_ln
    original_attn_ln = layer.attn_ln
    original_mlp_ln = layer.mlp_ln

    # Create a patched forward that uses npu_add_layer_norm
    # But we need to know the residual... the layer forward does:
    #   residual = hidden_states
    #   hidden_states = self.attn_ln(hidden_states)
    #   attn_output = self.attention(hidden_states, ...)
    #   hidden_states = residual + attn_output
    # So attn_ln is called WITHOUT residual. The residual add is separate.
    # npu_add_layer_norm fuses: LayerNorm(x) + residual -> output
    # But the layer forward does them separately.
    # We'd need to patch the LAYER forward, not just the LayerNorm.

    # For now, just try npu_layer_norm_eval (no residual fusion, just faster LN)
    w = original_attn_ln.weight
    b = original_attn_ln.bias
    eps = original_attn_ln.eps
    normalized_shape = original_attn_ln.normalized_shape

    def make_ln_eval(w, b, eps, nshape):
        def forward(x):
            return torch_npu.npu_layer_norm_eval(x, list(nshape), w, b, eps)
        return forward

    original_attn_ln.forward = make_ln_eval(w, b, eps, normalized_shape)
    original_mlp_ln.forward = make_ln_eval(w, b, eps, normalized_shape)
    patched_count += 2

print(f"  Patched {patched_count} LayerNorm modules")

# Test with patched LayerNorm
for _ in range(3):
    with torch.no_grad():
        _ = model(features)
torch.npu.synchronize()

t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(features)
torch.npu.synchronize()
ms_ln = (time.time() - t0) / 10 * 1000
print(f"  With npu_layer_norm_eval: {ms_ln:.1f} ms ({512*1000/ms_ln:.0f} docs/s)")
print(f"  Speedup: {ms_base/ms_ln:.3f}x, savings: {ms_base - ms_ln:.1f} ms")

# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  npu_gelu_mul: 1-input only (GELU(x)*x), NOT usable for SwiGLU")
print(f"  npu_fast_gelu: same speed as F.gelu ({ms_gelu:.2f} vs {5.87:.2f} ms)")
print(f"  npu_add_layer_norm: tested above")
print(f"  npu_layer_norm_eval: tested above")
print(f"  RoPE standalone: {ms_rope:.2f} ms per layer")
print(f"  GELU+mul standalone: {ms_gelu_mul:.2f} ms per layer")
print(f"  MLP component: up_gate={ms_ug:.2f} + gelu_mul={ms_gm:.2f} + down={ms_dp:.2f} = {ms_ug+ms_gm+ms_dp:.2f} ms")
print(f"\n  Baseline: {ms_base:.1f} ms")
print(f"  With npu_layer_norm_eval: {ms_ln:.1f} ms ({ms_base/ms_ln:.3f}x)")
