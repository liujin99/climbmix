#!/usr/bin/env python3
"""Read attention forward code + test with variable-length texts + check torch_npu fused APIs.

Usage:
    python scripts/profile_attn_code.py
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
import pathlib
import inspect
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu
import numpy as np

# ============================================================
# Fake xformers (production)
# ============================================================
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

_attn_calls = []
_fa_timings = []  # Detailed FA timings

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
            # Uniform path
            torch.npu.synchronize()
            _t0 = time.time()
            s = qs_list[0]; ks = ks_list[0]
            q_b = q.view(n, s, H, D).transpose(1, 2)
            k_b = k.view(n, ks, H, D).transpose(1, 2)
            v_b = v.view(n, ks, H, D).transpose(1, 2)
            out = _F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=p)
            out = out.transpose(1, 2).reshape(1, -1, H, D).contiguous()
            torch.npu.synchronize()
            _fa_timings.append(("uniform_sdpa", time.time() - _t0))
            return out
        else:
            # Variable-length path (npu_fusion_attention)
            torch.npu.synchronize()
            _t0 = time.time()
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
            torch.npu.synchronize()
            _fa_timings.append(("npu_fa_tnd", time.time() - _t0))
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
# 1. Read attention forward code from modeling.py
# ============================================================
print("=" * 70)
print("1. Attention forward code")
print("=" * 70)
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    with open(p) as f:
        lines = f.readlines()
    
    # Find NewAttention class and its forward
    in_attn_class = False
    in_mlp_class = False
    brace_depth = 0
    
    for i, line in enumerate(lines):
        # Find class definitions
        if "class NewAttention" in line:
            in_attn_class = True
            print(f"\n--- NewAttention (line {i+1}) ---")
        if "class NewGatedMLP" in line or "class NewMLP" in line or "class GatedMLP" in line:
            in_mlp_class = True
            print(f"\n--- MLP class (line {i+1}) ---")
        
        if in_attn_class or in_mlp_class:
            print(f"  {i+1}: {line.rstrip()}")
            if i > 0 and line.strip() == "" and brace_depth == 0:
                pass  # skip empty lines in class
        
        # End of class (next class at same indent)
        if (in_attn_class or in_mlp_class) and i > 0:
            if line.startswith("class ") and "NewAttention" not in line and "NewGatedMLP" not in line and "NewMLP" not in line and "GatedMLP" not in line:
                in_attn_class = False
                in_mlp_class = False
    
    # Also print the layer forward
    print("\n--- Layer forward (search) ---")
    for i, line in enumerate(lines):
        if "class NewLayer" in line or ("class " in line and "Layer" in line and "def forward" not in line):
            for j in range(i, min(i+80, len(lines))):
                print(f"  {j+1}: {lines[j].rstrip()}")
                if j > i and lines[j].strip().startswith("class ") and j > i+1:
                    break
            break
    
    # Print the model forward (unpad logic)
    print("\n--- Model forward (unpad logic) ---")
    for i, line in enumerate(lines):
        if "unpad_inputs" in line.lower() and "def forward" not in line.lower():
            start = max(0, i - 20)
            end = min(len(lines), i + 30)
            for j in range(start, end):
                print(f"  {j+1}: {lines[j].rstrip()}")
            print("  ...")
            break
    break

# ============================================================
# 2. Check torch_npu fused APIs
# ============================================================
print("\n" + "=" * 70)
print("2. torch_npu fused APIs")
print("=" * 70)
fused_apis = []
for name in dir(torch_npu):
    obj = getattr(torch_npu, name)
    if callable(obj) and any(kw in name.lower() for kw in ["gelu", "layer_norm", "ln", "fused", "fast", "flash"]):
        fused_apis.append(name)
        print(f"  torch_npu.{name}")

# Check torch_npu.npu
for name in dir(torch_npu.npu):
    obj = getattr(torch_npu.npu, name)
    if callable(obj) and any(kw in name.lower() for kw in ["gelu", "layer_norm", "ln", "fused", "fast", "flash"]):
        fused_apis.append(f"torch_npu.npu.{name}")
        print(f"  torch_npu.npu.{name}")

if not fused_apis:
    print("  (none found)")

# Check torch.nn.functional for NPU-specific
print("\n  torch.nn.functional GELU variants:")
print(f"    F.gelu exists: {hasattr(_F, 'gelu')}")
print(f"    F.gelu_approx: {hasattr(_F, 'gelu_approx')}")

# ============================================================
# 3. Load model
# ============================================================
print("\n" + "=" * 70)
print("3. Load model")
print("=" * 70)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

auto_model = model[0].auto_model
encoder = auto_model.encoder

# Print attention forward source code
print("\n  Attention forward source:")
print("  " + inspect.getsource(encoder.layer[0].attention.forward)[:2000])

# ============================================================
# 4. Test with UNIFORM texts (current)
# ============================================================
print("\n" + "=" * 70)
print("4. Uniform texts (all same length)")
print("=" * 70)
uniform_texts = [" ".join([f"word{(i*7+j*13)%500}" for j in range(400)]) for i in range(512)]
features_u = model.tokenize(uniform_texts)
for k in list(features_u.keys()):
    if isinstance(features_u[k], torch.Tensor):
        features_u[k] = features_u[k].to("npu")

# Check seqlens
attn_mask = features_u.get("attention_mask")
if attn_mask is not None:
    seqlens = attn_mask.sum(dim=1).cpu().numpy()
    print(f"  Seqlens: min={seqlens.min()}, max={seqlens.max()}, unique={len(set(seqlens))}")

for _ in range(3):
    with torch.no_grad():
        _ = model(features_u)
torch.npu.synchronize()

_attn_calls.clear()
_fa_timings.clear()
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(features_u)
torch.npu.synchronize()
ms_u = (time.time() - t0) / 10 * 1000
calls_u = len(_attn_calls) // 10
fa_types_u = [t for t, _ in _fa_timings[:10]]  # types from first fwd
fa_time_u = sum(t for _, t in _fa_timings) / 10 * 1000
print(f"  Total: {ms_u:.1f} ms ({512*1000/ms_u:.0f} docs/s)")
print(f"  FA calls/fwd: {calls_u}, types: {set(_attn_calls)}")
print(f"  FA actual time: {fa_time_u:.1f} ms ({fa_time_u/ms_u*100:.1f}%)")
print(f"  FA type sample: {fa_types_u[:3]}")

# ============================================================
# 5. Test with VARIABLE-length texts (production-like)
# ============================================================
print("\n" + "=" * 70)
print("5. Variable-length texts (production-like)")
print("=" * 70)
var_texts = []
for i in range(512):
    # Vary length: 50 to 500 words
    length = 50 + (i * 37) % 450
    var_texts.append(" ".join([f"word{(i*7+j*13)%500}" for j in range(length)]))

features_v = model.tokenize(var_texts)
for k in list(features_v.keys()):
    if isinstance(features_v[k], torch.Tensor):
        features_v[k] = features_v[k].to("npu")

attn_mask = features_v.get("attention_mask")
if attn_mask is not None:
    seqlens = attn_mask.sum(dim=1).cpu().numpy()
    print(f"  Seqlens: min={seqlens.min()}, max={seqlens.max()}, unique={len(set(seqlens))}, avg={seqlens.mean():.0f}")

for _ in range(3):
    with torch.no_grad():
        _ = model(features_v)
torch.npu.synchronize()

_attn_calls.clear()
_fa_timings.clear()
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        _ = model(features_v)
torch.npu.synchronize()
ms_v = (time.time() - t0) / 10 * 1000
calls_v = len(_attn_calls) // 10
fa_time_v = sum(t for _, t in _fa_timings) / 10 * 1000
fa_type_v = [t for t, _ in _fa_timings[:10]]
print(f"  Total: {ms_v:.1f} ms ({512*1000/ms_v:.0f} docs/s)")
print(f"  FA calls/fwd: {calls_v}, types: {set(_attn_calls)}")
print(f"  FA actual time: {fa_time_v:.1f} ms ({fa_time_v/ms_v*100:.1f}%)")
print(f"  FA type sample: {fa_type_v[:3]}")

# ============================================================
# 6. Hook: time individual operations inside attention forward
# ============================================================
print("\n" + "=" * 70)
print("6. Fine-grained attention timing (hooks on sub-ops)")
print("=" * 70)

hook_times = {}
def make_pre(name):
    def hook(module, args):
        torch.npu.synchronize()
        hook_times[name] = time.time()
    return hook
def make_post(name):
    def hook(module, args, output):
        torch.npu.synchronize()
        hook_times[name] = time.time()
    return hook

hooks = []
layer0 = encoder.layer[0]

# Hook attention module + sub-modules
hooks.append(layer0.attention.register_forward_pre_hook(make_pre("att_s")))
hooks.append(layer0.attention.register_forward_hook(make_post("att_e")))
hooks.append(layer0.attention.qkv_proj.register_forward_pre_hook(make_pre("qkv_s")))
hooks.append(layer0.attention.qkv_proj.register_forward_hook(make_post("qkv_e")))
hooks.append(layer0.attention.o_proj.register_forward_pre_hook(make_pre("op_s")))
hooks.append(layer0.attention.o_proj.register_forward_hook(make_post("op_e")))

# Also hook layer 0 total + LayerNorm
hooks.append(layer0.register_forward_pre_hook(make_pre("L0_s")))
hooks.append(layer0.register_forward_hook(make_post("L0_e")))
hooks.append(layer0.attn_ln.register_forward_pre_hook(make_pre("aln_s")))
hooks.append(layer0.attn_ln.register_forward_hook(make_post("aln_e")))
hooks.append(layer0.mlp_ln.register_forward_pre_hook(make_pre("mln_s")))
hooks.append(layer0.mlp_ln.register_forward_hook(make_post("mln_e")))
hooks.append(layer0.mlp.register_forward_pre_hook(make_pre("mlp_s")))
hooks.append(layer0.mlp.register_forward_hook(make_post("mlp_e")))
hooks.append(layer0.mlp.up_gate_proj.register_forward_pre_hook(make_pre("ug_s")))
hooks.append(layer0.mlp.up_gate_proj.register_forward_hook(make_post("ug_e")))
hooks.append(layer0.mlp.down_proj.register_forward_pre_hook(make_pre("dp_s")))
hooks.append(layer0.mlp.down_proj.register_forward_hook(make_post("dp_e")))

# Run with hooks (uniform texts)
hook_times.clear()
_attn_calls.clear()
_fa_timings.clear()
torch.npu.synchronize()
with torch.no_grad():
    _ = model(features_u)
torch.npu.synchronize()

print(f"\n  Layer 0 breakdown (uniform texts):")
lt = (hook_times["L0_e"] - hook_times["L0_s"]) * 1000
att_t = (hook_times["att_e"] - hook_times["att_s"]) * 1000
qkv_t = (hook_times["qkv_e"] - hook_times["qkv_s"]) * 1000
op_t = (hook_times["op_e"] - hook_times["op_s"]) * 1000
fa_hook_t = att_t - qkv_t - op_t
aln_t = (hook_times["aln_e"] - hook_times["aln_s"]) * 1000
mlp_t = (hook_times["mlp_e"] - hook_times["mlp_s"]) * 1000
mln_t = (hook_times["mln_e"] - hook_times["mln_s"]) * 1000
ug_t = (hook_times["ug_e"] - hook_times["ug_s"]) * 1000
dp_t = (hook_times["dp_e"] - hook_times["dp_s"]) * 1000
gelu_mul_t = mlp_t - ug_t - dp_t
other_t = lt - att_t - mlp_t - aln_t - mln_t

print(f"  Layer total:    {lt:6.1f} ms")
print(f"  attn_ln:        {aln_t:6.1f} ms")
print(f"  Attention:      {att_t:6.1f} ms  [qkv={qkv_t:.1f} + FA_overhead={fa_hook_t:.1f} + o_prj={op_t:.1f}]")
print(f"  mlp_ln:         {mln_t:6.1f} ms")
print(f"  MLP:            {mlp_t:6.1f} ms  [up_gate={ug_t:.1f} + gelu/mul={gelu_mul_t:.1f} + down={dp_t:.1f}]")
print(f"  Other (resid):  {other_t:6.1f} ms")
print(f"  FA actual (synced inside): {sum(t for _, t in _fa_timings)*1000:.1f} ms")
print(f"  FA overhead (att - qkv - o_prj - fa_actual): {fa_hook_t - sum(t for _, t in _fa_timings)*1000:.1f} ms")

# Run with hooks (variable texts)
hook_times.clear()
_attn_calls.clear()
_fa_timings.clear()
torch.npu.synchronize()
with torch.no_grad():
    _ = model(features_v)
torch.npu.synchronize()

print(f"\n  Layer 0 breakdown (variable texts):")
lt = (hook_times["L0_e"] - hook_times["L0_s"]) * 1000
att_t = (hook_times["att_e"] - hook_times["att_s"]) * 1000
qkv_t = (hook_times["qkv_e"] - hook_times["qkv_s"]) * 1000
op_t = (hook_times["op_e"] - hook_times["op_s"]) * 1000
fa_hook_t = att_t - qkv_t - op_t
aln_t = (hook_times["aln_e"] - hook_times["aln_s"]) * 1000
mlp_t = (hook_times["mlp_e"] - hook_times["mlp_s"]) * 1000
mln_t = (hook_times["mln_e"] - hook_times["mln_s"]) * 1000
ug_t = (hook_times["ug_e"] - hook_times["ug_s"]) * 1000
dp_t = (hook_times["dp_e"] - hook_times["dp_s"]) * 1000
gelu_mul_t = mlp_t - ug_t - dp_t
other_t = lt - att_t - mlp_t - aln_t - mln_t

print(f"  Layer total:    {lt:6.1f} ms")
print(f"  attn_ln:        {aln_t:6.1f} ms")
print(f"  Attention:      {att_t:6.1f} ms  [qkv={qkv_t:.1f} + FA_overhead={fa_hook_t:.1f} + o_prj={op_t:.1f}]")
print(f"  mlp_ln:         {mln_t:6.1f} ms")
print(f"  MLP:            {mlp_t:6.1f} ms  [up_gate={ug_t:.1f} + gelu/mul={gelu_mul_t:.1f} + down={dp_t:.1f}]")
print(f"  Other (resid):  {other_t:6.1f} ms")
print(f"  FA actual (synced inside): {sum(t for _, t in _fa_timings)*1000:.1f} ms")
print(f"  FA overhead (att - qkv - o_prj - fa_actual): {fa_hook_t - sum(t for _, t in _fa_timings)*1000:.1f} ms")

for h in hooks:
    h.remove()

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Uniform texts:   {ms_u:.1f} ms ({512*1000/ms_u:.0f} docs/s), FA={fa_time_u:.1f}ms, type={set(_attn_calls)}")
print(f"  Variable texts:   {ms_v:.1f} ms ({512*1000/ms_v:.0f} docs/s), FA={fa_time_v:.1f}ms")
print(f"  Ratio (var/uniform): {ms_v/ms_u:.2f}x")
