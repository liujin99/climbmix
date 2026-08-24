#!/usr/bin/env python3
"""Deep profile: hook correct attention/MLP modules + individual Linear layers.

Usage:
    python scripts/profile_deep.py
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
import numpy as np

# ============================================================
# Fake xformers (production code)
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
_attn_timings = []

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
            _attn_timings.append(time.time() - _t0)
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
# Helpers
# ============================================================
def make_texts(n, target_len=400):
    return [" ".join([f"word{(i*7+j*13)%500}" for j in range(target_len)]) for i in range(n)]

# ============================================================
# Load model
# ============================================================
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

auto_model = model[0].auto_model
encoder = auto_model.encoder
n_layers = len(encoder.layer)

texts = make_texts(512)
features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

# ============================================================
# 1. Baseline (no hooks)
# ============================================================
print("=" * 70)
print("1. Baseline (no hooks)")
print("=" * 70)
for _ in range(3):
    with torch.no_grad():
        _ = model(features)
torch.npu.synchronize()

_attn_calls.clear()
_attn_timings.clear()
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        output = model(features)
torch.npu.synchronize()
ms_base = (time.time() - t0) / 10 * 1000
calls = len(_attn_calls) // 10
fa_ms = sum(_attn_timings) / 10 * 1000 if _attn_timings else 0
print(f"  Total: {ms_base:.1f} ms ({512*1000/ms_base:.0f} docs/s)")
print(f"  FA calls/fwd: {calls}, FA total: {fa_ms:.1f} ms ({fa_ms/ms_base*100:.1f}%)")
print(f"  Non-FA: {ms_base - fa_ms:.1f} ms ({(ms_base-fa_ms)/ms_base*100:.1f}%)")

# ============================================================
# 2. Hook correct modules: attention, mlp, and their sub-modules
# ============================================================
print("\n" + "=" * 70)
print("2. Component timing (correct hooks on attention/mlp)")
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

# Embeddings
hooks.append(auto_model.embeddings.register_forward_pre_hook(make_pre("emb_s")))
hooks.append(auto_model.embeddings.register_forward_hook(make_post("emb_e")))

# Encoder
hooks.append(encoder.register_forward_pre_hook(make_pre("enc_s")))
hooks.append(encoder.register_forward_hook(make_post("enc_e")))

# Each layer: attention, mlp, attn_ln, mlp_ln
for i in range(n_layers):
    layer_i = encoder.layer[i]
    p = f"L{i}"

    hooks.append(layer_i.register_forward_pre_hook(make_pre(f"{p}_s")))
    hooks.append(layer_i.register_forward_hook(make_post(f"{p}_e")))

    # attention module (NewAttention)
    hooks.append(layer_i.attention.register_forward_pre_hook(make_pre(f"{p}_att_s")))
    hooks.append(layer_i.attention.register_forward_hook(make_post(f"{p}_att_e")))

    # attention sub-modules (qkv_proj, o_proj)
    hooks.append(layer_i.attention.qkv_proj.register_forward_pre_hook(make_pre(f"{p}_qkv_s")))
    hooks.append(layer_i.attention.qkv_proj.register_forward_hook(make_post(f"{p}_qkv_e")))
    hooks.append(layer_i.attention.o_proj.register_forward_pre_hook(make_pre(f"{p}_op_s")))
    hooks.append(layer_i.attention.o_proj.register_forward_hook(make_post(f"{p}_op_e")))

    # mlp module (NewGatedMLP)
    hooks.append(layer_i.mlp.register_forward_pre_hook(make_pre(f"{p}_mlp_s")))
    hooks.append(layer_i.mlp.register_forward_hook(make_post(f"{p}_mlp_e")))

    # mlp sub-modules
    hooks.append(layer_i.mlp.up_gate_proj.register_forward_pre_hook(make_pre(f"{p}_ug_s")))
    hooks.append(layer_i.mlp.up_gate_proj.register_forward_hook(make_post(f"{p}_ug_e")))
    hooks.append(layer_i.mlp.down_proj.register_forward_pre_hook(make_pre(f"{p}_dp_s")))
    hooks.append(layer_i.mlp.down_proj.register_forward_hook(make_post(f"{p}_dp_e")))

    # LayerNorms
    hooks.append(layer_i.attn_ln.register_forward_pre_hook(make_pre(f"{p}_aln_s")))
    hooks.append(layer_i.attn_ln.register_forward_hook(make_post(f"{p}_aln_e")))
    hooks.append(layer_i.mlp_ln.register_forward_pre_hook(make_pre(f"{p}_mln_s")))
    hooks.append(layer_i.mlp_ln.register_forward_hook(make_post(f"{p}_mln_e")))

# Pooling
if len(model) > 1:
    hooks.append(model[1].register_forward_pre_hook(make_pre("pool_s")))
    hooks.append(model[1].register_forward_hook(make_post("pool_e")))

print(f"  Registered {len(hooks)} hooks")

# Run
hook_times.clear()
_attn_calls.clear()
_attn_timings.clear()
torch.npu.synchronize()
t_start = time.time()
with torch.no_grad():
    _ = model(features)
torch.npu.synchronize()
t_total = (time.time() - t_start) * 1000

# ============================================================
# 3. Print detailed breakdown
# ============================================================
print(f"\n  Total (hooked): {t_total:.1f} ms")
print(f"  Baseline:       {ms_base:.1f} ms")
print(f"  Sync overhead:  {t_total - ms_base:.1f} ms")

# Embedding
emb_t = (hook_times["emb_e"] - hook_times["emb_s"]) * 1000
print(f"\n  Embedding:     {emb_t:6.1f} ms")

# Pre-encoder overhead
pre_enc = (hook_times["enc_s"] - hook_times["emb_e"]) * 1000
print(f"  Pre-encoder:   {pre_enc:6.1f} ms")

# Per-layer detail
attn_total = 0
mlp_total = 0
qkv_total = 0
op_total = 0
ug_total = 0
dp_total = 0
aln_total = 0
mln_total = 0
fa_in_attn = 0
layer_other_total = 0

print(f"\n  Per-layer breakdown (first 3 + last 3):")
print(f"  {'Lyr':>3} {'Total':>6} {'Attn':>6} {'qkv':>6} {'FA':>6} {'o_prj':>6} {'MLP':>6} {'ug_prj':>6} {'down':>6} {'aLN':>5} {'mLN':>5} {'othr':>6}")

for i in range(n_layers):
    p = f"L{i}"
    lt = (hook_times[f"{p}_e"] - hook_times[f"{p}_s"]) * 1000

    att_t = (hook_times[f"{p}_att_e"] - hook_times[f"{p}_att_s"]) * 1000
    qkv_t = (hook_times[f"{p}_qkv_e"] - hook_times[f"{p}_qkv_s"]) * 1000
    op_t = (hook_times[f"{p}_op_e"] - hook_times[f"{p}_op_s"]) * 1000
    fa_t = att_t - qkv_t - op_t  # FA = attention - qkv - o_proj

    mlp_t = (hook_times[f"{p}_mlp_e"] - hook_times[f"{p}_mlp_s"]) * 1000
    ug_t = (hook_times[f"{p}_ug_e"] - hook_times[f"{p}_ug_s"]) * 1000
    dp_t = (hook_times[f"{p}_dp_e"] - hook_times[f"{p}_dp_s"]) * 1000
    mlp_other = mlp_t - ug_t - dp_t  # GELU + element-wise

    aln_t = (hook_times[f"{p}_aln_e"] - hook_times[f"{p}_aln_s"]) * 1000
    mln_t = (hook_times[f"{p}_mln_e"] - hook_times[f"{p}_mln_s"]) * 1000

    other = lt - att_t - mlp_t - aln_t - mln_t

    attn_total += att_t
    mlp_total += mlp_t
    qkv_total += qkv_t
    op_total += op_t
    ug_total += ug_t
    dp_total += dp_t
    aln_total += aln_t
    mln_total += mln_t
    fa_in_attn += fa_t
    layer_other_total += other

    if i < 3 or i >= n_layers - 3:
        print(f"  {i:3d} {lt:6.1f} {att_t:6.1f} {qkv_t:6.1f} {fa_t:6.1f} {op_t:6.1f} {mlp_t:6.1f} {ug_t:6.1f} {dp_t:6.1f} {aln_t:5.1f} {mln_t:5.1f} {other:6.1f}")

print(f"\n  {'Sum':>3} {t_total:6.1f}")
print(f"  {'':>3} {'Attn':>6} = {attn_total:6.1f} ms ({attn_total/t_total*100:.0f}%)  [qkv={qkv_total:.1f} + FA={fa_in_attn:.1f} + o_prj={op_total:.1f}]")
print(f"  {'':>3} {'MLP':>6} = {mlp_total:6.1f} ms ({mlp_total/t_total*100:.0f}%)  [up_gate={ug_total:.1f} + down={dp_total:.1f} + gelu/mul={mlp_total-ug_total-dp_total:.1f}]")
print(f"  {'':>3} {'LN':>6} = {aln_total + mln_total:6.1f} ms ({(aln_total+mln_total)/t_total*100:.0f}%)  [attn_ln={aln_total:.1f} + mlp_ln={mln_total:.1f}]")
print(f"  {'':>3} {'Embed':>6} = {emb_t:6.1f} ms ({emb_t/t_total*100:.0f}%)")
print(f"  {'':>3} {'Other':>6} = {t_total - attn_total - mlp_total - aln_total - mln_total - emb_t:6.1f} ms")

# Pooling
if "pool_s" in hook_times and "pool_e" in hook_times:
    pool_t = (hook_times["pool_e"] - hook_times["pool_s"]) * 1000
    print(f"  {'':>3} {'Pool':>6} = {pool_t:6.1f} ms ({pool_t/t_total*100:.0f}%)")

# ============================================================
# 4. Linear layer shapes
# ============================================================
print("\n" + "=" * 70)
print("4. Linear layer shapes (FLOPs analysis)")
print("=" * 70)
layer0 = encoder.layer[0]
for name, mod in layer0.named_modules():
    if isinstance(mod, torch.nn.Linear):
        w = mod.weight
        print(f"  {name:30s}: in={w.shape[1]:5d} out={w.shape[0]:5d}  ({w.numel()/1e6:.2f}M params)")

# Estimate FLOPs per forward
T = features["input_ids"].shape[1]  # seq len (padded)
B = features["input_ids"].shape[0]  # batch
# But with unpadding, actual tokens ~ 230K
# Let's get from attention calls
print(f"\n  Batch: {B}, Padded seq: {T}")
print(f"  (With unpadding, total tokens ~230K)")

# FLOPs for Linear: 2 * M * N * K (M=batch, N=out, K=in)
# For packed: M=T_total, N=out, K=in
T_total = 230000  # approximate
qkv_flops = 2 * T_total * 1024 * 3072
o_flops = 2 * T_total * 1024 * 1024
ug_flops = 2 * T_total * 1024 * 8192  # up_gate: 1024 -> 2*4096=8192
dp_flops = 2 * T_total * 4096 * 1024

attn_proj_flops = qkv_flops + o_flops
mlp_flops = ug_flops + dp_flops
total_proj_flops = (attn_proj_flops + mlp_flops) * n_layers

print(f"\n  QKV proj:   {qkv_flops*n_layers/1e12:.2f} TFLOP")
print(f"  O proj:     {o_flops*n_layers/1e12:.2f} TFLOP")
print(f"  Up+Gate:    {ug_flops*n_layers/1e12:.2f} TFLOP")
print(f"  Down proj:  {dp_flops*n_layers/1e12:.2f} TFLOP")
print(f"  Total proj: {total_proj_flops/1e12:.2f} TFLOP")
print(f"\n  Attn proj time: {qkv_total+op_total:.1f} ms -> {attn_proj_flops*n_layers/1e12 / (qkv_total+op_total)/1e3:.0f} TFLOPS")
print(f"  MLP time:       {mlp_total:.1f} ms -> {mlp_flops*n_layers/1e12 / mlp_total/1e3:.0f} TFLOPS")
print(f"  (910B4 fp16 theoretical: ~320 TFLOPS)")

# ============================================================
# 5. torch.compile(backend="inductor")
# ============================================================
print("\n" + "=" * 70)
print("5. torch.compile(backend='inductor') — fusion attempt")
print("=" * 70)
for h in hooks:
    h.remove()

try:
    compiled_model = torch.compile(model, backend="inductor", mode="default")
    for _ in range(3):
        with torch.no_grad():
            _ = compiled_model(features)
    torch.npu.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = compiled_model(features)
    torch.npu.synchronize()
    ms_inductor = (time.time() - t0) / 10 * 1000
    print(f"  inductor: {ms_inductor:.1f} ms ({512*1000/ms_inductor:.0f} docs/s) [{ms_base/ms_inductor:.2f}x]")
except Exception as e:
    print(f"  inductor FAILED: {str(e)[:200]}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Baseline:              {ms_base:.1f} ms ({512*1000/ms_base:.0f} docs/s)")
print(f"  FA (npu_fusion_attn):  {fa_ms:.1f} ms ({fa_ms/ms_base*100:.1f}%)")
print(f"  Attention total:       {attn_total:.1f} ms ({attn_total/t_total*100:.0f}%)  [qkv+FA+o_proj]")
print(f"  MLP total:             {mlp_total:.1f} ms ({mlp_total/t_total*100:.0f}%)  [up_gate+gelu+down]")
print(f"  LayerNorm:             {aln_total+mln_total:.1f} ms ({(aln_total+mln_total)/t_total*100:.0f}%)")
print(f"  Embedding:             {emb_t:.1f} ms")
try:
    print(f"  inductor:              {ms_inductor:.1f} ms [{ms_base/ms_inductor:.2f}x]")
except:
    pass
