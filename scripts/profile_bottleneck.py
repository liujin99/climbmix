#!/usr/bin/env python3
"""Comprehensive profiling: find remaining bottleneck after npu_fusion_attention patch.

Tests:
  1. Module structure + modeling.py unpad code
  2. Component-level timing (embedding / per-layer / pooling)
  3. bf16 vs fp16
  4. NPU high_performance operator mode
  5. D2H strategy (NPU normalize vs CPU normalize)
  6. Batch size sweep (256 / 512 / 768)

Usage:
    python scripts/profile_bottleneck.py
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
import traceback
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu
import numpy as np

# ============================================================
# Fake xformers with npu_fusion_attention (production code)
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
# Helpers
# ============================================================
def make_texts(n, target_len=400):
    """Generate n texts of ~target_len words."""
    return [" ".join([f"word{(i*7+j*13)%500}" for j in range(target_len)]) for i in range(n)]

def benchmark_forward(model, features, N=10, warmup=3):
    """Benchmark forward pass, return ms/batch."""
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(features)
    torch.npu.synchronize()
    _attn_calls.clear()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            output = model(features)
    torch.npu.synchronize()
    ms = (time.time() - t0) / N * 1000
    calls = len(_attn_calls) // N
    return ms, calls, output

def benchmark_full(model, features, N=10, warmup=3):
    """Benchmark full pipeline: forward + float + normalize + D2H."""
    for _ in range(warmup):
        with torch.no_grad():
            output = model(features)
            emb = output["sentence_embedding"].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            _ = emb.cpu().numpy()
    torch.npu.synchronize()
    _attn_calls.clear()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(N):
            output = model(features)
            emb = output["sentence_embedding"].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.cpu().numpy()
    torch.npu.synchronize()
    ms = (time.time() - t0) / N * 1000
    calls = len(_attn_calls) // N
    return ms, calls

# ============================================================
# Section 1: Read stella modeling.py — unpad logic
# ============================================================
print("=" * 70)
print("1. stella modeling.py — unpad / attention / forward logic")
print("=" * 70)
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    with open(p) as f:
        lines = f.readlines()
    print(f"  File: {p} ({len(lines)} lines)")

    # Print key sections
    keywords = [
        ("def forward", 5, 30),
        ("unpad_inputs", 2, 15),
        ("BlockDiagonalMask", 2, 10),
        ("from_seqlens", 2, 10),
        ("nonzero", 3, 8),
        ("attention_mask_bool", 2, 10),
        ("memory_efficient_attention", 2, 10),
        ("qkv", 2, 8),
        ("intermediate", 2, 8),
        ("gelu", 2, 5),
        ("def embed", 2, 15),
    ]
    seen = set()
    for kw, before, after in keywords:
        for i, line in enumerate(lines):
            if kw.lower() in line.lower():
                start = max(0, i - before)
                end = min(len(lines), i + after)
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                print(f"\n  --- line {i+1} ({kw}) ---")
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j].rstrip()}")
    break

# ============================================================
# Section 2: Load model + print module structure
# ============================================================
print("\n" + "=" * 70)
print("2. Model module structure")
print("=" * 70)
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

auto_model = model[0].auto_model
print(f"Model type: {type(auto_model).__name__}")
print(f"\nTop-level modules:")
for name, mod in auto_model.named_children():
    print(f"  {name}: {type(mod).__name__}")

print(f"\nEncoder children:")
encoder = auto_model.encoder
for name, mod in encoder.named_children():
    print(f"  encoder.{name}: {type(mod).__name__}")

if hasattr(encoder, 'layer') and len(encoder.layer) > 0:
    layer = encoder.layer[0]
    print(f"\nLayer[0] children:")
    for name, mod in layer.named_children():
        print(f"  layer.{name}: {type(mod).__name__}")
        for sname, smod in mod.named_children():
            print(f"  layer.{name}.{sname}: {type(smod).__name__}")

# Find attention and MLP module names
attn_module_name = None
mlp_module_name = None
for name, mod in layer.named_children():
    mod_type = type(mod).__name__.lower()
    if "attention" in mod_type or "attn" in name.lower():
        attn_module_name = name
    if "intermediate" in name.lower() or "mlp" in name.lower() or "ffn" in name.lower() or "feed" in name.lower():
        mlp_module_name = name
if not attn_module_name:
    for name, mod in layer.named_children():
        if "attention" in name.lower() or "attn" in name.lower():
            attn_module_name = name
if not mlp_module_name:
    # Check for output module which often contains the second linear
    for name, mod in layer.named_children():
        if name.lower() in ("output", "outputdense", "dense"):
            mlp_module_name = name

print(f"\nDetected: attention='{attn_module_name}', mlp='{mlp_module_name}'")

# Print all module names for layer[0]
print(f"\nAll named modules in layer[0]:")
for name, mod in layer.named_modules():
    if name:
        print(f"  layer.{name}: {type(mod).__name__}")

# ============================================================
# Section 3: Baseline timing (fp16, no hooks)
# ============================================================
print("\n" + "=" * 70)
print("3. Baseline (fp16, npu_fusion_attention TND)")
print("=" * 70)
texts = make_texts(512)
features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

ms_fwd, calls, _ = benchmark_forward(model, features, N=10)
ms_full, _ = benchmark_full(model, features, N=10)
print(f"  Forward only: {ms_fwd:.1f} ms ({512*1000/ms_fwd:.0f} docs/s)")
print(f"  Full pipeline: {ms_full:.1f} ms ({512*1000/ms_full:.0f} docs/s)")
print(f"  D2H + normalize overhead: {ms_full - ms_fwd:.1f} ms")
print(f"  Attention calls/fwd: {calls}, types: {set(_attn_calls)}")

# ============================================================
# Section 4: Component-level timing with hooks
# ============================================================
print("\n" + "=" * 70)
print("4. Component-level timing (with sync hooks — total will be slower)")
print("=" * 70)

hook_times = {}

def make_hook(name):
    def hook(module, input, output):
        torch.npu.synchronize()
        hook_times[name] = time.time()
    return hook

# Register hooks
hooks = []
hooks.append(auto_model.register_forward_hook(make_hook("auto_model_start")))
# Wait, forward hooks fire AFTER forward. Let me use pre_forward hooks for start.
hooks.clear()

# Use pre-forward hook for start, forward hook for end
def make_pre_hook(name):
    def hook(module, args):
        torch.npu.synchronize()
        hook_times[name] = time.time()
    return hook

def make_post_hook(name):
    def hook(module, args, output):
        torch.npu.synchronize()
        hook_times[name] = time.time()
    return hook

hooks = []

# Embeddings
if hasattr(auto_model, 'embeddings'):
    hooks.append(auto_model.embeddings.register_forward_pre_hook(make_pre_hook("emb_start")))
    hooks.append(auto_model.embeddings.register_forward_hook(make_post_hook("emb_end")))

# Encoder
hooks.append(encoder.register_forward_pre_hook(make_pre_hook("enc_start")))
hooks.append(encoder.register_forward_hook(make_post_hook("enc_end")))

# Each layer
n_layers = len(encoder.layer) if hasattr(encoder, 'layer') else 0
for i in range(n_layers):
    layer_i = encoder.layer[i]
    hooks.append(layer_i.register_forward_pre_hook(make_pre_hook(f"layer{i}_start")))
    hooks.append(layer_i.register_forward_hook(make_post_hook(f"layer{i}_end")))

    # Attention sub-module
    if attn_module_name:
        attn_mod = getattr(layer_i, attn_module_name, None)
        if attn_mod:
            hooks.append(attn_mod.register_forward_pre_hook(make_pre_hook(f"layer{i}_attn_start")))
            hooks.append(attn_mod.register_forward_hook(make_post_hook(f"layer{i}_attn_end")))

    # MLP sub-module
    if mlp_module_name:
        mlp_mod = getattr(layer_i, mlp_module_name, None)
        if mlp_mod:
            hooks.append(mlp_mod.register_forward_pre_hook(make_pre_hook(f"layer{i}_mlp_start")))
            hooks.append(mlp_mod.register_forward_hook(make_post_hook(f"layer{i}_mlp_end")))

# Pooling (model[1])
if len(model) > 1:
    hooks.append(model[1].register_forward_pre_hook(make_pre_hook("pool_start")))
    hooks.append(model[1].register_forward_hook(make_post_hook("pool_end")))

# Normalize (model[2])
if len(model) > 2:
    hooks.append(model[2].register_forward_pre_hook(make_pre_hook("norm_start")))
    hooks.append(model[2].register_forward_hook(make_post_hook("norm_end")))

print(f"  Registered {len(hooks)} hooks on {n_layers} layers")

# Run with hooks
hook_times.clear()
torch.npu.synchronize()
t0_hook = time.time()
with torch.no_grad():
    _ = model(features)
torch.npu.synchronize()
t_total_hooked = (time.time() - t0_hook) * 1000

# Calculate component times
print(f"\n  Total (hooked, with sync): {t_total_hooked:.1f} ms")
print(f"  Total (baseline, no hooks): {ms_fwd:.1f} ms")
print(f"  Sync overhead: {t_total_hooked - ms_fwd:.1f} ms")

# Embedding time
if "emb_start" in hook_times and "emb_end" in hook_times:
    emb_t = (hook_times["emb_end"] - hook_times["emb_start"]) * 1000
    print(f"\n  Embedding: {emb_t:.1f} ms")

# Encoder time
if "enc_start" in hook_times and "enc_end" in hook_times:
    enc_t = (hook_times["enc_end"] - hook_times["enc_start"]) * 1000
    print(f"  Encoder total: {enc_t:.1f} ms")

# Per-layer times
attn_total = 0
mlp_total = 0
layer_total = 0
print(f"\n  Per-layer breakdown (first 5 + last 5):")
for i in range(n_layers):
    ls = f"layer{i}_start"
    le = f"layer{i}_end"
    if ls in hook_times and le in hook_times:
        lt = (hook_times[le] - hook_times[ls]) * 1000
        layer_total += lt

        at_s = f"layer{i}_attn_start"
        at_e = f"layer{i}_attn_end"
        ml_s = f"layer{i}_mlp_start"
        ml_e = f"layer{i}_mlp_end"
        at_t = (hook_times[at_e] - hook_times[at_s]) * 1000 if at_s in hook_times and at_e in hook_times else 0
        ml_t = (hook_times[ml_e] - hook_times[ml_s]) * 1000 if ml_s in hook_times and ml_e in hook_times else 0
        other_t = lt - at_t - ml_t
        attn_total += at_t
        mlp_total += ml_t

        if i < 5 or i >= n_layers - 5:
            print(f"    Layer {i:2d}: total={lt:6.1f}ms  attn={at_t:5.1f}ms  mlp={ml_t:5.1f}ms  other={other_t:5.1f}ms")

print(f"\n  Sum over {n_layers} layers:")
print(f"    Attention: {attn_total:.1f} ms ({attn_total/t_total_hooked*100:.0f}%)")
print(f"    MLP:        {mlp_total:.1f} ms ({mlp_total/t_total_hooked*100:.0f}%)")
print(f"    Layer other: {layer_total - attn_total - mlp_total:.1f} ms")
print(f"    Layer total: {layer_total:.1f} ms")

# Pooling + normalize
if "pool_start" in hook_times and "pool_end" in hook_times:
    pool_t = (hook_times["pool_end"] - hook_times["pool_start"]) * 1000
    print(f"\n  Pooling: {pool_t:.1f} ms")
if "norm_start" in hook_times and "norm_end" in hook_times:
    norm_t = (hook_times["norm_end"] - hook_times["norm_start"]) * 1000
    print(f"  Normalize: {norm_t:.1f} ms")

# Non-layer overhead
enc_and_pool = 0
if "emb_end" in hook_times and "enc_start" in hook_times:
    pre_enc = (hook_times["enc_start"] - hook_times["emb_end"]) * 1000
    print(f"\n  Pre-encoder (unpad): {pre_enc:.1f} ms")
if "enc_end" in hook_times:
    post_enc_t = (t_total_hooked/1000 - hook_times["enc_end"]) * 1000
    print(f"  Post-encoder (pool+norm+output): {post_enc_t:.1f} ms")

# Remove hooks
for h in hooks:
    h.remove()

# ============================================================
# Section 5: bf16 vs fp16
# ============================================================
print("\n" + "=" * 70)
print("5. bf16 vs fp16")
print("=" * 70)

# fp16 baseline already measured
print(f"  fp16: {ms_fwd:.1f} ms ({512*1000/ms_fwd:.0f} docs/s)")

# bf16
try:
    model.bfloat16()
    ms_bf16, _, _ = benchmark_forward(model, features, N=10)
    print(f"  bf16: {ms_bf16:.1f} ms ({512*1000/ms_bf16:.0f} docs/s)")
    print(f"  Speedup: {ms_fwd/ms_bf16:.2f}x")
    if ms_bf16 < ms_fwd:
        print(f"  *** bf16 is {ms_fwd - ms_bf16:.0f}ms FASTER")
    else:
        print(f"  *** bf16 is {ms_bf16 - ms_fwd:.0f}ms SLOWER")
    # Switch back to fp16 for subsequent tests
    model.half()
except Exception as e:
    print(f"  bf16 FAILED: {e}")
    model.half()

# ============================================================
# Section 6: NPU operator selection mode
# ============================================================
print("\n" + "=" * 70)
print("6. NPU operator selection mode")
print("=" * 70)

# Try different NPU options
npu_options_to_try = [
    {"ACL_OP_SELECT_IMPL_MODE": "high_performance"},
    {"ACL_OP_SELECT_IMPL_MODE": "high_precision"},
]

for opt in npu_options_to_try:
    try:
        torch.npu.set_option(opt)
        print(f"  Set: {opt}")
        ms_opt, _, _ = benchmark_forward(model, features, N=10)
        print(f"    {ms_opt:.1f} ms ({512*1000/ms_opt:.0f} docs/s), speedup vs baseline: {ms_fwd/ms_opt:.2f}x")
    except Exception as e:
        print(f"  {opt} FAILED: {str(e)[:100]}")

# Reset to default
try:
    torch.npu.set_option({"ACL_OP_SELECT_IMPL_MODE": "high_precision"})
except:
    pass

# ============================================================
# Section 7: NPU compile mode (graph mode)
# ============================================================
print("\n" + "=" * 70)
print("7. NPU compile mode (jit_compile)")
print("=" * 70)

try:
    torch_npu.npu.set_compile_mode(jit_compile=True)
    print("  Set jit_compile=True")
    ms_graph, _, _ = benchmark_forward(model, features, N=10)
    print(f"    {ms_graph:.1f} ms ({512*1000/ms_graph:.0f} docs/s), speedup: {ms_fwd/ms_graph:.2f}x")
    torch_npu.npu.set_compile_mode(jit_compile=False)
except Exception as e:
    print(f"  jit_compile FAILED: {str(e)[:100]}")
    try:
        torch_npu.npu.set_compile_mode(jit_compile=False)
    except:
        pass

# ============================================================
# Section 8: D2H strategy comparison
# ============================================================
print("\n" + "=" * 70)
print("8. D2H + normalize strategy")
print("=" * 70)

# Strategy A: Current (NPU float + normalize + D2H)
ms_a, _ = benchmark_full(model, features, N=10)
print(f"  A) NPU float+norm → D2H: {ms_a:.1f} ms")

# Strategy B: D2H fp16 → CPU float + normalize
torch.npu.synchronize()
for _ in range(3):
    with torch.no_grad():
        output = model(features)
        emb = output["sentence_embedding"].cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
torch.npu.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        output = model(features)
        emb = output["sentence_embedding"].cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / norms
torch.npu.synchronize()
ms_b = (time.time() - t0) / 10 * 1000
print(f"  B) D2H fp16 → CPU float+norm: {ms_b:.1f} ms")
print(f"  Savings: {ms_a - ms_b:.1f} ms ({(1 - ms_b/ms_a)*100:.0f}%)")

# Strategy C: Async D2H with NPU stream
print(f"\n  C) Async D2H with NPU stream...")
try:
    d2h_stream = torch.npu.Stream()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            output = model(features)
            with torch.npu.stream(d2h_stream):
                emb = output["sentence_embedding"].float()
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                emb_np = emb.cpu().numpy()
            d2h_stream.synchronize()
    torch.npu.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            output = model(features)
            with torch.npu.stream(d2h_stream):
                emb = output["sentence_embedding"].float()
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                emb_np = emb.cpu().numpy()
            d2h_stream.synchronize()
    torch.npu.synchronize()
    ms_c = (time.time() - t0) / 10 * 1000
    print(f"    {ms_c:.1f} ms")
    print(f"    vs A: {ms_a - ms_c:.1f} ms savings ({(1 - ms_c/ms_a)*100:.0f}%)")
except Exception as e:
    print(f"    FAILED: {str(e)[:100]}")

# ============================================================
# Section 9: Batch size sweep
# ============================================================
print("\n" + "=" * 70)
print("9. Batch size sweep")
print("=" * 70)

for bs in [256, 512]:
    if bs > 512:
        continue  # skip large to avoid OOM
    texts_bs = make_texts(bs)
    feats_bs = model.tokenize(texts_bs)
    for k in list(feats_bs.keys()):
        if isinstance(feats_bs[k], torch.Tensor):
            feats_bs[k] = feats_bs[k].to("npu")

    ms_bs, calls_bs, _ = benchmark_forward(model, feats_bs, N=10)
    docs_s = bs * 1000 / ms_bs
    print(f"  batch={bs}: {ms_bs:.1f} ms ({docs_s:.0f} docs/s, attn_calls={calls_bs})")
    del feats_bs

# Try 768
try:
    texts_768 = make_texts(768)
    feats_768 = model.tokenize(texts_768)
    for k in list(feats_768.keys()):
        if isinstance(feats_768[k], torch.Tensor):
            feats_768[k] = feats_768[k].to("npu")
    ms_768, calls_768, _ = benchmark_forward(model, feats_768, N=5, warmup=2)
    docs_768 = 768 * 1000 / ms_768
    print(f"  batch=768: {ms_768:.1f} ms ({docs_768:.0f} docs/s, attn_calls={calls_768})")
    del feats_768
except Exception as e:
    print(f"  batch=768: FAILED ({str(e)[:80]})")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Baseline (fp16):           {ms_fwd:.1f} ms ({512*1000/ms_fwd:.0f} docs/s)")
try:
    print(f"  bf16:                      {ms_bf16:.1f} ms ({512*1000/ms_bf16:.0f} docs/s) [{ms_fwd/ms_bf16:.2f}x]")
except:
    pass
try:
    print(f"  high_performance:          {ms_opt:.1f} ms ({512*1000/ms_opt:.0f} docs/s) [{ms_fwd/ms_opt:.2f}x]")
except:
    pass
try:
    print(f"  jit_compile:               {ms_graph:.1f} ms ({512*1000/ms_graph:.0f} docs/s) [{ms_fwd/ms_graph:.2f}x]")
except:
    pass
print(f"  D2H NPU norm (A):          {ms_a:.1f} ms")
print(f"  D2H CPU norm (B):          {ms_b:.1f} ms [{(1-ms_b/ms_a)*100:.0f}% savings]")
try:
    print(f"  D2H stream (C):            {ms_c:.1f} ms [{(1-ms_c/ms_a)*100:.0f}% savings]")
except:
    pass
