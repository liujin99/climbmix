#!/usr/bin/env python3
"""Test torch_npu fused APIs: npu_gelu_mul, npu_fast_gelu, npu_add_layer_norm.

Focus: Can we fuse GELU+mul (743ms) and LayerNorm+residual (166ms)?
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, time, types, types, warnings, inspect
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch_npu

print("=" * 70)
print("1. Test npu_gelu_mul API signature")
print("=" * 70)

# Create test tensors
T, D = 230000, 4096
gate = torch.randn(T, D, dtype=torch.float16, device="npu")
up = torch.randn(T, D, dtype=torch.float16, device="npu")

# Check signature
try:
    sig = inspect.signature(torch_npu.npu_gelu_mul)
    print(f"  npu_gelu_mul signature: {sig}")
except Exception as e:
    print(f"  Can't get signature: {e}")

# Test: current approach (separate GELU + mul)
torch.npu.synchronize()
for _ in range(5):
    _ = F.gelu(gate) * up
torch.npu.synchronize()

t0 = time.time()
for _ in range(20):
    out_ref = F.gelu(gate) * up
torch.npu.synchronize()
ms_sep = (time.time() - t0) / 20 * 1000
print(f"\n  F.gelu(gate) * up:     {ms_sep:.2f} ms")

# Test: npu_gelu_mul
try:
    out_fused = torch_npu.npu_gelu_mul(gate, up)
    torch.npu.synchronize()
    # Verify correctness
    out_ref_cpu = F.gelu(gate.float()).float() * up.float()
    diff = (out_fused.float() - out_ref_cpu).abs().max().item()
    print(f"  npu_gelu_mul output diff: {diff:.6f}")

    for _ in range(5):
        _ = torch_npu.npu_gelu_mul(gate, up)
    torch.npu.synchronize()

    t0 = time.time()
    for _ in range(20):
        out_fused = torch_npu.npu_gelu_mul(gate, up)
    torch.npu.synchronize()
    ms_fused = (time.time() - t0) / 20 * 1000
    print(f"  npu_gelu_mul(gate, up): {ms_fused:.2f} ms  ({ms_sep/ms_fused:.2f}x faster)")
except Exception as e:
    print(f"  npu_gelu_mul FAILED: {e}")

# Test: npu_gelu_mul with approximate flag?
try:
    out2 = torch_npu.npu_gelu_mul(gate, up, approximate="tanh")
    torch.npu.synchronize()
    print(f"  npu_gelu_mul with approximate='tanh': OK")
except Exception as e:
    print(f"  npu_gelu_mul with approximate: {e}")

# ============================================================
print("\n" + "=" * 70)
print("2. Test npu_fast_gelu")
print("=" * 70)
try:
    sig = inspect.signature(torch_npu.npu_fast_gelu)
    print(f"  npu_fast_gelu signature: {sig}")
except:
    pass

try:
    out_fast = torch_npu.npu_fast_gelu(gate)
    torch.npu.synchronize()
    diff = (out_fast.float() - F.gelu(gate.float())).abs().max().item()
    print(f"  npu_fast_gelu output diff vs exact: {diff:.6f}")

    for _ in range(5):
        _ = torch_npu.npu_fast_gelu(gate)
    torch.npu.synchronize()

    t0 = time.time()
    for _ in range(20):
        _ = torch_npu.npu_fast_gelu(gate)
    torch.npu.synchronize()
    ms_fast = (time.time() - t0) / 20 * 1000
    print(f"  npu_fast_gelu(gate):   {ms_fast:.2f} ms")

    # Compare with F.gelu
    t0 = time.time()
    for _ in range(20):
        _ = F.gelu(gate)
    torch.npu.synchronize()
    ms_gelu = (time.time() - t0) / 20 * 1000
    print(f"  F.gelu(gate):          {ms_gelu:.2f} ms  ({ms_gelu/ms_fast:.2f}x slower)")
except Exception as e:
    print(f"  npu_fast_gelu FAILED: {e}")

# ============================================================
print("\n" + "=" * 70)
print("3. Test npu_add_layer_norm (LayerNorm + residual)")
print("=" * 70)
H = 1024
x = torch.randn(T, H, dtype=torch.float16, device="npu")
weight = torch.randn(H, dtype=torch.float16, device="npu")
bias = torch.randn(H, dtype=torch.float16, device="npu")
residual = torch.randn(T, H, dtype=torch.float16, device="npu")

try:
    sig = inspect.signature(torch_npu.npu_add_layer_norm)
    print(f"  npu_add_layer_norm signature: {sig}")
except:
    pass

# Current: LayerNorm + residual
ln = torch.nn.LayerNorm(H).half().npu()
for _ in range(3):
    _ = ln(x) + residual
torch.npu.synchronize()

t0 = time.time()
for _ in range(20):
    out_ref = ln(x) + residual
torch.npu.synchronize()
ms_sep = (time.time() - t0) / 20 * 1000
print(f"\n  LayerNorm(x) + residual: {ms_sep:.2f} ms")

# npu_add_layer_norm
try:
    out_fused, _, _ = torch_npu.npu_add_layer_norm(
        residual, x, weight, bias, eps=1e-5
    )
    torch.npu.synchronize()
    diff = (out_fused.float() - (ln(x.float()).float() + residual.float())).abs().max().item()
    print(f"  npu_add_layer_norm diff: {diff:.6f}")

    for _ in range(3):
        _, _, _ = torch_npu.npu_add_layer_norm(residual, x, weight, bias, eps=1e-5)
    torch.npu.synchronize()

    t0 = time.time()
    for _ in range(20):
        _, _, _ = torch_npu.npu_add_layer_norm(residual, x, weight, bias, eps=1e-5)
    torch.npu.synchronize()
    ms_fused = (time.time() - t0) / 20 * 1000
    print(f"  npu_add_layer_norm:      {ms_fused:.2f} ms  ({ms_sep/ms_fused:.2f}x faster)")
except Exception as e:
    print(f"  npu_add_layer_norm FAILED: {e}")

# ============================================================
print("\n" + "=" * 70)
print("4. Test npu_layer_norm_eval")
print("=" * 70)
try:
    sig = inspect.signature(torch_npu.npu_layer_norm_eval)
    print(f"  npu_layer_norm_eval signature: {sig}")
except:
    pass

try:
    out = torch_npu.npu_layer_norm_eval(x, weight, bias, eps=1e-5)
    torch.npu.synchronize()
    diff = (out.float() - ln(x.float())).abs().max().item()
    print(f"  npu_layer_norm_eval diff: {diff:.6f}")

    t0 = time.time()
    for _ in range(20):
        _ = torch_npu.npu_layer_norm_eval(x, weight, bias, eps=1e-5)
    torch.npu.synchronize()
    ms = (time.time() - t0) / 20 * 1000
    print(f"  npu_layer_norm_eval: {ms:.2f} ms")

    t0 = time.time()
    for _ in range(20):
        _ = ln(x)
    torch.npu.synchronize()
    ms_ref = (time.time() - t0) / 20 * 1000
    print(f"  nn.LayerNorm:        {ms_ref:.2f} ms  ({ms_ref/ms:.2f}x slower)")
except Exception as e:
    print(f"  npu_layer_norm_eval FAILED: {e}")

# ============================================================
print("\n" + "=" * 70)
print("5. Test npu_fused_attention_layernorm_qkv_fwd")
print("=" * 70)
try:
    sig = inspect.signature(torch_npu.npu_fused_attention_layernorm_qkv_fwd)
    print(f"  npu_fused_attention_layernorm_qkv_fwd signature: {sig}")
except:
    pass

# ============================================================
print("\n" + "=" * 70)
print("6. Benchmark: full MLP with vs without npu_gelu_mul")
print("=" * 70)

# Simulate MLP operations
up_gate_w = torch.randn(1024, 8192, dtype=torch.float16, device="npu")
down_w = torch.randn(4096, 1024, dtype=torch.float16, device="npu")
hidden = torch.randn(T, 1024, dtype=torch.float16, device="npu")

# Current MLP forward
def mlp_current(x, up_gate_w, down_w):
    up_gate = F.linear(x, up_gate_w)  # (T, 8192)
    up_states, gate = torch.split(up_gate, 4096, dim=-1)
    gate = F.gelu(gate)
    gated_states = gate * up_states
    return F.linear(gated_states, down_w)

# Fused MLP forward
def mlp_fused(x, up_gate_w, down_w):
    up_gate = F.linear(x, up_gate_w)
    up_states, gate = torch.split(up_gate, 4096, dim=-1)
    gated_states = torch_npu.npu_gelu_mul(gate, up_states)
    return F.linear(gated_states, down_w)

# Warm up
for _ in range(3):
    _ = mlp_current(hidden, up_gate_w, down_w)
torch.npu.synchronize()
for _ in range(3):
    _ = mlp_fused(hidden, up_gate_w, down_w)
torch.npu.synchronize()

# Benchmark
t0 = time.time()
for _ in range(10):
    out_ref = mlp_current(hidden, up_gate_w, down_w)
torch.npu.synchronize()
ms_current = (time.time() - t0) / 10 * 1000

t0 = time.time()
for _ in range(10):
    out_fused = mlp_fused(hidden, up_gate_w, down_w)
torch.npu.synchronize()
ms_fused = (time.time() - t0) / 10 * 1000

# Correctness
diff = (out_fused.float() - out_ref.float()).abs().max().item()
print(f"  MLP current (GELU+mul separate): {ms_current:.2f} ms")
print(f"  MLP fused (npu_gelu_mul):        {ms_fused:.2f} ms  ({ms_current/ms_fused:.2f}x faster)")
print(f"  Output diff: {diff:.6f}")
print(f"  Savings: {ms_current - ms_fused:.2f} ms per forward, ×24 = {(ms_current - ms_fused)*24:.1f} ms total")

# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  npu_gelu_mul:     tested above")
print(f"  npu_fast_gelu:    tested above")
print(f"  npu_add_ln:       tested above")
print(f"  npu_ln_eval:      tested above")
print(f"\n  If npu_gelu_mul saves ~{ms_current - ms_fused:.0f}ms/layer × 24 = {(ms_current - ms_fused)*24:.0f}ms,")
print(f"  and npu_add_ln saves ~X ms × 24 = Y ms,")
print(f"  total savings ~{(ms_current - ms_fused)*24:.0f} + Y ms")
