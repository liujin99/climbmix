#!/usr/bin/env python3
"""FlashAttention 诊断脚本 — 在 NPU 服务器上运行。

用法：
    python scripts/diagnose_flash_attention.py

检查项：
1. torch_npu 是否有 npu_fusion_attention API
2. F.scaled_dot_product_attention 无 mask vs bool mask 性能对比
3. stella 模型的 attention 实际走哪条路径
4. HBM 用量对比
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["WANDB_SILENT"] = "true"

import sys
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch_npu

def check_api():
    print("=" * 60)
    print("1. 检查 torch_npu FlashAttention API")
    print("=" * 60)
    print(f"torch_npy version: {torch_npu.__version__}")
    print(f"torch version: {torch.__version__}")

    fa_attrs = [x for x in dir(torch_npu) if 'flash' in x.lower() or 'fusion' in x.lower()]
    print(f"FlashAttention-related: {fa_attrs}")

    fa_fn = getattr(torch_npu, 'npu_fusion_attention', None)
    if fa_fn is not None:
        print(f"npu_fusion_attention exists: {fa_fn}")
    else:
        fa_fn2 = getattr(torch_npu.functional, 'npu_fusion_attention', None)
        if fa_fn2:
            print(f"torch_npu.functional.npu_fusion_attention exists: {fa_fn2}")
        else:
            print("npu_fusion_attention NOT FOUND")

    compile_attrs = [x for x in dir(torch_npu) if 'compile' in x.lower() or 'jit' in x.lower()]
    print(f"Compile-related: {compile_attrs}")
    print()


def benchmark_sdpa():
    print("=" * 60)
    print("2. F.scaled_dot_product_attention 性能对比")
    print("=" * 60)

    B, H, S, D = 512, 12, 512, 128
    dtype = torch.float16

    q = torch.randn(B, H, S, D, dtype=dtype, device="npu")
    k = torch.randn(B, H, S, D, dtype=dtype, device="npu")
    v = torch.randn(B, H, S, D, dtype=dtype, device="npu")

    # warmup
    for _ in range(3):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.npu.synchronize()

    # 无 mask（应该走 FlashAttention）
    torch.npu.synchronize()
    t0 = time.time()
    N = 20
    for _ in range(N):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.npu.synchronize()
    t_no_mask = (time.time() - t0) / N * 1000
    print(f"无 mask:      {t_no_mask:.1f} ms/call")

    # bool mask（可能不走 FlashAttention）
    mask = torch.ones(B, 1, S, S, dtype=torch.bool, device="npu")
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(N):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.npu.synchronize()
    t_bool_mask = (time.time() - t0) / N * 1000
    print(f"bool mask:    {t_bool_mask:.1f} ms/call")

    # causal mask（可能走 FlashAttention）
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(N):
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.npu.synchronize()
    t_causal = (time.time() - t0) / N * 1000
    print(f"is_causal:    {t_causal:.1f} ms/call")

    ratio = t_bool_mask / t_no_mask if t_no_mask > 0 else 0
    print(f"\nbool mask / no mask = {ratio:.1f}x")
    if ratio > 2:
        print(">>> bool mask 显著慢于无 mask，说明 bool mask 路径未走 FlashAttention")
    else:
        print(">>> bool mask 和无 mask 速度接近，FlashAttention 可能已启用")

    # HBM 对比
    torch.npu.empty_cache()
    torch.npu.synchronize()
    hbm_before = torch.npu.memory_allocated()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.npu.synchronize()
    hbm_after = torch.npu.memory_allocated()
    print(f"\nHBM (无 mask): {hbm_before/1e9:.2f} GB -> {hbm_after/1e9:.2f} GB")

    del out
    torch.npu.empty_cache()
    torch.npu.synchronize()
    hbm_before = torch.npu.memory_allocated()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.npu.synchronize()
    hbm_after = torch.npu.memory_allocated()
    print(f"HBM (bool mask): {hbm_before/1e9:.2f} GB -> {hbm_after/1e9:.2f} GB")
    print(f"  (全量 attention 矩阵: {B*H*S*S*dtype.itemsize/1e9:.2f} GB)")
    print()


def check_stella_attention():
    print("=" * 60)
    print("3. 检查 stella 模型 attention 路径")
    print("=" * 60)

    # 安装 fake xformers
    import types
    import torch.nn.functional as _F

    class _BlockDiagonalMask:
        def __init__(self, q_seqlen, kv_seqlen=None, device=None):
            self.q_seqlen = list(q_seqlen) if isinstance(q_seqlen, (list, tuple)) else [q_seqlen]
            self.kv_seqlen = list(kv_seqlen) if kv_seqlen is not None else self.q_seqlen
            self.device = device
        @classmethod
        def from_seqlens(cls, q_seqlen, kv_seqlen=None, device=None, **kw):
            return cls(q_seqlen, kv_seqlen, device)

    class _LowerTriangularMask:
        def __init__(self, *a, **kw): pass

    def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
        print(f"  [FAKE XFORMERS] attn_bias type: {type(attn_bias).__name__}")
        print(f"  q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")

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

    model_name = "NovaSearch/stella_en_400M_v5"
    print(f"Loading {model_name}...")
    model = SentenceTransformer(model_name, device="npu", trust_remote_code=True)
    model.eval()
    model.max_seq_length = 512
    model.half()
    print("Model loaded.\n")

    # 用 3 条文本做一次前向，看 attention 走哪条路径
    texts = ["hello world", "this is a test sentence", "another document here"]
    print("Running tokenize + forward on 3 texts...")
    features = model.tokenize(texts)
    print(f"Tokenized features keys: {list(features.keys())}")
    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, torch.Tensor):
                    print(f"  {k}.{k2}: shape={v2.shape}, dtype={v2.dtype}")

    # 移到 NPU
    for k in features:
        if isinstance(features[k], torch.Tensor):
            features[k] = features[k].to("npu")
        elif isinstance(features[k], dict):
            for k2 in features[k]:
                if isinstance(features[k2], torch.Tensor):
                    features[k2] = features[k2].to("npu")

    print("\nRunning model forward (will print attention path)...")
    with torch.no_grad():
        output = model(features)
    print(f"Output keys: {list(output.keys())}")
    print(f"sentence_embedding shape: {output['sentence_embedding'].shape}")
    print()


def check_npu_fusion_attention():
    print("=" * 60)
    print("4. 直接测试 npu_fusion_attention（如果存在）")
    print("=" * 60)

    fa = getattr(torch_npu, 'npu_fusion_attention', None)
    if fa is None:
        fa = getattr(torch_npu.functional, 'npu_fusion_attention', None)
    if fa is None:
        print("npu_fusion_attention 不存在，跳过")
        return

    B, H, S, D = 512, 12, 512, 128
    dtype = torch.float16
    q = torch.randn(B, S, H, D, dtype=dtype, device="npu")
    k = torch.randn(B, S, H, D, dtype=dtype, device="npu")
    v = torch.randn(B, S, H, D, dtype=dtype, device="npu")

    # warmup
    for _ in range(3):
        try:
            out = fa(q, k, v)
        except Exception as e:
            print(f"npu_fusion_attention 调用失败: {e}")
            return
    torch.npu.synchronize()

    torch.npu.synchronize()
    t0 = time.time()
    N = 20
    for _ in range(N):
        out = fa(q, k, v)
    torch.npu.synchronize()
    t_fa = (time.time() - t0) / N * 1000
    print(f"npu_fusion_attention: {t_fa:.1f} ms/call")

    # 对比 SDPA 无 mask
    q_sdpa = q.transpose(1, 2).contiguous()
    k_sdpa = k.transpose(1, 2).contiguous()
    v_sdpa = v.transpose(1, 2).contiguous()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(N):
        out = torch.nn.functional.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
    torch.npu.synchronize()
    t_sdpa = (time.time() - t0) / N * 1000
    print(f"SDPA (无 mask):     {t_sdpa:.1f} ms/call")
    print(f"npu_fusion_attention / SDPA = {t_fa/t_sdpa:.2f}x")
    print()


if __name__ == "__main__":
    check_api()
    benchmark_sdpa()
    check_npu_fusion_attention()
    check_stella_attention()
