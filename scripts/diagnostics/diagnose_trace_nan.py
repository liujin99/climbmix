#!/usr/bin/env python3
"""Trace where NaN first appears in the model's forward pass.

SDPA works fine with random tensors, but all attention impls produce 100% NaN
with the real model. The NaN must come from elsewhere — weights, embeddings,
normalization, or MLP layers.

This script:
  1. Checks model weights for NaN after loading and after half()
  2. Runs a single forward pass with hooks on every submodule
  3. Reports the first module where NaN appears
  4. Also tests a standard (non-custom) model for comparison
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings, time
warnings.filterwarnings("ignore")

# Set up fake xformers first
import climbmix.core.embedding_cluster  # noqa: F401

import torch
import torch_npu
import numpy as np

def check_weights_for_nan(model, label):
    """Check all parameters for NaN/Inf."""
    n_nan_params = 0
    n_inf_params = 0
    total_params = 0
    nan_param_names = []
    for name, p in model.named_parameters():
        total_params += 1
        if torch.isnan(p).any():
            n_nan_params += 1
            if len(nan_param_names) < 10:
                nan_param_names.append(name)
        if torch.isinf(p).any():
            n_inf_params += 1
    print(f"  {label}: {n_nan_params}/{total_params} params have NaN, {n_inf_params} have Inf")
    if nan_param_names:
        print(f"    First NaN params: {nan_param_names}")
    return n_nan_params


def trace_nan_through_model(model, input_ids, attention_mask, label):
    """Run forward pass with hooks, find first module with NaN output."""
    hooks = []
    results = []

    def make_hook(name):
        def hook(module, input, output):
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor) and torch.isnan(inp).any():
                    results.append((name, "INPUT", i, True))
            if isinstance(output, torch.Tensor):
                has_nan = torch.isnan(output).any().item()
                if has_nan:
                    results.append((name, "OUTPUT", 0, True))
            elif isinstance(output, (tuple, list)):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                        results.append((name, "OUTPUT", i, True))
        return hook

    for name, module in model.named_modules():
        h = module.register_forward_hook(make_hook(name))
        hooks.append(h)

    print(f"\n  Tracing NaN through {label}...")
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        except Exception as e:
            print(f"  Forward pass error: {e}")
            # Try alternative call signature
            try:
                out = model(input_ids=input_ids)
            except Exception as e2:
                print(f"  Alternative call also failed: {e2}")
                return
        finally:
            for h in hooks:
                h.remove()

    if results:
        print(f"  First module with NaN: {results[0][0]} ({results[0][1]}[{results[0][2]}])")
        print(f"  Total modules with NaN: {len(results)}")
        print(f"  First 10:")
        for name, direction, idx, _ in results[:10]:
            print(f"    {direction}[{idx}] {name}")
    else:
        print(f"  No NaN detected in any module output!")
        if isinstance(out, torch.Tensor):
            print(f"  Final output: shape={out.shape}, NaN={torch.isnan(out).any().item()}")
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    print(f"  Output[{i}]: shape={o.shape}, NaN={torch.isnan(o).any().item()}")


# ════════════════════════════════════════════════════════════════════════
# Part 1: Check stella model weights
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Part 1: Check stella model weights")
print("=" * 70)

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()

print(f"\n  Model type: {type(m)}")
print(f"  Has attributes: {[a for a in dir(m) if not a.startswith('_')][:20]}")

# Get the underlying transformer
if hasattr(m, '_first_module') and hasattr(m._first_module, 'auto_model'):
    tm = m._first_module.auto_model
    print(f"  Transformer model: {type(tm).__name__}")
elif hasattr(m, 'model'):
    tm = m.model
    print(f"  Transformer model: {type(tm).__name__}")
else:
    print(f"  Model structure: {m}")
    # Try to find the transformer
    for name, module in m.named_modules():
        if 'transformer' in name.lower() or 'encoder' in name.lower():
            print(f"    Found: {name} → {type(module).__name__}")
    tm = None

if tm is not None:
    print(f"\n  Checking fp32 weights:")
    check_weights_for_nan(tm, "fp32 weights")

    print(f"\n  Converting to fp16...")
    m.half()
    check_weights_for_nan(tm, "fp16 weights")

# ════════════════════════════════════════════════════════════════════════
# Part 2: Tokenize and trace forward pass
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 2: Tokenize and trace forward pass")
print("=" * 70)

test_text = "This is a test sentence for embedding."
print(f"\n  Test text: '{test_text}'")

# Tokenize
if hasattr(m, 'tokenizer'):
    tok = m.tokenizer
elif hasattr(m, '_first_module') and hasattr(m._first_module, 'tokenizer'):
    tok = m._first_module.tokenizer
else:
    print("  Cannot find tokenizer")
    tok = AutoTokenizer.from_pretrained("NovaSearch/stella_en_400M_v5", trust_remote_code=True)

tok_res = tok([test_text], padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attention_mask = tok_res["attention_mask"].to("npu")
print(f"  Input IDs: shape={input_ids.shape}, dtype={input_ids.dtype}")
print(f"  Input IDs: {input_ids[0][:20]}")
print(f"  Attention mask: {attention_mask[0][:20]}")
print(f"  Input IDs NaN: {torch.isnan(input_ids.float()).any().item()}")
print(f"  Attn mask NaN: {torch.isnan(attention_mask.float()).any().item()}")

if tm is not None:
    # Move inputs to model dtype
    input_ids_m = input_ids.to(tm.dtype) if tm.dtype != torch.int64 else input_ids
    trace_nan_through_model(tm, input_ids, attention_mask, "stella transformer")

# ════════════════════════════════════════════════════════════════════════
# Part 3: Compare with a standard (non-custom) model
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 3: Compare with standard model (all-MiniLM-L6-v2)")
print("=" * 70)

try:
    m2 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="npu")
    m2.eval()
    m2.half()
    m2.max_seq_length = 512
    emb2 = m2.encode([test_text, "Another test sentence."], batch_size=2,
                     show_progress_bar=False, normalize_embeddings=True)
    emb2 = np.array(emb2, dtype=np.float32)
    n_nan2 = int(np.isnan(emb2).any(axis=1).sum())
    print(f"  all-MiniLM-L6-v2: NaN={n_nan2}/2 ({n_nan2/2*100:.1f}%)")
    print(f"  This model uses: {type(m2._first_module.auto_model).__name__}")
    print(f"  (non-custom, no trust_remote_code)")
except Exception as e:
    print(f"  Standard model test failed: {e}")

# ════════════════════════════════════════════════════════════════════════
# Part 4: Check stella model config and custom code
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 4: Stella model config")
print("=" * 70)

if tm is not None:
    config = tm.config if hasattr(tm, 'config') else None
    if config:
        print(f"  Config type: {type(config).__name__}")
        for attr in ['model_type', 'hidden_size', 'num_hidden_layers',
                      'num_attention_heads', 'hidden_act', 'layer_norm_eps',
                      'attention_implementation', 'unpad_inputs',
                      'use_memory_efficient_attention']:
            val = getattr(config, attr, "N/A")
            print(f"    {attr}: {val}")

    # Check what modules the model has
    print(f"\n  Top-level modules:")
    for name, module in tm.named_children():
        n_params = sum(p.numel() for p in module.parameters())
        print(f"    {name}: {type(module).__name__} ({n_params} params)")

    # Check for any custom/npu-specific operations
    print(f"\n  Looking for non-standard layers:")
    for name, module in tm.named_modules():
        mod_type = type(module).__name__
        if mod_type not in ['BertModel', 'BertEncoder', 'BertLayer',
                           'BertEmbeddings', 'BertSelfAttention', 'BertSelfOutput',
                           'BertAttention', 'BertIntermediate', 'BertOutput',
                           'BertPooler', 'ModuleList', 'Sequential', 'Linear',
                           'LayerNorm', 'Dropout', 'Embedding', 'Tanh',
                           'Module', 'NewAttention', 'NewModel']:
            print(f"    {name}: {mod_type}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
