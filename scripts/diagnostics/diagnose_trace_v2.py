#!/usr/bin/env python3
"""Trace NaN through stella model — find the exact layer where NaN first appears.

Key finding: standard BERT (all-MiniLM) = 0% NaN, stella custom = 100% NaN.
Problem is in stella's custom NewModel/NewAttention/NewGatedMLP code.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings, time
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # sets up fake xformers

import torch
import torch_npu
import numpy as np

def check_tensor(t, label):
    if not isinstance(t, torch.Tensor):
        print(f"    {label}: not a tensor ({type(t)})")
        return False
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    n_nan = torch.isnan(t).sum().item()
    n_elem = t.numel()
    print(f"    {label:60s}: shape={list(t.shape)} dtype={t.dtype} NaN={n_nan}/{n_elem} Inf={has_inf}")
    return has_nan

# ════════════════════════════════════════════════════════════════════════
# Load model and find the transformer
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("Loading stella model")
print("=" * 70)

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()

# Access the transformer module
print(f"\nSentenceTransformer modules: {list(m._modules.keys())}")
mod0 = m[0]
print(f"Module[0] type: {type(mod0).__name__}")
print(f"Module[0] attrs: {[a for a in dir(mod0) if not a.startswith('_') and 'model' in a.lower()]}")

# Try different ways to get the underlying HF model
tm = None
for attr in ['auto_model', 'model', '_model']:
    if hasattr(mod0, attr):
        val = getattr(mod0, attr)
        if isinstance(val, torch.nn.Module):
            tm = val
            print(f"  Found transformer via mod0.{attr}: {type(tm).__name__}")
            break

if tm is None:
    # Iterate children
    for name, child in mod0.named_children():
        if isinstance(child, torch.nn.Module) and hasattr(child, 'forward'):
            tm = child
            print(f"  Found transformer via named_children: {name} → {type(tm).__name__}")
            break

if tm is None:
    print("  ERROR: Cannot find transformer model")
    sys.exit(1)

# ════════════════════════════════════════════════════════════════════════
# Part 1: Check weights
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 1: Check weights for NaN/Inf")
print("=" * 70)

def check_weights(model, label):
    n_nan, n_inf, total = 0, 0, 0
    nan_names = []
    for name, p in model.named_parameters():
        total += 1
        if torch.isnan(p).any():
            n_nan += 1
            if len(nan_names) < 5:
                nan_names.append(name)
        if torch.isinf(p).any():
            n_inf += 1
    print(f"  {label}: {n_nan}/{total} NaN, {n_inf}/{total} Inf")
    if nan_names:
        print(f"    First NaN params: {nan_names}")
    return n_nan

check_weights(tm, "fp32 weights")
m.half()
check_weights(tm, "fp16 weights")

# Print config
config = getattr(tm, 'config', None)
if config:
    print(f"\n  Config: {type(config).__name__}")
    for attr in ['model_type', 'hidden_size', 'num_hidden_layers',
                  'num_attention_heads', 'intermediate_size', 'hidden_act',
                  'layer_norm_eps', 'attention_implementation',
                  'unpad_inputs', 'use_memory_efficient_attention',
                  'pad_token_id', 'vocab_size']:
        print(f"    {attr}: {getattr(config, attr, 'N/A')}")

# ════════════════════════════════════════════════════════════════════════
# Part 2: Tokenize and trace forward pass layer by layer
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 2: Trace forward pass")
print("=" * 70)

test_texts = ["This is a test sentence.", "Another sentence here."]

# Tokenize
tok = m.tokenizer if hasattr(m, 'tokenizer') else mod0.tokenizer
tok_res = tok(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attention_mask = tok_res["attention_mask"].to("npu")
print(f"  Input IDs: shape={input_ids.shape}")
print(f"  Attention mask: {attention_mask}")

# Check if unpad_inputs is active
if hasattr(tm, 'unpad_inputs'):
    print(f"  unpad_inputs: {tm.unpad_inputs}")
if hasattr(config, 'unpad_inputs'):
    print(f"  config.unpad_inputs: {config.unpad_inputs}")

# Check can_flatten_inputs (sentence-transformers v5+)
if hasattr(mod0, 'can_flatten_inputs'):
    print(f"  mod0.can_flatten_inputs: {mod0.can_flatten_inputs}")
if hasattr(mod0, '_can_flatten_inputs'):
    try:
        cfi = mod0._can_flatten_inputs()
        print(f"  mod0._can_flatten_inputs(): {cfi}")
    except Exception as e:
        print(f"  mod0._can_flatten_inputs(): ERROR {e}")

# Trace with hooks
print(f"\n  Running forward pass with hooks...")
hooks = []
nan_results = []

def make_hook(name):
    def hook(module, inp, out):
        # Check inputs
        for i, x in enumerate(inp):
            if isinstance(x, torch.Tensor) and torch.isnan(x).any():
                nan_results.append(f"INPUT[{i}] {name}")
        # Check output
        if isinstance(out, torch.Tensor):
            if torch.isnan(out).any():
                nan_results.append(f"OUTPUT {name}")
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor) and torch.isnan(o).any():
                    nan_results.append(f"OUTPUT[{i}] {name}")
        elif isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    nan_results.append(f"OUTPUT['{k}'] {name}")
    return hook

# Register on all submodules
for name, module in tm.named_modules():
    hooks.append(module.register_forward_hook(make_hook(name)))

with torch.no_grad():
    try:
        out = tm(input_ids=input_ids, attention_mask=attention_mask)
        print(f"  Forward completed. Output type: {type(out)}")
        if isinstance(out, torch.Tensor):
            check_tensor(out, "final output")
        elif isinstance(out, (tuple, list)):
            for i, o in enumerate(out):
                if isinstance(o, torch.Tensor):
                    check_tensor(o, f"output[{i}]")
        elif isinstance(out, dict):
            for k, v in out.items():
                if isinstance(v, torch.Tensor):
                    check_tensor(v, f"output['{k}']")
    except Exception as e:
        print(f"  Forward error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for h in hooks:
            h.remove()

if nan_results:
    print(f"\n  NaN found in {len(nan_results)} module hooks:")
    # Show first occurrence (deduplicated)
    seen = set()
    for r in nan_results:
        mod_name = r.split("] ")[-1] if "] " in r else r
        if mod_name not in seen:
            seen.add(mod_name)
            print(f"    {r}")
else:
    print(f"\n  No NaN found in any module hooks!")

# ════════════════════════════════════════════════════════════════════════
# Part 3: Manually trace layer 0 forward
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 3: Manual trace of layer 0")
print("=" * 70)

# Get embeddings first
print("  Getting embeddings...")
embeddings = tm.embeddings if hasattr(tm, 'embeddings') else None
if embeddings is None:
    for name, child in tm.named_children():
        if 'embed' in name.lower():
            embeddings = child
            print(f"  Found embeddings: {name}")
            break

if embeddings is not None:
    with torch.no_grad():
        emb_out = embeddings(input_ids=input_ids)
        if isinstance(emb_out, torch.Tensor):
            check_tensor(emb_out, "embeddings output")
        elif isinstance(emb_out, (tuple, list)):
            emb_out = emb_out[0]
            check_tensor(emb_out, "embeddings output[0]")
        
        # Get layer 0
        layer0 = tm.encoder.layer[0] if hasattr(tm, 'encoder') else None
        if layer0 is None:
            for name, child in tm.named_children():
                if hasattr(child, 'layer'):
                    layer0 = child.layer[0]
                    print(f"  Found layer0 via {name}")
                    break
        
        if layer0 is not None:
            print(f"\n  Layer 0 type: {type(layer0).__name__}")
            print(f"  Layer 0 children: {[n for n, _ in layer0.named_children()]}")
            
            # Run layer 0
            if isinstance(emb_out, torch.Tensor):
                print(f"  Running layer 0 with emb_out...")
                layer_out = layer0(emb_out, attention_mask=attention_mask)
                if isinstance(layer_out, torch.Tensor):
                    check_tensor(layer_out, "layer0 output")
                elif isinstance(layer_out, (tuple, list)):
                    for i, o in enumerate(layer_out):
                        if isinstance(o, torch.Tensor):
                            check_tensor(o, f"layer0 output[{i}]")
                
                # Trace attention specifically
                attn = layer0.attention if hasattr(layer0, 'attention') else None
                if attn is None:
                    for name, child in layer0.named_children():
                        if 'att' in name.lower():
                            attn = child
                            break
                
                if attn is not None:
                    print(f"\n  Attention type: {type(attn).__name__}")
                    print(f"  Attention attrs: {[a for a in dir(attn) if not a.startswith('_') and not callable(getattr(attn, a, None))][:20]}")
                    if hasattr(attn, 'unpad_inputs'):
                        print(f"  attn.unpad_inputs: {attn.unpad_inputs}")
                    if hasattr(attn, 'use_memory_efficient_attention'):
                        print(f"  attn.use_memory_efficient_attention: {attn.use_memory_efficient_attention}")
                    
                    # Run attention
                    print(f"  Running attention...")
                    try:
                        attn_out = attn(emb_out, attention_mask=attention_mask)
                        if isinstance(attn_out, torch.Tensor):
                            check_tensor(attn_out, "attention output")
                        elif isinstance(attn_out, (tuple, list)):
                            for i, o in enumerate(attn_out):
                                if isinstance(o, torch.Tensor):
                                    check_tensor(o, f"attn output[{i}]")
                    except Exception as e:
                        print(f"  Attention error: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Check QKV projection
                    if hasattr(attn, 'qkv_proj'):
                        print(f"\n  Running qkv_proj...")
                        qkv = attn.qkv_proj(emb_out)
                        check_tensor(qkv, "qkv_proj output")
                        if not torch.isnan(qkv).any():
                            # Split QKV
                            q, k, v = attn.separate_qkv(qkv) if hasattr(attn, 'separate_qkv') else (qkv, qkv, qkv)
                            check_tensor(q, "Q")
                            check_tensor(k, "K")
                            check_tensor(v, "V")
                
                # Trace MLP
                mlp = layer0.mlp if hasattr(layer0, 'mlp') else None
                if mlp is None:
                    for name, child in layer0.named_children():
                        if 'mlp' in name.lower():
                            mlp = child
                            break
                
                if mlp is not None:
                    print(f"\n  MLP type: {type(mlp).__name__}")
                    print(f"  Running MLP on emb_out...")
                    try:
                        mlp_out = mlp(emb_out)
                        if isinstance(mlp_out, torch.Tensor):
                            check_tensor(mlp_out, "MLP output")
                        elif isinstance(mlp_out, (tuple, list)):
                            for i, o in enumerate(mlp_out):
                                if isinstance(o, torch.Tensor):
                                    check_tensor(o, f"mlp output[{i}]")
                    except Exception as e:
                        print(f"  MLP error: {e}")
                        import traceback
                        traceback.print_exc()
                    
                    # Check MLP components
                    if hasattr(mlp, 'up_gate_proj'):
                        print(f"  Running up_gate_proj...")
                        up_gate = mlp.up_gate_proj(emb_out)
                        check_tensor(up_gate, "up_gate_proj output")
                        if hasattr(mlp, 'act_fn'):
                            print(f"  Running act_fn (GELU) on up_gate...")
                            # up_gate is (B, S, 2*hidden), split into up and gate
                            if up_gate.shape[-1] % 2 == 0:
                                up, gate = up_gate.chunk(2, dim=-1)
                                check_tensor(up, "up")
                                check_tensor(gate, "gate")
                                gated = mlp.act_fn(up) * gate if hasattr(mlp, 'act_fn') else up * gate
                                check_tensor(gated, "gated (act(up) * gate)")
                                if hasattr(mlp, 'down_proj'):
                                    down = mlp.down_proj(gated)
                                    check_tensor(down, "down_proj output")

# ════════════════════════════════════════════════════════════════════════
# Part 4: Check if fake xformers is even called
# ════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Part 4: Is fake xformers actually called?")
print("=" * 70)

# Monkey-patch the fake xformers to trace calls
import xformers.ops as _xf_ops
_orig_mea = _xf_ops.memory_efficient_attention
_call_count = [0]
_call_args = []

def _traced_mea(q, k, v, attn_bias=None, p=0.0, **kw):
    _call_count[0] += 1
    if _call_count[0] <= 3:
        info = {
            'q_shape': list(q.shape),
            'q_dtype': str(q.dtype),
            'attn_bias_type': type(attn_bias).__name__,
            'device': str(q.device),
        }
        _call_args.append(info)
        print(f"  [TRACE] xformers call #{_call_count[0]}: {info}")
        if isinstance(attn_bias, type(None)):
            print(f"    attn_bias is None")
    result = _orig_mea(q, k, v, attn_bias=attn_bias, p=p, **kw)
    if _call_count[0] <= 3:
        has_nan = torch.isnan(result).any().item()
        print(f"    result: NaN={has_nan}, shape={list(result.shape)}")
    return result

_xf_ops.memory_efficient_attention = _traced_mea
# Also patch the fmha version
import xformers.ops.fmha as _xf_fmha
_xf_fmha.memory_efficient_attention = _traced_mea

# Run full encode
print("  Running m.encode() with traced xformers...")
_call_count[0] = 0
emb = m.encode(test_texts, batch_size=2, show_progress_bar=False, normalize_embeddings=True)
emb_arr = np.array(emb, dtype=np.float32)
n_nan = int(np.isnan(emb_arr).any(axis=1).sum())
print(f"  xformers was called {_call_count[0]} times")
print(f"  Final embeddings: NaN={n_nan}/{len(emb_arr)}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
