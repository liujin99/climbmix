#!/usr/bin/env python3
"""Profile stella forward pass + try torch.compile + inspect modeling.py.

Usage:
    python scripts/profile_stella.py
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import time
import types
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as _F
import torch_npu

# ============================================================
# Fake xformers — exact copy from embedding_cluster.py
# ============================================================
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

_attn_call_log = {"count": 0, "attn_biases": [], "shapes": []}

def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, **kw):
    _attn_call_log["count"] += 1
    _attn_call_log["attn_biases"].append(type(attn_bias).__name__)
    _attn_call_log["shapes"].append(tuple(q.shape))

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

        if len(set(qs_list)) == 1 and len(set(ks_list)) == 1:
            s = qs_list[0]
            ks = ks_list[0]
            q_b = q.view(n, s, H, D).transpose(1, 2)
            k_b = k.view(n, ks, H, D).transpose(1, 2)
            v_b = v.view(n, ks, H, D).transpose(1, 2)
            out = _F.scaled_dot_product_attention(q_b, k_b, v_b, dropout_p=p)
            return out.transpose(1, 2).reshape(1, -1, H, D).contiguous()
        else:
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
            out = _F.scaled_dot_product_attention(
                q_pad.transpose(1, 2), k_pad.transpose(1, 2), v_pad.transpose(1, 2),
                attn_mask=kmask, dropout_p=p
            )
            out = out.transpose(1, 2)
            chunks = [out[i, :qs_i].unsqueeze(0) for i, qs_i in enumerate(qs_list)]
            return torch.cat(chunks, dim=1)

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
# Load model
# ============================================================
print("Loading stella model...")
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()
print("Model loaded.\n")

# ============================================================
# Create realistic input: 512 texts, ~200 tokens each (with padding to 512)
# ============================================================
texts = []
for i in range(512):
    n_words = 80 + (i * 13) % 320  # ~80-400 words ≈ 100-500 tokens
    texts.append(" ".join([f"word{j}" for j in range(n_words)]))

features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

real_tokens = features["attention_mask"].sum().item()
total_tokens = features["attention_mask"].numel()
print(f"Input: batch={features['input_ids'].shape[0]}, seq={features['input_ids'].shape[1]}")
print(f"Real tokens: {real_tokens}/{total_tokens} ({real_tokens/total_tokens*100:.1f}%)")

# ============================================================
# Check attention path (which BlockDiagonalMask path is taken?)
# ============================================================
print("\n=== Attention path check ===")
_attn_call_log["count"] = 0
_attn_call_log["attn_biases"] = []
_attn_call_log["shapes"] = []

with torch.no_grad():
    try:
        output = model(features)
    except Exception as e:
        print(f"Forward failed: {e}")

print(f"memory_efficient_attention calls: {_attn_call_log['count']}")
for i, (bias_type, shape) in enumerate(zip(_attn_call_log["attn_biases"], _attn_call_log["shapes"])):
    print(f"  call {i}: attn_bias={bias_type}, q_shape={shape}")

# Check if BlockDiagonalMask had variable lengths
# If q shape is (1, total_S, H, D), it's the packed path
for shape in _attn_call_log["shapes"]:
    if shape[0] == 1:
        total_s = shape[1]
        h = shape[2]
        d = shape[3]
        print(f"\nPacked path detected: total_S={total_s}, H={h}, D={d}")
        print(f"  Average seq per doc: {total_s/512:.1f} tokens")

# ============================================================
# Baseline timing
# ============================================================
print("\n=== Baseline timing ===")
_attn_call_log["count"] = 0
_attn_call_log["attn_biases"] = []
_attn_call_log["shapes"] = []

with torch.no_grad():
    for _ in range(3):
        _ = model(features)
torch.npu.synchronize()

N = 10
torch.npu.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        output = model(features)
torch.npu.synchronize()
t_baseline = (time.time() - t0) / N * 1000
print(f"Baseline: {t_baseline:.1f} ms/batch ({512*1000/t_baseline:.0f} docs/s)")

# ============================================================
# torch.profiler
# ============================================================
print("\n=== torch.profiler ===")
try:
    from torch.profiler import profile, ProfilerActivity
    activities = [ProfilerActivity.CPU]
    try:
        activities.append(ProfilerActivity.NPU)
        npu_avail = True
    except (KeyError, AttributeError):
        npu_avail = False
        print("NPU profiler activity not available, using CPU only")

    with profile(activities=activities, record_shapes=True) as prof:
        with torch.no_grad():
            for _ in range(5):
                _ = model(features)

    sort_key = "self_npu_time_total" if npu_avail else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))
except Exception as e:
    print(f"Profiler failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# torch.compile() attempts
# ============================================================
print("\n=== torch.compile() ===")

try:
    import torch._dynamo
    backends = torch._dynamo.list_backends()
    print(f"Available dynamo backends: {backends}")
except:
    print("Cannot list dynamo backends")

print(f"torch_npu._compiler attrs: {[x for x in dir(torch_npu._compiler) if not x.startswith('__')]}")

for mode_name in ["reduce-overhead", "default"]:
    print(f"\n--- torch.compile(mode='{mode_name}') ---")
    try:
        model2 = SentenceTransformer(
            "NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True
        )
        model2.eval()
        model2.max_seq_length = 512
        model2.half()

        model2[0].auto_model = torch.compile(model2[0].auto_model, mode=mode_name)

        print("  Compiling + warmup...")
        with torch.no_grad():
            for _ in range(5):
                _ = model2(features)
        torch.npu.synchronize()

        t0 = time.time()
        with torch.no_grad():
            for _ in range(N):
                _ = model2(features)
        torch.npu.synchronize()
        t_compile = (time.time() - t0) / N * 1000
        print(f"  {t_compile:.1f} ms/batch ({512*1000/t_compile:.0f} docs/s)")
        print(f"  Speedup: {t_baseline/t_compile:.2f}x")

        del model2
        torch.npu.empty_cache()
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# Read stella modeling.py — find unpad/attention logic
# ============================================================
print("\n=== stella modeling.py — attention/unpad code ===")
import pathlib
modeling_base = pathlib.Path.home() / ".cache/huggingface/modules/transformers_modules/NovaSearch/stella_en_400M_v5"
for p in sorted(modeling_base.rglob("modeling.py")):
    print(f"\nFile: {p}")
    with open(p) as f:
        lines = f.readlines()

    keywords = [
        "memory_efficient_attention", "BlockDiagonalMask", "from_seqlens",
        "unpad", "nonzero", "NonZero", "attention_mask_bool",
        "def forward", "class Bert", "class Stella", "def embed",
    ]
    seen_ranges = set()
    for i, line in enumerate(lines):
        for kw in keywords:
            if kw.lower() in line.lower():
                start = max(0, i - 3)
                end = min(len(lines), i + 12)
                range_key = (start, end)
                if range_key in seen_ranges:
                    break
                seen_ranges.add(range_key)
                print(f"\n--- line {i+1} (matched: '{kw}') ---")
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j].rstrip()}")
                break
    break

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline: {t_baseline:.1f} ms/batch ({512*1000/t_baseline:.0f} docs/s)")
print(f"Attention calls per forward: {_attn_call_log['count']}")
print(f"Attention path: {_attn_call_log['attn_biases'][:3]}")
print(f"Attention shapes: {_attn_call_log['shapes'][:3]}")
