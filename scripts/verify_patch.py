#!/usr/bin/env python3
"""Quick end-to-end verification: patched embedding_cluster.py forward pass.

Usage:
    python scripts/verify_patch.py
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import time
import warnings
warnings.filterwarnings("ignore")

# This import triggers the fake xformers + npu_fusion_attention patch
from climbmix.core.embedding_cluster import _HAS_NPU_FA
print(f"_HAS_NPU_FA: {_HAS_NPU_FA}")

import torch
import torch_npu
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

texts = [" ".join([f"word{j}" for j in range(80 + (i*13)%320)]) for i in range(512)]
features = model.tokenize(texts)
for k in list(features.keys()):
    if isinstance(features[k], torch.Tensor):
        features[k] = features[k].to("npu")

print("Warmup...")
with torch.no_grad():
    for _ in range(3):
        output = model(features)
torch.npu.synchronize()

emb = output["sentence_embedding"]
print(f"Output: shape={emb.shape}, dtype={emb.dtype}")
print(f"  norm sample: {emb[0].norm().item():.4f}, {emb[1].norm().item():.4f}")

N = 10
torch.npu.synchronize()
t0 = time.time()
with torch.no_grad():
    for _ in range(N):
        output = model(features)
torch.npu.synchronize()
t = (time.time() - t0) / N * 1000
print(f"\nPatched: {t:.1f} ms/batch ({512*1000/t:.0f} docs/s per NPU)")
print(f"Previous: 3540 ms/batch (143 docs/s)")
print(f"Speedup: {3540/t:.2f}x")
print(f"8-NPU projected: {512*1000/t*8:.0f} docs/s")
