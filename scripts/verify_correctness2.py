#!/usr/bin/env python3
"""Minimal correctness test: verify npu_fusion_attention produces valid embeddings.

Uses the EXACT same fake xformers from embedding_cluster.py (production code).
Tests with both small and large batches to isolate the NaN issue.
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

# Import the EXACT fake xformers from production code
import climbmix.core.embedding_cluster  # This sets up sys.modules["xformers"]

import torch
import torch_npu
import numpy as np

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
model.eval()
model.max_seq_length = 512
model.half()

# Check model config
auto_model = model[0].auto_model
encoder = auto_model.encoder
attn = encoder.layer[0].attention
print(f"use_memory_efficient_attention: {attn.use_memory_efficient_attention}")
print(f"memory_efficient_attention is None: {attn.memory_efficient_attention is None}")
print(f"unpad_inputs: {attn.config.unpad_inputs}")

# Test 1: Small batch (8 texts)
print("\n" + "=" * 60)
print("Test 1: Small batch (8 texts)")
print("=" * 60)
small_texts = [
    "Machine learning is a subset of artificial intelligence. " * 10,
    "The cat sat on the mat in the sun. " * 10,
    "Quantum mechanics deals with subatomic particles. " * 10,
    "Python is a popular programming language. " * 10,
    "The Renaissance was a period of cultural rebirth. " * 10,
    "DNA contains the genetic instructions for life. " * 10,
    "The Earth orbits the Sun once every 365 days. " * 10,
    "Economics studies the production and distribution of wealth. " * 10,
]

emb_small = model.encode(small_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
emb_small = np.array(emb_small, dtype=np.float32)
print(f"  Shape: {emb_small.shape}")
print(f"  NaN count: {np.isnan(emb_small).sum()}")
print(f"  Min: {emb_small.min():.6f}, Max: {emb_small.max():.6f}")
if np.isnan(emb_small).any():
    print("  FAILED: NaN detected!")
else:
    print("  PASS: No NaN")

# Test 2: Medium batch (64 texts)
print("\n" + "=" * 60)
print("Test 2: Medium batch (64 texts)")
print("=" * 60)
medium_texts = [f"Document {i} about topic {i%8}. " * 20 for i in range(64)]

emb_med = model.encode(medium_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
emb_med = np.array(emb_med, dtype=np.float32)
print(f"  Shape: {emb_med.shape}")
print(f"  NaN count: {np.isnan(emb_med).sum()}")
if np.isnan(emb_med).any():
    print("  FAILED: NaN detected!")
else:
    print("  PASS: No NaN")

# Test 3: Large batch (512 texts, production-like)
print("\n" + "=" * 60)
print("Test 3: Large batch (512 texts, production-like)")
print("=" * 60)
large_texts = [f"word{(i*7)%500} " * (50 + (i*37)%400) for i in range(512)]

emb_large = model.encode(large_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb_large = np.array(emb_large, dtype=np.float32)
print(f"  Shape: {emb_large.shape}")
print(f"  NaN count: {np.isnan(emb_large).sum()}")
if np.isnan(emb_large).any():
    print("  FAILED: NaN detected!")
else:
    print(f"  PASS: No NaN")
    print(f"  Embedding norm (first 5): {np.linalg.norm(emb_large[:5], axis=1)}")

# Test 4: Compare first 8 texts between small and large batch
if not np.isnan(emb_small).any() and not np.isnan(emb_large).any():
    print("\n" + "=" * 60)
    print("Test 4: Consistency check (small vs large batch)")
    print("=" * 60)
    # Use same 8 texts in large batch
    large_texts_8 = large_texts[:8]
    emb_large_8 = model.encode(large_texts_8, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
    emb_large_8 = np.array(emb_large_8, dtype=np.float32)

    if np.isnan(emb_large_8).any():
        print("  FAILED: NaN in large batch with 8 texts")
    else:
        # Compare embeddings of the same texts in different batch sizes
        # Use the same texts for both
        same_texts = [f"word{(j*13)%500} " * 200 for j in range(8)]
        emb_a = model.encode(same_texts, batch_size=8, show_progress_bar=False, normalize_embeddings=True)
        emb_a = np.array(emb_a, dtype=np.float32)
        emb_b = model.encode(same_texts, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
        emb_b = np.array(emb_b, dtype=np.float32)

        if np.isnan(emb_a).any() or np.isnan(emb_b).any():
            print(f"  NaN: batch_8={np.isnan(emb_a).sum()}, batch_512={np.isnan(emb_b).sum()}")
        else:
            diff = np.abs(emb_a - emb_b)
            cos_sim = np.sum(emb_a * emb_b, axis=1) / (np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1))
            print(f"  Max abs diff:  {diff.max():.6f}")
            print(f"  Mean abs diff: {diff.mean():.6f}")
            print(f"  Cosine sim:    min={cos_sim.min():.6f}, mean={cos_sim.mean():.6f}")
            if cos_sim.min() > 0.999:
                print("  PASS: Batch size doesn't affect embeddings")
            else:
                print("  WARNING: Batch size affects embeddings")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
