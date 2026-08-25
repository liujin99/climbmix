#!/usr/bin/env python3
"""Minimal embedding test — called by diagnose_env_vars.sh with different env vars.

Loads model, encodes 200 docs, reports NaN count. Designed to be fast (~30s).
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import climbmix.core.embedding_cluster  # sets up fake xformers
import torch, torch_npu, numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")

# Load 200 texts from first parquet
pf = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])[:1]
texts = []
for fname in pf:
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["text"])
    texts.extend([str(t) if t is not None else "" for t in table.column("text").to_pylist()[:300]])

rng = np.random.default_rng(42)
n = min(200, len(texts))
idx = rng.choice(len(texts), size=n, replace=False)
sample = [texts[i] for i in idx]

from sentence_transformers import SentenceTransformer
m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

emb = m.encode(sample, batch_size=512, show_progress_bar=False, normalize_embeddings=True)
emb = np.array(emb, dtype=np.float32)
n_nan = int(np.isnan(emb).any(axis=1).sum())
pct = n_nan / n * 100
status = "PASS" if n_nan == 0 else "FAIL"
print(f"RESULT: NaN={n_nan}/{n} ({pct:.1f}%) [{status}]")
