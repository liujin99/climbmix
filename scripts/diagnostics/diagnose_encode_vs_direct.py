#!/usr/bin/env python3
"""找出 model.encode() 和直接调用 tm() 产生不同 NaN 结果的原因。

已知:
- tm(input_ids, attention_mask) bs=1-2 → 0% NaN
- tm(input_ids, attention_mask) bs=8 → NaN
- m.encode(texts, bs=2) → 100% NaN

目标:
1. 找出 model.encode() 与直接调用的差异 (prompt? token_type_ids? 其他 kwargs?)
2. 找到 batch size 的精确阈值
3. 检查模型内部 attention 配置
"""
import os
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys, warnings
warnings.filterwarnings("ignore")

import climbmix.core.embedding_cluster  # 设置 fake xformers

import torch
import torch_npu
import numpy as np
import pyarrow.parquet as pq

DATA_DIR = os.environ.get("DATA_DIR", "/home/ma-user/work/100B_stem_parquet_filtered")

# 加载文本
pf = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".parquet")])[:1]
texts = []
for fname in pf:
    table = pq.read_table(os.path.join(DATA_DIR, fname), columns=["text"])
    texts.extend([str(t) if t is not None else "" for t in table.column("text").to_pylist()[:300]])

print(f"Loaded {len(texts)} texts")

from sentence_transformers import SentenceTransformer

m = SentenceTransformer("NovaSearch/stella_en_400M_v5", device="npu", trust_remote_code=True)
m.eval()
m.half()
m.max_seq_length = 512

tok = m.tokenizer
tm = m[0].auto_model

def check_tensor(t, label):
    if not isinstance(t, torch.Tensor):
        print(f"  {label:55s}: not a tensor ({type(t)})")
        return False
    n_nan = int(torch.isnan(t).sum().item())
    n_elem = t.numel()
    pct = n_nan / n_elem * 100 if n_elem > 0 else 0
    has_inf = torch.isinf(t).any().item()
    print(f"  {label:55s}: NaN={n_nan}/{n_elem} ({pct:.1f}%) Inf={has_inf} shape={list(t.shape)} dtype={t.dtype}")
    return n_nan > 0

def check_nan(emb, label):
    emb = np.array(emb, dtype=np.float32)
    n_nan = int(np.isnan(emb).any(axis=1).sum())
    pct = n_nan / len(emb) * 100
    status = "PASS" if n_nan == 0 else "FAIL"
    print(f"  {label:55s}: NaN={n_nan:4d}/{len(emb)} ({pct:5.1f}%) [{status}]")
    return n_nan

# ════════════════════════════════════════════════════════════════════════
# Part 0: 检查 prompt 配置
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 0: Prompt 配置检查")
print(f"{'='*70}")
print(f"  m.prompts: {m.prompts}")
print(f"  m.default_prompt_name: {m.default_prompt_name}")

# 检查 ST 模块的 forward_kwargs 和 model_forward_params
if hasattr(m[0], 'model_forward_params'):
    print(f"  m[0].model_forward_params: {m[0].model_forward_params}")
if hasattr(m[0], 'forward_kwargs'):
    print(f"  m[0].forward_kwargs: {getattr(m[0], 'forward_kwargs', 'N/A')}")
if hasattr(m[0], 'can_flatten_inputs'):
    print(f"  m[0].can_flatten_inputs: {m[0].can_flatten_inputs}")
if hasattr(m[0], 'unpad_inputs'):
    print(f"  m[0].unpad_inputs: {m[0].unpad_inputs}")

# 检查模型 config
print(f"\n  tm.config.unpad_inputs: {getattr(tm.config, 'unpad_inputs', 'N/A')}")
print(f"  tm.config.use_memory_efficient_attention: {getattr(tm.config, 'use_memory_efficient_attention', 'N/A')}")
print(f"  tm.config._attn_implementation: {getattr(tm.config, '_attn_implementation', 'N/A')}")

# 检查 attention 类型
for name, module in tm.named_modules():
    if 'attention' in name.lower() and 'NewAttention' in type(module).__name__:
        print(f"\n  {name}: {type(module).__name__}")
        print(f"    use_memory_efficient_attention: {module.use_memory_efficient_attention}")
        print(f"    memory_efficient_attention: {module.memory_efficient_attention}")
        print(f"    config.unpad_inputs: {module.config.unpad_inputs}")
        break

# ════════════════════════════════════════════════════════════════════════
# Part 1: 2 篇真实文档 — model.encode() vs tm() 直接调用
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 1: 2 篇真实文档 — encode() vs tm() 直接调用")
print(f"{'='*70}")

test_texts = texts[:2]

# 1a: model.encode()
print("\n  [1a] model.encode(texts[:2], bs=2):")
emb_encode = m.encode(test_texts, batch_size=2, show_progress_bar=False, normalize_embeddings=True)
check_nan(emb_encode, "encode() result")

# 1b: 手动 tokenize + tm()
print("\n  [1b] 手动 tokenize + tm():")
tok_res = tok(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = tok_res["input_ids"].to("npu")
attention_mask = tok_res["attention_mask"].to("npu")
print(f"    input_ids shape: {input_ids.shape}")
print(f"    attn_mask sums: {attn_mask.sum(dim=1).tolist()}")
print(f"    token_type_ids in tok_res: {'token_type_ids' in tok_res}")
if "token_type_ids" in tok_res:
    print(f"    token_type_ids unique: {tok_res['token_type_ids'].unique().tolist()}")

with torch.no_grad():
    out = tm(input_ids=input_ids, attention_mask=attention_mask)
    check_tensor(out.last_hidden_state, "tm() last_hidden_state")

# 1c: 用 m.preprocess() 来 tokenize (这是 encode() 内部用的)
print("\n  [1c] m.preprocess(texts[:2]) 的特征:")
features = m.preprocess(test_texts)
print(f"    keys: {list(features.keys())}")
for k, v in features.items():
    if isinstance(v, torch.Tensor):
        print(f"    {k}: shape={list(v.shape)} dtype={v.dtype}")
    else:
        print(f"    {k}: {type(v).__name__} = {repr(v)[:100]}")

# 1d: 用 preprocess 的 features 走 tm()
print("\n  [1d] 用 preprocess features 走 tm():")
features_device = {}
for k, v in features.items():
    if isinstance(v, torch.Tensor):
        features_device[k] = v.to("npu")
    else:
        features_device[k] = v

# 看看 tm 接受哪些参数
import inspect
tm_forward_params = inspect.signature(tm.forward).parameters
print(f"    tm.forward params: {list(tm_forward_params.keys())}")

# 过滤 features 只保留 tm 接受的参数
filtered = {}
for k, v in features_device.items():
    if k in tm_forward_params:
        filtered[k] = v
    else:
        print(f"    跳过参数 '{k}' (tm.forward 不接受)")
print(f"    传入参数: {list(filtered.keys())}")

with torch.no_grad():
    out = tm(**filtered)
    check_tensor(out.last_hidden_state, "tm(**filtered) last_hidden_state")

# 1e: 如果有 prompt, 尝试空 prompt
if m.default_prompt_name and m.prompts.get(m.default_prompt_name):
    prompt_text = m.prompts[m.default_prompt_name]
    print(f"\n  [1e] 有默认 prompt: '{prompt_text}'")
    print(f"  尝试 encode() 时显式传 prompt='' 强制禁用:")
    emb_no_prompt = m.encode(test_texts, batch_size=2, show_progress_bar=False,
                             normalize_embeddings=True, prompt="")
    check_nan(emb_no_prompt, "encode(prompt='') result")

# ════════════════════════════════════════════════════════════════════════
# Part 2: 用 m[0].forward() 跰完整的 ST Transformer 模块
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 2: m[0].forward() (ST Transformer 模块)")
print(f"{'='*70}")

features2 = m.preprocess(test_texts)
features2 = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in features2.items()}
with torch.no_grad():
    out2 = m[0](features2)
    te = out2.get('token_embeddings')
    if te is not None:
        check_tensor(te, "m[0] token_embeddings")
    else:
        print(f"  m[0] output keys: {list(out2.keys())}")

# ════════════════════════════════════════════════════════════════════════
# Part 3: 完整 forward — m.forward() (transformer + pooling + dense)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 3: m.forward() (完整 pipeline: transformer + pooling + dense)")
print(f"{'='*70}")

features3 = m.preprocess(test_texts)
features3 = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in features3.items()}
with torch.no_grad():
    out3 = m.forward(features3)
    se = out3.get('sentence_embedding')
    if se is not None:
        check_tensor(se, "m.forward sentence_embedding")
    else:
        print(f"  m.forward output keys: {list(out3.keys())}")

# ════════════════════════════════════════════════════════════════════════
# Part 4: 逐层检查 — preprocess 特征 vs 手动 tokenize 的差异
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 4: preprocess 特征 vs 手动 tokenize 的详细对比")
print(f"{'='*70}")

features4 = m.preprocess(test_texts)
tok_manual = tok(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

print(f"  preprocess keys: {sorted(features4.keys())}")
print(f"  manual keys:     {sorted(tok_manual.keys())}")

for k in sorted(set(list(features4.keys()) + list(tok_manual.keys()))):
    v_pre = features4.get(k)
    v_man = tok_manual.get(k)
    if isinstance(v_pre, torch.Tensor) and isinstance(v_man, torch.Tensor):
        same = torch.equal(v_pre, v_man) if v_pre.shape == v_man.shape else False
        print(f"  {k}: pre shape={list(v_pre.shape)} man shape={list(v_man.shape)} same={same}")
        if not same and v_pre.shape == v_man.shape:
            diff = (v_pre != v_man).sum().item()
            print(f"    differences: {diff}/{v_pre.numel()}")
    elif v_pre is not None:
        print(f"  {k}: only in preprocess = {repr(v_pre)[:80]}")
    elif v_man is not None:
        print(f"  {k}: only in manual = {repr(v_man)[:80]}")

# ════════════════════════════════════════════════════════════════════════
# Part 5: Batch size 精确阈值 (用 tm() 直接调用)
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 5: tm() 直接调用 — batch size 阈值")
print(f"{'='*70}")

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(texts), size=20, replace=False)
sample_texts = [texts[i] for i in sample_idx]

for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
    batch = sample_texts[:bs]
    tok_res = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = tok_res["input_ids"].to("npu")
    attention_mask = tok_res["attention_mask"].to("npu")
    with torch.no_grad():
        out = tm(input_ids=input_ids, attention_mask=attention_mask)
        lhs = out.last_hidden_state
        n_nan = int(torch.isnan(lhs).sum().item())
        pct = n_nan / lhs.numel() * 100
        status = "PASS" if n_nan == 0 else "FAIL"
        seq_lens = attention_mask.sum(dim=1).tolist()
        print(f"  bs={bs:2d}: NaN={n_nan:8d}/{lhs.numel()} ({pct:5.1f}%) [{status}]  seq_lens={seq_lens}")
    del out, lhs, input_ids, attention_mask, tok_res
    torch.npu.empty_cache()

# ════════════════════════════════════════════════════════════════════════
# Part 6: 用 preprocess+forward 走完整管道 (m.encode 内部路径) 对比 tm() 直接调用
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 6: 完整管道 (m.forward) vs tm() 直接调用 — bs=2")
print(f"{'='*70}")

batch2 = sample_texts[:2]

# 6a: 完整管道 (m.forward → transformer + pooling + dense + normalize)
print("\n  [6a] m.forward() 完整管道:")
features6a = m.preprocess(batch2)
features6a = {k: v.to("npu") if isinstance(v, torch.Tensor) else v for k, v in features6a.items()}
with torch.no_grad():
    out6a = m.forward(features6a)
    se = out6a.get('sentence_embedding')
    if se is not None:
        check_tensor(se, "m.forward sentence_embedding")

# 6b: tm() 直接调用同样数据
print("\n  [6b] tm() 直接调用:")
tok_res6b = tok(batch2, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids6b = tok_res6b["input_ids"].to("npu")
attention_mask6b = tok_res6b["attention_mask"].to("npu")
with torch.no_grad():
    out6b = tm(input_ids=input_ids6b, attention_mask=attention_mask6b)
    check_tensor(out6b.last_hidden_state, "tm() last_hidden_state")

# ════════════════════════════════════════════════════════════════════════
# Part 7: 如果 tm() 在 bs=2 也有 NaN (不太可能), 检查逐层
# ════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Part 7: 用 hooks 检查 tm() bs=2 的每一层输出")
print(f"{'='*70}")

# 检查 embeddings 输出
embeddings_layer = tm.embeddings
print(f"  embeddings 类型: {type(embeddings_layer).__name__}")

# 用 hook 检查每一层
hooks = []
layer_outputs = {}

def make_hook(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            n_nan = int(torch.isnan(output).sum().item())
            layer_outputs[name] = (n_nan, output.numel(), list(output.shape))
        elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            t = output[0]
            n_nan = int(torch.isnan(t).sum().item())
            layer_outputs[name] = (n_nan, t.numel(), list(t.shape))
    return hook

# 注册 hooks 到关键模块
for name, module in tm.named_modules():
    if name and any(x in name for x in ['embeddings', 'encoder', 'layer']) and 'NewAttention' not in name and 'NewGatedMLP' not in name:
        hooks.append(module.register_forward_hook(make_hook(name)))

with torch.no_grad():
    out7 = tm(input_ids=input_ids6b, attention_mask=attention_mask6b)

for h in hooks:
    h.remove()

print(f"  层输出 NaN 检查:")
for name in sorted(layer_outputs.keys()):
    n_nan, n_elem, shape = layer_outputs[name]
    pct = n_nan / n_elem * 100 if n_elem > 0 else 0
    status = " *** NaN ***" if n_nan > 0 else ""
    print(f"    {name:50s}: NaN={n_nan:8d}/{n_elem} ({pct:5.1f}%) shape={shape}{status}")

print(f"\n{'='*70}")
print("DONE")
print(f"{'='*70}")
