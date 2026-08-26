# NPU Embedding NaN 问题调查记录

> 记录时间: 2026-08-26（最终确认）
> 目的: 记录 stella embedding 模型在 NPU 上产生 100% NaN 的调查过程、根因分析和修复方案

---

## 1. 背景

ClimbMix 流水线中，`embed_documents` 函数（用于 subsample embedding + FAISS 聚类）在 NPU 上运行时，产生了大量 NaN embedding，导致 FAISS K-means 崩溃。

**触发场景**: smoke test / speedrun Phase 1（2000 docs subsample embedding → K-means K=1000）

---

## 2. 硬件环境

- **NPU**: 8×Ascend 910B4（32GB HBM/卡）
- **CANN**: 8.5.1
- **PyTorch**: 2.9.1 + torch_npu 2.9.0.post1
- **模型**: stella_en_400M_v5（NovaSearch/stella_en_400M_v5，1024维, 24层, H=16, D=64, RoPE, `unpad_inputs=True`）
- **数据**: STEM parquet, 116M docs, min=28 chars, max=793K chars, median=1311 chars

---

## 3. 根因

### 3.1 一句话总结

stella 模型用 `persistent=False` 注册了 `position_ids` 缓冲区（`torch.arange(max_position_embeddings)`），该缓冲区不保存在 checkpoint 中。在 torch_npu 环境下加载模型后，该缓冲区持有**未初始化的堆内存垃圾值**（如 `2166231420765151743`），导致 `rope_cos[position_ids]` 越界索引，NPU 静默返回 NaN，传播至全部 24 层 → 100% NaN embedding。

### 3.2 受影响的缓冲区

stella 的 `NewEmbeddings.__init__`（`modeling.py`）注册了以下 `persistent=False` 缓冲区：

| 缓冲区 | 所属模块 | 应有值 | 实际值（损坏后） |
|--------|----------|--------|------------------|
| `position_ids` | `NewEmbeddings` | `arange(8192)` | 堆垃圾（如 `2166231420765151743`） |
| `inv_freq` | `RotaryEmbedding` | `1.0 / (base ** (arange(0,dim,2)/dim))` | 堆垃圾 |
| `cos_cached` | `RotaryEmbedding` | `cos(freqs)` | 堆垃圾 |
| `sin_cached` | `RotaryEmbedding` | `sin(freqs)` | 堆垃圾 |

### 3.3 NaN 传播路径

```
NewEmbeddings.forward()
  → position_ids = self.position_ids[:l]        # 垃圾索引值
  → rope_cos = self.rotary_emb(...)             # cos_cached 也是垃圾
  → rope_cos[position_ids]                      # OOB 索引 → NPU 返回 NaN
  → rope_embeds = (NaN, NaN)
  → NewAttention.forward()
    → apply_rotary_pos_emb(q, k, NaN_cos, NaN_sin)  # q, k 变 NaN
    → attention(q=NaN, k=NaN, v=...)                 # 输出 NaN
  → 24 层传播 → last_hidden_state 全 NaN
  → pooling → dense → normalize → embedding = NaN
```

### 3.4 确认证据

在 CPU 上加载模型后检查 `position_ids` 缓冲区：

```
dtype= torch.int64
shape= (8192,)
first5= [2166231420765151743, -71213169107857921, 2312886558010579479, 792633534417207295, -1021598236673]
min= -71213169107857921  max= 2312886558010579479
any_nan= False
equals_arange= False    # ← 应为 True
```

- 损坏在 CPU 上就存在（不仅是 NPU 问题，是 torch_npu 环境下的 PyTorch 行为）
- 13 种环境变量配置测试全部 100% NaN（排除了环境变量因素）
- 模型权重检查 `any_nan=False, any_inf=False`（排除了权重问题）
- NPU 硬件检查 8 卡全部 OK 无故障（排除了硬件问题）

### 3.5 为什么 NPU 返回 NaN 而 CPU 报错

- **CPU**: `tensor[garbage_index]` 超出 bounds → 抛 `IndexError`（有 bounds check）
- **NPU**: `tensor[garbage_index]` 超出 bounds → 静默返回 NaN（无 bounds check）

这解释了为什么在 NPU 上表现为 100% NaN 而非崩溃。

### 3.6 责任归属

| 环节 | 责任 | 说明 |
|------|------|------|
| stella 模型作者 | 无责 | `register_buffer(..., persistent=False)` 是 PyTorch 标准用法，契约是：不存 checkpoint，但 `__init__` 重新运行时会正确赋值。stella 只是"触发者"——恰好用了 `persistent=False` + RoPE 索引的组合。 |
| PyTorch 本身 | 无责 | 在 CPU/CUDA 上 `persistent=False` 缓冲区的加载机制完全正常。 |
| **torch_npu** | **全责** | 两层缺陷：(1) 加载 `trust_remote_code` 模型时，`persistent=False` 缓冲区的 `__init__` 重新初始化未正确执行，残留未初始化堆内存；(2) NPU 索引操作无 bounds check，越界静默返回 NaN 而非报 `IndexError`，掩盖了问题的真正来源。 |

---

## 4. 调查过程（排除的假设）

### 4.1 环境变量（排除）

**脚本**: 环境变量二分法测试（脚本已删除，13 种配置全部 100% NaN）

测试了 13 种环境变量配置（`ASCEND_FUSION_ENABLE`, `ASCEND_LAUNCH_BLOCKING`, `TASK_QUEUE_ENABLE` 等），全部 100% NaN。

### 4.2 NPU 硬件故障（排除）

`npu-smi info`: 8 卡全部 `OK`，无运行进程，无故障记录。

### 4.3 模型权重损坏（排除）

C1 检查: 435M 参数 `any_nan=False, any_inf=False`，权重完好。

### 4.4 fp16 溢出（排除）

fp32 + 手动 attention 同样 100% NaN——bug 在 embedding 层，不在 attention。

### 4.5 npu_fusion_attention TND 路径（排除 — 旧文档的错误结论）

> **旧文档曾将此作为根因，这是错误的。** 旧文档观察到 batch_size 对 NaN 比例有影响，并归因于 TND 内核的数值不稳定。实际上，`position_ids` 缓冲区损坏是确定性的（每次加载都是垃圾值），旧文档观察到的 batch_size 相关性是不同代码版本（bf16/fp32/chunked attention 等多次改动）在不同 commit 上运行的混淆结果，并非真实属性。最终确认：所有 batch_size 在当前代码下都是 100% NaN（环境变量二分法测试证实）。

### 4.6 进程级非确定性（排除 — 旧文档的错误结论）

> **旧文档曾声称 npu_fusion_attention TND 内核在进程级别非确定性地产生 NaN。** 这是错误的。根因是 `position_ids` 缓冲区在每次加载时都确定性地损坏（只是垃圾值的具体内容因堆状态而异），NaN 是确定性的 100%。旧文档观察到的"某些进程 0% NaN"来自不同代码版本或不同加载路径，并非同一代码的非确定行为。

---

## 5. 修复方案

### 5.1 核心修复: `_repair_stella_buffers`（commit `d799f52`）

在 `embedding_cluster.py` 中新增 `_repair_stella_buffers(model)` 函数（line 120），在模型加载后、`half()` 之前调用，重新初始化所有损坏的非持久缓冲区：

```python
def _repair_stella_buffers(model):
    import torch
    try:
        inner = model[0].model          # NewModel
        embeddings = inner.embeddings
        cfg = inner.config
        device = inner.device
    except (AttributeError, IndexError, TypeError):
        return  # not a stella model; nothing to do

    max_pos = cfg.max_position_embeddings
    # position_ids — re-register as arange on the correct device
    embeddings.register_buffer(
        "position_ids", torch.arange(max_pos, device=device), persistent=False
    )
    # rotary_emb (inv_freq, cos_cached, sin_cached) — rebuild via _init_rope
    embeddings._init_rope(cfg)
    embeddings.rotary_emb = embeddings.rotary_emb.to(device)
    # Sanity check
    assert torch.equal(embeddings.position_ids,
                       torch.arange(max_pos, device=device)), \
        "position_ids repair failed — still not arange"
```

### 5.2 调用点（3 处）

| 函数 | 行号 | 位置 |
|------|------|------|
| `embed_documents` NPU 路径 | 229 | `_load_model` 后、`model.half()` 前 |
| `embed_documents` CPU 路径 | 254 | `_load_model` 后 |
| `_embed_streaming_worker` | 344 | `_load_model_stream` 后、`model.half()` 前 |

**放在 `half()` 之前的原因**: 让 `inv_freq`/`cos_cached`/`sin_cached` 等 float 缓冲区随 `half()` 一起转 fp16，与正常加载行为一致。

### 5.3 为何安全

- `persistent=False` 缓冲区不在 checkpoint 中，重新初始化为 `__init__` 值 = 恢复应有状态，语义等价于干净加载
- `_init_rope(cfg)` 是 stella 自身的方法，正确处理 base RoPE 与 NTK scaling，无需硬编码参数
- `assert` 防止未来回归

### 5.4 NaN 防护层（保留）

即使根因已修复，仍保留多层 NaN 防护作为安全网：

1. `embed_documents`: embedding 后检查 NaN → 逐条重试（bs=1）→ 仍 NaN 则 raise
2. `cluster_embeddings_faiss`: K-means 前检查 NaN/Inf
3. `cluster_embeddings_faiss`: K-means 后，零向量文档 label 设为 -1
4. `compute_cluster_quality` / `prune_clusters`: 跳过 -1 label
5. `select_data_by_mixture`: 只选 label ≥ 0 的文档

---

## 6. 验证结果

### 6.1 诊断测试（200 docs）

```
RESULT: NaN=0/200 (0.0%) [PASS]
```

修复前: NaN=200/200 (100.0%) [FAIL]
修复后: NaN=0/200 (0.0%) [PASS]

### 6.2 Speedrun 端到端（2000 docs）

```
[Embed] Encoding 2000 documents (batch_size=512)...
[Embed] Encoded 2000 docs in 16.4s, dim=1024
[Embed] No NaN detected (2000 docs)
[Embed] Cached embeddings to: .../embedding_cache.npz
[Cluster] FAISS K-means: K=1000, dim=1024, n_docs=2000
```

0 NaN，embedding 正常缓存，FAISS 聚类正常运行。

---

## 7. 诊断脚本

调查过程中创建了大量中间诊断脚本（14 个），均在错误假设下编写（TND 内核、batch size、padding、warmup 等），已全部删除。仅保留：

| 脚本 | 说明 |
|------|------|
| `diagnose_env_test.py` | 最小验证脚本（200 docs），已调用 `_repair_stella_buffers`，用于回归测试 |

---

## 8. 经验教训

1. **`persistent=False` 缓冲区在 torch_npu 上可能不初始化**: PyTorch 的 `register_buffer(..., persistent=False)` 在标准环境下会在 `__init__` 时正确赋值，但 torch_npu 环境下加载 trust_remote_code 模型时，这些缓冲区可能持有未初始化内存。需要在模型加载后显式验证或重新初始化。

2. **NPU 越界索引静默返回 NaN**: 与 CPU 的 `IndexError` 不同，NPU 的索引操作无 bounds check，越界返回 NaN 而非报错。这使得 bug 表现为"输出全 NaN"而非"崩溃"，增加定位难度。**教训: 在 NPU 上遇到 100% NaN 时，首先检查所有索引操作的输入是否合法。**

3. **先验证输入再怀疑计算内核**: 本案中大量时间花在排查 `npu_fusion_attention` TND 内核、fp16 溢出、环境变量等，而根因在 embedding 层的索引输入。**教训: 遇到全 NaN 时，从模型forward的第一行开始逐层检查输入，而非从最后一步的 attention 开始。**

4. **诊断脚本必须与生产代码使用相同的加载路径**: `diagnose_env_test.py` 直接用 `SentenceTransformer(...)` 加载模型，绕过了 `_load_model` 中的 `_repair_stella_buffers` 调用，导致修复后首次验证仍显示 100% NaN。**教训: 诊断脚本必须复用生产代码的模型加载逻辑，或在脚本中显式调用相同的修复函数。**

5. **跨 commit 的诊断数据不可比较**: 旧文档中 batch_size vs NaN 比例的数据来自不同 commit（bf16/fp32/chunked attention 等多次改动），导致得出"batch_size 影响 NaN"和"进程级非确定性"的错误结论。**教训: 诊断时锁定代码版本，避免在多次代码改动间比较结果。**

---

## 9. 相关文件

| 文件 | 说明 |
|------|------|
| `src/climbmix/core/embedding_cluster.py` | `_repair_stella_buffers` (line 120), fake xformers (lines 21-117), `embed_documents` NPU/CPU 路径 (lines 229, 254), `_embed_streaming_worker` (line 344), NaN retry (lines 272-282) |
| `scripts/diagnostics/diagnose_env_test.py` | 最小验证脚本（200 docs），已调用 `_repair_stella_buffers`，用于回归测试 |
| `runs/speedrun_climbmix.sh` | Speedrun 脚本（已验证 0 NaN 端到端通过） |
| stella `modeling.py` (remote HF cache) | `NewEmbeddings.__init__` (register_buffer position_ids, persistent=False), `NewEmbeddings.forward` (rope_cos[position_ids]), `RotaryEmbedding.__init__` (register_buffer inv_freq/cos_cached/sin_cached, persistent=False) |
