# NPU Embedding NaN 问题调查记录

> 记录时间: 2026-08-25
> 目的: 记录 `npu_fusion_attention` TND 路径产生 NaN 的调查过程、根因分析和修复方案，避免重复踩坑

---

## 1. 背景

ClimbMix 流水线中，`embed_documents` 函数（用于 subsample embedding + FAISS 聚类）在 NPU 上运行时，产生了大量 NaN embedding，导致 FAISS K-means 崩溃。

**触发场景**: smoke test Phase 1（2000 docs subsample embedding → K-means K=1000）

---

## 2. 硬件环境

- **NPU**: 8×Ascend 910B4（32GB HBM/卡）
- **CANN**: 8.5.1
- **PyTorch**: 2.9.1 + torch_npu 2.9.0.post1
- **模型**: stella_en_400M_v5（1024维, 24层, H=16, D=64, `unpad_inputs=True`）
- **数据**: STEM parquet, 116M docs, min=28 chars, max=793K chars, median=1311 chars

---

## 3. 调查过程

### 3.1 初始现象

Smoke test 在 FAISS K-means 阶段崩溃：
```
RuntimeError: '!(std::isfinite(x[i]))' failed: input contains NaN's or Inf's
```

embedding 本身完成（8/8 batches, 19.9s），但输出数组中存在 NaN。

### 3.2 第一步：排查数据

**脚本**: `scripts/diagnose_nan.py`

检查 parquet 数据中的文本长度分布：

| 类别 | 数量 | 占比 |
|------|------|------|
| Empty (0 chars) | 0 | 0.00% |
| Very short (1-9 chars) | 0 | 0.00% |
| Short (10-49 chars) | 73 | 0.03% |
| Normal (50+ chars) | 232,677 | 99.97% |

**结论**: 数据无问题，0 空文本，所有文本 ≥ 28 chars。

### 3.3 第二步：batch size 对比

在同一模型实例上（fp32），用不同 batch size 编码相同的 2000 个文档：

| batch_size | NaN 行数 | NaN 比例 |
|------------|----------|----------|
| 8 | 0 | 0.0% |
| 32 | 224 | 11.2% |
| 64 | 464 | 23.2% |
| 128 | 336 | 16.8% |
| 256 | 208 | 10.4% |
| 512 | — | OOM |

**发现**: NaN 不是小概率事件（10-23%），是系统性 bug。batch_size=8 完全无 NaN，32+ 大量 NaN。NaN 比例与 batch size 非单调，说明非确定性。

### 3.4 第三步：多变量隔离测试

**脚本**: `scripts/diagnose_nan2.py`

尝试测试 TND vs fallback、fp32 vs fp16、不同 max_seq_length。但脚本有设计缺陷：**多次 reload model 导致 NPU 状态崩溃**，从第二个测试开始 100% NaN（2,048,000 个值全部 NaN）。

**关键教训**: NPU 上不要反复 `del model; torch.npu.empty_cache(); SentenceTransformer(...)` 重新加载模型。多次 reload 后 NPU 内核状态会损坏，所有计算返回 NaN。

### 3.5 第四步：单次加载模型，干净测试

**脚本**: `scripts/diagnose_nan3.py`（保留为生产诊断脚本）

关键设计：**模型只加载一次**，在同一实例上测试所有配置，通过 `model.half()` 切换精度。

#### fp32 + TND 路径（当前 embed_documents 配置）

| batch_size | NaN 行数 | NaN 比例 | 状态 |
|------------|----------|----------|------|
| 8 | 0 | 0.0% | PASS |
| 32 | 1088 | 54.4% | FAIL |
| 64 | 400 | 20.0% | FAIL |
| 128 | 128 | 6.4% | FAIL |
| 256 | 512 | 25.6% | FAIL |

- bs=256 重复 3 次：每次都是 512 行 NaN（**确定性**，非随机）
- NaN 比例非单调，与 batch size 无线性关系

#### fp16 + TND 路径（model.half() 后）

| batch_size | NaN 行数 | NaN 比例 | 状态 |
|------------|----------|----------|------|
| 8 | 232 | 11.6% | FAIL |
| 64 | 64 | 3.2% | FAIL |
| 256 | 0 | 0.0% | PASS |
| 512 | 0 | 0.0% | PASS |

#### fp16 + TND + msl=512 + bs=512（streaming 配置）

| 重复次数 | NaN 行数 | 状态 |
|----------|----------|------|
| rep0 | 0 | PASS |
| rep1 | 0 | PASS |
| rep2 | 0 | PASS |

**完全稳定，0% NaN。**

---

## 4. 根因分析

### 4.1 问题定位

```
stella model (unpad_inputs=True)
  → NonZero boolean indexing (移除 padding)
  → q/k/v: (1, total_S, H, D) packed tensor
  → fake xformers _memory_efficient_attention()
    → npu_fusion_attention(input_layout="TND", ...)
```

**根因**: `torch_npu.npu_fusion_attention` 的 TND（Variable Length）路径在 fp32 精度下存在 bug，对特定的序列长度组合会产生 NaN。

### 4.2 为什么 streaming 路径没有问题

streaming 路径（`_embed_streaming_worker`）使用：
- `model.half()` → fp16
- `model.max_seq_length = 512`
- `batch_size = 512`

fp16 + 大 batch (≥256) 时 TND 内核稳定。而 `embed_documents` 之前使用 fp32 + bs=256，恰好落在不稳定的配置区间。

### 4.3 为什么 batch_size=8 无 NaN

推测：少量序列（8 条）时 TND 内核的内部实现路径不同（可能退化为逐序列处理），避免了触发 bug。batch_size ≥ 32 时，内核使用批量处理路径，序列长度组合的数值不稳定导致 NaN。

### 4.4 NaN 是确定性的（同一进程内）

同一进程内，同一配置（fp32, bs=256）重复 3 次，每次都是相同的 512 行 NaN。说明不是随机硬件错误，而是特定输入组合的确定性数值 bug。

### 4.5 跨进程非确定性（fp16+bs=512 仍可能产生 NaN）

**关键发现**：诊断脚本中 fp16+bs=512 连续 4 次运行都是 0% NaN，但 smoke test（不同进程、相同配置）产生了 143/2000 (7.15%) NaN。

说明 `npu_fusion_attention` TND 内核的行为在**进程级别非确定性**——不同进程启动时 NPU 内核初始化状态不同，即使配置完全相同，不同进程可能产生不同的 NaN 模式。

**影响**：
- 无法完全消除 NaN，只能降低概率（fp16+bs=512 显著降低但非零）
- 必须保留 NaN 防护层：检测 → 零向量替换 → label -1 → 排除
- streaming 路径也需 NaN 检查（已添加，commit `1950d84`）

---

## 5. 修复方案

### 5.1 已实施修复（commit `459a6d8`）

`embed_documents` 在 NPU 上使用与 streaming 路径一致的配置：

```python
model = _load_model(model_name, "npu")
model.eval()
model.half()                          # fp16（关键）
model.max_seq_length = 512
batch_size = max(batch_size, 512)     # 强制 ≥512（关键）
```

### 5.2 NaN 防护（commit `be42595`）

即使修复了主要问题，仍保留 NaN 检测作为安全网：

1. `embed_documents`: embedding 后检查 NaN，替换为零向量
2. `cluster_embeddings_faiss`: K-means 前检查 NaN/Inf
3. `cluster_embeddings_faiss`: K-means 后，零向量文档 label 设为 -1
4. `compute_cluster_quality` / `prune_clusters`: 跳过 -1 label
5. `_assign_remaining_by_domain`: 从比例分配中排除 -1，保留 -1 不被覆盖
6. `select_data_by_mixture`: 只选 label ≥ 0 的文档

整条链路确保 NaN 文档被正确追踪和排除，不会进入最终数据集。

---

## 6. 验证结果

修复后（fp16 + bs=512 + msl=512）在 2000 docs 上：
- 3 次重复：0 NaN
- 完全确定性

---

## 7. 经验教训

1. **NPU 不要反复 reload model**: 多次 `del model; empty_cache(); new SentenceTransformer(...)` 会导致 NPU 内核状态损坏，后续所有计算返回 100% NaN。诊断脚本必须单次加载模型。

2. **fp32 在 NPU 上不稳定**: `npu_fusion_attention` TND 路径在 fp32 下有 bug，fp16 更稳定。NPU 上应优先使用 fp16。

3. **batch size 影响数值稳定性**: 小 batch (8) 可能走不同内核路径而稳定；大 batch (≥256) 在 fp16 下也稳定。需要找到稳定区间。

4. **诊断脚本设计原则**:
   - 单次加载模型，避免 reload 污染
   - 同一配置重复多次，检查确定性
   - 隔离变量（精度、batch size、max_seq_length）逐一测试
   - 记录具体哪些输入产生 NaN（行号、文本特征）

5. **NaN 防护应该是多层防御**:
   - embedding 层：检测 + 替换
   - 聚类层：检测 + 标记
   - 选择层：过滤
   - 即使修复了根因，防护层仍需保留

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| `src/climbmix/core/embedding_cluster.py` | fake xformers (lines 21-127), `embed_documents` (line 153), `cluster_embeddings_faiss` (line 608) |
| `src/climbmix/core/cluster_merge.py` | `compute_cluster_quality` (line 32), `prune_clusters` (line 76), `merge_clusters_by_distance` (line 128) |
| `src/climbmix/core/discovery.py` | `_assign_remaining_by_domain` (line 116) |
| `scripts/diagnose_nan3.py` | 生产诊断脚本（单次加载，fp32→fp16 切换） |
| `docs/embedding_performance.md` | embedding 性能优化记录（npu_fusion_attention 1.32x 加速） |
