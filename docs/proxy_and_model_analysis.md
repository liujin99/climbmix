# Proxy 与 Model 大小分析文档

> 记录时间: 2026-07-16
> 目的: 系统记录 climbmix 项目中关于 proxy/target 模型大小的所有分析、数据、结论和设计决策

---

## 1. nanochat 架构特殊性: Value Embeddings (VE)

### 1.1 VE 是什么

nanochat GPT 模型在**交替层**上使用 Value Embeddings (VE)，将 V 投射从矩阵乘法替换为 embedding lookup。VE 不参与核心矩阵乘法计算（FLOPs），但占大量存储空间。

### 1.2 VE 对参数量的影响

VE 在小模型中占比极高，导致 total_params 和 scaling_params 差距巨大：

| depth | n_embd | scaling(M) | VE(M) | wte(M) | total(M) | VE占比 |
|-------|--------|-----------|-------|--------|----------|--------|
| 10 | 640 | 70.2 | 104.9 | 21.0 | 196.0 | 53.5% |
| 12 | 768 | 110.1 | 151.0 | 25.2 | 286.3 | 52.7% |
| 14 | 896 | 164.2 | 205.5 | 29.4 | 399.1 | 51.5% |
| 16 | 1024 | 234.9 | 268.4 | 33.6 | 536.9 | 50.0% |
| 18 | 1152 | 324.4 | 339.7 | 37.7 | 701.9 | 48.4% |
| 20 | 1280 | 435.2 | 419.4 | 41.9 | 896.5 | 46.8% |
| 22 | 1408 | 569.5 | 507.5 | 46.1 | 1123.2 | 45.2% |
| 24 | 1536 | 729.8 | 604.0 | 50.3 | 1384.1 | 43.6% |

### 1.3 scaling_params 定义

nanochat 中 `scaling_params` = `transformer_matrices + lm_head`，来源代码 `base_train.py:259-263`:

```python
def get_scaling_params(m):
    # transformer matrices + lm_head gives cleanest scaling laws (see dev/LOG.md Jan 27, 2026)
    params_counts = m.num_scaling_params()
    scaling_params = params_counts['transformer_matrices'] + params_counts['lm_head']
    return scaling_params
```

`gpt.py:345-372` 中 `num_scaling_params()` 返回的分组：
- **transformer_matrices**: 所有 transformer 层的 Q/K/V/O 投射 + MLP up/down 投射
- **lm_head**: 输出投射 (vocab_size × n_embd)
- **wte**: 词 token 嵌入 (vocab_size × n_embd) — 不计入 scaling
- **value_embeds**: VE 嵌入 — 不计入 scaling
- **scalars**: resid_lambda, x0_lambda, smear_gate 等 — 不计入 scaling

每层 transformer_matrices = 12 × n_embd² (4 attn projections + 8 MLP projections)

nanochat 注释说明: "transformer matrices + lm_head gives cleanest scaling laws"，即用这个组合做 Chinchilla scaling law 分析时拟合最好。因为 embedding lookup 不参与 FLOPs 计算，排除后 tokens/params ratio 更符合真实计算量关系。

### 1.4 nanochat 的惯例

- **报告模型大小**: 用 total_params（"196M 模型"）
- **计算训练量**: 用 scaling_params（ratio × scaling_params）
- **报告 Tokens:Scaling params ratio**: 用 scaling_params

所以 nanochat 的 ratio 计算流程：
```
target_tokens = ratio × scaling_params
d10: ratio=9.5 × 70.2M = 670M tokens (not 9.5 × 196M = 1.86B)
```

---

## 2. 与 CLIMB 论文的对标分析

### 2.1 CLIMB 论文模型配置

CLIMB 使用**标准 Transformer**（类似 Llama/GPT-2），**没有 VE**。所有参数都参与核心计算。

| 角色 | 参数量 | 说明 |
|------|--------|------|
| Proxy (main) | 350M | 标准Transformer, 所有参数是"有效计算"参数 |
| Proxy (ablation) | 62M | 消融实验中测试的小proxy |
| Target | 1.3B | 标准Transformer, 无VE |

标准 Transformer 中 wte+lm_head 只占 ~5-10%，所以 350M ≈ 320M "scaling equivalent"。total_params ≈ scaling_params，几乎无需区分。

### 2.2 nanochat 对标: 用 scaling_params 才公平

因为 VE 占 ~50% 参数但不参与核心计算，**必须用 scaling_params 对标**:

| nanochat depth | scaling(M) | vs CLIMB 350M proxy | vs CLIMB 1.3B target |
|---------------|-----------|--------------------|--------------------|
| d10 | 70.2 | 0.20× | 0.05× |
| d12 | 110.1 | 0.31× | 0.08× |
| d14 | 164.2 | 0.47× | 0.13× |
| d16 | 234.9 | 0.67× | 0.18× |
| d18 | 324.4 | **≈0.93×** | 0.25× |
| d24 | 729.8 | — | **0.56×** |

d18 的 scaling_params(324M) ≈ CLIMB 350M proxy 的有效计算力（1:1 对标）。

d24 的 scaling_params(730M) 只有 CLIMB 1.3B target 的 **56%**。虽然 total_params(1.38B) ≈ 1.07× CLIMB 1.3B，但 VE 占 43.6%，"虚胖"。

要真正对标 CLIMB 1.3B 需要 d30 (scaling=1.39B, total=2.4B)，但训练成本太高。

### 2.3 proxy:target 比例分析

CLIMB 的 proxy:target 比例（按 scaling_params）：
```
350M / 1300M ≈ 1:3.7
```

如果按比例缩减到我们的 target (730M scaling)：
```
proxy = 730M × (1/3.7) ≈ 197M scaling
```

对应 nanochat 深度：

| depth | scaling(M) | proxy:target 比例 | vs CLIMB 1:3.7 |
|-------|-----------|------------------|----------------|
| d10 | 70.2 | 1:10.4 | 太小 |
| d12 | 110.1 | 1:6.6 | 偏小 |
| **d14** | **164.2** | **1:4.4** | **接近** |
| d16 | 234.9 | 1:3.1 | 偏大 |
| d18 | 324.4 | 1:2.2 | 过大 |

d14 (scaling=164M) 比例最接近 CLIMB 的 1:3.7。

---

## 3. CLIMB 消融实验: 62M Proxy 有效性

### 3.1 论文数据 (Table 3 Abl.proxy)

1B target model, 40B tokens continual pre-training:

| Proxy size | piqa | arc_c | arc_e | hellaswag | winogrande | siqa | Avg. |
|-----------|------|-------|-------|-----------|------------|------|------|
| 62M | 75.41 | 40.56 | 72.82 | 65.76 | 63.23 | 42.89 | **60.11** |
| 132M | 75.56 | 40.93 | 72.94 | 65.57 | 63.09 | 43.07 | **60.19** |
| 350M | 75.78 | 40.98 | 72.97 | 66.01 | 63.32 | 43.37 | **60.41** |

### 3.2 与所有 baseline 对比

| 方法 | 1B target avg | vs 62M proxy |
|------|-------------|-------------|
| Base (before training) | 56.46 | +3.65 |
| Random | 57.93 | +2.18 |
| DoReMi | 59.16 | +0.95 |
| RegMix | 59.37 | +0.74 |
| **CLIMB 62M proxy** | **60.11** | — |
| CLIMB 350M proxy | 60.41 | -0.30 |

### 3.3 关键结论

**62M proxy 仍然明显优于所有 baseline**：
- 比 RegMix 高 +0.74
- 比 DoReMi 高 +0.95
- 比 Random 高 +2.18
- 62M vs 350M 差距仅 0.30

说明 **proxy 大小对 CLIMB 效果不是决定性因素**，小 proxy 也能找到有意义的优化方向。

---

## 4. d10 Base Training 实测数据

### 4.1 训练日志

d10, ratio=9.5 (base training), 8×910B3 NPU:

```
step 00999/01270 (78.66%) | loss: 2.999974 | lrm: 0.36 | dt: 1660.64ms | tok/sec: 315,713 | bf16_mfu: 6.11
Step 01000 | Validation bpb: 0.933707
```

### 4.2 CORE 评估结果 (22 tasks)

| Task | accuracy | centered | 信号质量 |
|------|----------|----------|---------|
| hellaswag_zeroshot | 0.3300 | 0.1067 | 中 |
| jeopardy | 0.0000 | 0.0000 | 无 |
| bigbench_qa_wikidata | 0.1000 | 0.1000 | 中 |
| arc_easy | 0.4800 | **0.3067** | 高 |
| arc_challenge | 0.2480 | -0.0027 | 无 |
| copa | 0.5000 | 0.0000 | 无 |
| commonsense_qa | 0.2880 | 0.1100 | 中 |
| piqa | 0.6360 | **0.2720** | 高 |
| openbook_qa | 0.2900 | 0.0533 | 低 |
| lambada_openai | 0.2520 | **0.2520** | 高 |
| hellaswag_10shot | 0.3100 | 0.0800 | 低 |
| winograd | 0.5495 | 0.0989 | 中 |
| winogrande | 0.5000 | -0.0000 | 无 |
| bigbench_dyck_languages | 0.0640 | 0.0640 | 低 |
| agi_eval_lsat_ar | 0.2217 | 0.0272 | 极低 |
| bigbench_cs_algorithms | 0.4360 | **0.4360** | 高(LM) |
| bigbench_operators | 0.0714 | 0.0714 | 低 |
| bigbench_repeat_copy_logic | 0.0000 | 0.0000 | 无 |
| squad | 0.1300 | 0.1300 | 中 |
| coqa | 0.1240 | 0.1240 | 中 |
| boolq | 0.5420 | **-0.2053** | 反信号 |
| bigbench_language_identification | 0.2520 | 0.1771 | 中 |

**CORE metric: 0.1001**

### 4.3 信号分类

- **高信号 (centered > 0.2)**: 7 个 — piqa, arc_easy, lambada_openai, cs_algorithms, commonsense_qa, squad, coqa
- **中信号 (0.1-0.2)**: 4 个 — hellaswag_zeroshot, wikidata, winograd, language_identification
- **近随机 (≈0)**: 5 个 — jeopardy, arc_challenge, copa, winogrande, repeat_copy_logic
- **反信号 (<0)**: 1 个 — boolq (-0.2053)

### 4.4 关键发现

1. 有信号的任务恰好是 CLIMB 论文搜索用的那几个 (piqa, arc_easy, hellaswag)
2. 22 任务 raw average 被 5 个近随机 + 1 个反信号严重稀释
3. CORE=0.1001 低但非零，数据混合差异可以显现
4. **应该用高信号子集做搜索指标**，不用全 22 任务平均

---

## 5. 设计决策

### 5.1 验证流程: 三阶段渐进

#### Stage 0: Dry-run (CPU)

CPU 上跑完整流程验证代码逻辑，不依赖 NPU。

#### Stage 1: 快速验证 (d10 proxy → d24 target)

**核心设计: 用最小 proxy 搜索，在最大 target 上验证可迁移性**

| 步骤 | 模型 | 操作 | 时间 |
|------|------|------|------|
| 搜索 | d10 (70M scaling) | 3轮 [8,4,2] = 14 configs × 500 steps | ~3h |
| 验证 | d24 (730M scaling) | 最优混合 vs baseline 混合，1-2 config × 1000 steps | ~2h |
| **总计** | | | **~5h** |

**为什么不在 d10 上自验证，而是在 d24 上验证？**
- 自验证 (d10→d10) 只能证明算法能跑通，不能证明 proxy→target 可迁移性
- d10 proxy→d24 target 验证了 CLIMB 的核心假设：小 proxy 找到的最优混合在大 target 上也有效
- d24 验证只需 1-2 config，时间可控 (~2h)
- 如果 d10 搜索的最优混合在 d24 上也优于 baseline → 通路成立，进入 Stage 2

**决策门 (Decision Gate)**:
- ✅ d24 上最优混合 > baseline → 进入 Stage 2
- ❌ d24 上无优势 → 分析原因，可能需要换 proxy 或调整搜索参数

#### Stage 2: 正式搜索 (d14/d18 proxy → d24 target)

Stage 1 通过后才进入。选择更大 proxy 做更精细搜索:

| proxy | scaling(M) | 对标 | 搜索量 | 时间 |
|-------|-----------|------|--------|------|
| d14 | 164M | ≈CLIMB 132M, 比例1:4.4 | [16,8,4] = 28 configs | ~6h |
| d18 | 324M | ≈CLIMB 350M | [16,8,4] = 28 configs | ~24h |

d24 最终验证: 3 configs × 1000-2000 steps

#### 每次实验完成后: 自动生成报告

每轮搜索/验证跑完后自动生成：
- Markdown 报告（搜索过程、最优混合、评估分数、与 baseline 对比）
- matplotlib domain 分布对比图

### 5.2 Proxy 选择: 三级候选

| 级别 | depth | scaling(M) | total(M) | 对标 | 搜索时间(28 configs) |
|------|-------|-----------|---------|------|--------------------|
| 快速验证 | d10 | 70.2 | 196.0 | ≈CLIMB 62M | ~3h |
| 比例匹配 | d14 | 164.2 | 399.1 | ≈CLIMB 132M, 比例1:4.4 | ~6h |
| 完全对标 | d18 | 324.4 | 701.9 | ≈CLIMB 350M | ~11h |

Stage 1 用 d10 (≈CLIMB 62M)，CLIMB 消融已证明 62M proxy 有效 (60.11 vs RegMix 59.37, 优势 +0.74)。

### 5.3 Target: d24 (scaling=730M, total=1.38B)

d24 是所有阶段的最终 target。虽然 scaling 只有 CLIMB 1.3B 的 56%，但:
- CLIMB 的核心发现（退火阶段数据混合影响最终质量）不依赖具体模型大小
- 730M scaling 是有效模型，比 d10 有足够信号
- proxy-target 一致性主要依赖同架构下的比例关系，不是绝对大小
- 报告中如实标注: **target=730M scaling / 1.38B total**

### 5.4 指标选择: 高信号任务子集

不用全 22 任务 CORE 平均（被噪声稀释），改用高信号子集:

**推荐 6 任务子集** (centered > 0.1):
- piqa (0.2720)
- arc_easy (0.3067)
- lambada_openai (0.2520)
- commonsense_qa (0.1100)
- squad (0.1300)
- coqa (0.1240)

**或用 CLIMB 论文的 3 任务**:
- piqa + arc_easy + hellaswag (zeroshot + 10shot)

搜索和验证使用**同一指标**，避免 distortion。

### 5.5 报告中的参数标注规范

报告中标注两种参数量:
- **total_params**: 遵循学术惯例（如"1.38B 模型"）
- **scaling_params**: 对标 CLIMB 和计算训练量时使用

必须加说明: "nanochat 使用 VE (value embeddings)，约占 50% 参数但不参与核心计算。与 CLIMB 350M/1.3B 对标时应看 scaling_params (324M/730M)，不看 total_params。"

---

## 6. 训练时间估算 (8×910B3 NPU)

### 6.1 Stage 1 快速验证时间

| 步骤 | 时间 |
|------|------|
| d10 base 预训练 | ~0.6h (已完成) |
| d10 proxy 搜索 (14 configs × 500 steps) | ~3h |
| d24 target 验证 (1-2 configs × 1000 steps) | ~2h |
| **总计** | **~5-6h** |

### 6.2 Stage 2 正式搜索时间

| 方案 | configs | d14 proxy | d18 proxy |
|------|---------|-----------|-----------|
| 折衷 [16,8,4] | 28 | ~6h | ~24h |
| 论文对齐 [64,32,16] | 112 | ~4天 | ~7天 |

d24 target 验证 (3 configs × 1000 steps): ~2h

### 6.3 全流程总时间

| 阶段 | 时间 |
|------|------|
| Stage 0 (dry-run) | ~0.5h |
| Stage 1 (d10→d24 快速验证 + 报告) | ~5-6h |
| Stage 2 (d14→d24 正式搜索 + 报告) | ~8-26h |
| **总计** | **~1-3天** |

---

## 7. LR Schedule 与训练语义

### 7.1 CLIMB mid-training = Annealing

CLIMB 论文使用 WSD schedule，在 stable stage 之后做 decay（annealing）。这是"退火"，不是"重新热身"。

- Annealing = "fine-tune model behavior preferences"，不是"learn new knowledge"
- 知识已在 base checkpoint 中，退火只调整输出偏好
- 数据混合影响的是模型"偏好"什么领域，不是"学习"什么知识

### 7.2 nanochat mid_train 默认参数 = 正确的 annealing

nanochat mid_train.py 的默认参数适合 CLIMB annealing:
- lr_scale=1.0 → peak LR = base's final LR ≈ 0.001 (不是 base peak 0.02)
- warmup=0.0 → 无重新热身，直接从 base 的 LR 开始
- warmdown=0.9 → 90% 步数用于退火衰减
- ratio=0.5 → 训练量 = 0.5 × scaling_params

### 7.3 lr_scale 语义澄清

**lr_scale 是相对于 base 的 final LR，不是 peak LR**:

```
optimizer.load_state_dict()  → group["lr"] = base's final LR ≈ 0.001
initial_lr = group["lr"] × lr_scale

lr_scale=1.0  → peak = 0.001 (温和延续退火)
lr_scale=0.5  → peak = 0.0005 (更低的退火, 几乎没有学习)
lr_scale=10   → peak = 0.01 (base peak × 0.5, 适合 continual pre-training)
lr_scale=20   → peak = 0.02 (= base peak, shock annealing)
```

CLIMB annealing 用 lr_scale=1.0 (默认) 即可。

### 7.4 ratio vs fixed tokens

nanochat 的 ratio 基于 scaling_params 计算:
```
d10: ratio=0.5 × 70.2M = 35M tokens (67 steps) ← 太少
d24: ratio=0.5 × 730M = 365M tokens (696 steps) ← 合理
```

对于 proxy 搜索，d10 的 67 steps 太少，差异无法显现。应该用 `--num-iterations` 固定训练量:
- proxy 搜索: 500-1000 steps (不依赖 ratio × scaling_params)
- target 验证: 1000-2000 steps

CLIMB 论文用固定 40B tokens，也是独立于模型大小的。

---

## 8. 未解决的问题与后续步骤

### 8.1 需要验证的事项

1. d10/d14/d18 的实际 step time（需要跑一个短训练测试）
2. d14/d18 的 base_eval CORE 分数（确认有足够信号）
3. d24 的 base checkpoint 预训练（Stage 1 验证需要）
4. proxy→target 可迁移性验证: d10 搜索的最优混合在 d24 上是否也优于 baseline

### 8.2 需要更新的代码和脚本

1. 修改 ProxyRunner: 使用 --num-iterations 固定训练量，不用 ratio
2. 修改搜索指标: 用高信号任务子集，不用全 22 CORE 平均
3. 更新 shell scripts: Stage 1 = d10 proxy + d24 target 验证
4. 更新 mid_train.py: 确保 annealing 参数正确 (lr_scale=1.0, warmup=0.0)
5. 写 d24 base 预训练脚本 (Stage 1 需要的 d24 checkpoint)

### 8.3 需要记录在报告中的说明

- nanochat VE 架构特殊性 + scaling_params vs total_params 的区别
- proxy 选择理由 (对标 CLIMB 62M/132M/350M 消融)
- target scaling=730M (0.56× CLIMB 1.3B) 的说明
- 指标选择理由 (高信号子集 vs 全 22 任务)
- Stage 1 用 d10→d24 验证 proxy→target 可迁移性的设计理由

---

## 附录: nanochat depth 详细参数表

偶数 depths, head_dim=128, aspect_ratio=64:

| depth | n_embd | n_head | scaling(M) | VE(M) | wte(M) | total(M) | VE占比 |
|-------|--------|--------|-----------|-------|--------|----------|--------|
| 4 | 256 | 2 | 3.2 | 4.2 | 0.8 | 8.2 | 51.2% |
| 6 | 384 | 3 | 17.6 | 22.6 | 1.3 | 41.5 | 54.5% |
| 8 | 512 | 4 | 40.4 | 51.7 | 1.7 | 93.8 | 55.0% |
| 10 | 640 | 5 | 70.2 | 104.9 | 21.0 | 196.0 | 53.5% |
| 12 | 768 | 6 | 110.1 | 151.0 | 25.2 | 286.3 | 52.7% |
| 14 | 896 | 7 | 164.2 | 205.5 | 29.4 | 399.1 | 51.5% |
| 16 | 1024 | 8 | 234.9 | 268.4 | 33.6 | 536.9 | 50.0% |
| 18 | 1152 | 9 | 324.4 | 339.7 | 37.7 | 701.9 | 48.4% |
| 20 | 1280 | 10 | 435.2 | 419.4 | 41.9 | 896.5 | 46.8% |
| 22 | 1408 | 11 | 569.5 | 507.5 | 46.1 | 1123.2 | 45.2% |
| 24 | 1536 | 12 | 729.8 | 604.0 | 50.3 | 1384.1 | 43.6% |
| 26 | 1664 | 13 | 918.4 | 708.8 | 54.5 | 1681.8 | 42.1% |
| 28 | 1792 | 14 | 1137.7 | 822.1 | 58.7 | 2018.5 | 40.7% |
| 30 | 1920 | 15 | 1390.0 | 943.7 | 62.9 | 2396.7 | 39.4% |

公式:
- n_embd = depth × 64 (偶数 depths)
- n_head = depth / 2
- per_layer_transformer = 12 × n_embd²
- lm_head = 32768 × n_embd
- VE = (depth/2) × 32768 × n_embd (交替层)
- wte = 32768 × n_embd
- scaling_params = transformer_matrices + lm_head
- total_params = scaling_params + VE + wte

奇数 depths 不推荐 (n_embd rounding 导致与下一个偶数 depth 共享宽度)。
