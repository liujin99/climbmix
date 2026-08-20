# Scoring Metric Design: SNR-Weighted Accuracy + NLL

> 记录时间: 2026-08-20
> 目的: 系统记录 ClimbMix 项目中评分指标的设计思路、公式推导、被否决方案、参考数据和最终实现

---

## 1. 背景与目标

ClimbMix 通过 **proxy 搜索** 寻找最优 STEM 数据混合配比。核心流程：

1. 采样多组数据混合配方（cluster 权重向量）
2. 每组配方训练一个 d20 proxy 模型（435M scaling, 500 步）
3. 在 6 个 STEM benchmark 上评测，得到每个 benchmark 的 accuracy 和 NLL
4. 将多 benchmark 的多指标合成一个 **score**
5. 用 LightGBM 学习"配比 → score"映射，引导下一轮采样
6. 3 轮迭代后输出最优配比

**本文档关注第 4 步**：如何将 accuracy 和 NLL 合成一个 score。

### 1.1 6 个 STEM Benchmark

| Benchmark | 类型 | 题目数 K | 评测方式 |
|-----------|------|---------|---------|
| arc_easy | multiple_choice | 2376 | 10-shot |
| arc_challenge | multiple_choice | 1172 | 10-shot |
| mmlu_stem | multiple_choice | 3545 | 0-shot (旧版) |
| gpqa_diamond | multiple_choice | 198 | 0-shot |
| gsm8k_cot | generation | 1319 | 5-shot |
| math_cot_500 | generation | 500 | 5-shot |

题目数 K 是评分公式中的关键参数——K 越大，二项噪声越小。

---

## 2. 两个评测指标

### 2.1 Accuracy（正确率）

- **定义**：模型回答正确的比例
- **Centering**：`centered_accuracy = accuracy - baseline`，baseline 为固定参考值（非 d14/d28），centering 不影响 z-score（z-score 自带去均值）
- **来源**：`nanochat-npu/scripts/base_eval.py` 输出 4 列 CSV（task_name, accuracy, centered_accuracy, nll）
- **数据流**：`base_eval.py` → CSV → `proxy_runner.py` 的 `_parse_eval_results()` 解析为 `per_task_accuracies: Dict[str, float]`
- **方向**：越高越好

### 2.2 NLL（负对数似然）

- **定义**：模型对正确答案的负对数似然损失。即使答案错误，NLL 也能反映模型对正确答案的"接近程度"
- **来源**：`nanochat-npu/nanochat/core_eval.py` 的 `judge_example()` 返回 `(is_correct, nll_gold)`；`evaluate_task()` 收集所有样本的 NLL，DDP all_reduce 汇总
- **数据流**：`core_eval.py` → NLL 列写入 CSV → `proxy_runner.py` 解析为 `per_task_nlls: Dict[str, float]`
- **方向**：越低越好（在公式中取负，统一成"越高越好"）

### 2.3 为什么需要两个指标

6 个 STEM benchmark 难度跨度极大：

- **简单 benchmark**（arc_easy 68.8%, mmlu_stem 29.1%）：accuracy 远超随机基线，config 间的 accuracy 差异是真实信号
- **难 benchmark**（gpqa 22.2%≈随机 25%, gsm8k 1.67%, math 0%）：accuracy 接近随机或为零，config 间的 accuracy 差异是噪声

**单靠 accuracy 无法区分难 benchmark 上的数据混合质量**。NLL 在这些 benchmark 上提供更平滑的梯度信号——即使模型答错，不同数据混合会影响模型对正确答案的置信度，从而影响 NLL。

---

## 3. 核心问题：Accuracy 在难 Benchmark 上是噪声

### 3.1 二项噪声模型

每个 benchmark 的 accuracy 观测值包含 **真实信号** 和 **二项噪声**：

```
σ²_noise = p(1-p) / K    # p = 真实正确率, K = 题目数
```

- K 大 → 噪声小 → accuracy 可靠
- K 小 → 噪声大 → accuracy 不可靠
- p ≈ 0.5 → 噪声最大（p(1-p) = 0.25）
- p ≈ 0 或 1 → 噪声最小

### 3.2 实际数据验证

以 d20 base model 为例：

| Benchmark | K | d20 acc | 随机基线 | p(1-p)/K (实际噪声) | 0.25/K (保守噪声) | 信号？ |
|-----------|------|---------|---------|---------------------|-------------------|--------|
| arc_easy | 2376 | 68.8% | 25% | 0.0000856 | 0.000105 | 强 |
| arc_challenge | 1172 | 36.9% | 25% | 0.000200 | 0.000213 | 中 |
| mmlu_stem | 3545 | 29.1% | 25% | 0.0000499 | 0.0000706 | 强(K大) |
| gpqa_diamond | 198 | 22.2% | 25% | 0.000870 | 0.001263 | 无 |
| gsm8k_cot | 1319 | 1.67% | 0% | 0.0000125 | 0.000190 | 无 |
| math_cot_500 | 500 | 0.00% | 0% | 0 | 0.000500 | 无 |

**关键发现**：
- 简单 benchmark：config 间 accuracy 方差（~0.0005）远大于噪声 → accuracy 有信号
- 难 benchmark：config 间 accuracy 方差接近或小于噪声 → accuracy 是噪声
- 0.25/K 比实际 p(1-p)/K 保守（最多 15 倍），确保不高估信号

---

## 4. 设计演进：被否决的方案

### 4.1 方案 A：纯 Accuracy

**思路**：仅用 accuracy z-score 的均值作为 score，不用 NLL。

**否决理由**：
- gsm8k (1.67%) 和 math (0%) 的 accuracy 几乎为零，z-score 不稳定
- gpqa (22.2%≈随机 25%) 的 accuracy 方差可能被噪声主导
- **浪费了 NLL 在难 benchmark 上的梯度信息**
- 如果 3/6 个 benchmark 的 accuracy 是噪声，score 会被噪声严重稀释

### 4.2 方案 B：两个 LightGBM 预测器（Acc + NLL, R² 加权）

**思路**：
1. 用一个 LightGBM 预测 accuracy，另一个预测 NLL
2. 各自计算 R²，按 R² 加权组合两个预测器的推荐

**否决理由**：
- **R² 在 N=27 时不可靠**：27 个样本训练 LightGBM，R² 的方差很大，无法可信地比较两个预测器的质量
- **NLL predictor 会主导**：NLL 的 R² 通常更高（NLL 是连续值，比 0/1 的 accuracy 更容易预测），会导致 NLL 主导推荐，违反 accuracy-primary 原则
- **聚合目标丢失 per-benchmark 信息**：两个 predictor 分别预测 aggregate accuracy 和 aggregate NLL，无法在 per-benchmark 层面做 SNR 加权

### 4.3 方案 C：固定权重

**思路**：对每个 benchmark 固定一个 accuracy/NLL 混合权重（如简单 benchmark w=0.8, 难 benchmark w=0.2）。

**否决理由**：
- 权重选择主观，需要人工调参
- 无法自适应：不同 proxy 深度、不同数据池规模下，最优权重可能不同
- 没有利用 noise 和 between-config variance 的统计信息

### 4.4 方案对比

| 方案 | 优点 | 缺点 | 结论 |
|------|------|------|------|
| A. 纯 Accuracy | 简单 | 难 benchmark 全是噪声 | 否决 |
| B. 双 Predictor | 理论上自适应 | R² 不可靠(N=27), NLL 主导 | 否决 |
| C. 固定权重 | 简单可控 | 主观, 不自适应 | 否决 |
| **D. SNR 加权** | **零人工参数, 自适应** | **依赖方差估计(N小时不稳定)** | **采用** |

---

## 5. 最终设计：SNR 加权评分

### 5.1 核心公式

对每个 benchmark b，在所有 N 个累积 config 上：

```
1. Z-score 标准化
   acc_z = (acc - mean(acc)) / (std(acc) + ε)
   nll_z = -(nll - mean(nll)) / (std(nll) + ε)    # 取负: NLL 越低越好 → z 越高越好

2. SNR 权重
   σ²_noise  = 0.25 / K                          # 二项噪声方差上界 (p=0.5 最差情况)
   σ²_between = var(acc) + ε                      # config 间真实 accuracy 方差
   f = 1 - σ²_noise / σ²_between                  # 信号占比, 可为负
   w = max(w_floor, min(1, max(0, (1+f)/2)))      # accuracy 权重, clamp 到 [w_floor, 1]

3. 加权评分
   score_b = w × acc_z + (1-w) × nll_z

4. 跨 benchmark 平均
   score = mean(score_b across 6 benchmarks)
```

### 5.2 权重 w 的行为

| f 值 | 含义 | w (w_floor=0) | w (w_floor=0.5) | 效果 |
|------|------|---------------|-----------------|------|
| f → 1 | noise << between | → 1.0 | → 1.0 | 纯 accuracy |
| f = 0 | noise = between | 0.5 | 0.5 | 50/50 |
| f → -∞ | noise >> between | → 0.0 | 0.5 | 纯 NLL / 50-50 |

**关键设计决策**：`f` 不 clamp 到 [0, 1]，而是允许为负。这样 w 可以低于 0.5，直到 0（纯 NLL）。clamp 作用在 w 上：`max(0, (1+f)/2)` 确保 w ≥ 0，然后 `max(w_floor, ...)` 施加下限。

### 5.3 0.25/K 的选择理由

二项分布方差为 `p(1-p)/K`，在 p=0.5 时取最大值 `0.25/K`。

| 方案 | 公式 | 优点 | 缺点 |
|------|------|------|------|
| p(1-p)/K | 精确噪声 | 精确 | 需要 p 估计; p≈0 时噪声≈0 导致 f→1 (误判为强信号) |
| **0.25/K** | **上界** | **保守, 无 p 依赖, 零参数** | **对 p≠0.5 的 benchmark 过度保守** |

选择 0.25/K 的理由：
1. **保守 = 安全**：宁可低估 accuracy 信号、多用 NLL，也不要高估噪声信号。过度保守的后果只是多用 NLL，而高估信号的后果是用噪声做决策
2. **零人工参数**：0.25 是数学常数（p=0.5 时的上界），不需要调参
3. **无 p 依赖**：不需要估计每个 config 的真实 p，避免 p 估计本身的噪声
4. **对 p≈0 的 benchmark 特别重要**：gsm8k (p=0.017) 的实际噪声 = 0.0000125，但 0.25/K = 0.000190（15 倍保守）。如果用精确噪声，f 会被误判为接近 1（强信号），实际上 accuracy 几乎全噪声

### 5.4 w_floor = 0.0 的选择理由

| w_floor | 效果 | 适用场景 |
|---------|------|---------|
| 0.0 | hard benchmark 可达 w=0 (纯 NLL) | 默认, 先让数据说话 |
| 0.5 | hard benchmark 至少 50/50 | 如果 NLL 信号也不够时的安全网 |

选择 0.0 作为默认值：
- **让数据决定**：如果 accuracy 确实是噪声，w 应该是 0，不应该人为设下限
- **第一轮后可调**：跑完 iteration 1 后检查实际 w 值，如果 NLL 信号不足，可以调到 0.5 作为安全网
- **w=0.5 是否足够是未知数**：在讨论中无法确定 0.5 是否足够保守，所以从 0.0 开始更安全

### 5.5 每轮重算全部 Score

z-score 和 SNR 权重依赖**全局统计量**（mean, std, var），加入新 config 后这些统计量会变。因此每轮迭代：

1. 将新 config 的 per-benchmark (acc, nll) 追加到 `_accumulated_per_benchmark`
2. 调用 `_compute_scores()` 重新计算**所有**累积 config 的 score（不只是新 config 的）
3. 用重算后的全部 score 作为 LightGBM 的训练目标

这意味着前几轮 config 的 score 会在后续轮次中变化，这是正确行为——全局统计量变了，相对排名也应变。

---

## 6. 参考代理模型得分

### 6.1 d14/d20/d28 三深度对比

d20 (scaling=435M) 是 proxy 深度，d14 (scaling=164M) 和 d28 (scaling=1138M) 是参考基线。

| Benchmark | K | d14 centered | d20 centered | d28 centered | d20 acc | 趋势 |
|-----------|------|------------|-------------|-------------|---------|------|
| arc_easy | 2376 | +0.5056 | +0.5842 | +0.6538 | 68.8% | d14 < d20 < d28 |
| arc_challenge | 1172 | +0.1115 | +0.1581 | +0.2787 | 36.9% | d14 < d20 < d28 |
| mmlu_stem | 3545 | +0.0247 | +0.0552 | +0.0597 | 29.1% | d14 < d20 < d28 |
| gpqa_diamond | 198 | -0.0505 | -0.0370 | +0.0236 | 22.2% | d14 < d20 < d28 |
| gsm8k_cot | 1319 | +0.0190 | +0.0167 | +0.0235 | 1.67% | d14 ≈ d20 < d28 |
| math_cot_500 | 500 | +0.0000 | +0.0000 | +0.0020 | 0.00% | d14 = d20 < d28 |

**关键结论**：
- d20 的 centered 分数在 d14 和 d28 之间（gsm8k 微幅波动属噪声），**d20 作为 proxy 深度合理**
- 简单 benchmark 有明显递增趋势 → accuracy 信号强
- 难 benchmark 趋势极弱（gsm8k 0.019→0.017→0.024, math 0→0→0.002）→ accuracy 信号弱

### 6.2 预期 SNR 权重

基于 d20 base model 的 accuracy 水平和 K，proxy 搜索中预期 w 值（w_floor=0.0）：

| Benchmark | K | σ²_noise (0.25/K) | 预期 σ²_between | 预期 f | 预期 w | 主导指标 |
|-----------|------|-------------------|----------------|--------|--------|---------|
| arc_easy | 2376 | 0.000105 | ~0.005 | ~0.98 | ~0.99 | accuracy |
| arc_challenge | 1172 | 0.000213 | ~0.003 | ~0.93 | ~0.97 | accuracy |
| mmlu_stem | 3545 | 0.0000706 | ~0.004 | ~0.98 | ~0.99 | accuracy |
| gpqa_diamond | 198 | 0.001263 | ~0.0003 | < 0 | ~0.0 | NLL |
| gsm8k_cot | 1319 | 0.000190 | ~0.00002 | < 0 | ~0.0 | NLL |
| math_cot_500 | 500 | 0.000500 | ~0 | -∞ | ~0.0 | NLL |

注：σ²_between 是 proxy 搜索中不同数据混合 config 间的 accuracy 方差，实际值需第一轮数据确认。上表为基于 d14→d28 变化幅度的估计。

**注意 mmlu_stem**：accuracy 29.1% 接近随机 25%，但因为 K=3545（最大），噪声极小（0.0000706），所以即使 accuracy 绝对值不高，w 也会很高。**SNR 公式关注的是"config 间方差 vs 噪声"的比值，不是绝对 accuracy 水平。**

---

## 7. LightGBM 预测器

### 7.1 输入与输出

| 维度 | 内容 | 说明 |
|------|------|------|
| 输入 X | mixture_weights (10 维) | K_enhanced=10 个 cluster 的权重向量 |
| 目标 y | -score | 取负（LightGBM 做 minimization, score 要 maximize） |

### 7.2 正则化参数

N=27（27 个 proxy config）时样本极少，必须强正则化：

| 参数 | 值 | 理由 |
|------|-----|------|
| max_depth | 3 | 限制树深度，防过拟合 |
| min_samples_leaf | 3 | 每叶至少 3 样本 |
| colsample | min(1.0, max(0.3, 20.0/K)) = 1.0 | K=10 → 1.0, 不做列采样 |
| n_estimators | 100 | 树数量 |
| learning_rate | 0.1 | 默认 |

### 7.3 搜索流程

1. **训练**：用所有累积 config 的 (weights, -score) 训练 LightGBM
2. **采样**：Dirichlet 采样 + 多浓度级别生成 100K+ 候选配比
3. **预测**：LightGBM 预测每个候选的 score
4. **精炼**：在 top 预测附近局部搜索 5K 候选
5. **输出**：选预测 score 最高的候选作为下一轮 config

### 7.4 验证集划分

- N < 10：不划分（全量训练）
- N >= 10：20% 验证集，用于报告 R²
- 固定 random_state=42，保证可复现

---

## 8. 完整 Pipeline 流程

### 8.1 三阶段设计

```
Stage 0: 数据加载
  └─ 加载预处理的 parquet shards + metadata

Stage 1: 聚类发现 (ClusterDiscovery)
  ├─ sentence embedding 聚类
  ├─ K_base=3 粗聚类 + K_enhanced=10 细聚类
  └─ 输出: cluster labels + per-cluster token counts

Stage 2: 迭代搜索 (IterativeBootstrapper)
  ├─ Iteration 1: Dirichlet 采样 15 个 config
  │   ├─ 每个 config 训练 d20 proxy (500 步, 30% 通用数据混合)
  │   ├─ 6 benchmark 评测 → per-benchmark (acc, nll)
  │   ├─ _compute_scores() → SNR 加权 score
  │   └─ 训练 LightGBM (X=weights, y=-score)
  │
  ├─ Iteration 2: LightGBM 引导采样 8 个 config → 训练 → 评测 → 重算全部 score → 重训 LightGBM
  ├─ Iteration 3: LightGBM 引导采样 4 个 config → 同上
  │
  └─ 最终: LightGBM 在 105K 候选中搜索最优配比

Stage 3: 输出最优数据混合权重
```

### 8.2 每轮迭代细节

1. **采样 config**：
   - Iteration 1：Dirichlet 采样（无先验信息）
   - Iteration 2+：LightGBM 预测引导采样

2. **Proxy 训练**：
   - 模型：d20 (scaling=435M, total=896M)
   - 训练量：500 steps（固定，不依赖 ratio × scaling_params）
   - LR：lr_scale=1.0, warmup=0.0, warmdown=0.9（annealing 语义）
   - 数据：30% 通用数据 + 70% 按 cluster 权重采样的 STEM 数据

3. **评测**：
   - 6 个 STEM benchmark
   - accuracy + NLL（per-benchmark dict）
   - DDP all_reduce 汇总

4. **评分**：
   - 追加 per-benchmark (acc, nll) 到累积列表
   - 调用 `_compute_scores()` 重算所有 score
   - score = SNR 加权的 per-benchmark z-score 平均

5. **预测器训练**：
   - X = mixture_weights, y = -score
   - LightGBM, max_depth=3, min_samples_leaf=3

6. **搜索**：
   - Dirichlet 多浓度采样 + LightGBM 预测
   - 105K 候选评估

### 8.3 资源估算

| 项目 | 数值 |
|------|------|
| 总 config 数 | 27 (15+8+4) |
| 每 config 训练时间 | ~2h (500 steps, 8×910B NPU) |
| 每 config 评测时间 | ~25min (6 benchmark) |
| 总搜索时间 | ~2.5 天 |
| 最终搜索候选数 | ~105K |

---

## 9. 实现细节

### 9.1 关键文件

| 文件 | 作用 | 状态 |
|------|------|------|
| `nanochat-npu/nanochat/core_eval.py` | NLL 收集 (`judge_example` 返回 `(is_correct, nll_gold)`) | 已提交 |
| `nanochat-npu/scripts/base_eval.py` | 4 列 CSV 输出 (task_name, accuracy, centered_accuracy, nll) | 已提交 |
| `climbmix/src/climbmix/pipeline/proxy_runner.py` | 解析 CSV → `per_task_accuracies` + `per_task_nlls` | 已提交 |
| `climbmix/src/climbmix/core/types.py` | `BENCHMARK_SIZES`, `ProxyResult`, `SearchConfig.w_floor` | 已修改 |
| `climbmix/src/climbmix/core/iterative_bootstrapper.py` | `_compute_scores()` SNR 公式实现 | 已修改 |
| `climbmix/scripts/run_climb.py` | CLI 默认值 (configs_per_iter, K_enhanced, proxy_depth) | 已修改 |

### 9.2 _compute_scores() 实现

```python
def _compute_scores(self) -> npt.NDArray[np.float64]:
    N = len(self._accumulated_per_benchmark)
    if N == 0:
        return np.array([], dtype=np.float64)

    benchmarks = self.config.val_tasks
    per_config_scores = np.zeros(N, dtype=np.float64)

    for b in benchmarks:
        accs = np.array([(d[0] or {}).get(b, 0.0) for d in self._accumulated_per_benchmark])
        nlls = np.array([(d[1] or {}).get(b, 0.0) for d in self._accumulated_per_benchmark])

        acc_z = (accs - accs.mean()) / (accs.std() + 1e-12)
        nll_z = -(nlls - nlls.mean()) / (nlls.std() + 1e-12)

        K = BENCHMARK_SIZES.get(b, 1000)
        sigma2_noise = 0.25 / K
        sigma2_between = float(accs.var()) + 1e-12
        f = 1.0 - sigma2_noise / sigma2_between
        w = max(self.w_floor, min(1.0, max(0.0, (1.0 + f) / 2.0)))

        per_config_scores += w * acc_z + (1.0 - w) * nll_z

    per_config_scores /= len(benchmarks)
    return per_config_scores
```

### 9.3 BENCHMARK_SIZES

```python
BENCHMARK_SIZES = {
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "mmlu_stem": 3545,
    "gpqa_diamond": 198,
    "gsm8k_cot": 1319,
    "math_cot_500": 500,
}
```

### 9.4 ProxyResult 结构

```python
@dataclass
class ProxyResult:
    mixture_config: MixtureConfig
    validation_loss: float
    validation_accuracy: float      # aggregate accuracy
    validation_nll: float           # aggregate NLL
    per_task_accuracies: Optional[Dict[str, float]]  # per-benchmark centered accuracy
    per_task_nlls: Optional[Dict[str, float]]        # per-benchmark NLL

    @property
    def score(self) -> float:
        return self.validation_accuracy  # 仅 aggregate, 实际评分用 _compute_scores()
```

`score` 属性保留兼容性，但实际评分由 `_compute_scores()` 从 `per_task_accuracies` 和 `per_task_nlls` 计算。

### 9.5 run_iteration() 评分流程

```python
# 1. 训练 + 评测 → 收集 per-benchmark 数据
if proxy_runner is not None:
    results = proxy_runner.run_batch(new_configs)
    for r in results:
        self._accumulated_configs.append(r.mixture_config)
        self._accumulated_per_benchmark.append(
            (r.per_task_accuracies, r.per_task_nlls)
        )

# 2. 重算所有 score
all_scores = self._compute_scores()
self._accumulated_scores = all_scores.tolist()

# 3. 当前轮的 score = 重算结果的最后 n_new 个
n_new = len(trained_configs)
scores_arr = all_scores[-n_new:]

# 4. 训练 predictor
predictor_targets = -all_scores  # maximize score → minimize -score
```

---

## 10. 验证计划

### 10.1 第一轮后检查项

| 检查项 | 预期 | 异常处理 |
|--------|------|---------|
| w 值 | easy ~0.99, hard ~0.0 | 如果 hard benchmark w > 0, 说明 between-config variance 大, 检查是否合理 |
| NLL z-score 分布 | hard benchmark 的 nll_z 应有区分度 | 如果 nll_z 全 ≈ 0, 说明 NLL 也没信号, 该 benchmark 贡献为 0 |
| score 分布 | 有正有负, 有区分度 | 如果 score 全接近 0, 说明数据混合差异不大 |
| predictor R² | 可能较低 (N=15) | 正常, 第三轮 (N=27) 后会改善 |
| w_floor 是否需调整 | w=0.0 应该 OK | 如果 hard benchmark 的 NLL 也不稳定, 考虑 w_floor=0.5 |

### 10.2 已验证的测试

- **类型检查**：BENCHMARK_SIZES, ProxyResult.score, SearchConfig, K_enhanced, max_depth, min_samples_leaf, proxy_depth ✅
- **_compute_scores() 单元测试**：fake data 验证 w 值范围、score 有限性 ✅
- **w_floor 效果测试**：0.0 → hard benchmark w=0; 0.5 → hard benchmark w=0.5 ✅
- **端到端 pipeline 测试**：2 轮迭代 + 105K 候选搜索, 无 proxy_runner (random fake data) ✅
- **语法检查**：types.py, iterative_bootstrapper.py, run_climb.py ✅

### 10.3 尚未验证

- **真实 proxy 训练 + 评测**：需要 8×910B NPU, 验证 per_task_accuracies 和 per_task_nlls 的真实值
- **NLL 的 between-config variance**：NLL 是否真的有区分度（核心假设）
- **proxy → target 可迁移性**：d20 proxy 搜索的最优混合在 d28 target 上是否也优
- **NPU embedding 兼容性**：sentence_transformers + stella_en_400M_v5 在 Ascend NPU 上的运行

---

## 11. 配置参数汇总

### 11.1 搜索参数

| 参数 | 值 | 文件 | 理由 |
|------|-----|------|------|
| configs_per_iter | [15, 8, 4] | types.py: SearchConfig | 27 总 configs, ~2.5 天 |
| num_iterations | 3 | types.py: SearchConfig | 3 轮迭代 |
| w_floor | 0.0 | types.py: SearchConfig | 无 floor, 第一轮后可调 |
| K_enhanced | 10 | types.py: ClusterDiscoveryConfig | 10 个细 cluster |
| K_base | 3 | types.py: ClusterDiscoveryConfig | 3 个粗 cluster |

### 11.2 Proxy 参数

| 参数 | 值 | 文件 | 理由 |
|------|-----|------|------|
| proxy_depth | 20 (d20) | types.py: ProxyConfig | scaling=435M, 在 d14/d28 之间 |
| proxy_iterations | 500 | types.py: ProxyConfig | 固定训练量, 不依赖 ratio |
| lr_scale | 1.0 | types.py: ProxyConfig | annealing 语义 |
| warmup | 0.0 | types.py: ProxyConfig | 无重新热身 |
| warmdown | 0.9 | types.py: ProxyConfig | 90% 步数退火 |

### 11.3 Predictor 参数

| 参数 | 值 | 文件 | 理由 |
|------|-----|------|------|
| max_depth | 3 | types.py: PredictorConfig | 强正则化 (N=27) |
| min_samples_leaf | 3 | types.py: PredictorConfig | 每叶至少 3 样本 |
| n_estimators | 100 | types.py: PredictorConfig | 默认 |
| learning_rate | 0.1 | types.py: PredictorConfig | 默认 |

### 11.4 评分参数

| 参数 | 值 | 文件 | 理由 |
|------|-----|------|------|
| noise_formula | 0.25/K | iterative_bootstrapper.py | 二项方差上界, 保守 |
| z_score_epsilon | 1e-12 | iterative_bootstrapper.py | 防 std=0 |
| var_epsilon | 1e-12 | iterative_bootstrapper.py | 防 var=0 |
| metric_direction | maximize | types.py: CLIMBConfig | accuracy 越高越好 |

---

## 附录: 术语表

| 术语 | 含义 |
|------|------|
| proxy | 小模型 (d20, 435M scaling), 快速验证数据混合效果 |
| target | 大模型 (d28, 1138M scaling), 最终训练用 |
| config | 一组数据混合配方 (cluster 权重向量) |
| score | 一个 config 的评分 (SNR 加权的 per-benchmark z-score 平均) |
| SNR | Signal-to-Noise Ratio, 信号噪声比 |
| w | accuracy 在评分中的权重, [w_floor, 1] |
| f | 信号占比 = 1 - noise/between, 可为负 |
| K | benchmark 题目数 |
| z-score | 标准化分数 = (x - mean) / std |
| w_floor | accuracy 权重下限, 默认 0.0 |
| annealing | 退火, 从 base checkpoint 的 final LR 开始衰减 |
| VE | Value Embeddings, nanochat 特有, 占 ~50% 参数但不参与 FLOPs |
| scaling_params | transformer matrices + lm_head, 用于对标计算量 |

---

## 12. Proxy 训练量对比分析

> 记录时间: 2026-08-20
> 目的: 对比论文与我们的 proxy/target 训练量，记录参数选择理由

### 12.1 论文设置 (arXiv:2504.13161)

论文 Section 3.1 + Appendix C.4 明确记载：

| 参数 | 值 |
|------|-----|
| Phase-1 预训练 | 10T tokens (DCLM + TxT360), 256 H100 |
| Batch size | 2M tokens (全程) |
| 优化器 | AdamW, LR=5e-5 (stable) → 1e-5 (anneal) |
| LR schedule | WSD (Warmup-Stable-Decay) |
| Proxy 模型 | 350M (scaling), 62M (ablation) |
| Target 模型 | 1B, 40B tokens |
| Proxy 训练时间 | 45 GPU hours (per proxy model) |
| Target 训练时间 | 6,400 GPU hours |

### 12.2 推算论文 proxy token 量

用 target model 反推 step time，再推算 proxy：

```
Target: 1B 模型, 40B tokens, 6,400 GPU hours, 256 H100
  wall clock = 6400/256 = 25h
  steps = 40B / 2M = 20,000 steps
  step time = 25*3600/20000 = 4.5 sec/step (256 H100, 1B 模型)

Proxy: 350M 模型, 45 GPU hours
  step time ≈ 4.5 * (350M/1B) ≈ 1.6 sec/step
  wall clock = 45/256 = 10.5 min
  steps = 10.5*60/1.6 ≈ 394 steps
  tokens = 394 * 2M ≈ 788M tokens
```

**论文 proxy ≈ 800M tokens（~400 steps at 2M batch）**

### 12.3 我们的设置

| 参数 | 论文 | 我们 |
|------|------|------|
| Proxy 模型 | 350M (scaling) | d20 (435M scaling, 896M total) |
| Batch size | 2M tokens | ~524K tokens (512 × 1024) |
| Proxy 步数 | ~400 | **1000** |
| Proxy token 量 | ~800M | **524M** |
| Token 比例 | 1.0× | **0.65×** (论文的 65%) |
| 每 config 时间 | 10.5 min (256 H100) | ~4h (8×910B NPU) |
| 27 configs 总时间 | 4.7h | ~108h (4.5 天) |
| Target 模型 | 1B | d28 (1138M scaling) |
| Target 步数 | 20,000 | 1000 |
| Target token 量 | 40B | 524M |

### 12.4 为什么选 1000 步

| 方案 | 步数 | Token 量 | 论文比例 | 总时间 | 评估 |
|------|------|---------|---------|--------|------|
| 原 500 步 | 500 | 262M | 33% | 2.25 天 | 太少，信号可能被噪声淹没 |
| **1000 步** | **1000** | **524M** | **65%** | **4.5 天** | **合理折中** |
| 2000 步 | 2000 | 1B | 125% | 9 天 | 匹配论文但时间过长 |

选择 1000 步的理由：
1. **Token 量 524M，为论文的 65%**——大部分信号应该能显现
2. **4.5 天可行**——8×910B NPU 上 ~4h/config × 27 configs
3. **2000 步 (1B tokens) 匹配论文但 9 天太长**——其他步骤（聚类、评测）也需要时间
4. **论文的 base 预训练 10T tokens，我们的 base ~2-3B tokens**——base 差距大，proxy 差距可接受

### 12.5 H100 vs 910B NPU 换算说明

论文用 256 H100 GPU，我们用 8×910B NPU：

| | H100 | 910B |
|--|------|------|
| BF16 FLOPS | ~989 TFLOPS | ~310-376 TFLOPS |
| 相对算力 | 1.0× | ~0.32× |
| 数量 | 256 | 8 |
| 总算力比 | 256 | 2.6 (相对) |

总算力差距 ~100×，但我们不需要匹配 wall clock——关键是匹配 **token 量**（训练数据量），而非计算速度。1000 步 × 524K tokens = 524M tokens 已接近论文的 800M。

### 12.6 Target 训练 1000 步的风险

Target 524M tokens vs 论文 40B tokens（76× 差距）。但场景不同：
- 论文: 1B 模型从 10T 预训练 base 上 mid-train 40B
- 我们: d28 (1138M) 从 ~2-3B 预训练 base 上 mid-train 524M

**如果 target 差异不显著，proxy 搜索可能无法迁移。** 需要在第一轮后检查：
- 不同 config 间的 score 差异是否大于噪声
- proxy 最优 config 在 target 上是否也最优

如果差异不显著，考虑增加 target 步数到 2000（1B tokens, ~8h × 2 模型 = 16h）。
