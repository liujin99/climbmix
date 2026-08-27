# Deviations from the Nemotron-CLIMB Paper

> Paper: arXiv:2504.13161 (v2, 2025-11-30) — *Nemotron-CLIMB: CLustering-based
> Iterative Data Mixture Bootstrapping for Language Model Pre-training*
> 本文档记录我们复现中**有意偏离**论文的每一处、偏离原因、以及已核对一致的项。
> 相关文档: `docs/scoring_metric_design.md`(评分指标 + proxy/target 训练量对比)、
> `TODO.md`(决策记录)。

论文原文关键段已核对(§2.2 子程序、§3.1 实现细节、§3 实验设置、附录 C/D)。

## 偏差总表

| # | 维度 | 论文 | 我们 | 原因 |
|---|------|------|------|------|
| D1 | 搜索预算 | 112 configs(64/32/16,4:2:1,3 轮) | 35 configs(20/10/5,3 轮) | NPU 算力:~4h/config × 8×910B vs 论文 45 GPU-h × 256 H100 |
| D2 | 引导采样 | "从预测排序 top-N 中随机采 M 个"(M/N 未给值) | top_n = 3 × sample_from_top_m(=32→96),且在 top-N 基础上做 Dirichlet 探索而非均匀抽取 | 论文未指定 N/M 与抽取方式;3×M 是我们的实现选择 |
| D3 | 最终选择 | 最终 predictor 在设计空间 A 上取 argmax(A 的枚举方式未说明) | 4 个浓度级(1/5/10/50)× 25K Dirichlet 候选 + argmax 附近 5K 精搜 | 同为 predictor argmax,我们把 A 的枚举具体化 |
| D4 | LightGBM 超参 | max_depth=4,min_samples_leaf≥5,L1+L2,early stopping(20 轮无提升)+ 独立验证集 | max_depth=3,min_samples_leaf=3,L1=L2=1.0,early_stopping=20(带验证集切分),n_estimators=500,lr=0.02,auto_adjust 公式 | N=27~35 小样本下更强的容量限制;结构(L1/L2+早停)与论文一致 |
| D5 | 聚类数 K | 固定 K:主实验 21 个超簇(1000→剪枝 240→合并);消融 15/30 | 带宽 `K_final = clamp(natural_K(τ), 3, 15)` 池自适应 | 我们的池(580K docs)结构与 800B-token 池不同;带宽避免距离守卫强并语义不同的簇 |
| D6 | 合并阈值 | 欧氏距离阈值 1.5(§3.1,v2 明文) | τ=0.9 欧氏距离,作用在 **L2 归一化后**的 stella 向量(≈cos 0.60) | 论文的 1.5 所在空间(是否归一化)未说明,数值不可直接换算;0.9 是我们按 cos≈0.6 语义校准的守卫 |
| D7 | Random 基线小簇策略 | 未记载(其规模天然不会遇到:800B/21 簇/40B 预算 → 每簇配额 ~1.9B) | 簇配额不足 → 全取、不复制、不重分配(与 CLIMB 臂同一函数同一策略,两臂对称退化) | App. C.1 只定义等权 1/K;小池必须补一个策略且两臂一致 |
| D8 | 优化目标 | 下游任务验证集 accuracy(PIQA/ARC_E/HellaSwag) | SNR 加权 acc+NLL z-score(6 个 STEM benchmark) | 难 benchmark 上 accuracy 是二项噪声;见 scoring_metric_design.md |
| D9 | 通用数据混合 | 无(纯簇内配比) | 70% STEM + 30% ClimbMix(anti-forgetting) | 我们是从小 base 起步的 mid-training 场景(参照 MAI-Thinking-1 / Apple Intelligence) |
| D10 | Proxy/Target 规模 | proxy 350M ≈800M tokens;target 1B/40B tokens | proxy d20(435M scaling)1000 步 ≈524M;target d28 524M/1000 步 | 算力;详细换算见 scoring_metric_design.md §12 |
| D11 | token 计量 | 精确 tokenize(池统计) | chars/4 估算(元数据预计算列) | 池扫描免 tokenize;配比/配额的近似 |
| D12 | 评测子采样 | 全量 | speedrun 100 题/任务(fixed shuffle seed 1337,跨实验可比较);生产 -1 全量 | speedrun 时间预算;生产无偏差 |

## 细节与出处

### D1 搜索预算
论文 §3.1 + D.6:"three iterations of search with 64, 32, and 16 candidates
evaluated in iterations 1, 2, and 3, respectively, giving a total of 112
searches"(4:2:1 分配)。我们的 `CONFIGS_PER_ITER=20,10,5` 共 35。论文 Table 3
显示预算升到 150%/200% 仍有增益——若生产预算允许,升 D1 是第一个该动的旋钮。

### D2 引导采样
论文 §2.2 子程序 1:"sort all configurations in the weight space A … randomly
sample M new configurations from the top N ranked configurations"。M/N 均未给
数值。我们:`iterative_bootstrapper.py` 中 `top_n = 3 × sample_from_top_m`,
抽取方式为 `dirichlet_sampler.sample_from_top_n`(以 top-N 配比为基座的
Dirichlet 扰动,天然保持在单纯形上),而非从 top-N 均匀抽。

### D3 最终选择
论文 §2.2:"one selects the best configuration predicted by the final
predictor as the final data mixture weight"——纯 predictor argmax,但设计空间 A
如何枚举未说明。我们 `_search_full_design_space`:4 个浓度级 × 25K 候选中取
argmin(预测损失),再在其附近精搜 5K。语义一致,枚举方式是我们的具体化。

### D4 LightGBM
论文 §3.1:"we set L1 and L2 regularization, early stopping, a maximum depth
of four, and require at least five samples per leaf … halting training after
20 rounds of no improvement"(配独立验证集)。我们(types.py `PredictorConfig`)
结构一致(L1=L2=1.0、early_stopping_rounds=20、20% 验证集切分供早停与 R²),
差异在容量参数(max_depth 3 vs 4、min_samples_leaf 3 vs 5)与
n_estimators=500/lr=0.02/auto_adjust。

### D5/D6 聚类与合并
论文:stella_en_400M_v5 嵌入 + FAISS 球面 K-means(K_init=1000)→ fasttext
质量剪枝(阈值 3.0,1000→240)→ 欧氏 1.5 合并 → 21 超簇(主实验)。
我们:同一嵌入与 K-means 与剪枝阈值;合并改为带宽制(见 TODO "Cluster-count
band"),τ=0.9 作用在归一化向量上。**勘误**:早前审计记录"1.5/3.0 不在论文
正文"是基于 v1 的结论;v2 §3.1 两者均有明文。

### D7 Random 基线
论文 App. C.1:"each cluster is assigned an equal and uniform weight"。我们的
实现 = 同一选择器 + α_k=1/K + 同一 token 上限;小簇不足配额时全取不复制
(`data_selector.select_data_by_mixture` 既有策略),`.done` 记录计划/实际权重
与短缺簇。详见 TODO "Random baseline = equal cluster weights"。

### D10 训练规模
论文 proxy ≈800M tokens(45 GPU-h,由 6400 GPU-h target 反推,§C.4)、target
1B/40B。我们 proxy 524M(1000 步)、target 524M——token 口径为论文的 65%/1.3%
(base 池 ~2-3B vs 论文 10T,场景不同)。完整推导见 scoring_metric_design.md §12。

## 已核对一致(正向审计)

- 嵌入模型 stella_en_400M_v5(§3.1)
- FAISS 球面 K-means,K_init=1000(§2.1)
- fasttext 质量剪枝阈值 3.0、仅按质量不按大小(§2.1/§3.1)
- Dirichlet 初始化按各簇 token 数(§3.1)
- 迭代 bootstrapping 结构:采样→proxy 训练→predictor 拟合→引导再采样(§2.2)
- 最终选择 = predictor 预测 argmax(§2.2;枚举方式见 D3)
- Random 基线 = 等簇权 1/K(App. C.1;短缺策略见 D7)
- WSD 退火语义:stable 阶段可恢复、数据混合研究聚焦 decay 阶段(§C.4;我们
  lr_scale=1.0/warmup=0.0/warmdown=0.9 从 base checkpoint 直接退火)
- LightGBM 作为 predictor(§2.2 实现注)
