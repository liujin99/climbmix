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
| D2 | 引导采样 | "从预测排序 top-N 中随机采 M 个"(M/N 未给值) | 一致:top-N(N=3×sample_from_top_m=96)中无放回**原样**抽 M 个(N=96 的具体化是我们的选择;2026-08-28 回归论文字面语义,此前为 Dir(5·w) 扰动) | 扰动被数值实验证伪,见细节节 D2 |
| D3 | 最终选择 | 最终 predictor 在设计空间 A 上取 argmax(A 的枚举方式未说明) | 4 个浓度级(1/5/10/50)× 25K Dirichlet 候选 + argmax 附近 5K 精搜 | 同为 predictor argmax,我们把 A 的枚举具体化 |
| D4 | LightGBM 超参 | max_depth=4,min_samples_leaf≥5,L1+L2,early stopping(20 轮无提升)+ 独立验证集 | max_depth=3,min_samples_leaf=3,L1=L2=1.0,early_stopping=20(带验证集切分),n_estimators=500,lr=0.02,auto_adjust 公式 | N=27~35 小样本下更强的容量限制;结构(L1/L2+早停)与论文一致 |
| D5 | 聚类数 K | 固定 K:主实验 21 个超簇(1000→剪枝 240→合并);消融 15/30 | **elbow 定 K_ENHANCED=14**(2026-09-04;带宽 clamp 机制保留为池自适应默认) | 我们的池(116M docs,方向盆地极偏斜)上 natural_K(τ) 不稳(0.7→49/0.8→4/0.9→≤3)且 floor=3 退化(C0 独占 99.86% docs);14 = merge-distance elbow(最大跳变 0.0816),在论文粒度带 15-30 下沿(我们的搜索预算 35 vs 论文 112) |
| D6 | 合并阈值 | 欧氏距离阈值 1.5(§3.1,v2 明文) | τ=0.9 欧氏距离,作用在 **L2 归一化后**的 stella 向量(≈cos 0.60) | 论文的 1.5 所在空间(是否归一化)未说明,数值不可直接换算;0.9 是我们按 cos≈0.6 语义校准的守卫 |
| D7 | Random 基线小簇策略 | 未记载(其规模天然不会遇到:800B/21 簇/40B 预算 → 每簇配额 ~1.9B) | 簇配额不足 → 全取、不复制、不重分配(与 CLIMB 臂同一函数同一策略,两臂对称退化) | App. C.1 只定义等权 1/K;小池必须补一个策略且两臂一致 |
| D8 | 优化目标 | 下游任务验证集 accuracy(PIQA/ARC_E/HellaSwag) | SNR 加权 acc+NLL z-score(6 个 STEM benchmark) | 难 benchmark 上 accuracy 是二项噪声;见 scoring_metric_design.md |
| D9 | 通用数据混合 | 无(纯簇内配比) | 70% STEM + 30% ClimbMix(anti-forgetting) | 我们是从小 base 起步的 mid-training 场景(参照 MAI-Thinking-1 / Apple Intelligence) |
| D10 | Proxy/Target 规模 | proxy 350M ≈800M tokens;target 1B/40B tokens | proxy d20(435M scaling)1000 步 ≈524M;target d28(~1.5B scaling)1000 步 ≈1B | 算力;详细换算见 scoring_metric_design.md §12。token 口径为论文的 65%/2.5%,但退火预算相对 base 论文 +0.4%(40B/10T) vs 我们 +33~50%(1B/2-3B)——场景不同(小 base mid-training) |
| D11 | token 计量 | 精确 tokenize(池统计) | chars/4 估算(元数据预计算列) | 池扫描免 tokenize;配比/配额的近似 |
| D12 | 评测子采样 | 全量 | speedrun 100 题/任务(fixed shuffle seed 1337,跨实验可比较);生产 -1 全量 | speedrun 时间预算;生产无偏差 |
| D13 | 剪枝规则 | 簇平均质量 < 3.0 即剪(fasttext,§2.1/§3.1) | 平均阈值 + **单列下限** hybrid:任一质量列的簇均值 < `PRUNE_COLUMN_FLOOR`(生产默认 2.0,0=关)即剪 | 平均线漏检"格式干净但知识贫瘠"的簇(2026-08-31 20-分片画像:69/1000 簇过均线但 knowledge_value 1.8-2.0;6.6% docs / 仅 1.5% tokens);标签未校验故取保守档 2.0,见细节 D13 |

## 细节与出处

### D1 搜索预算
论文 §3.1 + D.6:"three iterations of search with 64, 32, and 16 candidates
evaluated in iterations 1, 2, and 3, respectively, giving a total of 112
searches"(4:2:1 分配)。我们的 `CONFIGS_PER_ITER=20,10,5` 共 35。论文 Table 3
显示预算升到 150%/200% 仍有增益——若生产预算允许,升 D1 是第一个该动的旋钮。

### D2 引导采样
论文 §2.2 子程序 1:"sort all configurations in the weight space A … randomly
sample M new configurations from the top N ranked configurations"。M/N 均
未给数值。我们:`top_n = 3 × sample_from_top_m`(=96),从 top-N 无放回
**原样**抽 M 个(固定每轮种子);N=96 这一具体化是我们的选择。

**历史(2026-08-28 回归)**:此前我们以 top-N 为基座做 Dir(5·w) Dirichlet
扰动("好配置附近加密采样")。数值实验(4000 样本 L1 距离)证明该扰动
是 K 盲的:K=3 时扰动比初始池更紧(meanL1 0.47 vs 0.59,符合"附近采样"
意图);K=15 时扰动后的点离基座 meanL1=1.07,比池内任意两点的平均间距
(0.97)还宽 — 扰动稀释了排序信号而非增加探索,此时严格劣于原样抽取
(K=21:1.20 vs 池 0.72)。论文的探索性来自 top-N 带宽(N≫M)+ 候选池
自身的随机性,无需额外旋钮。据此回归论文字面语义(用户决策)。
`DirichletSampler.sample_from_top_n` 保留,仅供最终选择的 argmax 附近
精搜使用(见 D3)。

### D3 最终选择
论文 §2.2:"one selects the best configuration predicted by the final
predictor as the final data mixture weight"——纯 predictor argmax,但设计空间 A
如何枚举未说明。我们 `_search_full_design_space`:4 个浓度级 × 25K 候选中取
argmin(预测损失),再在其附近精搜 5K。语义一致,枚举方式是我们的具体化。
注(2026-08-29,smoke_search 合成真值实验):低预算下纯 argmin 存在 optimizer's
curse——树不外推,宽池角落的预测是 guided 带内叶值,argmin 专挑乐观的错,
合成场景中选中的配置差于搜索自己已实测的最好配置(u 0.723 vs 1.009,N=112)。
**决策:仍按论文原始做法**。候选修法"argmin 结果补训一次再与实测最好者比较"
被否——多卡并行场景下批尾多一个实验要独占整组卡,约 +10h 串行尾延迟;该缺口
随 predictor 精度上升而收窄(论文 N=112、ρ=94% 时预测最优≈真实最优)。

### D4 LightGBM
论文 §3.1:"we set L1 and L2 regularization, early stopping, a maximum depth
of four, and require at least five samples per leaf … halting training after
20 rounds of no improvement"(配独立验证集)。我们(types.py `PredictorConfig`)
结构一致(L1=L2=1.0、early_stopping_rounds=20、20% 验证集切分供早停与 R²),
差异在容量参数(max_depth 3 vs 4、min_samples_leaf 3 vs 5)与
n_estimators=500/lr=0.02/auto_adjust。
注:`_compute_colsample`(predictor.py)的 colsample_bytree 公式在当前域
(K_final≤15)恒被钳到 1.0,即休眠无操作——所有特征(簇配比)始终全量参与,
非遗漏。
注:小 N 下 held-out Spearman 必然 nan(非缺陷):K 簇只有 K+ 维特征,
guided 采样的候选又全部落在 top-N 带内、特征向量高度相似,浅树
(depth≤3)把它们路由到同一片叶子 → 预测恒定 →秩方差为零 → ρ 无定义。
论文 D.10 的 94% 出现在 N=112(候选分散)。搜索状态里该值存 null
(RFC-8259 不允许字面量 NaN),报告显示 N/A。

### D5/D6 聚类与合并
论文:stella_en_400M_v5 嵌入 + FAISS 球面 K-means(K_init=1000)→ fasttext
质量剪枝(阈值 3.0,1000→240)→ 欧氏 1.5 合并 → 21 超簇(主实验)。
我们:同一嵌入与 K-means 与剪枝阈值;合并改为带宽制(见 TODO "Cluster-count
band"),τ=0.9 作用在归一化向量上。**勘误**:早前审计记录"1.5/3.0 不在论文
正文"是基于 v1 的结论;v2 §3.1 两者均有明文。
**2026-09-04 生产修订(D5)**:prod1 全池(116M docs / ~92B tokens)实测带宽
退化——natural_K(τ) 在 τ=0.7/0.8/0.9 下为 49/4/≤3,floor-stop 得 K=3 且 C0
独占 99.86% docs(搜索退化为单参数 f(w_C0),random 臂 383M vs CLIMB 臂 ~1B
tokens 不可比)。改用 merge-distance elbow 定 K_ENHANCED=14(最大跳变
0.0816;12 宏簇 ~7.6B tokens/簇 + 2 尘埃簇 0.07% docs)——即论文"固定 K"
语义落在我们的池上,仍在带宽 cap 15 内。kmeans 缓存不受影响(键 = K_init=1000
+ pool,K_ENHANCED 只作用于其后的 prune+merge)。

### D7 Random 基线
论文 App. C.1:"each cluster is assigned an equal and uniform weight"。我们的
实现 = 同一选择器 + α_k=1/K + 同一 token 上限;小簇不足配额时全取不复制
(`data_selector.select_data_by_mixture` 既有策略),`.done` 记录计划/实际权重
与短缺簇。详见 TODO "Random baseline = equal cluster weights"。

### D10 训练规模
论文 proxy ≈800M tokens(45 GPU-h,由 6400 GPU-h target 反推,§C.4)、target
1B/40B。我们 proxy 524M(1000 步)、target ~1B(1000 步 × ~1M batch,TARGET_TOKENS=1B
STEM cap)——token 口径为论文的 65%/2.5%(base 池 ~2-3B vs 论文 10T,场景不同;
d28 scaling 修正为 ~1.5B,2026-09-04,meta auto-detect)。完整推导见
scoring_metric_design.md §12。

### D13 剪枝规则(2026-08-31)
论文用 fasttext 分类器簇均值 < 3.0 剪枝。我们保留均值阈值
(`PRUNE_THRESHOLD=3.0`),叠加**单列下限** `PRUNE_COLUMN_FLOOR`(生产默认
2.0,0=关闭即论文语义):任一质量列的簇均值低于下限即剪。动机:2026-08-31
20-分片全量标签画像(`prune_rule_analysis.py` 离线分析 `prune_profile.json`)
发现 69/1000 簇过了均线但 knowledge_value 1.8-2.0——notation/noise 满分把
知识贫瘠平均掉了(公式表/无解法习题干类);该种群 6.6% docs 但仅 1.5%
tokens(弱簇均为短文档)。取 2.0 而非更狠的 2.5(后者 21% docs 且会连带
弱在其他列的族)的原因:5 列 scorer 本身未校验,下限越保守对标签噪声越
稳健——先保守跑,人工抽验被剪簇原文后再提档(改 knob 重跑秒级,pool
缓存命中)。实现:`prune_clusters(column_mins, column_floor)` +
`compute_cluster_column_mins`;无质量标签时 floor 与均值剪枝一同失效
(返回 {})。speedrun 默认 0(只验管道形状);该 knob 进 search-stage
fingerprint(语义变更 → Steps 1-3 重跑,embedding pool 缓存不受影响)。
**已实证(2026-08-31, 20-分片子集重跑,缓存命中 87.3s)**:71/1000 剪枝
(69 过均线但 floor 不及)、157,917 docs(6.81%)/ 1.51% tokens 移除——
与 prune_rule_analysis.py 预测逐位吻合;merge 树随 K_init=929 微移
(natural_K(0.7) 45→39, elbow 6→31),K_final 仍钉 floor=3 无影响。

## 已核对一致(正向审计)

- 嵌入模型 stella_en_400M_v5(§3.1)
- FAISS 球面 K-means,K_init=1000(§2.1)
- fasttext 质量剪枝阈值 3.0、仅按质量不按大小(§2.1/§3.1)
- Dirichlet 初始化按各簇 token 数(§3.1)
- 迭代 bootstrapping 结构:采样→proxy 训练→predictor 拟合→引导再采样(§2.2)
- "剪枝"语义(§2.2/Fig.4,2026-08-28 核查):config 选择层面的渐进收窄 —
  predictor 排序使预测差的区域永不被采样训练 + 预算衰减(论文 64/32/16);
  非 proxy 训练中途终止。每个采样 config 恰好训练一次(其标签来源);
  我们同机制(非 top-N 不训练 + 20/10/5 衰减),亦无中途 kill
- predictor 质量口径(§D.10):论文报告留出集 Spearman 94%(112 configs,
  350M proxy);我们现记录留出集 R² + Spearman + (pred, actual) 对到
  search_state.json(`predictor_eval`,累计 N≥10 才有切分 — speedrun N=7
  无,生产 35 每轮都有)。预期管理:N=35 vs 112 + SNR z-score 标签更噪,
  留出相关性必然低于 94%,属预算差异而非缺陷
- 最终选择 = predictor 预测 argmax(§2.2;枚举方式见 D3)
- Random 基线 = 等簇权 1/K(App. C.1;短缺策略见 D7)
- WSD 退火语义:stable 阶段可恢复、数据混合研究聚焦 decay 阶段(§C.4;我们
  lr_scale=1.0/warmup=0.0/warmdown=0.9 从 base checkpoint 直接退火)
- LightGBM 作为 predictor(§2.2 实现注)
