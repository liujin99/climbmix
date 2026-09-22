# prod4 实验记录（d20 搜索 → d28 验证全家）

prod4 = CLIMBmix 算法的一轮完整验证实验（2026-09-15 → 09-20，d20 代理搜索 → d28 九臂验证）。目标是**算法设计的可迁移结论**——算法将运行于其他数据池（非 STEM 基准、更大的池），单轮跑赢是证据不是目的，要求结果一直的好。

## TL;DR

1. **算法核心前提在 d28 成立**：CLIMB 搜索产出的配比全面碾压 uniform/自然占比/域配比家族（超家族带上界 +0.014~+0.031），跨轮 gap 单调改善（prod1 −0.0036 → prod2 +0.0102 → prod4 0.016~0.032）。
2. **算法最弱环节定位**：终选机制的外推无增益——同为 CLIMB 搜索的输出，两个实测过的配置点（climb-cfg72 0.2142 / climb-cfg25 0.2066）均优于未实测的插值终选点（climb-终选 0.1972/0.1990），预告过的赢家诅咒软性坐实。
3. **下一轮前置 = 选择机制修复**（best-measured 降级 + no-claim 守卫 + 赢家 proxy 复测）；验证轴三选：非 STEM 基准池 / 扩池 / 20B 终形态。
4. **复现性机器全轮跑通**：锚点 4 位对账、seed-pair gap 带、协议冻结九臂同批可比、数据面交叉账全平。

## 1. 臂命名对照（先读这个，防误读）

**climb-终选 / climb-cfg25 / climb-cfg72 是同一个 CLIMB 搜索输出的三种配方**——不是三个方法竞争。L1a 回答的问题是：同一搜索产出的不同配方点，哪个在目标模型上真正最好。

| 臂名（本 doc 显示名） | 服务器产物名 | 是什么 |
|---|---|---|
| climb-终选 / climb-终选-rep | climb / climb_rep | CLIMB 搜索的**终选配方**：predictor 在设计空间上的 argmin 外推点，搜索阶段**从未实测过**；-rep = 换随机种子（43）复刻 |
| climb-cfg25 | cfg25 | CLIMB 搜索**实测过的配置**（搜索分第 1，历史点） |
| climb-cfg72 | cfg72 | CLIMB 搜索**实测过的配置**（搜索分第 2，终选配方的最近邻） |
| random3b / random3b-rep | random3b / random3b_rep | 均匀簇权重基线（论文 App. C.1 random baseline）；-rep 同上 |
| natural | natural | 池自然占比基线（均匀抽 doc 的期望分布） |
| domainfix | domainfix | 四域固定配比基线（外部手工值） |
| base | base_eval_check | 未续训的 d28 base 模型（参考线，非参赛） |

搜索空间里的配置 id（cfgXX）与由它构建的验证臂同名，臂显示名加 `climb-` 前缀以示血统。**下轮约定**：实测配置晋臂直接命名 climb-cfgXX（ARM_NAME 自带前缀），产物自解释。

## 2. 实验参数配置

### 2a. 共同训练配置（九臂同形）

| 项 | 值 |
|---|---|
| 模型 | d28 base（~2.4B 参数；fp32 master 7.38 GiB 混合 dtype 存储，计算 bf16） |
| 预算 | TARGET_TOKENS=3B → 2861 步 × total_batch 2²⁰ tokens/步（实测 1.048M tok/步 = tok/s×dt 闭合） |
| 并行 | 8 节点 × 8 卡 = ws=64；MID_DEVICE_BATCH_SIZE=1（生产锁）；grad_accum=8（seq 2048）[4] |
| 配方锁定 | LR/wd 链自预训练 meta 继承 × √(2²⁰/B_REF) batch 缩放；lrm 线性退火 1→0；--load-optimizer=0 |
| eval | STEM 套件：4 MC（arc_easy 2376 / arc_challenge 1172 / mmlu_stem 3545 / gpqa_diamond 198）+ 2 生成 CoT（gsm8k_cot 1319 / math_cot_500 500）；复合分 = 6 任务 Centered 均值；eval db=16 / core_bs=8、全量（max_per_task=-1）；**旧协议**（math cap 256、无停止串截断——D17 新协议激活后统一补测，附录 E）；锚点 = base ckpt 单节点远端复评 |

训练健康度（R1 锚点）：dt 3365 ms/步、bf16_mfu 14.42%、311,609 tok/s、175 min/臂、epoch 1——与 09-16 f1bad8c9 逐位一致；峰值内存 24962 MiB ≈ 24.4 GiB（余量 ~5.1G）。

### 2b. 各臂数据配方与作业台账

| 臂 | 数据配方 | 选样结果 |
|---|---|---|
| climb-终选 / -rep | 搜索终选权重 C10 .4737 + C5 .3540 + C12 .0641 + 12 簇 0.9% 地板（`optimal_mixture_weights.json`；-rep = SEED 43） | 配额命中 3.000B |
| climb-cfg25 | 实测配置 #1：C10 .78 + C0 .12 + C12 .07 + C8 .01 + 微量 4 簇（稀疏无地板，历史点） | 3.000B |
| climb-cfg72 | 实测配置 #2：C10 .66 + C11 .14 + C5 .08 + 9 簇 0.92% 地板（引导带产物） | 3.000B |
| random3b / -rep | 均匀簇权重 α_k=1/15（非均匀抽 doc）；-rep = SEED 43 | 3.000B |
| natural | 15 簇池 token 份额权重（`gen_natural_weights.py`，char/4 est；= 均匀抽 doc 的期望 token 分布） | 3,538,132 docs / 3,000,112,486 tok，全簇无 shortfall |
| domainfix | 四域固定配比 **数学 .60 / 物理 .15 / 化学 .125 / 生物学 .125**（quadmix manual_ratio 原值），域标签模式 token 级配额制（§4e） | 4,094,357 docs / 3,000,014,542 tok，四域配额逐字精确 |

作业台账（全部 SUCCEEDED、rc 全 0）：climb-终选 418f1301 / random3b e2fae0c0（09-17）；climb-cfg25 5bcebdb6 / climb-cfg72 727b08df（09-17 发射，62h 含排队）；natural b21c9cb2 / domainfix 0be0e8c5（09-18 12:07/12:08，worktree 发射）；climb-终选-rep / random3b-rep（09-19 晚 SEED=43）。落地时刻（eval CSV mtime）：natural/domainfix 09-20 04:10/04:23、random3b-rep/climb-终选-rep 08:01/08:08、climb-cfg72/climb-cfg25 10:59/11:06。

## 3. 搜索阶段执行记录（d20，09-15 → 09-16，总 8.35h）

### 3a. 搜索配置

| 项 | 值 |
|---|---|
| 数据池 | `100B_stem_parquet_filtered`：116.1M docs / 91.84B est tokens（char/4）；四域划分（数学 47.6% docs 最大域） |
| 聚类 | `embedding_cluster`（stella 嵌入，K_init=1000 → balanced 划分 K=15，D14；`balanced_profile.json`；结构闸门 max token share ≤ 50%） |
| proxy 模型 | d20（435.2M scaling / 1793M total；选型见 proxy_and_model_analysis.md） |
| 评分 | STEM 6 任务 SNR-weighted acc+NLL z-score（修正评分版） |
| proxy 预算 | 400M 单遍（2026-08-28 校准：训练消耗 524M ≤ 混合 571M） |
| 迭代 | 3 轮，configs_per_iter = [54, 36, 18]，Dirichlet α ∝ 簇 token 量 |
| 时长 | 总 30073.4s（8.35h）：stage3_search 29994.5s + 终选 40.1s |

### 3b. 每轮执行情况（111 实测点 = 54 历史 + 57 新）

| 轮 | 计划 | 实现 | 内容 | 时长 | 轮内最佳（实时口径）[1] |
|---|---|---|---|---|---|
| 1 | 54 | 54 | **历史注入**：prod3_20260915_162714 直读 54 点（pool_sha256 匹配、K=15、同 6 任务、w_floor 0.5、09-15T20:10 注入，0 丢弃 0 重复）——免训练复用 | — | 1.3335 |
| 2 | 36 | 38 | 新点：候选池 800 novel → top-96 带（12.0%）→ 采样 40 实现 38 | 21509.3s（~6.0h） | 1.1764 |
| 3 | 18 | 19 | 新点：候选池 500 novel → top-96 带（19.2%）→ 采样 22 实现 19 | 8472.5s（~2.35h） | 1.0467 |

剪枝（论文 §2.2 选择级）：两轮引导共排除 **1108 个候选从未 proxy 训练**。每轮实现数 ≥ 计划地板（[54, 38, 19] vs [54, 36, 18]），prod3 的"iter 地板静默失血"教训（9/20 有效点）未复现。

### 3c. d20 实验结果（重排口径 utility，higher = better；z 尺度全距 −1.70 ~ +1.26）

| # | 分数 | config | 来源 | 备注 |
|---|---|---|---|---|
| 1 | **1.2550** | cfg25 | 历史 | **最佳实测**；C10 .78 + C0 .12 + C12 .07 稀疏无地板 |
| 2 | 1.1297 | cfg72 | 第 2 轮 | 终选点最近邻（L1=0.65）；C10 .66 + C11 .14 + C5 .08，0.01 地板带 |
| 3 | 1.1134 | cfg60 | 第 2 轮 | |
| 4 | 1.0467 | cfg92 | 第 2 轮 | |
| 5 | 0.9003 | cfg29 | 历史 | **prod3 旧赢家，重排后让位**（修正评分裁决自洽） |
| 6 | 0.8791 | cfg101 | 第 3 轮 | 第 3 轮最佳（未超越第 2 轮） |
| 7 | 0.8526 | cfg47 | 历史 | |
| 8 | 0.7292 | cfg94 | 第 3 轮 | |
| 9-15 | .6973/.6970/.6749/.6675/.6298/.5906/.5747 | cfg58/98/76/95/106/96/13 | 第 2 轮×3 / 第 3 轮×4 / 历史 | |

最差 5 点：cfg22 −.8199 / cfg3 −.8223 / cfg52 −.8545 / cfg24 −1.5918 / cfg19 −1.7046（全为 prod3 历史坏点）。

结构观察：**前 4 名 = 1 历史点 + 第 2 轮引导带包揽 2-4 名**（C10 热区收敛的产物）；第 3 轮最好仅第 6——引导带窄化后第 3 轮没有再超越。

### 3d. LightGBM 拟合情况

| 时点 | val n | val R² | val ρ | online ρ (n) |
|---|---|---|---|---|
| 首拟合（54 历史，第 2 轮采样前） | 10 | −0.2172 | 0.2901 | — |
| 第 2 轮后 | 18 | 0.3763 | 0.5955 | None（n=38，退化未记）[2] |
| 第 3 轮后（终轮，驱动终选） | 22 | 0.3204 | 0.5008 | 0.4211 (19) |

- **pooled held-out ρ = 0.5799**（40 pairs，`predictor_scatter.png`）
- 论文对照 D.10：94% held-out Spearman @ 112 configs / 350M proxy——我们 N 相当但标签噪声高得多（6 任务含生成式 CoT vs 论文 3 个 MC 验证损失）；report 自带脚注"预算 artifact 非缺陷"
- 超参 = 论文 §3.1 语义（L1/L2 + 早停 + max_depth 4 + 叶样本 5）
- **特征重要性（终轮模型，split counts / share）**：

| Rank | 簇 | Splits | Share |
|---|---|---|---|
| 1 | C10 | 178 | 14.1% |
| 2 | C1 | 141 | 11.1% |
| 3 | C7 | 141 | 11.1% |
| 4 | C13 | 136 | 10.7% |
| 5 | C0 | 122 | 9.6% |
| 6 | C9 | 110 | 8.7% |
| 7 | C3 | 92 | 7.3% |
| 8 | C4 | 84 | 6.6% |
| 9 | C11 | 83 | 6.6% |
| 10 | C8 | 66 | 5.2% |
| 11 | C14 | 41 | 3.2% |
| 12 | C5 | 30 | 2.4% |
| 13 | C2 | 20 | 1.6% |
| 14 | C12 | 18 | 1.4% |
| 15 | C6 | 4 | 0.3% |

而终选赢家把 C5/C12 给到 .354/.064——分裂结构与设计空间 argmin 的分歧点 = 赢家外推的语境注记（预测器对 C5/C12 的响应来自少数分裂，外推到大权重缺乏实测支撑）。

### 3e. 终选：predictor_design_space 与赢家

- **Selection mode: predictor_design_space**（search.log:51585）——首次在"有信号"路径外发未实测插值点（prod1 无信号角落事故后加的守卫放行）。
- **climb-终选配方** = C10 .4737 + C5 .3540 + C12 .0641 + 12×0.0090 地板（`optimal_mixture_weights.json`，09-16 06:14）。12/15 簇钉 0.9% 地板（0.01 钳位+重归一产物；地板质量 ~10.8% 为防饿死压舱，非搜索输出）。
- **外推半径**：到最近实测 cfg72 的 L1 = 0.65，到 cfg25 = 0.87——不是任何实测点的精修，是大步外推；赢家邻域实测 1.05–1.13 全部低于最佳实测 0.12–0.40 → 赢家诅咒签名。改进方向四条（选择时 claim 报告 / 赢家 proxy 复测 / 不确定性感知获取 / 地板质量审计）记档 TODO 09-17。
- **终选簇表**（orig = 池，sel = 赢家 6B 口径选样，α = 赢家权重，ratio = sel/orig）：

| 簇 | 池 docs | 池 tokens | sel docs | α | ratio | 备注 |
|---|---|---|---|---|---|---|
| C10 | 12,970,306 | 5.05B | 7,293,100 | .4737 | 0.56 | 主引擎；~389 tok/doc（多而短） |
| C5 | 3,184,847 | 6.94B | 976,290 | .3540 | 0.31 | 二号；~2180 tok/doc（少而长） |
| C12 | 2,990,596 | 7.04B | 165,641 | .0641 | 0.06 | |
| 其余 12 | 76.9M 合计 | 4.7–7.0B/簇 | 1.8 万–12.3 万 | .0090 | ~0.01 | 地板压舱 |

配额交叉验证：C10 sel 7.29M × 389 tok ≈ 2.84B = 0.4737×6B ✓；C5 sel 976K × 2180 ≈ 2.13B = 0.354×6B ✓——终选产物 `sampled_dataset.parquet`（12.3 GB）为**赢家 6B 口径**选样 [3]。

## 4. d28 验证全景（9 臂 + base 参考线）

**管线校准**：锚点 d8a43a5d 远端 base = 0.1746 vs 本地 0.1738（PASS，阈值 0.002）；09-18 12:34 worktree 复评 = **0.174590**（4 位对账 PASS，兼验 worktree 远端 eval 通路）。

### 4a. 记分板（stem_metric = 6 任务 Centered 均值）

| 臂 | STEM | seed | 说明 |
|---|---|---|---|
| **climb-cfg72** | **0.2142** | 42 | CLIMB 实测点：终选点最近邻（搜索分 1.1297 #2，第 2 轮引导带） |
| **climb-cfg25** | **0.2066** | 42 | CLIMB 实测点：最佳实测（搜索分 1.2550 #1，历史点） |
| climb-终选-rep | 0.1990 | 43 | CLIMB 终选插值点 · 复刻 |
| climb-终选 | 0.1972 | 42 | CLIMB 终选插值点（§3e 配比，未实测过） |
| random3b-rep | 0.1833 | 43 | uniform · 复刻 |
| domainfix | 0.1804 | 42 | 四域固定配比（数学 60%） |
| base（参考） | 0.1746 | — | base ckpt 远端复评 |
| natural | 0.1705 | 42 | 池自然占比 |
| random3b | 0.1647 | 42 | uniform（低抽签，见 §4c） |

uniform 家族带（random3b ×2 + natural + domainfix）= .1647–.1833，四点均值 **.1747 ≈ base .1746**；CLIMB 配方四点 = .1972–.2142，高于家族带上界（.1833）+ .014（climb-终选）~+.031（climb-cfg72）。

### 4b. per-benchmark（raw acc；二项 SE ≈ arc_easy .0092 / arc_challenge .0145 / mmlu .0076 / gpqa .0298 / gsm8k .0120 / math .0074）

| bench | N | climb-cfg72 | climb-cfg25 | climb-终选 | climb-终选-rep | random3b | random3b-rep | natural | domainfix | base |
|---|---|---|---|---|---|---|---|---|---|---|
| arc_easy | 2376 | .7226 | .7365 | .7210 | .7184 | .7205 | .7197 | .7201 | .7294 | .7424 |
| arc_challenge | 1172 | .4497 | .4377 | .4428 | .4411 | .4326 | .4437 | .4292 | .4292 | .4608 |
| mmlu_stem | 3545 | .2931 | .2987 | .2818 | .2942 | .2818 | .2945 | .2827 | .2832 | .2979 |
| gpqa_diamond | 198 | .2424 | .2323 | .2273 | .2525 | .2121 | .2626 | .2475 | .2525 | .2626 |
| gsm8k_cot | 1319 | **.3116** | .2714 | .2578 | .2381 | .1077 | .1228 | .1069 | .1289 | .0273 |
| math_cot_500 | 500 | .0300 | .0280 | .0280 | .0140 | .0180 | .0160 | .0100 | .0280 | .0020 |

次级 NLL（gsm8k）：climb-cfg25 **.5313** / climb-cfg72 .5479 / climb-终选 .5559 / climb-终选-rep .5577——两个实测点的 gsm8k NLL 也低于插值终选点，宽基一致。5 任务 NLL 均值（mmlu nan 除外）：climb-cfg25 **1.4488** 最低 < random3b 1.5005 < climb-cfg72 1.5025 < random3b-rep 1.5055 < domainfix 1.5120 < natural 1.5204 < climb-终选 1.5209 < climb-终选-rep 1.5262 < base 1.6413——**random3b NLL 次低且 STEM 最差，NLL 反转在家族臂复现**。

### 4c. T1 复跑噪声标定（SEED 42/43 成对）

| 量 | seed42 | seed43 |
|---|---|---|
| climb-终选 | 0.1972 | 0.1990 |
| random3b | 0.1647 | 0.1833 |
| **gap** | **0.0324（z≈+3.8）** | **0.0157（z≈+1.8）** |

- climb 配方**逐种子稳定**（Δ+.0018），且高于全家所有臂的所有测量；自身任务涨跌相消（gsm8k −.0197 / math −.0140 vs gpqa +.0336 / mmlu +.0124）
- random3b 摆 +.0185，几乎全由 gpqa 单任务低抽贡献（Centered −.0505→+.0168，Δ+.067 ≈ 13/198 样本）
- **诚实表述：gap = 0.016–0.032 带，符号双种子一致**；单种子 0.0324 高估幅度，方向结论不变

### 4d. L1a 热区三点判决（同一 CLIMB 搜索的三种配方：终选插值点 vs 两个实测点）

| 点 | d20 搜索分 | d28 STEM（seed42） | 备注 |
|---|---|---|---|
| climb-cfg72 | 1.1297（#2，最近邻） | **0.2142** | d28 全场最佳；gsm8k .3116 领跑 |
| climb-cfg25 | 1.2550（#1，最佳实测） | 0.2066 | d28 次席 |
| climb-终选 | 未实测（预测最优） | 0.1972 / 0.1990（rep） | 热区四点中垫底 |

- **L1a 裁决：插值外推无增益（软性赢家诅咒坐实）**——两个实测点 d28 均不低于插值终选点；与 08-29 smoke_search（argmin 落后 best-measured 0.2–0.3 utility）、特征重要性分歧注记（§3d：C5/C12 split share 2.4%/1.4% vs 终选权重 .354/.064）全部呼应。
- **climb-cfg72 vs climb-终选：Δ+.0170（z≈2.0，边缘显著）**，gsm8k 主导（+.054，z≈3.1），6/6 任务同向为正；vs climb-终选-rep Δ+.0152（z≈1.8）。**单种子**——领先档位与 T1 噪声带（±0.01–0.02）贴近，加冕前建议补 climb-cfg72-rep（SEED=43）。
- climb-cfg25 vs climb-终选：Δ+.0094（z≈1.1，噪声级）。
- d20→d28 排名翻转（climb-cfg25 ↔ climb-cfg72：Δd20 0.125 / Δd28 −.0076，z≈0.9）：两侧均不显著，proxy ρ≈.42–.50 带内。**热区真实，热区内排序软**。
- 地板不对称注记落点：预登记的担忧是"若 climb-cfg25 胜出，地板伤赢家与点优劣并列解释"——实际 climb-cfg72（0.01 地板带，与终选同政策族）领先，地板政策不构成混淆方向。
- **20B 耦合挂账**：climb-cfg72 C10 .66 → 20B 配额 13.2B vs est 池 5.06B，**R′≈2.61 超 cap=2**；climb-cfg25 C10 .78 → R′≈3.09 更超；终选 C10 .474 → R′≈1.87 恰在线内。配方终选与 20B 可行性（oversampling 政策 / 扩池 / 权重再派生 / OVERSAMPLE_OK 越狱）为耦合决策，判读会须一并过。

### 4f. 赢家配方解剖（2026-09-22 补；工具 `scripts/diagnostics/recipe_report.py`——图 + 逐簇表 + 每基准明细已缝进 `result/prod4_current/report.md` "赢家配方解剖" 节，此处为读数摘录）

- **搜索的净价值 ≈ 发现 C10 并压满**：cfg72 = C10 .659（**12× 超采**）+ C11 .135 + C5 .082，其余 12 簇全压在采样地板 .009；cfg25 = C10 .781 + C0 .123 + C12 .071。四条独立证据同指 C10：实测前二重仓 / argmin 也给 .474 / 预测器第一特征（14.1% splits）/ 而舰队均值仅 .061（cfg72 偏离 +.60 = 全维度最大）。
- **赢家 vs cfg25 差异全在配角簇且为噪声级**：L1=.602 分解 = C11 +.135 / C10 −.122 / C0 −.114 / C5 +.082 / C12 −.060（C10 基本对消）；d28 差 .0076 在噪声带（.0085）内 → 可声明的是"C10 主导家族"整体赢基线，家族内 d20→d28 翻转（1.2550 vs 1.1297 → 反超）落在 proxy ρ≈.42–.50 已知带内。
- **L1a 的配方级面孔 = argmin 的 C5 豪赌失败**：argmin 在 C10 之外押 C5 .354（预测器证据仅 2.4% splits，§3e 老红旗），d28 上终选臂 .1972/.1990 落后 cfg72 ~.016；cfg72 只给 C5 .082——"外推错在 C5"。
- **C10 形态与语义缺口**：12.97M docs = 池内最多（12% 文档）但仅 5.5% token，户均 ~390 ch = 全池最短文档簇（短文档 + 高价值，与 gsm8k +.054 主导（§4d）形态相合但属推测——语义待 `cluster_peek.py` 抽样确认）。
- **质量轴缺席实锤**：本 run 质量过滤 = none → 簇质量分全零——质量维度在**整个配方空间**不存在（不止簇内无选择，连簇级质量分也未参与搜索），§6 指令②"簇内质量分"的立项证据增强。
- 验证记录：双臂 8 节点同形、rc 全 0、资产挂载与全家逐字节同（同旧协议批）；CSV ↔ CP4 逐位吻合、复合分算术闭合（climb-cfg72 六任务 Centered 均值 = .214226 ✓）。

### 4e. 专题核验

**domainfix 实现确认**（token 级配额制，每域配额 = w×3B，域内 seeded random 选样）：

| 域 | 配额 (w×3B) | 实选 docs | 池 docs | est tok/doc |
|---|---|---|---|---|
| 数学 | 1,800,000,000 | 2,703,413 | 55,255,374 | ~666 |
| 化学 | 375,000,000 | 767,202 | 28,874,219 | ~489 |
| 生物学 | 375,000,000 | 323,420 | 16,633,772 | ~1,160 |
| 物理 | 450,000,000 | 300,322 | 15,346,595 | ~1,498 |

交叉账全平：四域实选合计 4,094,357 = 总选样；池 docs 合计 116,109,960 = 全池 116.1M（四域完整划分）；实际 3,000,014,542 tok vs 3B = +14,542（0.0005%，文档粒度）；无 shortfall。结构观察：① 数学池内 token 份额 ~40% est → 配 60% = 主动加码 +20pp（手工值设计意图）；② 域文档长度异质（物理 ~1498 vs 化学 ~489 est tok/doc）。实现细节（标签源/代码位置）见附录 C。

**base 对比分析：能力换位（"训完低于 base 是否正常"）**——混合偏数学/代码簇的继续预训练做**能力换位**：生成类大涨、MC 知识类小遗忘；gsm8k 增益够大净赚（climb-终选 +.023），不够则净亏（random3b −.010）。

| 维度 | base | uniform 家族 | CLIMB 配方 |
|---|---|---|---|
| MC 三任务均值 acc（arc_e/arc_c/mmlu） | .5004 | .477–.486（−.014~−.023） | .482/.485 |
| MC NLL | arc_easy 2.342 / gpqa .742 | 2.35–2.38（+.01~.04）/ .85–.91（+.11~.17, N=198） | 同带 |
| gsm8k_cot acc / NLL | .0273 / .969 | .107–.129（×3.9–4.7）/ .633–.655 | .238–.312（×8.7–11.4）/ .531–.558 |
| math_cot_500 acc / NLL | .0020 / 1.325 | .010–.018 / .848–.859 | .014–.030 / .886–.891 |

random3b 净亏分解（Centered Δ vs base）：gpqa −.0673 / arc_challenge −.0375 / arc_easy −.0292 / mmlu −.0214 / gsm8k +.0804 / math +.0160 → 净 −.0098；gpqa 一项占 −.0112 > 全部净亏，复刻证明 seed42 gpqa 为低抽 → 去 gpqa 后 ≈ base 持平 + gsm8k 增益。**证据栈（非配置错误）**：① eval 链路锚点 4 位对账 + R1 双臂本地-远端逐位一致；② MC NLL 微漂而非爆炸；③ 全臂同向平滑 + 剂量效应；④ 数据面域 id 目检 + 配额精确（§4e）。**含义**：对比轴 = 同预算 vs uniform 家族；base = 换位效应参考线；若最终形态在意 MC 保留 → 下一轮可加通用域保底配额（设计问题，非 bug）。

## 5. 四个判决

| 搜索侧点 | 臂验证 | 状态 / 结果 |
|---|---|---|
| cfg72（终选点最近邻 1.1297） | climb-cfg72 | **0.2142——全场最佳**，高于终选点 Δ+.017（z≈2.0，gsm8k 主导） |
| cfg25（最佳实测 1.2550） | climb-cfg25 | **0.2066——次席**，高于终选点 Δ+.009（噪声级） |
| 终选（外推插值点） | climb-终选 / -rep | 0.1972 / 0.1990——双种子稳健高于 uniform 家族带，但热区内垫底（§4d） |
| uniform / natural / domainfix 家族 | random3b×2 / natural / domainfix | .1647–.1833 带，无一接近 CLIMB 配方 |

1. **L0（成立但被 L1a 超越）**：climb-终选双种子 .1972/.1990 高于 uniform 家族带全区间，gap 诚实带 0.016–0.032 符号双种子一致，噪声几乎全在基线侧；外推点未翻车（prod1 式事故不复现）——但不再是全场最优。
2. **L1a（本轮最重要的算法发现）**：插值外推无增益，配方领跑者易主 climb-cfg72（0.2142，单种子）——两个实测点均不低于插值终选点；热区内排序软（互差均噪声级，除 climb-cfg72 vs 终选 z≈2.0）；详见 §4d。后续搜索轮的赢家选择政策（best-measured 降级 / no-claim 守卫 / 赢家 proxy 复测）由本判决直接触发。
3. **L1b：域级配比解释出局**。natural .1705 / domainfix .1804 全落 uniform 带内；domainfix 把数学加码到 60%（池内 token 份额 ~40% est）后 gsm8k 仍只 .1289 vs CLIMB 配方 .238–.312——**赢家优势不在"更多数学"，在簇级选择（C10 那批文档的质/结构）**；MC 任务上全家族无分化，赢集中在 gsm8k_cot（约 2× 全家）。
4. **T1：赢家配方稳健，噪声在基线侧**。climb 配方逐种子稳定（Δ+.002）；random3b 摆 +.0185 几乎全由 gpqa 低抽贡献。

判读附注：**噪声结构**——gpqa（N=198）单种子 ±.03–.07 摆动主导复合分噪声，gsm8k 双种子逐位可复现且是全部分化主引擎；**NLL 反转复现**（random3b NLL 次低 + STEM 最差）；**协议注记**——本报告全为旧协议，本地重评已证新协议下 gap .0324→.0341（climb math +36% 截断救回 / random3b 不变 = 能力地板），激活补测后定版；**搜索侧注记**——终选的 C5/C12 大权重来自预测器响应最稀薄的方向（§3d），L1a 已证实此担忧。

## 6. 算法层结论与设计指令（可迁移八条）

目标是 CLIMBmix 算法本身的最优设计——算法将运行于其他数据池（非 STEM 基准、更大的池），单轮跑赢是证据不是目的，**要求结果一直的好**。

| # | prod4 证据 | 算法层结论 | 设计指令（下轮 / 换池） |
|---|---|---|---|
| 1 | 插值终选点热区垫底：climb-cfg72 0.2142 / climb-cfg25 0.2066 ≥ climb-终选 0.1972/0.1990（§4d） | 终选 argmin 外推无增益，赢家诅咒风险真实（预告 → 应验） | **终选改 best-measured 优先 + no-claim 守卫**（预测增益 ≤ 任务噪声即降级实测最优）+ 赢家 proxy 复测；TODO 09-17 改进方向 #1/#2 由"可选"升级为"下轮前置" |
| 2 | d20↔d28 热区内排序翻转（互差均噪声级）；online ρ≈.42–.50、pooled .58 | proxy 可靠定位"好区域"，不可靠排序"区域内点" | 搜索输出按**区域**消费（top-k 实测点直接进臂验证，不迷信单点 argmin）；不确定度感知获取（LightGBM 种子集成 + LCB）为中期方向 |
| 3 | CLIMB 配方全超 uniform 家族带上界 +.014~+.031；跨轮 gap 单调改善 | **CLIMB 核心前提在 d28 成立**：学习配比 ≫ 均匀/自然/域配比 | 聚类（K=15 balanced）+ 迭代引导 + predictor 剪枝的骨架有效，保持 |
| 4 | natural .1705 / domainfix .1804 全落 uniform 带；domainfix 数学加码 60% 后 gsm8k 仍 .1289 | 域级配比不可替代簇级选择——赢在簇级结构（C10 浓度） | 换池保留簇级搜索为一级机制；域标签通道（--label-source domain）仅作基线/对照用途 |
| 5 | uniform 家族均值 .1747 ≈ base .1746（能力换位：MC 遗忘 −.014~−.023 vs gsm8k ×3.9–9.4） | 偏科混合上的继续预训练是零和换位，复合分掩盖结构变化 | 若最终形态在意 MC 保留 → 通用域保底配额（设计选项，未验证，下轮可测） |
| 6 | gpqa N=198 单种子 ±.03–.07 主导复合分噪声；gsm8k 双种子逐位可复现且是全部分化的主引擎 | 小 N 任务噪声吞排序信号；大 N 生成任务是可靠判别器 | 臂间判读优先看大 N/可复现列；考虑复合分小 N 任务降权或多种子 eval（评分设计复核项） |
| 7 | C10 20B R′ 随配比 1.87→3.09（终选→climb-cfg25）；C5 贴线 −2% | 放大预算/换池时池约束成为活跃约束，且与配方选择耦合 | oversampling 三层政策（已设计 ~150 行）在放大轮实现；扩池/换池先重跑池表裁定贴线簇；fail-loud 语义保持 |
| 8 | 锚点 4 位对账、T1 seed-pair gap 带、协议冻结 9 臂同批可比、domainfix 交叉账全平 | 复现性机器是"一直的好"的制度保障 | seed-pair 复刻、锚点、冻结窗、交叉账验证作为**每轮标准动作**（本轮全跑通） |

## 7. 下一轮路线

1. **prod4 收官**：9 臂全景 + 四判决（L0 / L1a / L1b / T1）+ 锚点链闭环；复现性机器全轮零事故。
2. **下一轮前置（P1，先于任何新搜索）——选择机制修复**：① 终选 claim 报告（打印赢家预测分 vs 最佳实测分 + L1 距离 + 地板份额）+ no-claim 守卫（claimed gain ≤ 任务噪声 → 降级 best-measured）；② 赢家 proxy 复测。L1a 证据把 TODO 09-17 的改进方向 #1/#2 升级为必须做。
3. **验证轴选择（下轮主决策，三选一或组合）**：A = **非 STEM 基准池**（算法迁移性的直接检验，最贴合"一直的好"目标）；B = 增大数据池（C10/C5 供给约束松绑 + 池表重裁定）；C = 20B 终形态（oversampling 政策实现 + C10 R′ 冲突三出路抉择 + general 供给审计）。
4. **配方问题从属化**：若终形态需要"prod4 最优配方点"的稳健性证据 → 补 climb-cfg72-rep（SEED=43，队列空 ~4h）；算法迭代视角下**非阻塞项**——下轮终选走 best-measured 政策后，"哪一点加冕"由新政策回答。
5. **eval 协议激活**（收官后随时可执行，不阻塞新轮设计）：① pull 两树 + 重建 tarball + worktree 退役；② 8 ckpt（climb-终选/-rep、random3b/-rep、natural、domainfix、climb-cfg25/cfg72）新协议补测 ~40min/臂，新 CSV 名不覆盖；③ 此后新轮全走新协议，跨轮趋势链按 D17 脚注。climb-cfg72 的领先由 gsm8k（协议不变量、逐位可复现）承载，激活后排名大概率保持。
6. 收尾余项：OBS 孤儿清理、家族臂训练健康度抽查（dt/mfu vs R1 锚点）、dataset.py.bak 删除确认、口径确证与 round-2 可选项（热区配置 d20 逐任务表 / predictor 超参行 / 终选块全文）、CP4 渲染器 ref 占位债。

# 附录

## A. 复现性机器（每轮标准动作）

- **锚点**：远端 eval 管线用 base ckpt 对账——预期 0.1746（d8a43a5d，09-17），worktree 复评 0.174590（09-18），4 位吻合。
- **seed-pair 复刻**：关键臂双种子（42/43）测 gap 带而非单点（§4c）。
- **协议冻结窗**：一轮内 dispatch 绑定资产，全批同协议可比；新协议激活走统一补测（新 CSV 名不覆盖）。本轮冻结 09-18 定案（prod4 全轮收官前不 pull 两树/tarball），L1b 经 worktree 部署绕行（`climbmix_lb@b93bb97`，RUN_DIR 绝对路径穿透，产物同落主树 prod4_current）。
- **数据面交叉账**：域 id 目检、配额 token 精确、选样合计对账（domainfix 四域账全平，§4e）。
- **代码血统**：臂与搜索靠权重文件内容哈希衔接（`weights_id` 入 `.done` 身份，权重文件传路径按内容哈希）。

## B. 时间线

| 时刻 | 事件 |
|---|---|
| 09-15 20:10 | 历史注入（prod3 54 点，identity 入档） |
| 09-15 21:52 – 09-16 06:22 | 搜索 3 轮（8.35h）→ 终选 predictor_design_space |
| 09-16 10:28 | eval_random.csv 拷入（prod3 占位 ref，provenance 后闭环） |
| 09-17 10:30 | R1 三作业同时阵亡（boot shell 键化 bug × 双臂 + 锚点 EL0004） |
| 09-17 13:01/14:45 | db=4/2 step-0 OOM（内存墙判决输入） |
| 09-17 16:00/16:32 | db=1 双臂入轨（climb-终选 418f1301 / random3b e2fae0c0） |
| 09-17 16:46-17:24 | 锚点 d8a43a5d SUCCEEDED → base 0.1746 |
| 09-17 18:52/19:29 | 双臂落地 → R1 首报/终报；当晚 ref provenance 闭环 |
| 09-17 晚 | L1a（climb-cfg25/climb-cfg72）发射（8 节点 db=1 同形）；18:59 323855b4 冗余提交 |
| 09-18 11:22 | b93bb97 基线家族支持（natural/domainfix/label-source + 22 项测试） |
| 09-18 12:07/12:08 | L1b natural/domainfix 发射（worktree 绕行主树冻结；b21c9cb2/0be0e8c5） |
| 09-18 12:34 | base 复评 0.174590 PASS（worktree 远端 eval 通路验证） |
| 09-18 下午 | eval 协议合并 dev-data-mix `0c1229f` + 冻结窗定案 |
| 09-18 晚 | R1 本地重评验证 PASS（新协议默认档；协议效应不均匀入账） |
| 09-19 晚 | T1 双复刻发射（SEED=43） |
| 09-20 04:10/04:23 | natural/domainfix 落地 |
| 09-20 08:01/08:08 | random3b-rep/climb-终选-rep 落地 → 家族全景判读 |
| 09-20 10:59/11:06 | L1a 落地（727b08df/5bcebdb6，SUCCEEDED，62h 含排队）→ CP4 9 臂全景 |
| 09-20 | 判读会收官：四判决闭环 + 八条设计指令 + 下一轮路线 |

工程面：家族四臂全部训满 2861 步 + 训后 eval 存活；worktree 双部署（climbmix_lb/nanochat_lb）全轮零事故；本地 shards+mixed 自动清理（domainfix `.done` 即被清，验证改走发射日志）。

## C. 工程与口径细节（脚注集中区）

- **[1] Best Score 双口径**：report.md 迭代表的 Best Score 为**迭代时实时口径**；search_state.json accumulated 终值为重排口径（最佳实测 = cfg25 1.2550，§3c）。两口径差异（第 1 轮实时 1.3335 vs 重排后历史最佳 1.2550）与评分修正历史一致，判读一律以 state 终值 + TODO 09-17 条为准；口径确证列 round-2 可选项。
- **[2] online_eval 第 2 轮退化**：记录 n=38 / spearman=None（代码 nan→None 路径，退化情形未记）。
- **[3] 终选 6B 产物与 3B 臂的血统**：验证臂 3B 预算用 `optimal_mixture_weights.json` 经 `prepare_random_baseline --weights` 重新选样，不直接消费 `sampled_dataset.parquet`（12.3 GB 赢家 6B 口径选样）；血统靠权重文件内容哈希衔接（`weights_id` 入 `.done` 身份）。
- **[4] 落地 log 核对**：`Inherited max_seq_len=2048` / `Inherited total_batch_size=1048576` / `Grad accum steps: 8`，与 dispatch 守卫打印一致。
- **搜索数据源**：report.md 全文 + search_state.json 摘要 + search.log 迭代/终选行（2026-09-20 服务器提取）。
- **323855b4 冗余提交**（09-17 18:59 重复提交）：落地未留独立产物（同臂重复提交幂等语义），对账职能已由 09-18 复评完成。
- **ref provenance**：`eval_random.csv`（0.1781）= prod3 random 臂逐字节拷贝件（09-16 晨 CP4 手动渲染时拷入作占位 ref）。**09-20 判读会决定：prod3 random 移出 prod4 对比轴，仅作跨轮佐证**——climb-终选对其 +0.0191（z +2.25）；跨轮告示：prod3 ref 为 ws=32（形状不变量）且 eval 早于 53d0bcd，锚点 PASS + 复刻互差在噪声内缓释。CP4 渲染器仍以它为默认 ref（占位债，判读一律 prod4 内对比）。
- **NLL 聚合口径**：R1 原记录的 stem NLL 1.86x 系另一聚合口径（保留历史不混用）；本 doc §4b 用 5 任务均值（mmlu nan 除外）。
- **domainfix 实现细节**：标签源 = `prepare_random_baseline.py --label-source domain`（:200, :268-280）——parquet `category_name` → schema `domain_names` id（数学=0/化学=1/生物学=2/物理=3），读 metadata_cache.npz `cluster_labels` 键（与 15 簇 K-means 标签**不同源**）；配比出处 = quadmix `run_stem_quadmix_vs_manual.sh:40`（数学=60:物理=15:化学=12.5:生物学=12.5）；执行 = 与 CLIMB 臂同一选样器 `select_data_by_mixture`（:305-308），shortfall take-all 不重复不重分配（本次未触发）。`.done` 被 clean_derived_data 按设计清除 → 验证走发射日志 plan 块（`prod4_domainfix_arm.log`）。
- **家族臂训练健康度**：明细未逐一摘录（各臂 `mid_train_*.log`），收尾抽查项（dt/mfu vs R1 锚点）。

## D. 产物索引（服务器产物名 = artifact 名，与 §1 显示名对照）

- run 目录：`result/prod4_current/`；自动报告 `report.md` + `predictor_scatter.png` + `domain_distribution.png`
- 搜索结构化：`search_state.json`（111 点权重+分数+逐任务 acc/nll、predictor_eval 含 held-out (pred, actual) pairs、online_eval、pruning_history、history_seed identity）、`search.log`（13.4 MB）、`optimal_mixture_weights.json`、`pipeline_summary.json`、`cluster_info.json`、`balanced_profile.json`、`cluster_cache.npz`（929 MB，final_labels）、`sampled_dataset.parquet`（12.3 GB，赢家 6B 选样）；逐实验 `exp_0054`–`exp_0110`
- 臂 ckpt：`mid_checkpoints/d28_{climb,climb_rep,random3b,random3b_rep,natural,domainfix,cfg25,cfg72}_prod4`（另 prod3 `d28_random_prod3`）
- 臂 eval/audit：`eval_{climb,climb_rep,random3b,random3b_rep,natural,domainfix,cfg25,cfg72,base_remote}.csv`（`eval_random.csv` = prod3 拷贝件，佐证用）、`target_arm_*.json`（job id / node_count / rc / elapsed / OBS uri）、`launch_env.json`
- 提交端日志：`/home/ma-user/work/tmp/prod4_{climb8,random3b8_db1,natural,domainfix,climb_rep,random3b_rep}_arm.log`、`prod4_anchor_eval2.log`
- 本地重评（新协议预演）：`/home/ma-user/work/tmp/reeval_0918/{climb,random3b}_new_protocol.csv`
- worktree（保留至激活）：`/home/ma-user/work/climbmix_lb@b93bb97`、`/home/ma-user/work/nanochat_lb@0c1229f`
- OBS：`<prod-prefix>/prod4/target_arms/{climb,random3b,natural,domainfix,climb_rep,random3b_rep}/mixture_data_k*`（完整桶前缀属内部值，见服务器 audit / launch_env.json，不入公开仓）

## E. 记录沿革与代码基线

- **版本**：v1 = 09-17 R1 首记 → v2 = 09-20 家族全景 + 搜索各自成档 → v3 = 合并"一轮一文档" → v4 = L1a 落地判决 → v5 = 收官 + 算法层指令 → v6 = 双读者重排 → v7 = 头部去杂 → **v8 = §4f 赢家配方解剖补节（recipe_report.py 工具产出读数）**
- **日期线**：09-15 池/聚类/历史注入 → 09-16 搜索收官 + 赢家终选 → 09-17 R1 双臂落地 + L1a 发射 → 09-18 L1b 发射 + eval 协议合并 → 09-19 T1 双复刻发射 → 09-20 家族落地 → 09-20 10:59/11:06 L1a 落地 → 9 臂全景 → 判读会收官
- **代码基线**：搜索 = 指纹 `.fingerprint_search`（09-15 21:52；历史注入 identity 记 scoring_commit `dda6239`）；臂 R1 = climbmix `bf9690c`→`3df386a` / nanochat-npu `20b94b0` / boot shell climbmix-ma `146f465` / worker tar `a7c56792`；L1b+T1 = climbmix `b93bb97`（worktree 部署绕行主树冻结，产物同落 prod4_current）
- **协议口径**：全批臂 eval = 旧协议（dispatch 时绑定资产，同批直接可比）；新协议（dev-data-mix `0c1229f`，D17）激活后统一补测（§7）
- **关联记档**：proxy 选型 = `docs/proxy_and_model_analysis.md`；评分设计 = `docs/scoring_metric_design.md`；D14（balanced K15）/ D17（eval 协议）= `paper_deviations.md`；TODO 09-17「db≥2 内存墙判决」（R1 = 判决后首个全净轮）等条目
