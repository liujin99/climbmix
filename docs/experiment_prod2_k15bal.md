# prod2_k15bal 实验结果记录

**日期**：2026-09-08 启动（priority 修复后重启）→ 2026-09-09 20:03 完成
**归档**：`result/prod2_k15bal_20260909_200323`（跑完自动从 `_current` 改名）
**代码基线**：commit `043549f`（floor 语义/紧凑画像/锚点修复当日落地，本 run 未受影响）

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 聚类 | embedding_cluster，K_enhanced=15（k15bal 平衡基线） |
| Proxy | d20，远程舰队（8 卡作业 × 并发上限 10，priority=2 修复后） |
| 计划迭代 | [20, 10, 10]；**实际 [22, 4, 4] = 30 点**（iter2/3 缩水 = C_eff 中位污染 bug，已在 043549f 修复） |
| 臂 | d28 远程并行（climb + random），**实际墙钟 ~6h/臂并行**（远低于预估 13h；prod1 本地串行 20h） |
| 端到端 | 搜索 ~13:30 结束 → 20:03 归档（臂+eval ≈ 6.5h） |

## 2. 主结果（CP4：climb vs random，d28）

| 指标 | climb | random | Δ |
|---|---|---|---|
| stem_metric (centered) | **0.1752** | 0.1650 | **+0.0102** (SE 0.0085, z=+1.20, 单侧 p=0.115) |
| vs base 0.1738 (本地) | +0.0014 | -0.0088 | — |
| prod1 对照 | 0.1623 | 0.1659 | **-0.0036**（方向翻转！） |

**判定：INCONCLUSIVE（z<1.65），但符号首次翻正。**

Per-benchmark（raw acc，二项 SE）：

| bench | climb | random | Δ | z |
|---|---|---|---|---|
| arc_easy | .7525 | .7449 | +.0076 | +0.60 |
| arc_challenge | .4582 | .4437 | +.0145 | +0.71 |
| mmlu_stem | .2925 | .2951 | -.0025 | -0.23 |
| gpqa_diamond | .2475 | .2121 | +.0354 | +0.84 |
| gsm8k_cot | .0500 | .0561 | -.0061 | -0.69 |
| math_cot_500 | .0000 | .0060 | -.0060 | **-1.74** |

模式：climb 赢知识型（arc/gpqa）、输推理型（gsm8k/math）；符号检验 3/6。远端锚点未跑成（见 §5），绝对水平未校准，但 Δ 对比有效。

## 3. 搜索内部状态

- **predictor 全程无信号**：held-out R² = -0.05 / -1.29 / -0.198（各迭代），val spearman 0.6 / 0.3 / 0.486（弱正），在线 spearman **0.0 / -0.2**。
- **B3 无信号守卫正确触发**（R²≤0）→ 最终选择 = `no_signal_best_measured`：30 个实测点取最优，**不是**预测器 argmax（避免了 prod1 的"未测角落"事故——安全机制按设计工作）。
- 赢家来自 **iter1 随机点**（iter best：0.7446 > 0.7037 > 0.5892），guided 两轮最优点未超过随机轮最好点；guided sd（0.004）≈ 随机轮（0.0096）一半——确实收缩了采样区域，但该区域不更好。
- 赢家配比：**C10 62.3% + C11 29.6% + C1 6.0%**（~98% 集中在 3 簇），1.93M docs。

## 4. 信号与噪声分析（本 run 最重要的产出）

### 4.1 实际目标函数（iterative_bootstrapper.py `_compute_scores`）
score = Σ_b [w_b·acc_z + (1-w_b)·nll_z]，w 由各 bench SNR 决定。实测 w：

| bench | w | f | 计分方式 |
|---|---|---|---|
| arc_easy | 0.915 | +0.830 | 几乎纯 acc |
| arc_challenge | 0.624 | +0.247 | 混合 |
| mmlu_stem | 0.597 | +0.195 | acc-only（**NLL 全 null**） |
| gpqa_diamond | 0.210 | -0.581 | 大部分 NLL |
| gsm8k_cot | **0.000** | -7.14 | **纯 NLL** |
| math_cot_500 | **0.000** | -83.4 | **纯 NLL** |

### 4.2 acc SNR（30 点 config 间散布 / 二项噪声地板）

| bench | sd_cfg | SE_binom | **SNR** | p_raw |
|---|---|---|---|---|
| arc_easy | 0.0225 | 0.0127 | **1.77** | 0.68 |
| mmlu_stem | 0.0099 | 0.0100 | 1.00 | 0.27 |
| arc_challenge | 0.0157 | 0.0191 | 0.82 | 0.40 |
| gpqa | 0.0283 | 0.0411 | 0.69 | 0.25（chance） |
| gsm8k | 0.0048 | 0.0163 | 0.29 | 0.27（chance） |
| math | 0.0024 | 0.0259 | **0.09** | 0.25（chance） |

合成（N 加权）：sd 0.0086 vs 地板 0.0064 → **SNR 1.35**，eval 噪声解释 55% 观测方差；best-of-30 胜者诅咒 ≈ 2.04σ ≈ **0.018**。

### 4.3 NLL 补盲验证（剔除 4 个退化点后 n=26 的相关结构）

| 相关对 | ρ | 含义 |
|---|---|---|
| gsm8k NLL ↔ arc_easy acc | **-0.66** | hard bench NLL 携带连续质量信号 |
| math NLL ↔ arc_easy acc | **-0.67** | 同上 |
| gsm8k NLL ↔ math NLL | **+0.95** | 冗余（同一信号） |
| arc_easy acc ↔ mmlu acc | **0.00** | acc 互不相关 = 各自噪声淹没 |
| blended ↔ acc-only 合成 | +0.882 | NLL 部分重排了名次 |

另外：4 个退化配比（NLL 离群，gsm8k NLL 1.07–1.50 vs 中位 0.85）acc 完全看不见，NLL 一眼抓住——**离群检测有效**。

## 5. 发现的问题清单

> 2026-09-10 修复状态：#1、#2（cp4_report fallback 部分）、#3（PYTHONPATH 部分）已修复并过回归；#4 已在 043549f 修复；mmlu NLL 缺失本身（nanochat 0-shot 无 gold-span）待上游。

1. **f 单位 bug**：σ²_noise=0.25/K 是 raw 单位、between 方差算在 centered（÷0.75）上 → 噪声低估 1.78×，w 偏高（修正后 mmlu 0.60→~0.29，arc_ch 0.62→~0.33，自动把权重从噪声 acc 转向 NLL）。
2. **mmlu_stem NLL 全 null**（nanochat 0-shot 无 gold-span）→ 最大 bench 只能 acc-only；eval CSV STEM 行 NLL=nan 同源，cp4_report 需 per-task 聚合 fallback。
3. 锚点臂远端失败（私有 eval 目录缺 `base_checkpoints/` 链接）——043549f 已修；手动重发另因缺 PYTHONPATH（climbmix_ma 不在 sys.path）失败——dispatch 脚本需自带 vendored 路径引导。
4. iter2/3 只跑 4 配置（C_eff 末窗污染）——043549f 已修。
5. 运维：wave1 死作业在 fleet monitor 里显示 UNKNOWN（无害噪声）；跑完目录自动改名导致次日监控扑空（by design，需记住）。

## 6. 结论

1. **信号存在但我们欠采样**：目标函数有真信号（NLL 相关结构证明），但每点复合 SNR ≈ 1.0–1.3、14 维单纯形，30 点低于可学阈值（粗估需 3–4×，≈100 点）——predictor "R²<0 但 spearman 正"正是欠采样签名，而非无信号。guided 迭代本身有价值，前提是实验数到论文量级。
2. **方向翻转但未达显著**：prod1 -0.0036 → prod2 +0.0102，胜者诅咒（~2σ）解释了 proxy 优势未完全传导到 d28。
3. **d20 对推理 acc 结构性全盲**（gsm8k/math/gpqa 全在 chance），NLL 补盲设计被数据验证有效。
4. **零成本改进可得**：修 f 单位 bug 即可提升每点 SNR（w 自动从噪声 acc 转向 NLL）。
5. 运维上：远程并行臂 ~6h（非 13h）→ 24h 预算里搜索可占 ~17h（compact ~5 波 ≈ 55 点）。
