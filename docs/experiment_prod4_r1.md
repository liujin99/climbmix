# prod4 R1 实验结果记录（赢家验证轮：climb vs random 基线）

**日期**：2026-09-17（当日事故链三次重发后 16:00/16:32 双臂入轨，19:29 三作业全部落地）
**归档**：`result/prod4_current`（CP4 报告由 arm_engine 自动渲染）
**代码基线**：climbmix `bf9690c`（发射时服务器树；守卫 rev3 `3df386a` 当晚落地后已 pull）/ nanochat-npu `20b94b0`（dev-data-mix）/ boot shell climbmix-ma `146f465` / 资产 tar sha256 `a7c56792`
**关联记档**：TODO 2026-09-17「db≥2 内存墙判决」（本轮 = 该判决后首个全净验证轮）；`paper_deviations.md` D16/D17

---

## 1. 实验配置

| 项 | 值 |
|---|---|
| 轮次语义 | prod4 R1：搜索赢家配比（climb）vs 均匀簇基线（random）在 d28 mid-train 上验证 |
| 模型 | d28 base（~2.4B 参数；fp32 master 7.38 GiB 混合 dtype 存储，计算 bf16） |
| 预算 | TARGET_TOKENS=3B → 2861 步 × total_batch 2²⁰ tokens/步（实测 1.048M tok/步 = tok/s×dt 闭合） |
| 并行 | 8 节点 × 8 卡 = ws=64；MID_DEVICE_BATCH_SIZE=1（生产锁）；grad_accum=8（seq 2048）[1] |
| 配方锁定 | LR/wd 链自预训练 meta 继承 × √(2²⁰/B_REF) batch 缩放；lrm 线性退火 1→0；--load-optimizer=0 |
| 数据-climb | 赢家配比 C10 .4737 + C5 .3540 + C12 .0641 + 12 簇 0.9% 地板（搜索 111 实测点 = 54 历史 + 57 新；末轮 val R² .32 / Spearman .50 / 在线 ρ .421；Selection = predictor_design_space） |
| 数据-random3b | 均匀簇权重 α_k=1/15（doc 层面按池加权，非均匀抽 doc）；与 climb 同日同配方重跑 |
| OBS | 内容键 `target_arms/{climb,random3b}/mixture_data_k{da5cb5340f9c, 8bed4b50e884}`；本地 shards+mixed ~15 GB/臂，落地后自动清理 |
| eval | 4 MC（arc_easy 2376 / arc_challenge 1172 / mmlu_stem 3545 / gpqa_diamond 198）+ 2 生成 CoT（gsm8k_cot 1319 / math_cot_500 500，D17 协议：stop strings + math cap 1024）；锚点 = base ckpt 单节点远端复评 |
| 产出 | ckpt `mid_checkpoints/d28_{climb,random3b}_prod4`；audit `result/prod4_current/target_arm_{climb,random3b}.json` |

[1] seq/GA 取自当日 dispatch 守卫打印（db=1→GA=8；晨间 db=2/4→GA=4/2 均整除放行后死于 OOM）。**已核对（2026-09-17 晚）**：落地 log 打印 `Inherited max_seq_len=2048` / `Inherited total_batch_size=1048576` / `Grad accum steps: 8`，与守卫打印一致。

训练健康度：dt 3365 ms/步、bf16_mfu 14.42%、311,609 tok/s、175 min/臂、epoch 1——与 09-16 f1bad8c9（同配方 8 节点 db=1）逐位一致，复现性锚点。峰值内存 24962 MiB ≈ 24.4 GiB（29.49G 卡，余量 ~5.1G）。

## 2. 主结果（CP4 三臂全景，ref = random）

**管线校准**：锚点 323855b4 远端 base stem = **0.1746** vs 本地 0.1738（|Δ|=0.0008 → PASS，阈值 0.002）。

**ref provenance（2026-09-17 晚闭环）**：`eval_random.csv`（mtime 09-16 10:28）与 prod3 归档**逐字节相同**（diff SAME_AS_PROD3）——它是 **prod3 random 臂**（1bda1c2f，SUCCEEDED，4 节点 ws=32、3B/2861 步、cluster_info 与 prod4 逐字节相同、同均匀 α=1/15 家族）的 eval，09-16 晨「cp4 手动渲染」准备期被拷入 prod4_current 作占位 ref。prod4 本轮名为 random 的两次尝试**全部失败**：10c9fc37（7m，ws=64 Muon OOM，即 09-15 夜三臂事故②）与 d9b8dae3（16 节点 random3b 09-16，训练段跑满 2861 步但 eval 死、ckpt 无 salvage 丢失——「后台看着成功」即此）。判定影响见判读 5。

headline（stem_metric，centered；SE_Δ=√2×0.006=0.0085）：

| 臂 | stem | Δvs ref | z | p | tag |
|---|---|---|---|---|---|
| climb | **0.1972** | +0.0191 | +2.25 | 0.012 | WINS** |
| random（ref*，prod3 复刻） | 0.1781 | — | — | — | — |
| random3b | 0.1647 | -0.0134 | -1.57 | 0.942 | 噪声 |

per-benchmark（raw acc ± 二项 SE）：

| bench | N | climb | random | random3b |
|---|---|---|---|---|
| arc_easy | 2376 | .7210±.0092 | .7391±.0090 | .7205±.0092 |
| arc_challenge | 1172 | .4428±.0145 | .4275±.0145 | .4326±.0145 |
| mmlu_stem | 3545 | .2818±.0076 | .2795±.0075 | .2818±.0076 |
| gpqa_diamond | 198 | .2273±.0298 | .2525±.0309 | .2121±.0291 |
| gsm8k_cot | 1319 | **.2578**±.0120 | .1190±.0089 | .1077±.0085 |
| math_cot_500 | 500 | .0280±.0074 | .0180±.0059 | .0180±.0059 |

聚合：climb raw mean Δ **+0.0205**（SE .0089，z +2.30，p 0.011，4/6 胜，精确单侧 p 0.344）；random3b Δ -0.0105（z -1.21，2/6，p 0.891）。
stem NLL（次级）：climb 1.8637 / random 1.8609 / random3b 1.8527。

## 3. 判读

1. **CLIMB WINS——主判据为预算匹配同日对照 climb vs random3b Δ=+0.0325（z≈+3.8）**；vs prod3 random 复刻 +0.0191（z +2.25，~95% 单边）为第二佐证。系列跨轮单调改善：prod1 -0.0036（z≈-0.4，噪声）→ prod2 +0.0102（z +1.20）→ prod4 +0.0325（同日同码对照）。管线逐轮成熟（评分修正、D17 协议、数据面修复、ws=64 Muon 墙拆除）的红利兑现。
2. **预算匹配对照更强**：climb vs random3b（同为 3B、同日、同 8 节点 db=1 配方）Δ=+0.0325，z≈+3.8。
3. **赢点结构**：gsm8k_cot .258 vs .119/.108（**2.2-2.4×，主引擎**）+ arc_challenge +.015 + math +.010；负项仅 gpqa（N=198，SE~.03 噪声主导）与 arc_easy -.018。收益集中在推理/CoT 任务——与搜索目标（stem 含双 CoT）一致。
4. **climb 超基线 +0.023**（0.1972 vs base 0.1738）；random 0.1781 ≈ 基线；random3b .1647 低于基线——提升来自 mixture 本身而非预算。
5. **噪声标定（provenance 闭环后更干净）**：0.1781（prod3 复刻）与 0.1647（prod4 random3b）= **两个独立 3B 均匀复刻的纯复跑噪声**（差 -0.0134，z -1.57 不显著；此前担心的预算混杂不存在——prod3 亦为 2861 步）。climb 对两个复刻分别 +0.0191/+0.0325 双压。跨轮小告示：prod3 ref 为 ws=32（total_batch 形状不变量，GA 补偿、训练数学同构）且其 eval 早于 53d0bcd/20b94b0——锚点 PASS（0.1746 vs 0.1738）+ 两复刻互差在噪声内，无系统性偏差证据。
6. **NLL 反转**：random3b NLL 最低（1.8527）但 stem 最差——次级指标与目标背离又一例，stem_metric 作搜索目标的正确性再确认。
7. gsm8k 绝对水平较 prod2（.05-.06）大幅上移（.11-.26）：D17 协议（stop strings + math cap 1024）放大 CoT 任务分辨率，混合物间差异同步放大。

[2] 已闭环（2026-09-17 晚）：ref random = prod3 臂（audit：SUCCEEDED、node_count=4、elapsed 18461s、csv 指向 prod3 归档）；prod4_current 里的 FAILED audit（node_count=8）属 10c9fc37 那次 7 分钟阵亡，此前误当 ref 配置。另：实测峰值内存 24.4 GiB 高于「静态 15G + db=1 前向 6.4G」朴素和约 3G（分配器 workspace/HCCL 缓冲/碎片），已同步回填 TODO db 判决条目。

## 4. 时间线（当日事故链 → 落地）

| 时刻 | 事件 |
|---|---|
| 10:30 | 三作业同时阵亡（boot shell 键化 bug × 双臂 + 锚点 EL0004，见 TODO 09-17 上午条） |
| 13:01 | db=4 双臂 step-0 OOM 27.84G（climb 6b2d962b / random3b 65972ece） |
| 14:45 | db=2 同点 OOM 27.61G（climb e218f73a） |
| 16:00/16:32 | db=1 双臂入轨（climb 418f1301 / random3b e2fae0c0） |
| 18:52 | climb SUCCEEDED 175m → CP4 首报（2 臂） |
| 18:59 | 锚点重发 323855b4（TARGET_ARM_NODES=1 修复节点数解析链） |
| 19:27/19:29 | random3b SUCCEEDED 175m → CP4 三臂终报；锚点出分 PASS（晨间 EL0004 未复现，待查项保留） |
| 当晚 | ref provenance 闭环：eval_random.csv = prod3 臂拷贝件（diff 逐字节同）；prod4 random 两试全败（10c9fc37 7m OOM / d9b8dae3 训满死于 eval） |

工程面：双臂满训 2861 步 + 训后 eval 存活（对照 09-16 16 节点双 run 训满即死于 eval——53d0bcd+20b94b0 修复的首次生产验证）；salvage（bc2cdcf）备而未触发；本地 shards+mixed 29.9 GB 自动清理；服务器树当晚 pull 至 3df386a。

## 5. 结论与下一步

1. **L0 裁决：A（climb 赢家配比）胜出**——搜索信号传导至 d28 mid-train 成立；插值赢家（predictor_design_space 首发外推点）未翻车，对照 prod1 的未测角落事故。
2. **L1a = cfg25 + cfg72**（已定案）：热区内部三点对照（插值赢家 / 实测冠军 cfg25 1.2550 / 最近邻 cfg72 1.1297，搜索尺度分）；权重已备 `tmp/cfg{25,72}_weights.json`；可选 16 节点 db=1 赶墙钟（2.12s/步，1.58×，+39% node-minutes）。
3. 收尾清单：OBS legacy 孤儿清理（无键 `mixture_data` + 6B random 1136 片）；#11 mfu 探针；可选——把 prod3 复刻 ref 显式登记进 CP4 渲染（当前 eval_random.csv 为拷贝件，报告不知情；防未来误读）。

## 附录：产物路径

- ckpt：`/home/ma-user/work/nanochat_model_dir/mid_checkpoints/d28_{climb,random3b}_prod4`
- audit/日志：`result/prod4_current/`（`target_arm_*.json`、`mid_train_*.log`、`eval_*.csv`、`launch_env.json`）
- 提交端日志：`/home/ma-user/work/tmp/prod4_{climb8,random3b8_db1}_arm.log`、`prod4_anchor_eval2.log`
- OBS：`<prod-prefix>/prod4/target_arms/{climb,random3b}/mixture_data_k{da5cb5340f9c,8bed4b50e884}`（完整桶前缀属内部值，见服务器 audit / launch_env.json，不入公开仓）
