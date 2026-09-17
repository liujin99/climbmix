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

[1] seq/GA 取自当日 dispatch 守卫打印（db=1→GA=8；晨间 db=2/4→GA=4/2 均整除放行后死于 OOM）；落地 log 的 `Inherited max_seq_len` / `Grad accum steps` 行可复核。

训练健康度：dt 3365 ms/步、bf16_mfu 14.42%、311,609 tok/s、175 min/臂、epoch 1——与 09-16 f1bad8c9（同配方 8 节点 db=1）逐位一致，复现性锚点。

## 2. 主结果（CP4 三臂全景，ref = random）

**管线校准**：锚点 323855b4 远端 base stem = **0.1746** vs 本地 0.1738（|Δ|=0.0008 → PASS，阈值 0.002）。

headline（stem_metric，centered；SE_Δ=√2×0.006=0.0085）：

| 臂 | stem | Δvs ref | z | p | tag |
|---|---|---|---|---|---|
| climb | **0.1972** | +0.0191 | +2.25 | 0.012 | WINS** |
| random（ref） | 0.1781 | — | — | — | — |
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

1. **CLIMB WINS（z=+2.25，~95% 单边）——系列首轮显著**，且跨轮单调改善：prod1 -0.0036（z≈-0.4，噪声）→ prod2 +0.0102（z +1.20）→ prod4 +0.0191（z +2.25）。管线逐轮成熟（评分修正、D17 协议、数据面修复、ws=64 Muon 墙拆除）的红利兑现。
2. **预算匹配对照更强**：climb vs random3b（同为 3B、同日、同 8 节点 db=1 配方）Δ=+0.0325，z≈+3.8。
3. **赢点结构**：gsm8k_cot .258 vs .119/.108（**2.2-2.4×，主引擎**）+ arc_challenge +.015 + math +.010；负项仅 gpqa（N=198，SE~.03 噪声主导）与 arc_easy -.018。收益集中在推理/CoT 任务——与搜索目标（stem 含双 CoT）一致。
4. **climb 超基线 +0.023**（0.1972 vs base 0.1738）；random 0.1781 ≈ 基线；random3b .1647 低于基线——提升来自 mixture 本身而非预算。
5. **噪声/预算标定**：random 两落点差 -0.0134（z -1.57 不显著）——混杂复跑噪声与预算差（ref random 为早前落地臂，计划预算 6B/4 节点恢复路径，以 audit 为准[2]）；无论取哪个 random，climb 双压。
6. **NLL 反转**：random3b NLL 最低（1.8527）但 stem 最差——次级指标与目标背离又一例，stem_metric 作搜索目标的正确性再确认。
7. gsm8k 绝对水平较 prod2（.05-.06）大幅上移（.11-.26）：D17 协议（stop strings + math cap 1024）放大 CoT 任务分辨率，混合物间差异同步放大。

[2] 待核：`result/prod4_current/target_arm_random.json`（requested_target_tokens / node_count）。若 ref 实为 3B 同配方，则第 5 条退化为纯复跑噪声标定，不影响 1-4。

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

工程面：双臂满训 2861 步 + 训后 eval 存活（对照 09-16 16 节点双 run 训满即死于 eval——53d0bcd+20b94b0 修复的首次生产验证）；salvage（bc2cdcf）备而未触发；本地 shards+mixed 29.9 GB 自动清理；服务器树当晚 pull 至 3df386a。

## 5. 结论与下一步

1. **L0 裁决：A（climb 赢家配比）胜出**——搜索信号传导至 d28 mid-train 成立；插值赢家（predictor_design_space 首发外推点）未翻车，对照 prod1 的未测角落事故。
2. **L1a = cfg25 + cfg72**（已定案）：热区内部三点对照（插值赢家 / 实测冠军 cfg25 1.2550 / 最近邻 cfg72 1.1297，搜索尺度分）；权重已备 `tmp/cfg{25,72}_weights.json`；可选 16 节点 db=1 赶墙钟（2.12s/步，1.58×，+39% node-minutes）。
3. 收尾清单：峰值内存实测回填 db 判决条目；ref random 预算核对（脚注[2]）；OBS legacy 孤儿清理（无键 `mixture_data` + 6B random 1136 片）；#11 mfu 探针。

## 附录：产物路径

- ckpt：`/home/ma-user/work/nanochat_model_dir/mid_checkpoints/d28_{climb,random3b}_prod4`
- audit/日志：`result/prod4_current/`（`target_arm_*.json`、`mid_train_*.log`、`eval_*.csv`、`launch_env.json`）
- 提交端日志：`/home/ma-user/work/tmp/prod4_{climb8,random3b8_db1}_arm.log`、`prod4_anchor_eval2.log`
- OBS：`obs://bucket-pangu-green-guangzhou/s00944147/l00916525/prod/climbmix/prod4/target_arms/{climb,random3b}/mixture_data_k{da5cb5340f9c,8bed4b50e884}`
