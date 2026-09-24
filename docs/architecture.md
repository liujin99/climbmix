# CLIMBmix 作业流程设计图（architecture）

> **一句话**：供给入口把 OBS 上的全池嵌入拼成本地性能层；实验入口在三层缓存
> 上取簇空间，用"远程舰队 + 本地整槽"的混合编队跑 112 个 d20 proxy 搜索点，
> 再用 d28 臂族验证赢家配方——报告全程自动组装。
>
> 本文是**设计图**不是操作手册：发射命令、对账、停发重发等操作步骤见
> `docs/prod5_runbook.md`；与论文的偏差见 `docs/paper_deviations.md`。
> 图示为 ASCII（可 grep、可 diff）；数字以 prod5 形态（100B 池）为示例。

---

## 0 · 读图须知：三个视角

整张设计图从三个正交视角展开，各回答一个问题，先框架后细节：

| 视角 | 图 | 回答的问题 |
|---|---|---|
| 拓扑 | 图 0 | 谁在哪里、数据放在哪、谁负责算 |
| 生命周期 | 图 1 | 什么阶段发生什么、什么时候做 |
| 数据分层 | 图 2 | 什么能删、删了多心疼（磁盘管理） |

之后是三张细节图：Stage 1 缓存瀑布（图 3）、搜索内部（图 4）、臂族与终报（图 5）。

---

## 1 · 术语拆解（先拆词，再看图）

历史上最坑人的是"merge"一词两用——文档里出现 merge 时先问自己指哪一个：

| 词 | 指什么 | 在哪跑 | 何时发生 | 耗时 |
|---|---|---|---|---|
| **embed_merge**（供给合并） | OBS 嵌入单元 → 本地块文件的**字节搬运**，零聚类语义 | `preprocess_pool.sh` | 本地层被清 / 换池 | 40min |
| **kmeans**（细簇聚类） | 1000 个细簇 = 聚类**基底**；只依赖池内容，与用户旋钮无关 | run_experiment Stage 1 | 基底 npz 缺失时（跑完永久入账） | ~40min（prod5 实测 38.8min） |
| **merge 段**（簇重塑） | 1000 细簇 → prune（质量过滤）→ balanced（容量划分）→ 15 大簇 | run_experiment Stage 1 | 缓存 miss 时（已向量化） | ~2min |

簇空间的两级结构——先有基底，再重塑：

```
116M 文档嵌入 ──kmeans──▶ 1000 细簇（基底；池级缓存 kmeans_K1000.npz）
                             │
                             ▼  merge 段（用户旋钮在此生效）
               prune      : 簇均质量分 < 3.0（或列底线 < 2.0）的细簇淘汰
               balanced   : 容量约束划分到恰好 K_ENHANCED=15 个大簇
                             │
                             ▼
               15 个大簇 = 搜索的坐标系（混合权重就定义在它上面）
```

其余高频术语：

- **泳道** = 一个并发训练槽。搜索期 = 远程 1 个作业（1 节点）或本地整槽；
- **臂** = 验证阶段的一个 d28 训练（搜索赢家配方或某基线，8 节点，3B tokens）；
- **锚点** = 1 节点 eval-only 作业（不训练，出校准读数）；
- **供给 / 消费** = preprocess_pool.sh / run_experiment.sh 两条入口（用户裁决：
  供给不属于实验，实验不内联做 40h 级供给动作）。

---

## 2 · 图 0 · 拓扑：三个执行位置

```
══════════════════════════════════════════════════════════════════
 OBS 对象存储（耐久层 — 一切数据的最终归宿，跨一切存活）
══════════════════════════════════════════════════════════════════
 · embed_units/u0000..u0062     全池嵌入 ~475GB（嵌入波一次性产出，永不清删）
 · climbmix_resource_package/    d20/d28 基础 ckpt + tokenizer + eval 资源
                                 （以 asset_mounts 挂载给每个远程作业）
 · prod/climbmix/<EXP>/exps/     实验数据面：spec.json、混料分片、结果 CSV
                                 （远程作业与编排机之间的中转站）
══════════════════════════════════════════════════════════════════
        ▲│                          ▲│                     ▲
        ││ 供给：下载嵌入块            ││ 实验：上传混料/下载结果  │ 资源只读挂载
        ││ （embed_merge，40min）      ││ （全程川流不息）         │
        ││                          ││                     │
┌───────┘▼──────────────────────────┘▼─────────────────────┘──────┐
│                本地开发机 = 编排者 + 备料场 + 一条本地算力              │
│                （192 vCPU · 8 NPU · 1.6T 盘）                      │
│                                                                  │
│  [供给入口] runs/preprocess_pool.sh   ← 本地层冷了才跑（幂等）        │
│  [实验入口] runs/run_experiment.sh    ← 每轮实验的入口                │
│              Stage 0-3 搜索 · 收官 · 臂族派发 · 报告组装              │
│  [本地整槽] 8 NPU 作为第 11 条搜索泳道参与训练（混合舰队的一等公民）      │
└───────┬────────────────────────────────────────────┬──────────────┘
        │ 派发（搜索 10 并发 / 验证臂族）                  │ 本地整槽训练
        ▼                                            ▼
┌───────────────────────────┐        ┌──────────────────────────────┐
│ ModelArts 舰队（主力算力）    │        │ 远程作业内部（scripts/worker）   │
│                           │        │ 下载混料分片 → symlink 基础ckpt │
│ 搜索期：1 节点/作业 × 10 并发 │        │ → proxy 训练 → eval → 上传结果  │
│ 验证期：8 节点/臂，在飞节点   │        │ （作业间零通信，OBS 是唯一中介）  │
│         总和 ≤ 16（自动排队）│        └──────────────────────────────┘
└───────────────────────────┘
```

要点：本地机是**编排者**（搜索/验证的调度、备料、报告都在这），算力大头在舰队；
OBS 是唯一的数据中转（远程作业之间互不通信）；本地 8 NPU 不是闲置的——
它以整槽身份参与搜索，混合编队 = 10 远程 + 1 本地 = 11 条泳道。

---

## 3 · 图 1 · 生命周期：从池到终报

```
【一次性投资 · 仅换池/换嵌入模型时重复】
  100B 池数据（1000 个 parquet 分片）
      │
      ▼  嵌入波（scripts/embed_dispatch，多节点分片波，~40h）
  OBS embed_units（~475GB，从此永不清删）
      │
【供给 · 本地性能层冷了才跑】
      ▼  runs/preprocess_pool.sh（幂等：热则秒退）
  磁盘守卫 → embed_merge（40min）→ 本地块 443GB
      （block_*.npy + manifest.json + 每块 sha256）
      │
【每轮实验 · 实验入口】
      ▼  runs/run_experiment.sh（显式 5 键：EXP_NAME + 4 个 REMOTE 环境身份键）
  Stage 0   载入池元数据（质量分/token 数，116M docs，~2min）
  Stage 1   簇空间 ★缓存瀑布（四级，见图 3）
  Stage 2   质量过滤（filter=none：直通）
  Stage 3   迭代搜索 ★见图 4
      │      112 个 d20 proxy 实验（64,32,16 × 640M tokens/点，~16h）
      ▼
  收官      最优混合权重 → optimal_mixture_weights.json + 终选采样集
      │
      ▼  验证阶段 ★见图 5
  d28 臂族   top-3 赢家配方 + uniform/natural/domainfix 基线 + base 锚点
      │      （8 节点/臂 × 3B tokens，在飞 ≤16 节点自动排队）
      ▼
  终报      最后一臂落地 → report.md 自动盖 FINAL 章（零人工组装）
```

时间感的锚点：嵌入 40h（一次性）→ 供给 merge 40min（冷了才跑）→
Stage 1 秒级~2min（缓存热时）→ 搜索 ~16h（每轮大头）→ 臂族 ~1-2 天（可并行）。

---

## 4 · 图 2 · 数据分层与文件地图

### 4.1 三层存储：寿命与丢失代价

```
寿命最长 ◀──────────────────────────────────────────▶ 随时可删

 OBS 耐久层        本地性能层         派生缓存层            运行目录
 embed_units       block_*.npy        kmeans_K1000.npz      cluster_cache
 ~475GB 永存        443GB 可再生        stage1_<hash>/        搜索状态/报告
                                     各 ~1GB 跨轮常驻        每轮归档重建

 丢了 = 40h 重嵌入   丢了 = 40min 重merge  丢了 = 2~40min 重算     丢了 = 本轮重跑
 （设计上永不清删）  （收官后清理省盘）     （保留！清理只删块）    （指纹失配即归档）
```

设计指令（用户裁决）：**本地盘只放可再生暂存**——不是"零本地"：discovery
（③a 起）要对 116M×1024 做多趟全量扫描，需要本地读吞吐；但这层只是
40min 可再生的燃料，不是常驻资产。"上量必爆"拆成两个问题：

- **跨轮累积（已解决）**：块文件收官即清（`rm -f <key>/block_*.npy
  <key>/manifest.json`），稳态常驻只剩 ~1.5GB 派生缓存；臂暂存成功后
  自动清（~31GB/臂）；远程备料上传即删——轮与轮之间零残留。且 ② 命中时
  块根本不读：收官清块后只要旋钮+代码不变，重发连 preprocess 都免。
- **单轮占用（上量才触发）**：性能层随池大小线性——prod5 形态 merge 后
  盘余 ~414G 放得下，池翻倍必爆。结构性解 = OBS 直读流式层（Stage 1
  按块流读、不再全量落盘）+ 自动清块——已排队，触发条件 = 换池/上量。

### 4.2 文件地图

```
本地盘:
  ~/work/100B_stem_parquet_filtered/            池数据（1000 分片 + 元数据缓存）
  ~/work/climbmix/
    cache/embeddings/<key>/                     池缓存（key = 池内容哈希）
      ├─ block_u00XX.npy + manifest.json        性能层（收官可删）
      ├─ kmeans_K1000.npz                       派生 · 保留（细簇基底标签）
      └─ stage1_<hash16>/                       派生 · 保留（Stage 1 整段产物）
    result/<EXP>_current/                       运行目录（指纹失配整目录归档）
      ├─ launch_env.json / remote_config.json   发射实录（对账用）
      ├─ cluster_cache.npz + cluster_info_cache.json   run 级簇缓存
      ├─ search_state.json / exp_XXXX/ / report.md    搜索状态与产物
      └─ {arm}_shards/ {arm}_mixed/             臂暂存（成功后自动清理）
OBS:
  {obs_prod_base}/embed_units/u00XX/            耐久层（~475GB）
  {obs_prod_base}/prod/climbmix/<EXP>/exps/     实验数据面
  .../climbmix_resource_package/{d20,...}/      资源包（挂载给远程作业）
```

---

## 5 · 图 3 · Stage 1：簇空间从哪来（瀑布，命中即止）

Stage 1 的全部产物只有两样：**final_labels**（116M 文档 → 15 大簇的标签，
930MB）和 **cluster_info**（15 个簇的档案）。从哪拿，四级瀑布：

```
 ① run 目录 cluster_cache.npz 存在?        ◀ 种子/resume 路径（优先级最高）
    │是 → 载入 [守卫: 行数=池 / doc 总和自洽 / 标签 id 越界] → 完成
    │否
 ② 池目录 stage1_<hash>/ 存在?             ◀ 键 = 全量 discovery 旋钮
    │是 → 载入 [行数守卫] → 完成             + 全仓源码哈希（代码漂移自动重键）
    │否                                       （命中时嵌入根本不用载）
 ③ 现算（进入 discovery）:
     3a. 载入本地块（嵌入矩阵，只读 memmap）
         └─ 完整性校验（只守这道门，不是常规路径的一环）:
              stat（默认）: 63 块 size 对账 + 2M 行抽样 ≈ 1min
              sha: 并行重哈希（抓同尺寸位翻转）
              抽样异常 → 回退全扫（带 quarter 进度）
              ✗ 不符 → FATAL 指路 preprocess（绝不自动重建——两入口裁决）
              块整体缺失 → 同上 FATAL（绝不内联重嵌 ~40h）
     3b. kmeans_K1000.npz 存在?              ◀ 细簇基底，与旋钮无关
         │是 → 载入标签（~40min 免付）
         │否 → prescan（块级并行 ~3min）→ 训练 + 全池指派 → 写 npz
                （~40min，prod5 实测 38.8min；大头 = 116M 全池指派）
    3c. merge 段（~2min，已向量化）:
        1000 细簇 → prune（质量分）→ balanced（容量划分）→ 15 大簇
    3d. 写回: run 级 cluster_cache + 晋升池级 stage1_<hash>/
```

四种典型场景对账：

| 场景 | 走到哪 | 代价 |
|---|---|---|
| 旋钮 + 代码都没变（块清没清无所谓，② 不读块） | ② 命中 | 秒级 |
| 旋钮变了（K / 阈值）或代码变了，块在 | ③b 命中 → ③c | ~3min（载块 + merge 段） |
| 块被清 + 代码变更（stage1 重键） | preprocess + ③b 命中 → ③c | ~40min + ~3min |
| 池变了（新 key 目录，kmeans 一并重算） | preprocess + ③ 全程 | ~40min + ~45min |
| 嵌入文件被动过（异常） | 3a 抓住 | FATAL → 人工重 merge |

### 常见误读（历史困惑点存档）

- preprocess **不做任何聚类**——它只是字节搬运（两个 merge 的区别，见 §1）；
- kmeans **在 run_experiment 里跑**——种子轮看不到它，是因为 run 级缓存
  命中把 Stage 1 整个绕过了；
- sha 校验**不是常规路径的一环**——它只守 3a 的门，stat/样本是默认快路径。
- ①② 不做 sha 不是遗漏——校验强度**与信任模型成比例**：443GB 块是外部
  过程产出的信任根才配 sha/stat 分级；<1GB 的 npz 由 zip CRC 抓位腐
  （坏得响 → 重算）+ 一致性守卫（① 行数 / doc 和自洽 / 标签 id 越界；
  ② 池 key 目录绑定 + 内容键 + 行数）覆盖同一"坏得响"语义，加 sha 不
  改变任何行为；
- ② 的键哈希**包内全部源码**（src/climbmix；scripts/runs 不参与簇计算，
  不入键）——误重键只付 ~3min（③b 命中 → merge 段），漏重键会静默供
  旧代码算出的簇；风险不对称，宁粗勿细。

---

## 6 · 图 4 · Stage 3：迭代搜索内部

```
轮次计划 64 → 32 → 16（论文 §2.2，共 112 点；自适应模式 e_i 是下限，
                    舰队有空槽时自动补点到满载）

每轮循环:
  配置生成 ──▶ 本地备料 ──▶ 提交 ──▶ 训练+评测 ──▶ 结果落地 ──▶ 预测器
     ▲                                                        │
     └────────── 引导采样（下一轮）◀── LightGBM 拟合 ◀──────────┘

  配置生成:  轮 1 = 簇 token 占比加权的 Dirichlet 随机
             轮 2+ = 预测器排序 top-N 中抽 M 个（论文语义，无扰动）
  本地备料:  按权重从池选 STEM 文档（70%）+ 混 general（30%）→ 写分片
             → 上传 OBS；并发 = REMOTE_MAX_PREP（耦合搜索节点数）
  泳道:      远程 10 作业（1 节点/作业）+ 本地 1 整槽（8 NPU）= 11 并发
  作业内部:  下载分片 → symlink d20 基础 ckpt → proxy 训练（640M tokens）
             → eval（stem 基准）→ 结果 CSV 上传 → 本地落地 exp_XXXX/
  单点成本:  ~1.5-2h → 112 点全程 ~16h
  鲁棒性:    单点失败 = inf/0 分进预测器（NaN 正确丢弃）；
             resume 三级（权重精确复用 / mid-train 续 eval / 重跑）；
             单遍守卫防混料池被重复消费
```

---

## 7 · 图 5 · 验证阶段：d28 臂族与终报

```
搜索收官 → topk_mixture_candidates.json

候选臂（全部 d28 · 3B tokens · 8 节点/臂）:
  climb-cfgXX × 3    搜索赢家配方（top-3）
  uniform            基线：簇等权
  natural            基线：逐文档等概率 ≡ 簇 token 占比
  domainfix          基线：领域固定
  base 锚点          1 节点 eval-only（不训练，校准读数用）

发射: scripts/dispatch_target_arm.py —— 全部臂命令可连发（无需人工排队）
  └─ 臂族注册表 .validation_fleet/：在飞训练臂节点总和 ≤ 16
     （REMOTE_MAX_VALIDATION_NODES，默认 16 = 2 臂 × 8 节点）
     超限的派发自动排队轮询；1 节点锚点入册可见但免计（1 与 8 不可比）

臂落地（eval CSV + ckpt 回传）时自动:
  ├─ report.md 幂等刷新（CP4 判定节 + 赢家配方节，flock 串行化）
  ├─ 本地暂存自动清理（{arm}_shards + {arm}_mixed，~31GB/臂；
  │   守卫双保险拒清未成功臂；CLIMBMIX_ARM_AUTOCLEAN=0 关闸）
  └─ 最后一臂落地 → 终报自动盖 FINAL 章（预期臂清单自核查，
     expected_arms.txt 可覆盖）—— 整个大实验的最后一步由机器自判
```

---

## 8 · 速查

### 8.1 发射成本矩阵（缓存分层落地后的预期）

| 场景 | 路径 | 发射 → 首批派发 |
|---|---|---|
| 旋钮 + 代码不变（常规重发；块清没清无所谓） | ② 命中 | **~20min**（爬坡主导） |
| 代码变更后首发（块在） | ③b 命中 + merge 段 | ~22min |
| 收官清块后 + 代码变更 | preprocess + ③b + merge 段 | ~65min |
| 换池 / 换嵌入模型 | 嵌入波 + 全链 | ~40h + 上述 |

### 8.2 常用命令（详见 runbook）

```
bash runs/preprocess_pool.sh                  # 供给入口（幂等，冷才跑）
EXP_NAME=prodN REMOTE_ENABLED=1 REMOTE_BACKEND=modelarts \
  REMOTE_BACKEND_MODULE=climbmix_ma:create_backend \
  REMOTE_FLAVOR=modelarts.pool.visual.8xlarge \
  bash runs/run_experiment.sh                 # 实验入口（5 键）
python3 scripts/check_launch_parity.py <src_run> <dst_run>   # 发射后 1min 对账
bash scripts/diagnostics/prod2_watch.sh result/<EXP>_current # 监控
python3 scripts/backfill_block_hashes.py cache/embeddings/<key>  # 存量缓存补哈希
```
