# Embedding 性能优化记录

> 记录时间: 2026-08-24
> 目的: 记录 NPU embedding 阶段性能优化的尝试过程、数据分析和结论，避免重复踩坑

---

## 1. 背景

ClimbMix 流水线中，embedding 聚类阶段需要对 116M 文档进行向量化，使用 8×Ascend 910B4 NPU（32GB HBM/卡）并行推理 stella_en_400M_v5 模型。

**性能目标**: 750 docs/s（ETA ~43h）

---

## 2. 硬件环境

- **NPU**: 8×Ascend 910B4（32GB HBM/卡）
- **CANN**: 8.5.1
- **PyTorch**: 2.9.1 + torch_npu 2.9.0.post1
- **CPU**: 192 vCPUs
- **内存**: 1.5TB
- **模型**: stella_en_400M_v5（1024维, fp16, max_seq_len=512）
- **数据**: 1000 个 parquet 文件，116M 文档，4 个 domain

---

## 3. 优化尝试时间线

### 3.1 `e0036e1` — 基线版本（~750 docs/s）✅

**改动**:
- `model.half()`（fp16）
- `model.max_seq_length = 512`（截断到 512 token）
- batch_size 256→512
- `output.float()` → `.normalize()` → `.cpu().numpy()`（float+normalize 在 NPU 上）
- tok_pool `max_workers=1`
- **流水线式预取**：NPU 处理 batch N 时，CPU 同时 tokenize batch N+1（只提交 1 个 future）

**结果**: ~750 docs/s，HBM ~28GB，无 OOM，AICore 0-100% 跳动

---

### 3.2 `43d4007` — 预提交全部 futures（~420 docs/s）❌

**改动**（在 `e0036e1` 基础上）:
- tok_pool `max_workers` 1→2
- **一次性提交整个 shard 的 ALL tokenization futures**（代替逐个流水线预取）
- `del texts, tok_futures`

**结果**: ~420 docs/s（比基线慢 1.8x），最终 OOM

**分析**: 预提交 ALL futures 导致：
1. 所有 batch 的 tokenization 结果同时驻留内存
2. tok_pool=2 的两个线程与主线程竞争 CPU（D2H 传输、memmap 写入、numpy 运算）
3. GIL 竞争：多个 tokenization 线程抢占 Python GIL，延迟 `.result()` 调用
4. NPU 是瓶颈（~5s/batch），tokenization 只需 <1s，额外线程纯属浪费

**结论**: **流水线式预取（1 个 future）优于预提交全部 futures**

---

### 3.3 `9ece26c` — timing 诊断 + float 移到 CPU（OOM）❌

**改动**（在 `43d4007` 基础上）:
- tok_pool 2→4
- float+normalize 从 NPU 移到 CPU（`.cpu()` → `.float()` → `.normalize()` → `.numpy()`）
- 添加 per-batch timing 输出（wait/npu/cpu）

**结果**: 立即 OOM

**分析**:
- tok_pool=4 加剧了 CPU 竞争
- float 移到 CPU 本身不影响 NPU 速度，但 `.cpu()` 提前触发同步可能改变 NPU 执行流
- timing prints 本身可忽略（`time.time()` 微秒级）

**timing 数据**（关键发现）:
- `wait=0ms`（batch 1+）→ tokenization 不是瓶颈
- `cpu=4ms` → CPU 后处理可忽略
- `npu=10-16s` → NPU 前向是唯一瓶颈（基线版本约 5s）

---

### 3.4 `5c720c7` — batch 256 + empty_cache 每个 shard（~233 docs/s）❌

**改动**:
- batch_size 512→256
- 每个 shard 结束后 `torch.npu.empty_cache()`

**结果**: ~233 docs/s（比基线慢 3.2x）

---

### 3.5 `421d4a6` — batch 512 + empty_cache 每 5 batch（~211 docs/s）❌

**改动**:
- batch_size 256→512
- 每 5 个 batch 调用 `torch.npu.empty_cache()`

**结果**: ~211 docs/s，NPU 时间 11-17s/batch

---

### 3.6 `7ec188f` — 删 empty_cache + 内存限制 0.92（~213 docs/s, OOM）❌

**改动**:
- 删除所有 `empty_cache()` 调用
- 添加 `torch.npu.set_per_process_memory_fraction(0.92)`（限制 30.1GB）

**结果**: ~213 docs/s，OOM

**分析**: `set_per_process_memory_fraction(0.92)` 限制分配器到 30.1GB，但 batch 512 需要 ~31GB。分配器永远无法稳定，不断释放/重分配——和 `empty_cache()` 完全相同的机制。

---

### 3.7 `719ba05` — 回退到 `43d4007` 状态（~420 docs/s, OOM）❌

**改动**: 删除 `set_per_process_memory_fraction`，tok_pool 4→2，float+normalize 移回 NPU，删除 timing

**结果**: ~420 docs/s，OOM（代码与 `43d4007` 完全一致，`git diff` 零差异）

---

### 3.8 `00c45da` — 回退到 `e0036e1` 状态（~659+ docs/s, 仍在加速）✅

**改动**: 完全回退到 `e0036e1`（tok_pool=1，流水线式预取，float+normalize 在 NPU）

**结果**: 150s 时 659 docs/s（仍在 ramp up），HBM 23-28GB（舒适），AICore 88-100%（大多数 NPU）

---

## 4. 关键结论

### 4.1 绝对不要做的事

| 操作 | 影响 | 原因 |
|------|------|------|
| **`torch.npu.empty_cache()`** | 3x 慢（750→211 docs/s） | 强制分配器释放并重新分配 ~3.3GB 内存块，破坏 block 复用 |
| **`set_per_process_memory_fraction(0.92)`** | 3x 慢 + OOM | 限制分配器到 30.1GB，但 batch 512 需要 ~31GB，分配器永远无法稳定 |
| **预提交 ALL futures** | 1.8x 慢（750→420 docs/s） | 所有 tokenized batch 驻留内存，CPU 线程竞争 + GIL 竞争 |
| **tok_pool > 1** | ~1.8x 慢 | NPU 是瓶颈（5s/batch），tokenization 只需 <1s，额外线程纯属竞争资源 |
| **batch_size < 512** | 按比例慢 | NPU 启动开销固定，batch 越小吞吐越低 |

### 4.2 最佳配置

| 参数 | 值 | 说明 |
|------|------|------|
| `model.half()` | fp16 | 必须开，显存减半 |
| `model.max_seq_length` | 512 | CLIMB 论文 ablation <0.35% |
| `batch_size` | 512 | 平衡吞吐和显存（~28GB HBM） |
| `tok_pool.max_workers` | 1 | 流水线预取，不要并行 |
| 预取策略 | 1 个 future | NPU 处理 batch N 时 CPU tokenize batch N+1 |
| float+normalize | NPU 上做 | `.float()` → `.normalize()` → `.cpu().numpy()` |
| `empty_cache()` | 不用 | 分配器自动复用 batch 0 的 block |
| `set_per_process_memory_fraction` | 不用 | 让分配器使用全部 32GB |
| 进程并行 | 8 进程，每进程 1 NPU | 不用 DataParallel，用 `mp.Process` |

### 4.3 原理：CANN NPU 缓存分配器

CANN/PyTorch NPU 的缓存分配器（caching allocator）行为：
1. 第一次 batch 时，分配 ~28GB 的内存块（模型 + 激活）
2. 后续 batch **复用同一批内存块**，零开销
3. `del features, output` 只是把 tensor 引用清掉，block 仍在分配器缓存中
4. 下次 `model(features)` 直接复用缓存的 block，无需 `malloc`

**`empty_cache()` 破坏了这个机制**：
- 强制分配器释放所有缓存的 block 回操作系统
- 下一个 batch 必须重新 `malloc` ~3.3GB
- 每次释放/重分配耗时数秒 → 3x 慢

### 4.4 为什么 tok_pool=1 + 流水线预取最优

```
时间线（tok_pool=1, 流水线预取）:
  CPU: [tokenize b0] [wait] [tokenize b1] [wait] [tokenize b2] ...
  NPU: [...........] [forward b0] [forward b1] [forward b2] ...
                         ↑ NPU 处理 b0 时，CPU 同时 tokenize b1

时间线（tok_pool=2, 预提交 ALL）:
  CPU: [tokenize b0,b1,b2,b3,... 全部完成] [竞争: D2H + memmap + tokenize]
  NPU: [...........] [forward b0] [gap] [forward b1] [gap] ...
                                              ↑ CPU 竞争导致批次间 gap
```

- NPU 前向需要 ~5s/batch
- Tokenization 只需 ~0.5s/batch
- 1 个 tokenization 线程足够跟上 NPU 节奏
- 额外的 tokenization 线程只是在和主线程抢 CPU（D2H 传输、memmap 写入）

---

## 5. 性能对比汇总

| 版本 | tok_pool | 预取方式 | empty_cache | mem_fraction | batch | 速度 (docs/s) | OOM |
|------|----------|----------|-------------|--------------|-------|-------------|-----|
| `e0036e1` | 1 | 流水线 | 无 | 无 | 512 | ~750 | 否 |
| `43d4007` | 2 | ALL | 无 | 无 | 512 | ~420 | 是 |
| `9ece26c` | 4 | ALL | 无 | 无 | 512 | OOM | 是 |
| `5c720c7` | 4 | ALL | 每 shard | 无 | 256 | ~233 | 否 |
| `421d4a6` | 4 | ALL | 每 5 batch | 无 | 512 | ~211 | 否 |
| `7ec188f` | 4 | ALL | 无 | 0.92 | 512 | ~213 | 是 |
| `719ba05` | 2 | ALL | 无 | 无 | 512 | ~420 | 是 |
| `00c45da` | 1 | 流水线 | 无 | 无 | 512 | ~659+ | 否 |

---

## 6. 未解决的问题

### 6.1 AICore 利用率跳动

即使最优配置，AICore 仍在 0-100% 跳动（基线版本）。原因可能是：
- 批次间存在小 gap（CPU 处理 D2H + memmap 写入）
- CANN runtime kernel 启动开销
- FlashAttention mask 处理开销

### 6.2 NPU 前向比理论值慢 8x

- batch 512 × seq 512 理论 ~660ms（910B4 算力估算）
- 实际 ~5s（基线版本）
- 可能原因：stella 自定义 `modeling.py` 中有算子未走 NPU kernel、SDPA 未用 FlashAttention
- 未深入调查（HuggingFace 不可达，模型文件仅存于 NPU 服务器）

### 6.3 `model.tokenize()` 已废弃

- 警告："The `tokenize` method is deprecated, please use `preprocess` instead"
- 尚未切换到 `preprocess`，功能正常但有潜在风险

---

## 7. At-scale 聚类阶段的内存墙与分块指派（2026-08-29）

嵌入完成后的聚类阶段(K-means 训练 + 全量指派)在 116M docs × 1024 dim
fp32(≈475 GB)下不能整矩阵进内存。逐项分析(`cluster_assign.py` /
`cluster_embeddings_faiss` memmap 路径):

| 操作 | 整内存路径峰值 | 分块路径峰值 |
|---|---|---|
| `np.isnan(emb)`/`emb == 0` 全矩阵布尔 | ~119 GB ×2 次 | 一个块 |
| `np.nan_to_num` 整拷贝 | ~475 GB | 0(干净池)/一个块(边车) |
| `kmeans.train` | faiss 内部采样 ≤256×K_init 点(~1 GB) | 同左(不变) |
| `index.search` 标签输出 | 输入已在 RAM + ~1.4 GB | 一个块 + 0.93 GB 标签 |
| **合计(新跑,memmap 输入)** | **>500 GB(OOM 风险)** | **≈ 块大小 + ~1 GB** |

关键事实:
- **指派逐行独立**(每行与 1000 质心算内积取 argmax),分块只改 I/O 批次,
  标签与整内存路径**逐元素相等**(测试断言 exact equality,非近似)。
- 块大小 = min(MemAvailable, cgroup limit)/8,钳 [64 MB, 8 GB];
  `CLIMB_ASSIGN_CHUNK_GB` 可覆盖。>8 GB 无收益(总 I/O 与计算量不变,
  faiss 百万行批即打满线程);cgroup 项防容器误读(/proc/meminfo 报宿主机)。
- 路由:**memmap 输入(新跑)→ 分块;ndarray(npz 缓存命中/子样本)→ 原路径
  一字不改**。当前服务器 1.5 TB RAM 下两路径结果一致。
- 磁盘共存期提醒:新跑时 memmap(475 GB)与 npz 缓存(475 GB)短暂共存
  ≈950 GB;盘不够时改 OBS 存储(见 TODO E)。
- RLIMIT_AS 实测(1.6 GB memmap、2.5 GB 地址空间上限):分块路径通过,
  整载路径 MemoryError——墙真实存在且解有效。

---

## 8. 聚类阶段 CPU 吞吐实测与线程钉扎（2026-08-29, 192-vCPU aarch64）

`scripts/diagnostics/cluster_bench.py` 在生产服务器实测(faiss 1.15.0,
BLAS 钉 1 线程,搜索形状 = 一个 kmeans 训练迭代: 256K×1000×1024 = 524
GFLOP/call):

| threads | GFLOP/s | 说明 |
|---:|---:|---|
| **24** | **281** | **甜点**(11.7 GFLOP/s/核,≈Kunpeng sgemm 实际峰值) |
| 48 | 150 | 崩塌(非平台) |
| 64 | 141 | 旧代码默认 min(cpu,64) |
| 96 | 99 | |
| 192 | 114 | |

结论与动作:
- **机制更正 + 钉扎 clobber 二次修复(同日晚)**: 前一版把首要瓶颈记为
  "嵌套线程池踩踏"——**错**。这版 wheel(faiss 1.14.3 x86 / 1.15.0
  aarch64)捆绑的是 **OpenMP 构建的 OpenBLAS(libopenblaso),与 faiss
  共享同一份 libgomp**;OpenMP 默认 nested=off → 4316d64 直跑时的
  真实形态是"64 个 worker 各自串行跑 1 线程 sgemm"(≈bench 64 线程
  的 141 GFLOP/s,与 480.3s 反推的 112 量级吻合),没有踩踏。而
  a3a075a 引入的 threadpoolctl 钉扎反而把 faiss 自己的 OMP 池一起
  踩成 1——`openblas_set_num_threads(1)` 在 OpenMP 构建上转发到
  共享 libgomp 的 `omp_set_num_threads(1)`。生产实测: 日志照打
  `threads=24, blas=1(pinned)`,实际 1% CPU、118s/迭代(≈4.4
  GFLOP/s),500K 训练预计 ~3.3h,比 4316d64 还慢。修复:
  `_blas_single_thread` 在进钉扎后与退出时各**重申**一次
  `faiss.omp_set_num_threads`(日志改 `blas=1(pinned+reassert)`),
  重申不生效打 WARNING 哨兵;bench 新增 `_pin_probe`(按生产顺序
  omp_set→threadpool_limits 的探针,本机 x86 实测报 CLOBBERED)。
  本地回归: 修复路径 95 GFLOP/s vs clobber 模拟 25 GFLOP/s
  (8 线程 3.8×)。
- **甜点 24 ≠ 旧默认 64**: >24 线程吞吐崩塌(块调度粒度 query_bs=4096
  → 256K 行只有 64 个工作块 + NUMA),追机制不值——直接钉。
  库默认已改为 `min(cpu, 24)`(`_cluster_thread_count`),所有入口
  (run 脚本/直接跑 prune_report/discovery)统一落在甜点;
  run 脚本仍显式 export 24(双保险);换机器先跑 bench,
  甜点不同则 `CLIMBMIX_CLUSTER_THREADS=N` 覆盖。
- **新旧代码对账**(勘误: 原预测"新代码 ~192s"从未兑现,见首条):
  4316d64 的 480.3s 测量与 112 GFLOP/s 反推成立(机制如上更正);
  a3a075a..7483ee5 的"pinned"路径因 clobber 实际退化为 1 线程
  (~3.3h,从未端到端跑完过);clobber 修复后 24 线程才真正落地
  ——实测终局见下条。
  曾记的"本地嵌套惩罚 1.4-2.5×"测的是 numpy 的 pthread 版 OpenBLAS
  路径,不适用于 faiss 捆绑的 sgemm,已作废。
- **重跑终局与"未达预期"的裁决**(同日, 500K 重跑 + 判别实验):
  clobber 修复实测生效(118s→3.6s/迭代 ≈33×,哨兵静默),但全程
  有效吞吐 ~130-151 GFLOP/s(每 redo 72-88s,全程 ~410-430s),
  未达 bench 的 281。判别实验(fresh 进程 × {bench-env, nopin,
  pin-reassert}, 256K×1000×1024): T=24 得 236/230/190,T=12 得
  197/159/185——**三种模式同窗内无软件差异**(pin 内 pools=
  [openblas:1, openmp:24],与 env 钉扎同终态;nopin@12 离群证明
  单发方差 ~20%),nopin@48=146 复现 >24 崩塌。**剩余差距 = 宿主
  有效核数**: cgroup 配额 192 核/affinity 192 均无限制,但
  busy-cores 饱和于 ~12-13(T=24)且无每核吞吐塌陷——是"只有
  ~12 个 vCPU 在真跑、每个满速",不是线程互相拖慢;vCPU 超卖/
  功耗墙/邻居负载的组合(loadavg ~9: diag 期间疑有 prune_report
  尾段或其他负载同跑)。bench 的 281 测于 ~23 有效核的好窗口,
  该容器有效核在 ~12-23 间波动;4316d64 的 480.3s 同样被这堵墙
  封顶(64 线程从未有过 64 核可用)——两代代码实测都在 400-480s
  量级主要是宿主窗口,不是代码。聚类阶段就此定格: 500K 训练
  ~7 min(与池规模无关,固定 256K 子采样)+ 全量指派(见下条)——
  生产瓶颈是嵌入(实测 ~800 docs/s → 116M 全池 ~40h),聚类在任何
  规模都非瓶颈。
  判窗口方法: 空闲时重跑 cluster_bench(banner 现打印 cgroup
  配额/affinity/loadavg/steal%;若 24 线程回到 ~280 即窗口波动
  实锤,若 steal% 持续非零即宿主超卖实锤)。
- **空闲窗口复测定案**(同日, 服务器实测): 12/24/48 线程 =
  192/**274**/181 GFLOP/s——24 线程复现原始 bench 的 281(方差内),
  >24 崩塌第三次复现,甜点恒为 24;steal 0.0% 排除宿主超卖通道;
  loadavg ~8.9 与低谷期相同却一个 146 一个 274 → 波动源在容器
  可观测范围之外(宿主侧邻居/lxcfs 遮蔽),内部不可判也不可控。
  aarch64 wheel 的 pin probe 同样报 CLOBBERED(set 24→inside 1)
  ——确认服务器 wheel 与 x86 同病,生产 re-assert 修复在该机
  实打实扛住 clobber。**定格: 本机 CPU 聚类吞吐 130-280 GFLOP/s
  随宿主窗口波动,500K 训练 3.2-7 min,聚类在任何规模非瓶颈。**
- **对全量跑的量级**(按重跑终局 ~130-151 GFLOP/s 折算): kmeans
  训练固定采样 256K 点(与池大小无关)≈ 100×3.5-4s+预处理 ≈ **~7
  min**(宿主好窗口下 ~3.5 min);全池(116M docs)指派 = 238 TFLOP
  → **≈15-50 min**(130-280 GFLOP/s 窗口带)。聚类阶段在任何
  规模下都退出瓶颈名单。
- 进程 fan-out 探针(8×24)162 GFLOP/s:单发测量被 spawn+建数据开销
  淹没,稳态不可判;因指派已分钟级,不值得再测或实现。
- **池规模二次勘误(2026-08-31, 20-分片冒烟实测)**: 08-29 那次
  "勘误"本身错了——它把 prune_report 100-分片子集的元数据
  (11,610,081 docs)当成了全池。20 分片全池模式冒烟实测
  2,319,341 docs → **116K docs/分片 × 1000 分片 ≈ 116M docs /
  ~93B tokens**(801 tokens/doc,与池名 100B 相符)——§1/§2/§7
  与 TODO E 的原始 "116M" 一直是对的。连带修正: 单节点 8 卡
  全量嵌入按实测 ~800 docs/s ≈ **40h**(非 4.3h);全池嵌入磁盘
  峰值 ~950 GB(475 GB npz + 475 GB tmp);E 里程碑(弹性多节点,
  40/J h)回到全池嵌入的正路——但**首个生产 run 不被阻塞**:
  抽样发现模式(500K 精标签 + domain 映射全池)在任意池规模可用,
  全量嵌入是一次性缓存投资,单节点 40h(周末量级)或多节点 E
  二选一(见 §9 与 TODO E)。

## 9. 全池 prune_report 跑法与预算（2026-08-31 二次勘误后, 池 = 116M docs）

```bash
python3 scripts/prune_report.py \
    --data-dir /home/ma-user/work/100B_stem_parquet_filtered \
    --sample-shards 0 --sample-size 0
```

- **规模前提(20-分片冒烟实测, 2026-08-31)**: 2,319,341 docs / 20
  分片 → 116K/分片 × 1000 ≈ **116M docs / ~93B tokens**。全量跑
  = 一次性 ~40h 嵌入 + ~950 GB 磁盘峰值,**不是过夜量级**——决策:
  单节点 40h(周末挂机,缓存一次付费)或多节点 E(40/J h),二选一;
  首个生产 run 不等它(抽样模式 500K 精标签 + domain 映射即可)。
- **流式路径冒烟已验证(20 分片, ~232K docs)**: 8 worker 均衡
  (2-3 分片/卡),速率爬坡 ~5 min 后稳定 **780-820 docs/s**
  (500K 采样路径的 750 预测被证实,流式无额外开销);memmap 写入/
  validate/分块聚类链路全通。子集缓存键与全池键不同,零污染,
  测完可删。
- **时间线(全池)**: ~20-40 min 全量 metadata 扫描(一次性,进缓存)
  + **~40h 嵌入**(116M ÷ 800 docs/s;memmap,分片级断点续跑,
  monitor 15s 打 docs/s+ETA) + ~7 min 聚类训练(固定 256K 子采样)
  + 15-50 min 全量指派 + 分钟级 profile。
- **磁盘**: 峰值 ~950 GB(475 GB embedding_cache.npz + 475 GB 临时
  memmap,进程退出释放);常驻 475 GB(npz)+ ~100 MB(kmeans)。
  **跑前必须 df 确认**;不够则先清空间或改走 E。
- **RAM**: 嵌入/聚类全程 <20 GB(memmap + 分块指派)。缓存命中的
  重跑(调阈值/生产 Step 1)np.load 475 GB 进 RAM——本机 1.5 TB
  可容但分钟级 I/O;npy-memmap 旁车是更优解(列入 TODO,非阻塞)。
- **验收标志**: `[Embed-Stream] Encoded ~116,000,000 docs ...
  ~800 docs/s` → `[Embed] Validate: NaN=0 ...` → `[Cluster] memmap
  input (~116,000,000 x 1024) — chunked prescan/assign` →
  `[Cluster] FAISS K-means ... threads=24, blas=1(pinned+reassert)`
  → `[Cluster] Done in ...` → clusters.csv 全量 1000 行。
- **生产复用**: 缓存键 = 分片清单 + model + truncate-len(sample=0
  无 sample 分量),与 run_climbmix.sh Step 1(EMBEDDING_SAMPLE_SIZE=0)
  完全一致——40h 一次付费,discovery 与生产共用;fingerprint 归档
  只动 OUTPUT_DIR,不碰 pool 缓存。
