# 跨 Run 复用设计（reuse design）

2026-09-10 定稿。目标只有三条：

1. **完成的实验可保存、可追踪、可复查**；
2. **同一大实验支持复用已完成内容、中断继续**（d20 搜索 / d28 臂任一步骤中断后，重跑同一命令从断点继续，不从头开始）；
3. **跨 run 复用已完成的 d20 实验结果**（参数配置确认未变的前提下，新 run 把历史点当作已有观测）。

设计原则：**简单优先**。低概率场景明确"不做"而不是做半吊子机制；机制数量最小化。

---

## 1. 两层三区的存储架构

```
服务器磁盘（1 区）                     内网 OBS（2 区）
┌────────────────────┐         ┌────────────────────────────────────┐
│ result/<run>/       │         │ {obs_prefix}/                       │
│  工作视图 = 缓存     │  ─────▶ │  exps/ + target_arms/  运行时数据平面 │
│  (指纹/.done/日志/   │         │  archive/<run>/        精选归档区    │
│   交互分析产物)      │         └────────────────────────────────────┘
└────────────────────┘
```

| | 服务器磁盘 | OBS |
|---|---|---|
| 角色 | 计算的**伴生视图** | **持久层** |
| 为什么 | run 期间需要文件系统语义（指纹、`.done` 断点、日志追加）；run 后交互分析（grep / cp4_report / python） | 容器没有内网磁盘——master↔worker 全部数据交换本来就经过 OBS；对象存储不绑登录节点生命周期 |
| 承诺等级 | **缓存**：ops 清盘/机器回收即失效，不作为归档承诺 | **归档载体** |

OBS 内部分两区，判据 = 生产者 + 再生成本 + 生命周期不同：

| | `exps/` + `target_arms/`（运行时区） | `archive/<run>/`（归档区） |
|---|---|---|
| 生产者 | worker/executor 运行时自动写 | run 完成时写一次 |
| 内容 | 重资产（模型权重、混合数据）+ 过程产物 | 精选科学记录（KB–MB：search_state、eval CSV、最优权重、pipeline_summary） |
| 再生成本 | 臂权重高 / 混合数据中 / d20 proxy 低 | **极高**——N 个配比的实测分数不可再生（随机性 + 平台状态） |
| 生命周期 | 分级保留（见 §6） | 永不 GC |

现状缺口：OBS 上有全部重资产、却没有最关键的轻资产（search_state/eval CSV 是 master 产物，从不离开磁盘）。`archive/` 区补这个洞。归档动作目前手动（`obsutil cp`），自动化待做（见 §7）。

**OBS 前缀必须按 run 隔离**：`REMOTE_OBS_PREFIX` 末尾必须带 run 名（如 `…/prod/climbmix/prod3`）。原因：exp id 每 run 从 0 重编，跨 run 共用前缀时——搜索 exp 混合数据是先删后传（remote_executor `_run_remote_experiment`），旧 run 的 OBS 记录会被摧毁；臂数据用 `upload_dir_if_missing`（存在即跳过），新 run 可能直接用旧 run 的混合数据训练（静默正确性事故）。preflight 校验待加（§7；`run_search_arms.sh` 入口已强制）。

---

## 2. 配置两层模型与复用凭证

一个 d20 实验的本质 = **一个混合配比 → 在确定的训练配置下训练 → 在确定的评估协议下测量**。复用 = 测量结果对新上下文仍有效。配置分两层：

| 层 | 内容 | 变了能否复用 |
|---|---|---|
| **不可变层**（决定"这个测量是什么"） | 混合权重；d20 训练配置（模型规模/步数/超参/种子）；eval 协议（bench 集、`--eval-max-per-task`、评估代码语义）；**候选池定义**（cluster_cache 内容） | ❌ 测量失去意义 |
| **可变层**（决定"怎么用这个测量"） | 评分公式（w/score）；采样器/搜索算法；分析代码版本 | ✅ 分数可重算（rescore）、采样器可重 fit（warm-start） |

**复用凭证 = 不可变层一致**。操作层面的校验手段：

- **K（池维度）**：注入工具硬校验 + `_load_state` 已有的 n_clusters 守卫（不匹配 → 丢弃状态重新开始）。
- **池内容**：新 run 的 `cluster_cache.npz` **必须从旧 run 复制，不重新生成**（cluster 重生成有随机性，参数相同池子也可能不同）。注入工具对 `--pool` 指定的 cache 计算 sha256 记入种子溯源块。
- **训练/eval 配置**：由新 run 自己的指纹机制守住（改了配置 = 新指纹 = 全新实验，注入自然失效）。跨配置引用历史点属于操作员责任，发射谱（§8）里有检查清单。

**数据模型：raw 与 derived 分离**。`search_state.json` 里的 `accumulated_per_benchmark`（每任务原始 acc/NLL）是**不可变事实**；`accumulated_scores` 是 raw × 评分公式的**派生物**，可随公式版本重算（这正是 rescore 存在的理由；prod2 的 w 单位 bug 教训）。

---

## 3. 场景枚举与裁决

| # | 场景 | 频率 | 复用什么 | 裁决 |
|---|---|---|---|---|
| 5 | 扩点 warm-start（prod3 在 prod2 的 30 点上采新点） | 高 | 历史点 raw + 种子观测 | **已落地**：`inject_history.py`（§4.2） |
| 1 | 换/补基准集评分（如 mmlu NLL 上游修复后补测） | 高 | d20 checkpoint（OBS 已有） | **待实现**：re-eval 入口（§4.3） |
| 3 | 最优配比 + 改参数重训目标模型（训练量/模型大小/超参） | 高 | 配比 + 臂混合数据 | **已落地**：custom 臂（§4.4） |
| 4 | 新增对比基线（固定比例等） | 中 | 同 3 | **已落地**：同 §4.4；CP4 成对比较可用 |
| 2 | 评分公式变更（修 bug 或改设计） | 中低 | raw | **已落地**：`rescore_search.py`（§4.1）——成本≈0，两类变更统一覆盖，不值得按频率简化 |
| 7 | 纯事后分析/离线仿真（不训练） | 中 | raw | **零**：归档 + 脚本已支持 |
| 6 | 中断恢复/失败重试 | — | 全部 | **零**：已有（exp 级 `.done`/指纹、迭代级 search_state、`.mid_train_ok`、`--retry-failed`） |
| 8 | 锚点/base 分数复用 | 低 | 一个数 | **手动**：docs 记录 + 引用 |
| 9 | 候选池/K/数据源变化 | 低 | — | **明确不做**：权重相对于池定义，池变了测量失去意义；老点映射新池是复杂度陷阱。指纹不匹配即全新开始 |
| 10 | d20 代理训练配置变化（规模/步数/超参） | 低 | — | **明确不做**：测量仪器变了，读数不可比；当独立实验跑 |

净增量 = 2 个新机制（注入、re-eval）+ 1 个便宜入口（rescore）+ 1 个参数化（custom 臂）+ 一份"不做"清单。

**复用粒度 = 点，不是 run**：注入按点 cherry-pick（指纹匹配即可用），可合并多个源 run（K 一致 + 权重去重）。

**复用点的轮次放置**：全部作为**初始观测池**（种子块计入 `realized_configs_per_iter[0]`，相当于一个超大的 iter1），不恢复原轮次。理由：轮次反映旧 run 的采样路径（iter3 的点是在"只有 26 个观测"条件下采的），对新采样器无约束力；采样器 fit 只需要全部观测，不关心来源轮次。配套：guided 采样已有对 accumulated configs 的 4 位小数去重（`run_iteration` 的 `existing_flats`），新点自动避开历史配比；exp id 分配 = `len(accumulated_configs)`（`run_batch(experiment_id_base)`），新 exp 自动从历史点数之后编号，无碰撞。

---

## 4. 落地设计

### 4.1 rescore（`scripts/rescore_search.py`）

```
python3 scripts/rescore_search.py --state <search_state.json> [--top 10] [--w-floor 0.05]
```

- 读 state 的 raw（`accumulated_per_benchmark`），用**当前代码**的 `_compute_scores` 重算（bench 列表默认取 raw 键并集，可 `--benchmarks` 覆盖）。
- 打印：每任务 w/f 表、新旧分数差、新公式下的 top-N。
- 写 sidecar `<state>.rescored.json`（公式版本 = 评分代码的 git SHA、新分数、排名、w/f 表）。**绝不改动原 state**。
- 用途：评分公式变更后的影响评估（修 bug / 改设计同一入口）；注入前的预检（注入工具内部走同一条重算路径）。

### 4.2 warm-start 注入（`scripts/inject_history.py` + bootstrapper `history_seed`）

```
python3 scripts/inject_history.py \
  --source <旧run>/search_state.json        # 可重复（合并多个源 run） \
  --target-dir result/prod3_current          # 新 run 输出目录（会被创建） \
  [--pool <cluster_cache.npz>]               # 新旧 run 共用的池缓存（复制而来） \
  [--dry-run]
```

行为：

1. **校验**：源 state 的 `n_clusters` 必须自洽（= 权重维度）；若给 `--pool`，从 `final_labels` 数出 K 并要求与源一致；目标 `search_state.json` 已存在则拒绝（除非 `--force`）。
2. **选点**：丢弃无任何有限测量的点（失败实验，零信息）；按 4 位小数权重去重；每点保留 raw acc/NLL。
3. **重算**：当前公式重算分数（打印 w/f 表 + top-N，即内置 rescore）。
4. **写种子**：`<target-dir>/search_state.json`：
   - `last_completed_iter = 1`、`realized_configs_per_iter = [N_history]`、`pending = None`；
   - `n_clusters = K`、`accumulated_configs / per_benchmark / scores` = 历史点；
   - **丢弃**源 run 的 `predictor_eval / online_eval / pruning_history / last_c_eff`（旧 run 的诊断记录，轮次编号与新上下文冲突，只会误导报告；溯源进种子块）；
   - `history_seed` 溯源块：`{source_runs, n_points, pool_sha256, scoring_commit, injected_at}`。
5. 新 run 发射时 shell 无条件传 `--resume-search`（run_climbmix.sh Step 3），bootstrapper `_load_state` 读到种子 → `_refit_predictor` 在历史点上重拟合 → **iter2 起直接 guided 采样**，exp id 从 N+ 开始，`_reconstruct_iteration_results` 把历史点归位为 iter1（后续崩溃恢复的归因也正确）。

bootstrapper 改动（最小）：`_save_state`/`_load_state` 增加 `history_seed` 字段往返；`_load_state` 看到种子时大声打印来源。种子在后续每次 `_save_state` 中原样保留。

**发射预算语义**：`CONFIGS_PER_ITER` 的第 1 槽位 = 历史点数。prod3 想要"30 历史 + 60 新点" → `CONFIGS_PER_ITER="30,20,10"`（iter1 由种子占据，iter2/3 各采 20/10 新点）。发射线即真实总预算，无隐藏账。

### 4.3 re-eval（换/补基准集，`scripts/re_eval_history.py`，待实现）

对已完成 run 的 d20 点用新 bench 协议补测，**不重训**：

```
python3 scripts/re_eval_history.py --run-dir <旧run> --remote-config remote_config.json \
  --benchmarks stem_v2 --exp 0-29 [--max-per-task -1]
```

- 对每个 exp：从旧 run 的 OBS `exps/exp_XXXX/result/mid_checkpoint` 读 checkpoint（只读，不动旧前缀），构造 `eval_only=True` 的 spec（eval argv 换新 bench 协议），结果写到**新子前缀** `exps/exp_XXXX/reval_<benchtag>/`（不覆盖旧 result）。
- 产出：每点的增量 per-task 表 + 合并后的扩展 state（raw 的新 bench 列并入，**不覆盖旧列**——不同协议的测量分列存放）。
- 前置：mmlu_stem NLL 上游修复后启用（30 个 checkpoint 都在 OBS 上）。
- 与注入的组合：先 re-eval 得到扩展 raw → 注入时 bench 并集进新 run 的 `val_tasks`。

### 4.4 custom 臂（任意混合 + 训练参数覆盖，已落地）

场景 3（赢家重训，改训练量/模型大小/超参）与场景 4（固定比例基线）共用机制，**三件已有/新增部件**：

**① 选点：`prepare_random_baseline.py --weights`（新增）**——把均匀 1/K 泛化为任意固定配比：

```bash
python3 scripts/prepare_random_baseline.py \
  --data-dir $DATA_DIR --output-dir result/<run>/fixratio_shards \
  --cluster-cache result/<run>/cluster_cache.npz \
  --schema config/schema_stem.yaml --target-tokens 2B --num-npu 8 \
  --weights "0.2,0.2,0.3,..."        # 或 optimal_mixture_weights.json（赢家重训）
```

接受逗号列表 / JSON 数组文件 / `optimal_mixture_weights.json`（`{"C0": w, ...}` dict，按 C 序号排序）；非负、和 > 0，自动归一；维度必须等于池的 K。shortfall 政策与 CLIMB/random 臂完全一致（配额不足的簇取全部、不复制不重分配，`--target-tokens` 上限同 CLIMB 臂）。不带 `--weights` = 原均匀行为，逐字节不变。

**② 混合：`mix_general_data.py`（已有）**——与主脚本 Step 5 相同：

```bash
NANOCHAT_REPO=$NANOCHAT_DIR python3 scripts/mix_general_data.py \
  --stem-dir result/<run>/fixratio_shards --output-dir result/<run>/fixratio_mixed \
  --climbmix-dir $GENERAL_DATA_DIR --stem-ratio 0.7 --num-workers 8 --num-npu 8
```

**③ 发射：`dispatch_target_arm.py --arm <name>`（开放任意名）**——`--arm` 不再限死三选一，任何 `[A-Za-z0-9_-]+` 的名字走通用臂路径（锁/`.done`/audit/tag/OBS `target_arms/<name>/` 全部按名字泛化；random 的选点等待和 base_eval_check 的单节点约束只对这两个名字生效）：

```bash
# 固定比例基线臂
python3 scripts/dispatch_target_arm.py --arm fixratio_v1 \
  --data-dir result/<run>/fixratio_mixed

# 赢家重训：换训练参数 —— CLI 环境变量优先于 launch_env.json（load_launch_env 既有语义）
TARGET_STEPS=3000 MID_DEVICE_BATCH_SIZE=1 python3 scripts/dispatch_target_arm.py \
  --arm winner_v2 --data-dir result/<run>/winner_mixed
```

**复用语义**：臂混合数据走 `upload_dir_if_missing`（同配比 + 同池 → 同 OBS 地址 → 存在即跳过 = 复用混合数据，省重新混合）；这正是把跨 run 前缀碰撞的"地雷"反转成复用特性的前提——前提是 §1 的 per-run 前缀规则先落地。

**报告**：`cp4_report.py` 已有 `--arm-a/--arm-b`，custom 臂今天就能做成对比较（如 `--arm-a fixratio_v1 --arm-b climb`）；单报告 N 臂全景泛化进待办。

---

## 5. 明确不做清单

1. **池/K/数据源变化后的历史点映射**（场景 9）——复杂度陷阱，指纹不匹配即全新开始。
2. **d20 训练配置变化后的跨配置复用**（场景 10）——仪器变了读数不可比。
3. **按频率区分"修 bug / 改设计"的评分变更处理**——rescore 一个入口统一覆盖。
4. **历史点的原轮次恢复**——轮次是旧采样路径的产物，对新 run 无意义。
5. **OBS exp 存储的内容寻址全局重构**——注入走 search_state（观测层面），不需要 OBS 层的大改；per-run 前缀规则已消除写冲突。
6. **归档自动化**（mark_completed 尾部自动上传 archive/ + MANIFEST）——待做但不阻塞复用主线，见 §7。

---

## 6. 保留分级

| 资产 | 位置 | 保留 |
|---|---|---|
| 臂权重（d28 mid_checkpoint） | OBS `target_arms/` | 永久 |
| d20 checkpoint | OBS `exps/exp_*/mid_checkpoint` | 永久（re-eval 依赖） |
| exp 结果（result.json/CSV/日志） | OBS + 服务器 run 目录 | 永久 |
| 搜索混合数据（每点 ~几十 GB） | OBS `exps/exp_*/mixture_data` | 可清理（可由权重 + 池重建） |
| 臂混合数据 | OBS `target_arms/*/mixture_data` | 长期（重训复用） |
| run 目录分析产物 | 服务器 `result/<run>/` | 磁盘=缓存；精选产物入 OBS `archive/` |

---

## 7. 待办（不阻塞复用主线）

- ~~preflight 校验 `REMOTE_OBS_PREFIX` 末尾含 run 名~~（已由 `run_search_arms.sh` 启动校验覆盖；preflight 层——直接跑 `run_climbmix.sh` 的路径——仍待加）；
- `mark_completed` 尾部自动上传精选产物 → `{prefix}/archive/<run>/` + `MANIFEST.json`；
- re-eval 工具实现（§4.3，等 mmlu 上游）；
- `cp4_report.py` 单报告 N 臂全景（今天用 `--arm-a/--arm-b` 成对比较）；
- `_save_state` 增补 `pool_cache_sha256`（当前由注入工具写入 `history_seed` 承担）。

---

## 8. prod3 发射谱（复用 prod2 的 30 点）

**一键方式**（nanochat 式状态驱动，一个命令覆盖三态）：

```bash
# run_search_arms.sh 按目标目录状态自动分派:
#   无 search_state.json + 无 HISTORY_RUN  → 从零
#   无 search_state.json + HISTORY_RUN=<旧run> → 热启动 (校验+继承池+注入)
#   已有 search_state.json (种子或真实进度) → 续跑 (重跑同命令, 不重注入)
HISTORY_RUN=result/prod2_k15bal_20260909_200323 EXP_NAME=prod3 \
  ./runs/run_search_arms.sh                 # 编辑文件顶部 EDIT 块后直接跑
LAUNCH=0 ./runs/run_search_arms.sh          # 干跑: 校验+注入+打印发射线
```

壳内自动完成（即下面的手动谱）：源 run 完整性检查 → **K 一致性预检**（balanced_profile 的 K_final vs K_ENHANCED，不一致直接拒绝）→ 复制池缓存（继承，不重新聚类）→ inject_history（池 K 硬校验 + 当前公式重算 + 溯源）→ **槽位核对**（CONFIGS_PER_ITER 第 1 槽 vs 历史点数，不符大声警告）→ 调 run_climbmix.sh（可选并行预发 random 臂）。

**阶段脚本全家**（quadmix 式 `run_<阶段>`：从该阶段开始跑；`LAUNCH=0` 干跑）：

| sh | 从哪开始 / 覆盖场景 |
|---|---|
| `runs/run_search_arms.sh` | 主实验（d20 搜索+两臂）：从零 / 中断续跑（重跑同命令）/ 热启动（`HISTORY_RUN`，场景 5、6） |
| `runs/run_arm_only.sh` | 臂阶段：自定义配比（`WEIGHTS`）/ 赢家重训（`WEIGHTS` + 训练参数覆盖）/ 已有臂重发（`WEIGHTS` 空 + `--retry-failed`，场景 3、4） |
| `runs/run_eval_only.sh` | 评测阶段：base 锚点补发（场景 8）；（待 mmlu 上游）d20 re-eval 挂这（场景 1，§4.3） |
| `runs/run_report_only.sh` | 报告阶段：分数重算 sidecar（场景 2）+ 任意两臂 CP4 对比（含自定义臂） |

手动谱（等价于壳内动作，留作参考）：

```bash
# 0) 服务器拉最新代码
cd ~/work/climbmix && git pull

# 1) 建 prod3 输出目录，复制池缓存（不重新生成！），注入历史
mkdir -p result/prod3_current
cp result/prod2_k15bal_20260909_200323/cluster_cache.npz result/prod3_current/
cp result/prod2_k15bal_20260909_200323/balanced_profile.json result/prod3_current/
python3 scripts/inject_history.py \
  --source result/prod2_k15bal_20260909_200323/search_state.json \
  --target-dir result/prod3_current \
  --pool result/prod3_current/cluster_cache.npz

# 2) 检查清单（不可变层，逐项人工确认与 prod2 一致）
#    - K_ENHANCED=15、MERGE_STRATEGY=balanced、FILTER_METHOD/PRUNE_THRESHOLD/MERGE_DISTANCE 同 prod2
#    - PROXY_*（深度/步数/tokens/lr/warmup/warmdown）、EVAL_BENCHMARKS=stem、EVAL_MAX_PER_TASK=-1
#    - 候选池 = 复制的 cluster_cache.npz（不是重新聚类）
#    - 代码：评分公式修复已含（e8f22f2 之后）

# 3) 发射（CONFIGS_PER_ITER 第 1 槽 = 30 历史；总计 30+60=90 点预算）
EXP_NAME=prod3 K_ENHANCED=15 NPU_PER_EXP=8 REMOTE_MAX_JOBS=10 \
CONFIGS_PER_ITER="30,20,10" ADAPTIVE_CONFIGS=1 ADAPTIVE_COMPACT=1 REMOTE_LOCAL_PARALLEL=1 \
REMOTE_OBS_PREFIX=obs://<bucket>/<user>/climbmix/prod3 \
nohup bash runs/run_climbmix.sh &

# 4) 臂（照旧，可选多节点）
nohup python3 scripts/dispatch_target_arm.py --arm random &
```

注意：`ADAPTIVE_COMPACT=1` 进指纹（新 run 目录，无冲突）；adaptive 的 iter2/3 波预算按新 e_i（20/10）计算，不受历史影响。
