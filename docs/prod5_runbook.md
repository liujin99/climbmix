# prod5 发射 runbook（2026-09-21 打磨窗定稿）

> **整体作业流程设计图（拓扑 / 生命周期 / 数据分层 / Stage 1 缓存瀑布 /
> 搜索 / 臂族）见 `docs/architecture.md`**——本文是操作手册，那张是设计图。

> **顺序**：D1 激活 → smoke 彩排 → prod5 发射 → 发射窗卫生。
> （D2 prod4 八臂补测**缓做**——2026-09-21 裁决：先做 prod5 单轮内基线横评；
> §3 命令备好，需要严格同代跨轮比较时触发。）
> 每步带验证点，命令按序贴服务器；执行窗口 = 用户发令后（排期裁决 2026-09-21：
> 打磨收官、版本定案后再发射）。
> 判读规则与发射配置 = `docs/experiment_prod5.md`（预注册，发射前已定稿）。
> 注意顺序依赖：smoke 与 prod5 搜索的本地槽（REMOTE_LOCAL_PARALLEL）抢同一批
> 本地 8 NPU——串行执行。（D2 若触发，同样抢本地 NPU，排在非关键路径。）

## 0. 前置状态

- 服务器无残留进程（`pgrep -f "run_experiment.sh\|dispatch_target_arm"` 为空）；
  完整自检见 4.3 的 preflight_launch。
- `result/prod4_current` 已收官：主树 pull 后其指纹过期属预期，**无需任何动作**。

## 1. D1 激活（~30 min，无 NPU）

**1.1 climbmix 主树**（冻结基线 = 拉取后的 main HEAD）：

```
cd /home/ma-user/work/climbmix
git status --short
git pull origin main
git log --oneline -3
```

`git status` 须干净；记录 HEAD SHA（prod5 代码基线，事后入档）。

**1.2 nanochat-npu 主树**（预期 = `6e5baa2`，D17 协议 + D18 b16 尾）：

```
cd /home/ma-user/work/nanochat-npu
git status --short
git fetch origin
git checkout dev-data-mix
git pull origin dev-data-mix
git log --oneline -3
grep -n "gen-batch-size" scripts/base_eval.py | head -2
```

最后一行验证新协议在位：`default=16`（b16 era 单默认，D18）。

**1.3 退役 prod4 冻结窗 worktree**（使命完成）：

```
git -C /home/ma-user/work/climbmix_lb status --short
git -C /home/ma-user/work/climbmix worktree remove --force /home/ma-user/work/climbmix_lb
git -C /home/ma-user/work/nanochat_lb status --short
git -C /home/ma-user/work/nanochat-npu worktree remove --force /home/ma-user/work/nanochat_lb
```

（实际路径 = `/home/ma-user/work/{climbmix_lb,nanochat_lb}`，**无 `@SHA` 后缀**——
2026-09-22 激活实测；status 应为空输出，有输出先停下来人工核。）

**1.4 同步 nanochat 代码 → prod5 前缀**（远端 worker 的 eval 代码 = 新栈）：
prod5 用**新 OBS 前缀**（prod4 remote_config 的 `obs_prefix` 末段 `prod4` → `prod5`），
新前缀下无任何资产——必须先同步新代码 tarball，否则首个远端作业裸死或回退旧协议
（毒化 b16 era 前提）。上传走 **assets/ 新鲜通道**（boot shell 优先于 assets_big
一次性包）。三步（2026-09-22 激活实测补全；所有命令带 PYTHONPATH 前缀——vendored
后端 `climbmix-ma/` 在仓库目录内但不在 git 跟踪内，dispatch/sync 脚本的 bare-shell
自举只在首个 climbmix import 失败时才补路径，shell 已可导入 src 时会被跳过）：

```
# (a) 诊断：prod4 前缀 assets/ 里有什么（重点看有没有 .whl 平台轮子）
cd /home/ma-user/work/climbmix
PYTHONPATH=/home/ma-user/work/climbmix/src:/home/ma-user/work/climbmix/climbmix-ma python3 -c "from climbmix.remote.remote_executor import RemoteConfig; from climbmix.remote.backends import resolve_backend; rc=RemoteConfig.from_json_file('result/prod4_current/remote_config.json'); obs=resolve_backend(rc).make_obs_storage(rc); print('\n'.join(sorted(obs.list_objects(rc.obs_prefix.rstrip('/')+'/assets/'))))"

# (b) 派生 prod5 的 remote-config（后端身份照抄 prod4，仅换前缀；输出 = prod5 前缀，记下来）
mkdir -p /home/ma-user/work/tmp
python3 -c "import json; rc=json.load(open('result/prod4_current/remote_config.json')); rc['obs_prefix']=rc['obs_prefix'].replace('/prod4','/prod5'); json.dump(rc, open('/home/ma-user/work/tmp/remote_config.prod5.json','w'), indent=2); print(rc['obs_prefix'])"

# (c) 同步：pull + tar + 上传到 prod5 前缀
PYTHONPATH=/home/ma-user/work/climbmix/src:/home/ma-user/work/climbmix/climbmix-ma python3 climbmix-ma/scripts/ma_sync_code.py --remote-config /home/ma-user/work/tmp/remote_config.prod5.json --repo /home/ma-user/work/nanochat-npu
```

（(a) 若见 `.whl` → (c) 加 `--wheel <本地轮子路径>`（aarch64 cp311 容器离线装依赖用）；
(c) 输出须为 `nanochat-npu @ 6e5baa2 (clean)` + **tar sha256 记档**（= prod5 worker
tar 基线，接替 prod4 的 a7c56792 时代注记）。）

**1.5 验证（prod5 前缀）**：

```
cd /home/ma-user/work/climbmix
PYTHONPATH=/home/ma-user/work/climbmix/src:/home/ma-user/work/climbmix/climbmix-ma python3 scripts/dispatch_remote.py --remote-config result/prod4_current/remote_config.json --check-assets --obs-prefix <1.4(b) 打印的 prod5 前缀>
```

发射前的**预期形态**：direct mounts 全 OK + **`nanochat-npu code (assets/ (fresh))` OK**
+ 两个 worker 文件 MISSING（**预期**——RemoteExecutor 首次发射自动上传）→ 末行报
"2 missing" 属正常；唯一硬要求 = nanochat code 行为 OK。

## 2. smoke 彩排（~2.5h——12 实验 × ~52 min 训练 + eval 两轮串行，8 NPU 全本地零排队；2026-09-22 实测校准，看门狗默认 4h）

```
cd /home/ma-user/work/climbmix
bash scripts/diagnostics/smoke_round.sh
```

自动完成：聚类缓存继承（源 `result/prod4_current`）→ 12 实验（8+4）× 50M
tokens × 100 题/任务 × `NPU_PER_EXP=1`（8 卡 8 路，2 波）→ `REMOTE_ENABLED=0`
零远端提交 → Step 1-3 完成即停 → 16 项验证清单 → **重件自动清理**（~30G 的
d20 ckpt/mixture/parquet；`SMOKE_KEEP=1` 可保留；验证报告留存
`result/smoke5_verification.txt`）。

- 通过判据：清单全绿（终选 mode 落盘、topk JSON 与 state 对账、report 渲染、
  A2 重拟合触发、权重和、Step 1-3 完成门）。
- 失败：产物自动保留，`result/smoke5_engine.log` 尾部有引擎现场；修复后**同命令
  重跑 = 续跑**（search_state 在，已完成实验不重训）。
- 预期注记：tiny 预算下可能走 no-signal 守卫路径而非 claim 路径——两条都是真
  路径，验证目标是管线跑通；claim/A2 正常路径另有单测 + 审计重放覆盖。

## 3. D2 prod4 八臂补测（**缓做·触发式**；~4h，本地 8 NPU，eval-only；b16 era 基线端）

**2026-09-21 裁决缓做**：先做 prod5 单轮内基线横评（本轮判决主战场）；本节
命令保留备用，触发条件 = 需要严格同代跨轮比较（experiment_prod5.md V2/V3
定版）时执行。

目的：跨轮趋势链的同代基准端（prod5 全程新协议，prod4 八臂需同代分数才可比）。
**新 CSV 落独立目录，不覆盖旧件**。

```
cd /home/ma-user/work/nanochat-npu
mkdir -p /home/ma-user/work/climbmix/result/prod4_current/reeval_d18
python3 - <<'EOF'
import glob, os, shutil, subprocess, time

ARMS = ["climb", "climb_rep", "random3b", "random3b_rep",
        "natural", "domainfix", "cfg25", "cfg72"]
base_eval_dir = os.path.expanduser("~/work/nanochat_model_dir/base_eval")
out_dir = "/home/ma-user/work/climbmix/result/prod4_current/reeval_d18"

def snap():
    return {f: os.path.getmtime(f)
            for f in glob.glob(os.path.join(base_eval_dir, "mid_model_*.csv"))}

for arm in ARMS:
    tag = f"d28_{arm}_prod4"
    print(f"=== {tag} ===", flush=True)
    before = snap()
    t0 = time.time()
    cmd = ["torchrun", "--standalone", "--nproc_per_node=8",
           "-m", "scripts.base_eval", "--",
           "--eval=core", "--eval-benchmarks=stem", "--max-per-task=-1",
           "--device-batch-size=16", "--core-eval-batch-size=8",
           f"--model-tag={tag}", "--model-type=mid"]
    rc = subprocess.call(cmd)
    after = snap()
    new = [f for f, m in after.items() if before.get(f, -1) < t0]
    if rc != 0 or not new:
        print(f"  FAILED rc={rc} new_csvs={len(new)}", flush=True)
        continue
    dst = os.path.join(out_dir, f"{arm}.csv")
    shutil.copy2(sorted(new)[-1], dst)
    print(f"  rc=0 in {time.time()-t0:.0f}s -> {dst}", flush=True)
print("done — 把 8 个 CSV 的 stem 行贴回")
EOF
```

预期（D17/D18 A/B 实证外推）：NLL 列与旧 CSV **逐位一致**；gsm8k ±1 题（b16
近平局翻面）；math_cot 在 climb 型臂 +30%±、random 型 ≈0（能力地板）。结果贴回
后入 experiment_prod5.md V3 基准端。

## 4. prod5 发射（搜索 ~10h @ 10 并发 + 臂族）

**入口 = `runs/run_experiment.sh` 本身**（2026-09-22 用户裁决：大实验以
EXP_NAME 区分、直接跑引擎，不设每轮 launcher 包装——曾建的
launch_prod5.sh 已删；其机器对照价值降级为只读工具
`scripts/check_launch_parity.py`，见 4.4）。

引擎默认值定稿（2026-09-22 用户裁决系列）：CONFIGS_PER_ITER=64,32,16
（论文 §2.2 值）/ PROXY_TARGET_TOKENS=640M（原生默认，**prod5 按默认跑**
——E1 修订，prod4 实跑 400M）/ TARGET_TOKENS=3B / TARGET_ARM_NODES=8 /
REMOTE_MAX_PREP=**耦合 `$REMOTE_MAX_SEARCH_NODES`**（⑬q A：动态提交语义下
同时在飞备料数 ≤ 在飞槽位 → 相等恰好永不短板且零浪费；显式 env 可覆盖）/
REMOTE_JOB_TIMEOUT_H=**0（不限制**——隐藏运行时天花板
是雷，楔死交监控检测；DISPATCH_RANDOM_ARM 旋钮已删）。**本轮显式 5 个键**
（4 个环境身份键故意保持本地安全默认——裸跑引擎不会误发真集群；
REMOTE_OBS_PREFIX 留空自动拼 `{obs_prod_base}/prod5`；REMOTE_MAX_SEARCH_NODES
默认 10 = 搜索阶段节点上限，留卡给同池租户——作业恒 1 节点 → 节点数=作业数；
更名自 REMOTE_MAX_JOBS，旧名被发射最早期 guard fail-loud 拦截）：

```
cd /home/ma-user/work/climbmix
EXP_NAME=prod5 \
REMOTE_ENABLED=1 REMOTE_BACKEND=modelarts \
REMOTE_BACKEND_MODULE=climbmix_ma:create_backend \
REMOTE_FLAVOR=modelarts.pool.visual.8xlarge \
bash runs/run_experiment.sh             # 干跑门（LAUNCH 默认 1, 加 LAUNCH=0 只看计划）
```

真发射 = 同一命令行（引擎默认 LAUNCH=1 即真发；后台化按需 nohup/setsid）。

 5 键之外全部吃默认，值得知道的六个：
- `CONFIGS_PER_ITER=64,32,16`（**默认 = 论文值**（§2.2 共 112 点）——
  忘设得到的是推荐计划；smoke 8,4 / 缩水几何属显式偏离）
- `PROXY_TARGET_TOKENS=640M`（**默认，prod5 按默认跑**——E1 修订：
  prod4 实跑 400M/381 步, 本轮 610 步, d20 分数跨轮不同预算不可比）
- `TARGET_TOKENS=3B`（默认。与 prod4 引擎实录 6B 的差异: prod4 终选产物
  6B 口径 12.3GB, 本轮 3B 口径 —— V1-V5 全部臂级比较不受影响（臂本来
  都是 3B）。臂派发因 launch_env 实录 3B/2861 而无需任何 env 覆盖）
- `TARGET_ARM_NODES=8`（默认 = prod4 臂形态 8 节点 ws=64；多节点 ws≠8 →
  派生 --load-optimizer=0 冷启 = prod4 2a 表同形。`REMOTE_JOB_TIMEOUT_H=0`
  = 搜索作业无运行时上限（楔死交监控——log 流 30s 上传停滞可见；
  opt-in 正值 = 防楔死天花板）；臂作业超时独立: dispatch `--job-timeout-h`
  缺省 **0 = 不限制**（同裁决哲学; 2026-09-28 对齐——原注释写"缺省 9h/13h"
  但 argparse default=0.0 使该分支不可达, 实跑一直是不限制）, 正值 = 天花板
  (sizing: ws=64 3B 臂 ≈ 3-4h 训练 + eval + 上传)）
- `REMOTE_QUEUE_TIMEOUT_H=0`（**默认 = 排队不设限**,2026-09-28 裁决——
  假期拥堵实测烧满 72h 耐心无卡,排队是平台的事,自设上限只会自伤(见
  §4.7);opt-in 正值 = 排队上限,`REMOTE_QUEUE_RETRY` 仅在正值下生效。
  prod5 当次续跑显式用了 8760(≈一年)作零代码保险丝,⑬t pull 后即默认 0）
- `FLEET_ARMS=climb,topk,uniform,natural,domainfix,base`（⑬t 全自动
  臂族,Step 4 一次调用发射全部验证臂,见 §4.6;空串 = 跳过臂族段）

要点：
- **池嵌入的两入口设计（2026-09-23 用户裁决：供给/消费分离 + 40h 陷阱
  封死）**：① `runs/preprocess_pool.sh` = **供给入口**（全幂等：缓存已热
  秒级退出；冷则磁盘守卫 → 从 OBS 耐久层 `{root}/embed_units/`（waves
  产出 ~475GB，设计上永不清删）merge 到本地 `cache/embeddings/<key>/`
  分片缓存，~1-2h → 探针行数对账；`DRY_RUN=1` 预览）；②
  `run_experiment.sh` = **消费入口**（Stage 1 读 `<key>/`；全池 miss →
  **fail-loud 指路 preprocess**——内联嵌入守卫
  `embedding_cluster._INLINE_EMBED_MAX_DOCS=2M`，`EMBED_INLINE_FULL_POOL=1`
  知情越权；小池/采样/waves worker 不受影响）。历史：09-22 smoke 孤儿
  归档事故、09-23 prod5 两次踩中 40h fallback 后封死
- **发射姿态（2026-09-23 用户裁决：未备好新代码不发射）**：prod5 的使命 =
  新代码从头到尾全新验证 → **标准路径 = 池级缓存复热后无种子发射**。
  供给 = `bash runs/preprocess_pool.sh`（~475GB，~1-2h，幂等）→
  之后 Stage 1 走新代码完整 discovery：分片缓存直读（discovery.py:38，
  manifest 优先级高于 npz）→ kmeans 确定性重跑（产 kmeans_K*.npz）→
  merge 段 → 本轮自产 cluster_cache.npz。**种子拷贝 = 池缓存冷时的
  fallback 快路径**（cluster cache 命中整个绕过 Stage 1——会给"全新验证"
  声明带星号：新代码 discovery 路径没跑到）；stage-gate 的 cache-seed
  豁免保种子存活。当日教训（⑬k 根因）：池缓存本地层在 09-04 合并后被
  磁盘清理清掉、恢复动作未补做、种子谱系掩盖
- **加载时快速校验（2026-09-23，⑬o）**：merge 发布 manifest 时逐块捎带
  sha256+bytes（验证同趟，零额外读）→ 引擎加载分片缓存默认走
  **stat+样本快路径**（~1min，代替曾白烧 ~55min 的全池单线程重扫，且
  旧路径补了 25% 步进进度打印——全量扫描永不静默）。`CLIMBMIX_EMB_VERIFY
  =sha`（并行重哈希，抓同尺寸位翻转）/`=full`（强制旧全扫）；size/sha
  不符 → fail-loud 点名 block 指路重 merge。存量无哈希缓存跑一次
  `python3 scripts/backfill_block_hashes.py cache/embeddings/<key>`
  （纯本地 ~3-5min，幂等）即启用快路径。**本地 443GB = 可再生暂存**
  （用户设计指令：本地盘只放可再生数据，跑完即清；OBS units 耐久层
  40min 重 merge 再生）——⑬r 起**收官清理自动化**：Stage 1 完成
  （①②③ 任何路径）后引擎自动删块文件（`CLIMBMIX_EMB_AUTOCLEAN=0` 关闸），
  **manifest.json 保留**（它是流式行图）。旧 manifest（无流式字段）下
  自动清会跳过并提示——手动配方仅限旧缓存：
  `rm -f cache/embeddings/<key>/block_*.npy`
  （**kmeans_K*.npz 与 stage1_key_*/ 存在 key 目录内**——prod5 实测
  `f8dcb9d7c29b/kmeans_K1000.npz`，整目录 rm 会连带删掉，下轮白付
  39min kmeans；保留 = 重 merge 后 kmeans/stage1 直接命中）
- **OBS 流式直读 + manifest-only 供给（⑬r，上量爆盘的结构性解）**：
  ① preprocess 上量模式 `PREPROCESS_MODE=manifest-only bash
  runs/preprocess_pool.sh` ——只对账 units 并发布 manifest（记录
  units_obs_prefix + remote_config），**零块落盘**；② 引擎 ③a 在块缺失
  时走 `StreamingShardedEmbeddingCache`：unit partials 按需 LRU 暂存
  （默认 2 单元 ≈ 15GB RAM，`CLIMBMIX_EMB_STREAM_LRU`），磁盘占用与池
  大小解耦；每趟全池扫描 = 一遍网络（prescan/训练采样/指派 ~3 趟）。
  `CLIMBMIX_EMB_STREAM=0` 关流式（回退 fail-loud 指路 preprocess）。
  kmeans-first：③ 先查 kmeans npz，命中则嵌入矩阵整个不载——代码变更
  重发 = 纯 merge 段 ~2min，块在不在都无所谓
- **池级 Stage-1 整段缓存（⑬p）**：merge 段产物（大簇标签 +
  cluster_info）内容键控存 `key 目录/stage1_key_<hash>/`（⑬r 改名，
  消除"哈希校验文件"误读；无存量目录，免迁移），键 = 全量
  discovery 配置（K_init/K_enhanced/K_max/prune 阈值/merge 策略…）+
  **全仓 climbmix 源码哈希**——代码漂移自动重键（重算，绝不把旧代码
  算的簇当新代码产物）。优先级 = run 级 macro_labels（种子/resume
  路径，保持最高）> 池级 stage1_key_ > 现算；**仅现算结果晋升池级**（run
  级种子命中不晋升——其簇可能出自旧代码谱系）。旋钮+代码不变时重发 =
  Stage 1 整段秒过（~23min merge 免付）。**⑬r 内容命名**：产物文件改叫
  `macro_labels.npz` + `macro_info.json`（旧名 cluster_cache.npz /
  cluster_info_cache.json 仍可读——旧 run 目录与旧种子免迁移；新写入用
  新名并清掉旧名副本）。
- **重发射注意**：`_current` 已有指纹且其间代码/参数变过 → 引擎归档整个
  目录后空目录重来 → 先 `rm -rf result/prod5_current` 再发

**4.4 发射后 1 分钟对账（机器对照，只读）**：引擎落 launch_env.json +
remote_config.json 后立刻 diff 源轮，预期外的偏离趁早发现：

```
python3 scripts/check_launch_parity.py result/prod4_current result/prod5_current
```

预期 delta = **9 个**：EXP_NAME / OUTPUT_DIR / TARGET_TOKENS 6B→3B /
TARGET_STEPS 5722→2861（launch_env）+ OBS 前缀末段 / job_timeout_s
28800→0 / max_concurrent_jobs 18→本轮值 / max_prep_parallel 6→8 /
asset_mounts 记录性缺失（remote_config）。**盲区须知**：CONFIGS_PER_ITER
与 PROXY_TARGET_TOKENS 不在 launch_env.json 记录清单 → 对账对这两个形状键
零可见——形状核对靠发射横幅日志两行（"轮次计划: 64,32,16" +
"PROXY_TARGET_TOKENS=640M -> 610 steps"）；补记两键排队 post-search 窗口。
清单之外出现偏离 → 按 4.4b 停发重发。

**4.4b 停发重发（对账红灯 / 错形发射时）**：对账超出预期 delta → 趁早
止损——警报挂一小时 = 多烧一小时卡。流程（2026-09-23 prod5 首用：recall
prod4 旧命令行残留 `CONFIGS_PER_ITER=54,36,18` + `PROXY_TARGET_TOKENS=
400M` 两键未删，实跑 108 点 @ 400M/381 步，iter1 跑完才处置）：

```
# 0) 停引擎（孤儿安全 trap；远端在跑作业不受影响、各自跑完上传）
pkill -f run_experiment.sh
#    …等 ≥2h 让在跑作业排空（~1.3h/作业 + 余量）…
# 1) 清 OBS 该轮 exps（守卫：三方同名 + 已收官拒扫 + 只删 exp_XXXX；
#    默认 dry-run，清单确认后 --apply）
python3 scripts/wipe_obs_exps.py result/prod5_current --exp-name prod5
python3 scripts/wipe_obs_exps.py result/prod5_current --exp-name prod5 --apply
# 2) 清本地（发射姿态见 §4 要点：池缓存已复热 → no-seed 直接发；
#    未复热 → 下面两行种子重拷 = fallback 快路径；旧名直接拷——⑬r
#    双读兼容 cluster_cache.npz / cluster_info_cache.json 旧名）
rm -rf result/prod5_current
mkdir -p result/prod5_current
cp result/prod4_current/cluster_cache.npz result/prod4_current/cluster_info_cache.json result/prod4_current/balanced_profile.json result/prod5_current/
# 3) 干净 shell 重发——先确认无残留 env 覆盖（输出必须为空，有输出=开新终端）
    env | grep -E "^(CONFIGS_PER_ITER|PROXY_TARGET_TOKENS|TARGET_TOKENS|TARGET_STEPS|TARGET_ARM_NODES|REMOTE_MAX_PREP|REMOTE_JOB_TIMEOUT_H|REMOTE_QUEUE_TIMEOUT_H|REMOTE_QUEUE_RETRY|REMOTE_MAX_SEARCH_NODES|REMOTE_MAX_VALIDATION_NODES|REMOTE_MAX_JOBS|EXP_NAME|OUTPUT_DIR|FLEET_ARMS|RESUME_ON_CODE_DRIFT)="
# ↑ 必须零输出（含旧名 REMOTE_MAX_JOBS 双查）——有输出 = shell 脏，开新终端
# 4) §4 发射线重发（新开或已确认干净的终端）+ §4.4 对账复核
```

两点说明：迟到上传自愈——作废轮 id 空间 ⊆ 重发 id 空间，扫描后残照的
残留会被同 id 覆盖，终态干净；发射终端 ≠ 体检终端时对账在体检终端跑
（引擎前台占住发射终端）。

**4.5 监控**：

```
bash scripts/diagnostics/prod2_watch.sh result/prod5_current
```

CP0 聚类结构 / CP1 SNR / CP2 online ρ / CP3 Selection mode / CP4 臂+锚点。
搜索关键行：每轮 Best Score（**口径 = 本轮新测批次的最佳，非舰队最佳**——
逐轮数字下降属预期形态不代表搜索退步，prod4 形态 1.33/1.18/1.05 三批不可比；
舰队最佳以收官 state 为准）、Pruning 排除数、终选
`Selection mode`（D19：`best_measured_no_claim` / `…claimed`）、
`Top-k arm candidates`（预期 3 个 climb-cfgXX）。

**4.6 搜索收官 → 臂族（验证阶段，⑬t 起全自动）**：臂族 =
`FLEET_ARMS`（默认 `climb,topk,uniform,natural,domainfix,base`）——
终选臂 climb + `topk_mixture_candidates.json` top-3 晋臂（自动展开
`climb-cfgN`，权重键控同 optimal_mixture_weights.json 约定）+
**uniform**（簇等权基线，2026-09-21 更名裁决；引擎自动流 ⑬t 起以
`--arm uniform` 发射，旧 random 名退役）+ natural / domainfix 同批单种子
+ base 锚点。**引擎 Step 4 = 一次调用**（`scripts/dispatch_fleet.py`）：
逐臂幂等备料（.done 门控）→ 并发后台 dispatch → 节点预算自动排队 →
落地/report 刷新/暂存清理全自动 → 全齐写 `.done_fleet`（完成标记）+
`final_report --auto` 兜底。**remote-only fail-loud**：自动流无本地
torchrun 兜底（7 臂全 fallback 到主节点 8 卡 = ~70h 串行灾难）；失败臂
重跑同一发射命令即幂等重试，已落地臂零成本秒过；本地路径 =
手动应急（`runs/lib/target_arm.sh`）。**臂预算零覆盖**：引擎默认
TARGET_TOKENS=3B、launch_env 实录 3B/2861 → 臂派发无需任何 env 覆盖
（prod4 时代"成对覆盖"已成历史）。**排队不设限**（2026-09-28 裁决，
REMOTE_QUEUE_TIMEOUT_H=0；臂运行时天花板独立 = dispatch `--job-timeout-h`
缺省 0 = 不限制（2026-09-28 对齐: 9h/13h 分支自 f600701 起不可达）, 正值 = 天花板）。

**prod5 桥接（本稿特例，⑬u 后 = 一条命令走默认路径）**：prod5 搜索跑在
⑬t 之前的代码上，其收官段 climb 远程派发因 asset-mount 缺失失败并误落
本地兜底（已击杀、零损失；根因 = run 的 remote_config 不记录
asset_mounts，新代码已带三级回退链）。**首选序列 = kill → pull → 代码
漂移续跑**（新代码的默认路径全程实战验证 = ⑬t/⑬u 的真实彩排）：

```
# 0) 击杀本地兜底训练 + 老驱动（无标记落盘, 零损失）
pkill -f "run_experiment.sh"; pkill -f "torchrun --standalone"
# 1) 拉新代码（工作树干净, 快进）
git pull
# 2) 代码漂移续跑 —— RESUME_ON_CODE_DRIFT=1 断言"本次 delta 仅执行形态"
RESUME_ON_CODE_DRIFT=1 EXP_NAME=prod5 REMOTE_ENABLED=1 \
  REMOTE_BACKEND=modelarts REMOTE_BACKEND_MODULE=climbmix_ma:create_backend \
  REMOTE_FLAVOR=modelarts.pool.visual.8xlarge bash runs/run_experiment.sh
```

预期：stage_gate 打 `⚠ CODE DRIFT acknowledged`（search+target 各一条,
落账 .code_drift.json）→ RESUME → Steps 1-3 跳过（sampled_dataset 在）→
Step 4 = dispatch_fleet 全自动臂族（climb 数据已备秒过、uniform 重备
~6min、cfg101=no-claim 终选重复臂自动剔除、base 先行；节点预算 16 自动
排队、排队不设限、挂载三级回退）→ 全齐 .done_fleet → Step 8 报告 +
final_report --auto 重挂三节 → mark_completed → 目录改名 prod5_<ts>。
老 random_shards/random_mixed 残料（~30GB）收官后手动清。

**4.6b 代码漂移续跑窗（⑬u, 通用机制）**：指纹 = sha256(代码子集 + 语义
参数)。默认代码漂移 = 归档重来（fail-loud——⑬p "绝不把旧代码算的簇当
新代码产物"）。`RESUME_ON_CODE_DRIFT=1` = 操作者显式断言"本次 delta 仅
执行形态（排队/编排/挂载类），零实验语义"→ 不归档续跑，.done 标记决定
重跑范围；**参数漂移不吃这套**（.fingerprint_*_params 逐项对账, 实验形
状变了覆盖也硬归档）；每次使用落账 .code_drift.json（旧/新指纹 + git
SHA + 参数核验状态）。用前先 diff 确认 delta 真的是 infra-only。

**臂族并发 = 自动排队（2026-09-23 裁决，机制不变）**：机器侧
`REMOTE_MAX_VALIDATION_NODES`（默认 **16** = 2 臂 × 8 节点；dispatch CLI
`--max-validation-nodes`）管在飞**训练臂节点总和**——超限的派发自动排队
（`--queue-poll-s` 默认 60s 轮询，先到先得无 FIFO，Ctrl+C 干净退出）；
1 节点作业（base 锚点等 eval-only）入册可见但**免计**（1 与 8 节点不同类
不可比）。与 `TARGET_ARM_NODES`（每臂形状）的区分：后者 = 单臂长什么样
（8 节点/臂），前者 = 全族在飞总和。注册表 `.validation_fleet/` 随派发
进程存活（atexit 注销 + 死 pid 清扫——dispatch 被杀 = 该臂脱离记账）；
report.md 刷新已 flock 串行化（`.report_refresh.lock`），并发臂落地无写入
竞争。**臂成功即自动清本地暂存（⑬q C）**：eval CSV + report 刷新落地后，
dispatch 自动调 `clean_derived_data.py --arms <arm> --apply` 释放
`{arm}_shards` + `{arm}_mixed`（~31GB/臂 @3B；OBS 有内容键化完整副本，
守卫双保险拒清未成功臂）；`CLIMBMIX_ARM_AUTOCLEAN=0` 关闸，手动路径照旧
可用。**大报告自更新（2026-09-22 用户裁决：最终要看整个实验跑完的
大报告）**：report.md = 搜索子报告 + **CP4 判定节** + **赢家配方节**，每个臂
（含 base 锚点）的 eval CSV 落地时 dispatch_target_arm 自动幂等刷新两节
（走 cp4_report，自带配方链；--ref 默认已改 uniform，锚点预期值缺省不做
PASS/FAIL 只报数）——最后一臂落地时 report.md 自动成为完整大报告，零人工
组装。人工判读仍建议跑一次 `cp4_report.py --ref uniform --base-expected
<本轮校准值>`（锚点判定 + 显著性正式读数）。`cluster_peek.py` 为按需调研
工具（簇语义抽样，单独手动跑）。

**终报（全自动，最后一臂落地时自动盖章）**：落臂钩子每次自动推进
report.md（判定节 + 配方节），并自查预期臂清单（topk 的 3 个 climb-cfgN
+ 基线 uniform/natural/domainfix；`RUN_DIR/expected_arms.txt` 可覆盖——
历史命名或 no-claim 条件臂精确控制时每行写一个臂名）——**全齐的那一刻
自动盖 FINAL 终报章**，大报告零人工步骤完成。手动 `final_report.py
RUN_DIR [--arms …] [--base-expected …]` 仅用于：强制盖章 / DRAFT 预览 /
锚点正式判定（`--base-expected` 校准值，自动模式锚点只报数不判定）。
report.md 最终结构：搜索子报告 → CP4 判定 → 赢家配方 → 终报印章。

**4.7 事故记录（2026-09-28,iter 3 排队饥饿）**：假期平台拥堵,末轮
16 个子任务 11 个跑完、exp 110–114 五个在平台队列 72h（24h × 3 次重提）
无卡 → 驱动排队上限自伤取消 → floor 守卫拒收 11/16 → 干净崩溃（state
已存,meta.json 复用,resume 精确重跑缺的 5 个,零科学损失）。**三连裁决**：
① 排队时钟 0=不限制（同轮搜索+臂,`ed34fd8`——旧代码直接设 0 会秒烧,
须 pull 后生效;当次续跑用 `REMOTE_QUEUE_TIMEOUT_H=8760` 环境保险丝零
代码续命）；② 臂族全自动发射（⑬t,`0f94058`）；③ 老驱动自动 random 臂
旧名缺陷——**收官前已贴抑制标记**（`.done_mid_train_random` +
`.done_eval_random`,touch 于搜索收官前,数据备料不受影响）,uniform 基线
臂走 4.6 桥接命令补发。教训：引擎自动流的臂名曾与 P-0 预注册命名脱节
（random vs uniform）,smoke 只验到 Step 3 从未彩排 Step 4-7——⑬t 重写
后该段首次进入自动验证面。

## 5. 发射窗顺手卫生（非阻塞批处理，F2）

- **mid optim 存量回收（2026-09-22 裁决：全砍）**——写入端已过滤（worker
  上传/下载/归档/训练侧只保 weights+meta，~1.5× 权重/实验的死重不再产生），
  历史存量用 `scripts/sweep_optim.py`（默认 dry-run，`--apply` 真删；本地
  roots + OBS `--remote-config/--obs-prefix` 两模式；结构性守卫永不碰
  base_checkpoints，`--min-age-hours 12` 保护在跑实验）：
  ```
  python3 scripts/sweep_optim.py /home/ma-user/work/nanochat_model_dir/mid_checkpoints result/prod4_current
  python3 scripts/sweep_optim.py --remote-config result/prod4_current/remote_config.json
  # 清单确认后加 --apply
  ```
- OBS 孤儿清理（prod4/target_arms 旧无键路径 1136 片 ≈6B 等，一次性 obsutil）；
- 中央 `mid_checkpoints` 清点：10 个残留 proxy ckpt（climbmix_prod1_×5 /
  prod2_k15bal_×3 / prod4_0054/0092，~25G）+ 旧轮 d28_smoke4n_* / d28_speedrun
  （optim 部分并入上面的 sweep，整目录清点仍走原条目）；
- `dataset.py.bak.20260914_104216` 删除确认；
- guard scratch clone `~/work/tmp/nanochat-perf` 清理；
- **PAT 撤销（用户动作，独立于本窗口）**。

## 记录沿革

- 2026-09-21 v1：定稿。D1/D2/smoke/prod5 四段 + 卫生批处理；smoke 参数 =
  用户裁决（诊断目录、重件不保存）；eval-only 命令形态 = `nanochat_cmds.
  build_target_eval_cmd`（prod4 臂 eval 同款 argv）；激活 pull 目标 =
  nanochat `6e5baa2` + climbmix main（发射时 HEAD）。
- 2026-09-22 v1.1：激活实测两处修正——worktree 实际路径无 `@SHA` 后缀 +
  remove 前加 status 检查；check-assets 命令必须带 PYTHONPATH 前缀
  （src + vendored climbmix-ma，自举跳过陷阱）。
- 2026-09-22 v1.2：1.4 重写为具体三步（诊断 prod4 assets 轮子 → 派生 prod5
  remote-config → ma_sync_code 同步到 **prod5 新前缀**）+ 1.5 验证与发射前预期
  形态。修正 v1 的两处错误认知：tarball 走 assets/ 新鲜通道而非 assets_big；
  check-assets 绿 ≠ 代码是新的（它只查在场，且 prod4 前缀的存量资产与 prod5
  无关）。
- 2026-09-22 v1.3：§4 手动序列（4.1-4.4）整体替换为**版本化发射器**
  `runs/launch_prod5.sh`（用户裁决：正确指令固化成脚本，不贴 LLM 现拼命令）。
  缓存种子存活依赖 stage-gate cache-seed 豁免（`3f99ef2`，同日）。
