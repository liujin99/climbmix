# prod5 发射 runbook（2026-09-21 打磨窗定稿）

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

## 4. prod5 发射（搜索 ~16h + 臂族）

**版本化发射器 `runs/launch_prod5.sh`（2026-09-22 新增）**——原 4.1-4.4 的
手动序列（缓存继承 / 参数对照 / preflight / 发射）全部内置：

```
cd /home/ma-user/work/climbmix
bash runs/launch_prod5.sh            # 干跑（默认）：环境重建 + 逐键对照表 + 引擎干跑门
LAUNCH=1 bash runs/launch_prod5.sh   # 真发射：preflight 过后后台起引擎，打印监控命令
```

要点（全部内置于脚本，此处仅备忘）：
- **缓存种子**：3 个聚类缓存文件 cp 进 result/prod5_current——stage-gate 的
  cache-seed 豁免（`3f99ef2`）保它存活。修复前该流程会把缓存 orphan 归档、
  空目录重来 → 重嵌入 116M 文档池（prod3 救援记录里的"预写指纹"手工步骤
  即为绕此坑，本修复使其消失）。
- **机器对照**：30 个 target/共享旋钮从 prod4 的 launch_env.json 重建、
  fleet 旋钮（REMOTE_*）从其 remote_config.json 重建（秒→小时回换算）、
  OBS 前缀末段换名（与 D1 的代码同步目标一致）；`TARGET_TOKENS` 与 SRC
  记录不一致直接熔断（E1 同规模裁决的机器化）。
- **EDIT 块 = 本轮有意变更**：`CONFIGS_PER_ITER=64,32,16`（E2）/
  `PROXY_TARGET_TOKENS=400M`、`TARGET_TOKENS=3B`（E1；launcher 默认 640M/
  2B，必须显式）/**`DISPATCH_RANDOM_ARM=0`**（launcher 默认 1 会预发已更名
  的 random 臂——手动流程必踩坑）。
- **重发射注意**：`_current` 已有指纹且其间代码/参数变过 → 引擎归档整个
  目录（含种子）后空目录重来 → 先 `rm -rf result/prod5_current` 再跑脚本。

**4.4 发射**：`LAUNCH=1 bash runs/launch_prod5.sh`（上面的版本化发射器；后台
起引擎 + 打印监控命令）。

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

**4.6 搜索收官 → 臂族**：`topk_mixture_candidates.json` top-3 晋臂
（`ARM_NAME=climb-cfgXX`，权重文件直接 `--weights` 可用）；**uniform**
（簇等权基线，2026-09-21 更名裁决；prod1-4 臂名 random3b）/ natural /
domainfix 同批单种子；base 锚点先行；no-claim 若放行外推 → 额外 +1 臂。
臂发射沿用 prod4 流程（dispatch_target_arm / arm_engine，含单遍守卫与磁盘
preflight）。**大报告自更新（2026-09-22 用户裁决：最终要看整个实验跑完的
大报告）**：report.md = 搜索子报告 + **CP4 判定节** + **赢家配方节**，每个臂
（含 base 锚点）的 eval CSV 落地时 dispatch_target_arm 自动幂等刷新两节
（走 cp4_report，自带配方链；--ref 默认已改 uniform，锚点预期值缺省不做
PASS/FAIL 只报数）——最后一臂落地时 report.md 自动成为完整大报告，零人工
组装。人工判读仍建议跑一次 `cp4_report.py --ref uniform --base-expected
<本轮校准值>`（锚点判定 + 显著性正式读数）。`cluster_peek.py` 为按需调研
工具（簇语义抽样，单独手动跑）。

## 5. 发射窗顺手卫生（非阻塞批处理，F2）

- OBS 孤儿清理（prod4/target_arms 旧无键路径 1136 片 ≈6B 等，一次性 obsutil）；
- 中央 `mid_checkpoints` 清点：10 个残留 proxy ckpt（climbmix_prod1_×5 /
  prod2_k15bal_×3 / prod4_0054/0092，~25G）+ 旧轮 d28_smoke4n_* / d28_speedrun；
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
