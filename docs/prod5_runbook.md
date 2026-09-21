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
git -C /home/ma-user/work/climbmix worktree list
git -C /home/ma-user/work/climbmix worktree remove --force "/home/ma-user/work/climbmix_lb@b93bb97"
git -C /home/ma-user/work/nanochat-npu worktree list
git -C /home/ma-user/work/nanochat-npu worktree remove --force "/home/ma-user/work/nanochat_lb@0c1229f"
```

（路径以 `worktree list` 实际输出为准。）

**1.4 重建 nanochat 代码 tarball → assets_big**（远端 worker 跑的 eval 代码 =
新栈；climbmix 侧的两文件 worker bundle 由 executor 每次发射自动上传，无需手动）：
按 climbmix-ma README 既有惯例重建上传（布局同历史发射，prod4 记录的 worker tar
基线为 `a7c56792` 时代），然后验证：

```
cd /home/ma-user/work/climbmix
python3 scripts/dispatch_remote.py --remote-config result/prod4_current/remote_config.json --check-assets
```

全绿为准（`remote_config.json` 若不在 prod4_current，用任一历史 run 的；它只是
obs 前缀 + 后端身份的载体）。

## 2. smoke 彩排（~30-45 min，8 NPU，全本地零排队）

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

**4.1 聚类缓存继承**（全新跑 ≠ 重新聚类；簇与 prod4 逐位一致）：

```
cd /home/ma-user/work/climbmix
mkdir -p result/prod5_current
cp result/prod4_current/cluster_cache.npz result/prod5_current/
cp result/prod4_current/cluster_info_cache.json result/prod5_current/
cp result/prod4_current/balanced_profile.json result/prod5_current/
```

**4.2 发射参数核对**：除 `EXP_NAME=prod5`、`CONFIGS_PER_ITER=64,32,16` 外，
全部与 prod4 发射参数相同（同规模裁决 E1）。先 dump 对照：

```
python3 -c "import json; print(json.dumps(json.load(open('result/prod4_current/launch_env.json')), indent=1, ensure_ascii=False))"
```

逐键过一遍 run_experiment.sh 的 EDIT 块（REMOTE_* / NPU_PER_EXP /
ADAPTIVE_* / 预算旋钮 …）。

**4.3 发射前自检**（固化 prod2 五坑：flavor 卡数 / priority / 资产挂载 / 残留
进程 / 历史作业）：

```
python3 scripts/diagnostics/preflight_launch.py --run-dir result/prod5_current --main-log prod5.log
```

须全绿。

**4.4 发射**（生产形态：远端舰队 + 本地混合；REMOTE_* 按 4.2 核对的原值）：

```
setsid nohup env EXP_NAME=prod5 CONFIGS_PER_ITER=64,32,16 REMOTE_ENABLED=1 \
  [其余按 prod4 原值] \
  bash runs/run_experiment.sh > prod5.log 2>&1 &
tail -f prod5.log
```

**4.5 监控**：

```
bash scripts/diagnostics/prod2_watch.sh result/prod5_current
```

CP0 聚类结构 / CP1 SNR / CP2 online ρ / CP3 Selection mode / CP4 臂+锚点。
搜索关键行：每轮 Best Score、Pruning 排除数、终选
`Selection mode`（D19：`best_measured_no_claim` / `…claimed`）、
`Top-k arm candidates`（预期 3 个 climb-cfgXX）。

**4.6 搜索收官 → 臂族**：`topk_mixture_candidates.json` top-3 晋臂
（`ARM_NAME=climb-cfgXX`，权重文件直接 `--weights` 可用）；random3b / natural /
domainfix 同批单种子；base 锚点先行；no-claim 若放行外推 → 额外 +1 臂。
臂发射沿用 prod4 流程（dispatch_target_arm / arm_engine，含单遍守卫与磁盘
preflight）。

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
