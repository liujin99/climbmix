#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: STEM 数据混合优化 — 单脚本全流程 (d20 → d28)
#
#  用法:   bash runs/run_climbmix.sh
#  实验:   EXP_NAME=myexp bash runs/run_climbmix.sh   (输出 result/myexp_current)
#
#  断点续跑 (直接重跑同一命令即可):
#    - 阶段指纹匹配 → 自动续跑: 聚类/搜索状态/已完成实验/target 训练/eval 全部复用
#    - search 指纹变(搜索语义代码或参数变更) → 归档 result/${EXP_NAME}_stale_search_<ts> 后全新开始
#    - target 指纹变(仅 target 语义变更) → 只归档 target 产物 (result/${EXP_NAME}_stale_target_<ts>),
#      搜索结果保留, Steps 4-8 重跑
#  恢复粒度: 步骤级(.done) / 迭代级(search_state.json) / 实验级(exp_*/meta.json)
#            / embedding 分片级(进度账本) / 训练内部不支持(整次重跑)
#  停止: pkill -f run_climbmix.sh —— 后台启动(自有进程组, 建议配 setsid)时
#        TERM 连带全部子进程; 前台直接 Ctrl-C
#  生命周期: 活跃 = result/${EXP_NAME}_current; 正常跑完自动改名 result/${EXP_NAME}_<ts>
#    (重跑同命令 → 自动恢复已完成 run, 全程跳过); 每个归档目录带 archive_meta.json。
#  旧版单一 .fingerprint 目录: MIGRATE_LEGACY_FINGERPRINT=1 采纳(不校验)。
#  num_npu 不进指纹(并行形状可变, 见 runs/lib/stage_gate.sh)。
#  注意: nanochat-npu 侧代码变更、同名数据文件内容变化不在指纹检测范围内
# ═══════════════════════════════════════════════════════════════════════
# Source CANN env BEFORE set -euo pipefail (set_env.sh may have commands
# that fail under strict mode, causing incomplete env setup)
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

# ── Orphan-safe teardown ──
# Background launches (nohup ... &: interactive job control gives this script
# its own process group — setsid in the launch command makes it airtight)
# install a group-kill trap: TERM/INT/exit takes every descendant
# (python/torchrun/dataloader workers) down with us, so
# `pkill -f run_climbmix.sh` cannot strand NPU-holding grandchildren (the
# 2026-09-04 prod1 stop left ~a dozen orphans to clean by hand). Foreground
# runs share the caller's process group: skip there — kill 0 would TERM the
# interactive shell itself, and the terminal already delivers Ctrl-C to all.
_pgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d '[:space:]')"
if [ -n "$_pgid" ] && [ "$$" = "$_pgid" ]; then
    trap 'trap - TERM INT EXIT; kill 0' TERM INT EXIT
fi

# ── Configuration ──
CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-main}"
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

PROXY_DEPTH="${PROXY_DEPTH:-20}"
TARGET_DEPTH="${TARGET_DEPTH:-28}"
# Token caps for data selection (full pool ≈ 100B tokens / 116M docs — NOT capped means
# every proxy exp would select the whole pool; that's the default 0, so always set these).
# Proxy: PROXY_TARGET_TOKENS 是搜索小实验的预算真源 (步数由它派生, 见下方推导块)。
# 真实 total_batch_size 来自 ckpt meta (2026-09-11 服务器实测): d20 与 d28 都是
# 1,048,576 — mid_train 从预训练 meta 继承 tbs, 历史上假设的 "d20 = 524,288" 是
# 回退值误当真值 (它导致 prod2 proxy 实测 3-4 epoch wrap, loader 日志已证)。
# 默认 640M → 610 步 = 论文 proxy 消耗 (~800M) 的 80% (资源受限, 2026-09-11 拍板);
# 池 = 预算/0.7 (mix 定尺寸修复后) ≈ 1.43×消耗 → 单遍 (epoch≈0.7)。
# 800M → 762 步 = 论文等量; 400M → 381 步 = prod2 原样 (信号减半)。
# Target: TARGET_TOKENS 是唯一真源 (退火预算), 步数由它派生 (见下方推导块),
# 池子=预算/STEM_RATIO, 消耗=预算 → 恒单遍 (epoch≈0.7)。
# 默认 2B → 1907 步 — 与 proxy 等比: d28/d20 参数 3.4× (scaling 435M→1.5B,
# scoring_metric_design §12), 2B/640M = 3.1× → 臂与搜索同 tokens/参数 regime,
# predictor 选出的配比在臂预算下保持最优 (配比转移保真)。
PROXY_TARGET_TOKENS="${PROXY_TARGET_TOKENS:-640M}"
TARGET_TOKENS="${TARGET_TOKENS:-2B}"

# TARGET_STEPS 派生 (单一真源): steps = TARGET_TOKENS / total_batch_size (d28 ckpt meta)。
# 消耗 ≈ 预算、池子 = 预算/STEM_RATIO ≈ 1.43×消耗 → 恒单遍 (epoch≈0.7, 守卫恒过)。
# TARGET_STEPS 不再是用户旋钮: 外部设置 = 遗留配置, 就地报错 (防静默指纹漂移)。
# 臂级复用 (run_arm_only.sh) 用同一 CLI 派生; 直接 dispatch 的"同数据改步数"走
# env 优先级, 由 single-pass 守卫把关。
# 历史复现: prod1/prod2 实际跑的是 1000 步 (tbs 1,048,576) = TARGET_TOKENS=1000Mi。
if [ -n "${TARGET_STEPS:-}" ]; then
    echo "✗ TARGET_STEPS=${TARGET_STEPS} is no longer a knob — it is DERIVED from TARGET_TOKENS."
    echo "  unset TARGET_STEPS and set the budget instead (TARGET_TOKENS=2B → 1907 steps @ 1,048,576)."
    exit 1
fi
if [ "${TARGET_TOKENS}" = "0" ]; then
    echo "✗ TARGET_TOKENS=0 ('all available') is invalid for target arms — set an explicit budget (e.g. 2B)."
    exit 1
fi
TARGET_STEPS="$(python3 scripts/derive_target_steps.py \
    --target-tokens "$TARGET_TOKENS" \
    --ckpt-dir "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}")" \
    || { echo "✗ TARGET_STEPS derivation failed"; exit 1; }
echo "  TARGET_STEPS derived from TARGET_TOKENS=$TARGET_TOKENS -> $TARGET_STEPS steps"

# PROXY_NUM_ITERATIONS 派生 (与 TARGET_STEPS 同规则): 搜索小实验步数由
# PROXY_TARGET_TOKENS 派生 — token 预算是唯一真源 (说 token 直观, 说 steps 不直观)。
# 外部设置 = 遗留配置, 就地报错; 需要更强/更省的搜索信号时改预算即可
# (640M → 610 步 = 默认/论文 80%; 800M → 762 步 = 论文等量; 400M → 381 步 = prod2 原样)。
if [ -n "${PROXY_NUM_ITERATIONS:-}" ]; then
    echo "✗ PROXY_NUM_ITERATIONS=${PROXY_NUM_ITERATIONS} is no longer a knob — it is DERIVED from PROXY_TARGET_TOKENS."
    echo "  unset it and set the budget instead (PROXY_TARGET_TOKENS=640M → 610 steps @ 1,048,576)."
    exit 1
fi
PROXY_NUM_ITERATIONS="$(python3 scripts/derive_target_steps.py \
    --target-tokens "$PROXY_TARGET_TOKENS" \
    --ckpt-dir "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}")" \
    || { echo "✗ PROXY_NUM_ITERATIONS derivation failed"; exit 1; }
echo "  PROXY_NUM_ITERATIONS derived from PROXY_TARGET_TOKENS=$PROXY_TARGET_TOKENS -> $PROXY_NUM_ITERATIONS steps"
python3 - "$PROXY_TARGET_TOKENS" <<'PYEOF'
import sys
sys.path.insert(0, "src")
from climbmix.utils.token_estimate import parse_token_count
t = parse_token_count(sys.argv[1])
paper = 800_000_000
print(f"  ⚠ proxy 信号强度: 每实验消耗 {t:,} tokens ≈ 论文 proxy ~800M 的 {t/paper:.0%}")
print("    调强度只改预算: 400M → 381 步 (更省) / 800M → 762 步 (论文等量; 默认 640M = 80%)")
PYEOF
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10,5}"
# prod2 B++: 期望列表语义 — ADAPTIVE_CONFIGS=1 时 configs_per_iter 视为
# "期望每轮实验数", 由实测并发 (RemoteExecutor 探测) 浮动到
# w_i = round(e_i / S0) 个满波 (probe-truncate / straggler-eviction /
# rolling C_eff 三机制, 见 docs/parallel_k_selection.md §5.2)。
# 默认 0 = 字面语义 (每轮恰好 n 个, 老行为)。
ADAPTIVE_CONFIGS="${ADAPTIVE_CONFIGS:-0}"
# prod3 时间盒 (需 ADAPTIVE_CONFIGS=1): 池 = e_i+4 不吃 S0 填充、admit
# buffer 0 → 探针把每轮落在恰好 w_i 波 (波 ≈ 3h)。贪心模式每轮至少
# 2 波 (池 ≥ S0+4); 紧凑用 overshoot 样本换墙钟: [20,10,10] ≈ 22/11/11
# 承认 4 波 ~13h vs 贪心 6-7 波 ~19h。默认 0 = 贪心 (闲卡是免费样本)。
ADAPTIVE_COMPACT="${ADAPTIVE_COMPACT:-0}"
SEARCH_NUM_ITERATIONS="${SEARCH_NUM_ITERATIONS:-3}"
K_ENHANCED="${K_ENHANCED:-3}"
# balanced 模式下 K_max 语义等同 K_ENHANCED (容量约束划分恰好到 K);
# distance 模式仍可显式覆盖。默认跟随 K_ENHANCED。
K_CLUSTER_MAX="${K_CLUSTER_MAX:-$K_ENHANCED}"
K_INIT="${K_INIT:-1000}"
FILTER_METHOD="${FILTER_METHOD:-none}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-3.0}"
# Per-column floor: prune clusters whose WEAKEST quality-column mean falls
# below this (catches what the flat average washes out — the 69 escape
# clusters / 6.6% docs / 1.5% tokens population from the 2026-08-31 profile).
# 2.0 = conservative: only kills knowledge_value <2.0 even under scorer
# noise; raise after eyeballing pruned samples (0 = off).
PRUNE_COLUMN_FLOOR="${PRUNE_COLUMN_FLOOR:-2.0}"
MERGE_DISTANCE="${MERGE_DISTANCE:-0.9}"
# prod2: balanced = 容量约束平衡划分到恰好 K_ENHANCED 个宏簇(本池嵌入空间为单一
# 连续流形,距离合并在任何 (K,tau) 下都塌成 ~99% 巨簇 — 见 paper_deviations.md
# D14;2026-09-07 起为默认)。distance 仍可用(本池已死,仅作对照)。
MERGE_STRATEGY="${MERGE_STRATEGY:-balanced}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
# Stable pool-keyed cache for embeddings + K-means (survives fingerprint
# resets; K/merge knob changes reuse embeddings instead of re-embedding)
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$CLIMBMIX_DIR/cache/embeddings}"
DISCOVERY_METHOD="${DISCOVERY_METHOD:-embedding_cluster}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-npu}"
EMBEDDING_SAMPLE_SIZE="${EMBEDDING_SAMPLE_SIZE:-0}"
# Proxy/target training dynamics (semantic: change → fingerprint → fresh run)
PROXY_LR_SCALE="${PROXY_LR_SCALE:-1.0}"
PROXY_WARMUP="${PROXY_WARMUP:-0.0}"
PROXY_WARMDOWN="${PROXY_WARMDOWN:-0.9}"
TARGET_LR_SCALE="${TARGET_LR_SCALE:-1.0}"
TARGET_WARMUP="${TARGET_WARMUP:-0.0}"
TARGET_WARMDOWN="${TARGET_WARMDOWN:-0.9}"
# d28 Step-6 OOM: 对齐 quadmix/nanochat_mid_compare/run_stem_experiment.sh
# (dev/dataset-schema 分支, 同一 nanochat-npu repo + 同一 d28 ckpt) 的实证配置:
#   DEVICE_BATCH_SIZE=1 + 完整 NPU env 块 (runs/lib/npu_env.sh, 含 unified
#   memory) + --sample-every=-1 + --eval-every=-1 (quadmix 两者都显式关)。
# 根因 (2026-08-28): DistMuonAdamW Phase-1 为每个 Muon shape 组 stack
# 全量梯度副本 (optim.py:515-519, 当前组另需 2× 最大组 ~4G 瞬时) → 峰值
# ≈ 静态 16G + 副本 5.3G + 2G + 通讯 ~1G ≈ 24.2G, 距 torch 实际天花板
# ~24.5G (29.49 − ~4.7G CANN/HCCL) 余量 <0.3G; 而 --eval-every 默认 100
# 在 step 0 必跑 1280 个 val forward → allocator 段碎片化 → 2G 连续分配
# 失败 (dbs=1 实测 22.24G alloc OOM; quadmix 干净路径 390 步全过)。
# dbs=8/4/2 在第一个 forward 撞 26.9-27.5G 墙 (静态+激活, 实测全灭)。
# dbs 只影响 micro-batch 切分 (total batch 524,288 不变), 两臂同值 → 可比。
# 生产若想升 dbs: 先看 speedrun 日志的 "Peak memory usage" 实测余量。
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-1}"
# flat = 零裁剪文档打包 (DeepSeek V3 式), 与 proxy 搜索阶段及 quadmix 实验同口径
MID_TRAIN_LOADER="${MID_TRAIN_LOADER:-flat}"
# BPB-only 旋钮: base_eval 只在 bpb 分支读 --device-batch-size
# (base_eval.py:514/:522), 本流程 --eval=core 下是空操作; 32 是 8x910B3
# (64G HBM) 时代默认, 16 对齐 quadmix 同硬件 d28 实证值, 防将来开 bpb 踩坑。
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"
# core eval 的真实显存旋钮: --core-eval-batch-size (base_eval.py:417, 默认16)
# 把 chunk 内样本 pad 到最长序列一次 forward (峰值主体是 logits B×T×V)。
# 2026-08-28 speedrun Step-7 OOM 实证: 默认 16 的整块 forward 顶满 torch 池
# (~24.5G), 任务末尾 dist.barrier() 处 HCCL 申请 401MiB allreduce 通信缓冲
# 失败 (EL0004, allocator 记账之外, core_eval.py:412; 每任务后的
# empty_cache 在 barrier 之后才跑)。生产 EVAL_MAX_PER_TASK=-1 时每卡条数
# 更多, 但单 forward 峰值同样由 core_bs 决定。8x910B3(64G)→8x910B4(32G)
# 显存减半 → batch 同步减半 16→8。
EVAL_CORE_BATCH_SIZE="${EVAL_CORE_BATCH_SIZE:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
STEM_RATIO="${STEM_RATIO:-0.7}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"
# Eval subsample cap per task: -1 = FULL eval sets (production default).
# base_eval shuffles each task with a fixed seed (1337) before truncating,
# so any cap still yields comparable scores across experiments.
EVAL_MAX_PER_TASK="${EVAL_MAX_PER_TASK:--1}"
NUM_NPU="${NUM_NPU:-8}"
NPU_PER_EXP="${NPU_PER_EXP:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/result/${EXP_NAME}_current}"
# 终态标记: 全部存在 => run 完整跑完, 末尾 mark_completed 把活跃目录
# 改名为已完成形态 result/${EXP_NAME}_<ts> (stage_gate.sh 生命周期)
COMPLETION_MARKERS=(".done_eval_climb" ".done_eval_random")

# ── Remote execution fleet (remote jobs + OBS data plane) ──
# 生产混合舰队: 本地 8 卡 + 远端作业并行跑 proxy 实验。
# 全部为"执行形态"参数 (传输/配额/路径), 不改变实验语义 — 与 NUM_NPU
# 同一策略, 刻意不进 stage 指纹 (stage_gate.sh:51 先例: 池形状可变)。
# REMOTE_ENABLED=1 时 Step 1-3 的 search 用 RemoteExecutor:
#   - 本地: 混数据 + 上传分片到 OBS + 提交作业 + 回收结果为本地 exp_XXXX
#   - 远端作业: 下载分片 -> torchrun mid_train -> base_eval -> 结果上 OBS
#   - REMOTE_LOCAL_PARALLEL=1: 主节点本地卡也加入舰队 — 前
#     NUM_NPU/NPU_PER_EXP 个配置走本地并行, 其余远端作业, 全程并发。
#     NPU_PER_EXP == NUM_NPU 时本地是 1 个 k 卡"整槽"(ProxyRunner 串行
#     全卡路径, prod2 的 k=8 形态); 默认 0 = 本地卡空闲(保守, 老语义)
#   - REMOTE_NPU_PER_JOB 默认 = NPU_PER_EXP (k 全舰队一致, 分数可比性,
#     docs/parallel_k_selection.md); 想让本地卡出力需 NPU_PER_EXP <= NUM_NPU
#   - 动态提交 (池容量波动): 提交被配额/频控拒绝时指数退避重试,
#     配置不因瞬时拒绝烧毁; 一个迭代的作业随配额释放分多轮落地。
#     在飞上限 = REMOTE_MAX_JOBS, 池变大时调高即可; 本地混料/上传并发
#     由 REMOTE_MAX_PREP 限流, 不随作业上限放大。
# 平台后端在独立的 (私有) 适配仓实现, 经 REMOTE_BACKEND_MODULE 注册 —
# 见 docs/remote_setup.md "Writing a backend"。
# 前置 (M1, 后端仓 README): 平台配置文件 (网关/凭证/镜像) + moxing 可用
#   + 大资产上 OBS (nanochat-npu 代码, d20 ckpt, tokenizer, eval_bundle/stem)。
# 验证 (M3): 后端仓的 hello-world 校准脚本打通网关 → dispatch_remote.py
#   单发 exp + Δstem_metric < 0.002 → 并发波。
REMOTE_ENABLED="${REMOTE_ENABLED:-0}"
REMOTE_LOCAL_PARALLEL="${REMOTE_LOCAL_PARALLEL:-0}"
REMOTE_OBS_PREFIX="${REMOTE_OBS_PREFIX:-}"
REMOTE_BACKEND="${REMOTE_BACKEND:-mock}"           # mock (本地仿真) | 平台后端名
REMOTE_BACKEND_MODULE="${REMOTE_BACKEND_MODULE:-}" # 后端工厂 "pkg:attr" (见后端仓 README); pip 安装的后端可留空走 entry point
REMOTE_PLATFORM_CONFIG="${REMOTE_PLATFORM_CONFIG:-}" # 平台配置 JSON 路径 (默认由后端解析, 如 ~/.config/climbmix/...)
REMOTE_IMAGE="${REMOTE_IMAGE:-}"                   # 镜像 URI (可留空=用平台配置 image_url)
REMOTE_FLAVOR="${REMOTE_FLAVOR:-}"                 # 规格名 (可留空=用平台配置 default_flavor)
REMOTE_POOL_NAME="${REMOTE_POOL_NAME:-}"          # 专属池 (可空=用配置文件 pool_id)
REMOTE_NPU_PER_JOB="${REMOTE_NPU_PER_JOB:-$NPU_PER_EXP}"  # 每作业卡数 (单 exp 不跨节点)
REMOTE_MAX_JOBS="${REMOTE_MAX_JOBS:-14}"          # 在飞作业上限 (动态提交的上界)
REMOTE_SUBMIT_RETRY_H="${REMOTE_SUBMIT_RETRY_H:-24}" # 提交被拒重试时限 (小时)
REMOTE_MAX_PREP="${REMOTE_MAX_PREP:-4}"           # 本地混料/上传并发
REMOTE_STORAGE_KIND="${REMOTE_STORAGE_KIND:-moxing}"  # 容器内存储后端
REMOTE_STORAGE_ROOT="${REMOTE_STORAGE_ROOT:-}"    # mock 后端专用: 假 OBS 根目录
REMOTE_JOB_TIMEOUT_H="${REMOTE_JOB_TIMEOUT_H:-6}" # 单作业 RUNTIME 超时 (小时, 排队时间不计 — 首个 RUNNING 起算)
REMOTE_QUEUE_TIMEOUT_H="${REMOTE_QUEUE_TIMEOUT_H:-24}" # 排队超时 (提交→起跑, 小时; 池满时作业可在平台队列里等卡)
REMOTE_QUEUE_RETRY="${REMOTE_QUEUE_RETRY:-2}" # 排队超时后重提次数 (新排队时钟; 总排队耐心 = 超时 × (1+次数))
# 自适应驱逐的 PENDING 宽限 (分钟, 仅 ADAPTIVE_CONFIGS=1 生效): 已提交作业
# 排队超过该时长且同批已有作业在跑 = 舰队超额信号 → cancel + 该配置从本轮
# 永久移除 (pending 重写, resume 不重跑)。臂作业不受影响 (必做交付, 24h 耐心)。
REMOTE_PENDING_GRACE_MIN="${REMOTE_PENDING_GRACE_MIN:-30}"
REMOTE_CODE_WHEELS="${REMOTE_CODE_WHEELS:-}"  # 离线 wheel 本地路径 (逗号分隔, executor 自动补传)
# per-launch 直挂资产 (JSON 对象 {"name":"obs://..."}), 替换平台配置的全局
# asset_mounts — search 舰队只挂自己要的 (d20/tokenizer/eval_*), 不带 embed
# 专属的 stella/pool/d28。空 = 继承全局 (兼容旧行为)。
REMOTE_ASSET_MOUNTS="${REMOTE_ASSET_MOUNTS:-}"

# ── Target-arm execution (Step 6+7 的执行形态, 刻意不进指纹) ──
# remote (默认): scripts/dispatch_target_arm.py 把臂作为远端 8 卡作业提交
#   (与搜索舰队同池/同镜像/同 argv 构建器, 与本地唯一差异 = 在哪跑);
#   random 臂可由该脚本单独提前发射(搜索期间并行), climb 臂在搜索结束后
#   由 Step 6 发出。远端失败 → 自动回退本地 torchrun (三层兜底)。
# local: 永远本地跑 (prod1 行为)。
TARGET_ARM_MODE="${TARGET_ARM_MODE:-remote}"
# 多节点目标臂 (Phase 1): TARGET_ARM_NODES=4 → ws=32, 实测 6.1s/step (单节点
# 18.2s, 3.0x; Phase-0 job 440f760e)。约束: 2 的幂 (1/2/4) —— d28 优化器
# reduce_scatter 断言 shape[0] % world_size == 0, 全部维数是 2 的幂, 故
# ws=8×N 必须是 2 的幂 (ws=24 已实测失败, run 601cdb67)。与 TARGET_ARM_MODE
# 一样是执行形态, 刻意不进指纹 —— 但它派生的 target_load_optimizer 是训练
# 语义, 进指纹 (见 FP_TARGET_PARAMS)。
TARGET_ARM_NODES="${TARGET_ARM_NODES:-1}"
# ws≠8 无法加载 8-shard d28 优化器 (形状断言) → 多节点臂冷启动优化器; 单节点
# 保持 prod1 行为 (加载)。本地兜底 (target_arm.sh) 读同一变量: 两臂无论走
# 远端还是本地兜底, 优化器语义一致, 对比才公平。派生量, 非用户旋钮。
if [ "$TARGET_ARM_NODES" -gt 1 ]; then
    TARGET_LOAD_OPTIMIZER=0
else
    TARGET_LOAD_OPTIMIZER=1
fi
# 远端 d28 基座资产 (obs:// 目录): dispatch 首次引导时把本地
# base_checkpoints/d28 上传到该 URI (一次性); 缺省 {prefix}/assets_big/d28。
REMOTE_D28_ASSET_URI="${REMOTE_D28_ASSET_URI:-}"

# ── HF download endpoint ──
# The managed runtime's egress proxy selectively rejects Python's bare
# CONNECT tunnels to huggingface.co (observed: 90+ consecutive 503s across
# two independent runs / 80 min, while curl to the same host AND Python to
# hf-mirror.com both succeeded). hf-mirror.com serves the same bytes (Range
# resume verified, 206). Covers ClimbMix shards + eval_stem.zip (nanochat
# reads HF_ENDPOINT at import time in dataset.py AND base_eval.py).
# Override to use the origin: HF_ENDPOINT=https://huggingface.co bash runs/...
# (proxy details: the backend repo's README)
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── NPU Environment (deliberately minimal — matches proven train_base_model.sh) ──
# See runs/speedrun_climbmix.sh: the allocator block that used to live here
# (memory_pool:True + PYTORCH_NPU_ALLOC_MAX_SIZE=60G + friends) filled device 0
# outside torch_npu's allocator accounting → kernel-load OOMs (aclnnMean 207001
# / EL0004) in every proxy exp and Step 6 on 2026-08-26. Do not re-add
# allocator overrides unless a specific need is proven on this hardware.
export OMP_NUM_THREADS=1 WANDB_MODE=offline NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
# Cluster-stage CPU threads (faiss kmeans/assign): measured sweet spot on the
# 192-vCPU aarch64 host (scripts/diagnostics/cluster_bench.py, 2026-08-29):
# 24 threads = 281 GFLOP/s vs 141 at the old default min(cpu,64)=64 — sgemm
# throughput collapses past ~24 threads on this box. Override by exporting
# before launch. OMP_NUM_THREADS=1 above stays: it guards the NPU training
# stage; cluster_embeddings_faiss re-raises the cap at call time.
export CLIMBMIX_CLUSTER_THREADS="${CLIMBMIX_CLUSTER_THREADS:-24}"
mkdir -p "$NANOCHAT_BASE_DIR"
export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200 HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_NPU - 1)))
export RANK_SIZE=$NUM_NPU MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export PYTHONUNBUFFERED=1
export NANOCHAT_DTYPE="$NANOCHAT_DTYPE" PYTHONWARNINGS="ignore::UserWarning:torch_npu"

# ── Stage-scoped fingerprints (code + semantic params → auto-reset on change) ──
# search guards Steps 1-3 products (embedding/cluster/search_state/exp_*/
# sampled_dataset); target guards Steps 4-8 products (shards, mixes, .done_*).
# num_npu deliberately NOT fingerprinted — parallel shape only, see
# runs/lib/stage_gate.sh. Params are split by which stage consumes them;
# shared params (data mix, eval sets, dtype, data dirs) enter BOTH stages.
FP_SEARCH_PARAMS=(
    "proxy_depth=$PROXY_DEPTH"
    "proxy_num_iterations=$PROXY_NUM_ITERATIONS"
    "proxy_target_tokens=$PROXY_TARGET_TOKENS"
    "configs_per_iter=$CONFIGS_PER_ITER"
    # 期望列表语义开关是搜索语义的一部分 (进指纹); TARGET_ARM_MODE /
    # REMOTE_* 是执行形态, 刻意不进 (num_npu 先例)。
    "adaptive_configs=$ADAPTIVE_CONFIGS"
    "adaptive_compact=$ADAPTIVE_COMPACT"
    "search_num_iterations=$SEARCH_NUM_ITERATIONS"
    "K_enhanced=$K_ENHANCED"
    "K_cluster_max=$K_CLUSTER_MAX"
    "K_init=$K_INIT"
    "filter_method=$FILTER_METHOD"
    "prune_threshold=$PRUNE_THRESHOLD"
    "prune_column_floor=$PRUNE_COLUMN_FLOOR"
    "merge_distance=$MERGE_DISTANCE"
    "merge_strategy=$MERGE_STRATEGY"
    "embedding_model=$EMBEDDING_MODEL"
    "discovery_method=$DISCOVERY_METHOD"
    "embedding_device=$EMBEDDING_DEVICE"
    "embedding_sample_size=$EMBEDDING_SAMPLE_SIZE"
    "proxy_lr_scale=$PROXY_LR_SCALE"
    "proxy_warmup=$PROXY_WARMUP"
    "proxy_warmdown=$PROXY_WARMDOWN"
    "npu_per_exp=$NPU_PER_EXP"
    "stem_ratio=$STEM_RATIO"
    "eval_benchmarks=$EVAL_BENCHMARKS"
    "eval_max_per_task=$EVAL_MAX_PER_TASK"
    "nanochat_dtype=$NANOCHAT_DTYPE"
    "data_dir=$DATA_DIR"
    "general_data_dir=$GENERAL_DATA_DIR"
)
FP_TARGET_PARAMS=(
    "target_depth=$TARGET_DEPTH"
    "target_steps=$TARGET_STEPS"
    "target_tokens=$TARGET_TOKENS"
    "target_lr_scale=$TARGET_LR_SCALE"
    "target_warmup=$TARGET_WARMUP"
    "target_warmdown=$TARGET_WARMDOWN"
    "mid_device_batch_size=$MID_DEVICE_BATCH_SIZE"
    "mid_train_loader=$MID_TRAIN_LOADER"
    # 优化器加载语义 (TARGET_ARM_NODES>1 派生出 0=冷启动): 改变训练语义,
    # 必须进指纹 —— 否则旧 .done 会跳过重新训练。nodes 本身不进 (执行形态,
    # num_npu/TARGET_ARM_MODE 先例)。
    "target_load_optimizer=$TARGET_LOAD_OPTIMIZER"
    "eval_device_batch_size=$EVAL_DEVICE_BATCH_SIZE"
    "eval_core_batch_size=$EVAL_CORE_BATCH_SIZE"
    "core_metric_every=$CORE_METRIC_EVERY"
    "stem_ratio=$STEM_RATIO"
    "eval_benchmarks=$EVAL_BENCHMARKS"
    "eval_max_per_task=$EVAL_MAX_PER_TASK"
    "nanochat_dtype=$NANOCHAT_DTYPE"
    "data_dir=$DATA_DIR"
    "general_data_dir=$GENERAL_DATA_DIR"
)

source "$CLIMBMIX_DIR/runs/lib/stage_gate.sh"
run_stage_gate

# ── Target-arm shared lib + launch env snapshot ──
# target_arm.sh 是本地臂路径 (Step 6+7 / 远端兜底) 的唯一 argv 来源,
# 与 dispatch_target_arm.py 的 python 构建器逐 token 对齐 (测试保证)。
source "$CLIMBMIX_DIR/runs/lib/target_arm.sh"
export TARGET_BASE_CKPT="$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}"
# launch_env.json: 独立发射的 dispatch 进程 (nohup ... --arm random) 读它
# 获得全部所需变量 — 不依赖启动 shell 的环境传递。每次发射刷新。
export EXP_NAME DATA_DIR CLIMBMIX_DIR NANOCHAT_DIR NANOCHAT_BASE_DIR \
       GENERAL_DATA_DIR PROXY_DEPTH TARGET_DEPTH TARGET_STEPS TARGET_TOKENS \
       TARGET_LR_SCALE TARGET_WARMUP TARGET_WARMDOWN CORE_METRIC_EVERY \
       MID_DEVICE_BATCH_SIZE MID_TRAIN_LOADER EVAL_BENCHMARKS \
       EVAL_MAX_PER_TASK EVAL_DEVICE_BATCH_SIZE EVAL_CORE_BATCH_SIZE \
       STEM_RATIO NUM_NPU NPU_PER_EXP K_ENHANCED HF_ENDPOINT \
       REMOTE_D28_ASSET_URI NANOCHAT_DTYPE OUTPUT_DIR \
       TARGET_ARM_NODES TARGET_LOAD_OPTIMIZER
python3 - "$OUTPUT_DIR/launch_env.json" "$TARGET_BASE_CKPT" <<'PYEOF'
import json, os, sys
out, target_base_ckpt = sys.argv[1], sys.argv[2]
keys = ["EXP_NAME", "DATA_DIR", "CLIMBMIX_DIR", "NANOCHAT_DIR",
        "NANOCHAT_BASE_DIR", "GENERAL_DATA_DIR", "PROXY_DEPTH", "TARGET_DEPTH",
        "TARGET_STEPS", "TARGET_TOKENS", "TARGET_LR_SCALE", "TARGET_WARMUP",
        "TARGET_WARMDOWN", "CORE_METRIC_EVERY", "MID_DEVICE_BATCH_SIZE",
        "MID_TRAIN_LOADER", "EVAL_BENCHMARKS", "EVAL_MAX_PER_TASK",
        "EVAL_DEVICE_BATCH_SIZE", "EVAL_CORE_BATCH_SIZE", "STEM_RATIO",
        "NUM_NPU", "NPU_PER_EXP", "K_ENHANCED", "HF_ENDPOINT",
        "REMOTE_D28_ASSET_URI", "NANOCHAT_DTYPE", "OUTPUT_DIR",
        "TARGET_ARM_NODES", "TARGET_LOAD_OPTIMIZER"]
env = {k: os.environ.get(k, "") for k in keys}
env["TARGET_BASE_CKPT"] = target_base_ckpt
with open(out, "w") as f:
    json.dump(env, f, indent=2)
PYEOF

# ── Pre-flight ──
# Remote fleet config generation (REMOTE_* -> RemoteConfig JSON). Execution-
# shape only; deliberately absent from FP_SEARCH_PARAMS (num_npu precedent).
REMOTE_CONFIG_ARG=""
if [ "$REMOTE_ENABLED" = "1" ]; then
    case "$REMOTE_OBS_PREFIX" in
        obs://*) ;;
        "") echo "✗ REMOTE_ENABLED=1 requires REMOTE_OBS_PREFIX (obs://bucket/prefix)"; exit 1 ;;
        *) echo "✗ REMOTE_OBS_PREFIX must start with obs:// (got: $REMOTE_OBS_PREFIX)"; exit 1 ;;
    esac
    mkdir -p "$OUTPUT_DIR"
    REMOTE_CONFIG_PATH="$OUTPUT_DIR/remote_config.json"
    REMOTE_CONFIG_ARG="--remote-config $REMOTE_CONFIG_PATH"
    # Export for the config-gen heredoc below (namespaced, harmless).
    export REMOTE_OBS_PREFIX REMOTE_BACKEND REMOTE_BACKEND_MODULE \
           REMOTE_PLATFORM_CONFIG REMOTE_IMAGE REMOTE_FLAVOR \
           REMOTE_POOL_NAME REMOTE_NPU_PER_JOB REMOTE_MAX_JOBS \
           REMOTE_SUBMIT_RETRY_H REMOTE_MAX_PREP REMOTE_LOCAL_PARALLEL \
           REMOTE_STORAGE_KIND REMOTE_STORAGE_ROOT REMOTE_JOB_TIMEOUT_H \
           REMOTE_QUEUE_TIMEOUT_H REMOTE_QUEUE_RETRY REMOTE_PENDING_GRACE_MIN \
           REMOTE_CODE_WHEELS REMOTE_ASSET_MOUNTS
    python3 - "$REMOTE_CONFIG_PATH" "$REMOTE_PLATFORM_CONFIG" "$REMOTE_IMAGE" "$REMOTE_FLAVOR" <<'PYEOF'
import json, sys, os
cfg_path, platform_config, image, flavor = sys.argv[1:5]
cfg = {
    "obs_prefix": os.environ["REMOTE_OBS_PREFIX"],
    "backend": os.environ["REMOTE_BACKEND"],
    "backend_module": os.environ["REMOTE_BACKEND_MODULE"],
    "platform_config": platform_config,
    "image": image,
    "flavor": flavor,
    "pool_name": os.environ["REMOTE_POOL_NAME"],
    "npu_per_job": int(os.environ["REMOTE_NPU_PER_JOB"]),
    "max_concurrent_jobs": int(os.environ["REMOTE_MAX_JOBS"]),
    "submit_retry_timeout_s": float(os.environ["REMOTE_SUBMIT_RETRY_H"]) * 3600.0,
    "max_prep_parallel": int(os.environ["REMOTE_MAX_PREP"]),
    "local_parallel": os.environ["REMOTE_LOCAL_PARALLEL"] == "1",
    "storage_kind": os.environ["REMOTE_STORAGE_KIND"],
    "storage_root": os.environ["REMOTE_STORAGE_ROOT"],
    "job_timeout_s": float(os.environ["REMOTE_JOB_TIMEOUT_H"]) * 3600.0,
    "queue_timeout_s": float(os.environ["REMOTE_QUEUE_TIMEOUT_H"]) * 3600.0,
    "queue_resubmit_attempts": int(os.environ["REMOTE_QUEUE_RETRY"]),
    "pending_grace_min": float(os.environ["REMOTE_PENDING_GRACE_MIN"]),
    "job_env": {"HF_ENDPOINT": os.environ["HF_ENDPOINT"]},
}
wheels = [w for w in (os.environ.get("REMOTE_CODE_WHEELS") or "").split(",") if w]
if wheels:
    cfg["code_wheels"] = wheels
am_raw = os.environ.get("REMOTE_ASSET_MOUNTS") or ""
if am_raw.strip():
    am = json.loads(am_raw)
    if (not isinstance(am, dict)
            or not all(isinstance(k, str) and k
                       and isinstance(v, str) and v.startswith("obs://")
                       for k, v in am.items())):
        print(f"✗ REMOTE_ASSET_MOUNTS must be a JSON object "
              f"{{name: obs://...}} (got: {am_raw[:200]})", file=sys.stderr)
        sys.exit(1)
    cfg["asset_mounts"] = am
if os.environ["REMOTE_BACKEND"] != "mock" and os.environ["REMOTE_STORAGE_KIND"] != "local":
    # Real backend fail-fast: resolve the backend bundle + run its
    # validate() HERE, not mid-search (gateway/auth/image mistakes die
    # at launch with a clear message; values are never printed).
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    from climbmix.remote.backends import resolve_backend
    from climbmix.remote.remote_executor import RemoteConfig
    rc = RemoteConfig.from_dict(cfg)
    bundle = resolve_backend(rc)
    if bundle.validate:
        bundle.validate(rc)
    print("  backend resolved + platform config OK (secrets not printed)")
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2)
PYEOF
    echo "  Remote fleet: ${REMOTE_MAX_JOBS} jobs x ${REMOTE_NPU_PER_JOB} NPU (backend=${REMOTE_BACKEND}, prefix=${REMOTE_OBS_PREFIX})"
    if [ "$REMOTE_LOCAL_PARALLEL" = "1" ]; then
        if [ "$NPU_PER_EXP" -lt 1 ] || [ "$NPU_PER_EXP" -gt "$NUM_NPU" ] || [ $((NUM_NPU % NPU_PER_EXP)) -ne 0 ]; then
            echo "  ⚠ REMOTE_LOCAL_PARALLEL=1 but NPU_PER_EXP=${NPU_PER_EXP} does not slice NUM_NPU=${NUM_NPU}: master-node NPUs will IDLE (need a divisor of NUM_NPU, <= NUM_NPU)"
        elif [ "$REMOTE_NPU_PER_JOB" != "$NPU_PER_EXP" ]; then
            echo "  ⚠ remote k (REMOTE_NPU_PER_JOB=$REMOTE_NPU_PER_JOB) != local k (NPU_PER_EXP=$NPU_PER_EXP): k should stay fleet-wide fixed for score comparability"
        elif [ "$NPU_PER_EXP" -eq "$NUM_NPU" ]; then
            echo "  Hybrid fleet: local 1 x ${NPU_PER_EXP}-NPU whole-node slot (serial full-card path) + ${REMOTE_MAX_JOBS} remote jobs x ${REMOTE_NPU_PER_JOB} NPU"
        else
            echo "  Hybrid fleet: local $((NUM_NPU / NPU_PER_EXP)) x ${NPU_PER_EXP} NPU + ${REMOTE_MAX_JOBS} remote jobs x ${REMOTE_NPU_PER_JOB} NPU"
        fi
    fi
fi

echo -e "\n════════════════════════════════════════════════════════════"
echo "  ClimbMix: d${PROXY_DEPTH} proxy → d${TARGET_DEPTH} target  |  $OUTPUT_DIR"
echo "  NPU: ${NUM_NPU}x910B4, npu_per_exp=${NPU_PER_EXP} ($((NUM_NPU / NPU_PER_EXP)) parallel)"
echo "════════════════════════════════════════════════════════════"

python3 -c "import torch_npu; import torch; assert torch.npu.is_available(), 'NPU not available'" || { echo "✗ NPU not available"; exit 1; }
[ -d "$NANOCHAT_DIR" ] || { echo "✗ nanochat-npu not found at $NANOCHAT_DIR"; exit 1; }
for d in "$PROXY_DEPTH" "$TARGET_DEPTH"; do
    ckpt="$NANOCHAT_BASE_DIR/base_checkpoints/d${d}"
    [ -d "$ckpt" ] && ls "$ckpt"/model_*.pt >/dev/null 2>&1 || { echo "✗ d${d} checkpoint not found"; exit 1; }
    echo "✓ d${d} checkpoint"
done

( cd "$NANOCHAT_DIR" && python3 -c "from scripts.base_eval import prepare_eval_data; prepare_eval_data('stem')" 2>/dev/null ) || true

# ═══════════════════════════════════════════════════════════════════════
#  Step 1-3: Embedding + Proxy Search + Data Selection
#  (embedding 分片级续跑 + 聚类缓存 + search_state 迭代级续跑 +
#   exp_*/meta.json 实验级复用 — 均自动)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$OUTPUT_DIR/sampled_dataset.parquet" ]; then
    echo -e "\n===== Step 1-3: Proxy Search — already complete (sampled_dataset.parquet), skip =====\n"
else
    echo -e "\n===== Step 1-3: Proxy Search (d${PROXY_DEPTH}) =====\n"

    ADAPTIVE_ARGS=()
    [ "$ADAPTIVE_CONFIGS" = "1" ] && ADAPTIVE_ARGS+=(--adaptive-configs)
    [ "$ADAPTIVE_COMPACT" = "1" ] && ADAPTIVE_ARGS+=(--adaptive-compact)
    python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
        --data-dir "$DATA_DIR" \
        --nanochat-dir "$NANOCHAT_DIR" \
        --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
        --general-data-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" \
        --eval-benchmarks "$EVAL_BENCHMARKS" \
        --eval-max-per-task "$EVAL_MAX_PER_TASK" \
        --proxy-depth "$PROXY_DEPTH" \
        --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
        --proxy-target-tokens "$PROXY_TARGET_TOKENS" \
        --proxy-lr-scale "$PROXY_LR_SCALE" --proxy-warmup "$PROXY_WARMUP" --proxy-warmdown "$PROXY_WARMDOWN" \
        --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
        --target-depth "$TARGET_DEPTH" \
        --target-tokens "$TARGET_TOKENS" \
        --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
        --K-enhanced "$K_ENHANCED" \
        --K-max "$K_CLUSTER_MAX" \
        --K-init "$K_INIT" \
        --filter-method "$FILTER_METHOD" \
        --prune-threshold "$PRUNE_THRESHOLD" \
        --prune-column-floor "$PRUNE_COLUMN_FLOOR" \
        --merge-distance "$MERGE_DISTANCE" \
        --merge-strategy "$MERGE_STRATEGY" \
        --embedding-model "$EMBEDDING_MODEL" \
        --num-iterations "$SEARCH_NUM_ITERATIONS" \
        --discovery-method "$DISCOVERY_METHOD" \
        --embedding-device "$EMBEDDING_DEVICE" \
        --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
        --configs-per-iter "$CONFIGS_PER_ITER" \
        ${ADAPTIVE_ARGS[@]+"${ADAPTIVE_ARGS[@]}"} \
        --device-type npu --npu-devices "$NUM_NPU" --npu-per-exp "$NPU_PER_EXP" \
        --output-dir "$OUTPUT_DIR" \
        --exp-name "$EXP_NAME" \
        --cluster-cache-dir "$OUTPUT_DIR" \
        --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
        --resume-search \
        --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
        $REMOTE_CONFIG_ARG \
        --skip-target 2>&1 | tee "$OUTPUT_DIR/search.log"
fi

[ ! -f "$OUTPUT_DIR/sampled_dataset.parquet" ] && { echo "✗ No sampled_dataset.parquet"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Prepare Target Data (shards + random baseline)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 4: Prepare Target Data =====\n"

CLIMB_SHARDS="$OUTPUT_DIR/climb_shards"
RANDOM_SHARDS="$OUTPUT_DIR/random_shards"

# Random 臂的数据准备 (random baseline + mix) 与独立发射的 dispatch 进程
# (scripts/dispatch_target_arm.py --arm random, 搜索期间提前并行) 共享同一
# 组 .done 产物 — flock 串行化, 双方都做 .done 双检, 后到者秒过。
random_arm_lock() {
    if command -v flock >/dev/null 2>&1; then
        ( flock -x 9; "$@" ) 9>"$OUTPUT_DIR/.random_arm.lock"
    else
        echo "  (flock unavailable — random-arm prep not serialized with dispatch)"
        "$@"
    fi
}

prep_random_baseline() {
    if [ -f "$RANDOM_SHARDS/.done" ]; then
        echo "  Random baseline: already complete (.done), skip"
        return
    fi
    # Paper App. C.1: equal uniform cluster weights (1/K), same token cap as
    # the CLIMB arm — NOT a doc-uniform draw (that would weight clusters by
    # their natural size). Shortfall policy mirrors the CLIMB arm's selector.
    python3 "$CLIMBMIX_DIR/scripts/prepare_random_baseline.py" \
        --data-dir "$DATA_DIR" --output-dir "$RANDOM_SHARDS" \
        --cluster-cache "$OUTPUT_DIR/cluster_cache.npz" \
        --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
        --target-tokens "$TARGET_TOKENS" \
        --seed 42 --num-npu "$NUM_NPU"
}

python3 "$CLIMBMIX_DIR/scripts/prepare_shards.py" \
    --input "$OUTPUT_DIR/sampled_dataset.parquet" \
    --output-dir "$CLIMB_SHARDS" --num-npu "$NUM_NPU"

random_arm_lock prep_random_baseline

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Mix STEM + General Data (anti-forgetting)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 5: Mix STEM + General Data (ratio=$STEM_RATIO) =====\n"

mix_one() {
    local stem_dir="$1" out_dir="$2" label="$3"
    [ -d "$stem_dir" ] || return 0
    [ -f "$out_dir/.done" ] && { echo "  $label: already mixed (.done), skip"; return; }
    NANOCHAT_REPO="$NANOCHAT_DIR" python3 "$CLIMBMIX_DIR/scripts/mix_general_data.py" \
        --stem-dir "$stem_dir" --output-dir "$out_dir" \
        --climbmix-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" --num-workers "$NUM_NPU" --num-npu "$NUM_NPU" \
        || { echo "✗ Mix failed for $label"; exit 1; }
}

mix_one "$CLIMB_SHARDS" "$OUTPUT_DIR/climb_mixed" "CLIMB"
random_arm_lock mix_one "$RANDOM_SHARDS" "$OUTPUT_DIR/random_mixed" "Random"
CLIMB_DATA="$OUTPUT_DIR/climb_mixed"
RANDOM_DATA="$OUTPUT_DIR/random_mixed"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6+7: Target Arms — train + eval, per arm, interleaved (S1)
#
#  每臂三层执行: .done 标记 → 远端 dispatch (TARGET_ARM_MODE=remote 且
#  REMOTE_ENABLED=1 时, scripts/dispatch_target_arm.py; 远端作业内完成
#  训练+评测并落标记) → 本地 torchrun 兜底 (runs/lib/target_arm.sh)。
#  训完立刻评 (S1 交错: eval 结果提前 ~10h 可见); eval 也标记级幂等 —
#  远端已评过的臂直接跳过。random 在前: 独立发射的 dispatch
#  (--arm random) 通常已在搜索期间完成它, 双臂均秒过或各走各的路径。
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6+7: Target Arms (d${TARGET_DEPTH}) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_climb_${EXP_NAME}"
RANDOM_TAG="d${TARGET_DEPTH}_random_${EXP_NAME}"

run_arm() {
    local data_dir="$1" tag="$2" name="$3"
    if [ -f "$OUTPUT_DIR/.done_mid_train_$name" ]; then
        echo "  mid_train $name: already done, skip"
    else
        # 单遍 (epoch<=1) 守卫: 消耗 (TARGET_STEPS × total_batch_size) 不得
        # 超过混合池实际 token 量 — nanochat loader 跑完会静默循环重采,
        # 破坏论文单遍退火语义且两臂不对称。发射/兜底训练前统一拦截
        # (见 scripts/check_single_pass.py)。
        if ! python3 "$CLIMBMIX_DIR/scripts/check_single_pass.py" \
            --data-dir "$data_dir" \
            --num-iterations "$TARGET_STEPS" \
            --ckpt-dir "$TARGET_BASE_CKPT" \
            --stem-ratio "$STEM_RATIO" \
            --context "$name arm"; then
            echo "✗ [$name] single-pass guard failed — refusing to train"
            exit 1
        fi
        if [ "$TARGET_ARM_MODE" = "remote" ] && [ "$REMOTE_ENABLED" = "1" ]; then
            echo "  [$name] remote target-arm dispatch (waits for the job)..."
            if python3 "$CLIMBMIX_DIR/scripts/dispatch_target_arm.py" \
                --arm "$name" --data-dir "$data_dir" --tag "$tag"; then
                touch "$OUTPUT_DIR/.done_mid_train_$name"
            else
                echo "  [$name] remote dispatch did not complete -> local fallback"
            fi
        fi
        if [ ! -f "$OUTPUT_DIR/.done_mid_train_$name" ]; then
            target_arm_train "$data_dir" "$tag" "$name"
            touch "$OUTPUT_DIR/.done_mid_train_$name"
        fi
    fi
    # S1 interleaved eval: right after THIS arm's train. 远端作业内已完成
    # 评测的臂 (dispatch 落了 .done_eval_<name>) 直接跳过。
    if [ -f "$OUTPUT_DIR/.done_eval_$name" ]; then
        echo "  eval $name: already done, skip"
    else
        target_arm_eval "$tag" "$name"
        touch "$OUTPUT_DIR/.done_eval_$name"
    fi
}

run_arm "$RANDOM_DATA" "$RANDOM_TAG" "random"
run_arm "$CLIMB_DATA" "$CLIMB_TAG" "climb"

# ═══════════════════════════════════════════════════════════════════════
#  Step 8: Report (幂等, 总是重新生成)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 8: Report =====\n"

python3 "$CLIMBMIX_DIR/src/climbmix/pipeline/report_generator.py" \
    --result-dir "$OUTPUT_DIR" \
    --climb-train-log "$OUTPUT_DIR/mid_train_climb.log" \
    --random-train-log "$OUTPUT_DIR/mid_train_random.log" \
    --climb-eval-log "$OUTPUT_DIR/eval_climb.log" \
    --random-eval-log "$OUTPUT_DIR/eval_random.log" \
    --base-model-tag "d${TARGET_DEPTH}" \
    --climb-model-tag "$CLIMB_TAG" \
    --random-model-tag "$RANDOM_TAG"

echo -e "\n════════════════════════════════════════════════════════════"
echo "  Done! → $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

# 正常跑完 → 活跃目录转已完成形态 (result/${EXP_NAME}_<ts>); 缺终态标记则保持活跃
mark_completed
