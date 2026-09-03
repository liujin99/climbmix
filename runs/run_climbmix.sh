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
PROXY_NUM_ITERATIONS="${PROXY_NUM_ITERATIONS:-1000}"
TARGET_STEPS="${TARGET_STEPS:-1000}"
# Token caps for data selection (full pool ≈ 100B tokens / 116M docs — NOT capped means
# every proxy exp would select the whole pool; that's the default 0, so always set these).
# Proxy: 400M tokens/exp = single-pass calibration (2026-08-28, TODO.md): training
# consumes 524M (1000 iters × 524,288); mix = 400M STEM + 171M general ≈ 571M ≥
# consumption → no silent loader cycling (200M gave epoch≈1.8; run-4 target log
# showed epoch:4). Paper's 800M is its proxy CONSUMPTION (~394 × 2M batch), not a
# cap; our 524M = 65% of paper (documented deviation, scoring_metric_design §12.3).
# Target: 1B tokens ≈ d28 anneal budget (1000 iters × ~1M) ≈ 1 epoch, no repetition.
PROXY_TARGET_TOKENS="${PROXY_TARGET_TOKENS:-400M}"
TARGET_TOKENS="${TARGET_TOKENS:-1B}"
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-20,10,5}"
SEARCH_NUM_ITERATIONS="${SEARCH_NUM_ITERATIONS:-3}"
K_ENHANCED="${K_ENHANCED:-3}"
K_CLUSTER_MAX="${K_CLUSTER_MAX:-15}"
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
#   - REMOTE_LOCAL_PARALLEL=1 (默认): 主节点本地卡也加入舰队 — 前
#     NUM_NPU/NPU_PER_EXP 个配置走本地并行, 其余远端作业, 全程并发
#   - REMOTE_NPU_PER_JOB 默认 = NPU_PER_EXP (k 全舰队一致, 分数可比性,
#     docs/parallel_k_selection.md); 想让本地卡出力需 NPU_PER_EXP < NUM_NPU
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
REMOTE_LOCAL_PARALLEL="${REMOTE_LOCAL_PARALLEL:-1}"
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
REMOTE_JOB_TIMEOUT_H="${REMOTE_JOB_TIMEOUT_H:-6}" # 单作业超时 (小时)
REMOTE_CODE_WHEELS="${REMOTE_CODE_WHEELS:-}"  # 离线 wheel 本地路径 (逗号分隔, executor 自动补传)

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
    "search_num_iterations=$SEARCH_NUM_ITERATIONS"
    "K_enhanced=$K_ENHANCED"
    "K_cluster_max=$K_CLUSTER_MAX"
    "K_init=$K_INIT"
    "filter_method=$FILTER_METHOD"
    "prune_threshold=$PRUNE_THRESHOLD"
    "prune_column_floor=$PRUNE_COLUMN_FLOOR"
    "merge_distance=$MERGE_DISTANCE"
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
           REMOTE_CODE_WHEELS
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
    "job_env": {"HF_ENDPOINT": os.environ["HF_ENDPOINT"]},
}
wheels = [w for w in (os.environ.get("REMOTE_CODE_WHEELS") or "").split(",") if w]
if wheels:
    cfg["code_wheels"] = wheels
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
        if [ "$NPU_PER_EXP" -lt 1 ] || [ "$NPU_PER_EXP" -ge "$NUM_NPU" ] || [ $((NUM_NPU % NPU_PER_EXP)) -ne 0 ]; then
            echo "  ⚠ REMOTE_LOCAL_PARALLEL=1 but NPU_PER_EXP=${NPU_PER_EXP} does not slice NUM_NPU=${NUM_NPU}: master-node NPUs will IDLE (need a proper divisor < NUM_NPU)"
        elif [ "$REMOTE_NPU_PER_JOB" != "$NPU_PER_EXP" ]; then
            echo "  ⚠ remote k (REMOTE_NPU_PER_JOB=$REMOTE_NPU_PER_JOB) != local k (NPU_PER_EXP=$NPU_PER_EXP): k should stay fleet-wide fixed for score comparability"
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
        --embedding-model "$EMBEDDING_MODEL" \
        --num-iterations "$SEARCH_NUM_ITERATIONS" \
        --discovery-method "$DISCOVERY_METHOD" \
        --embedding-device "$EMBEDDING_DEVICE" \
        --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
        --configs-per-iter "$CONFIGS_PER_ITER" \
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

python3 "$CLIMBMIX_DIR/scripts/prepare_shards.py" \
    --input "$OUTPUT_DIR/sampled_dataset.parquet" \
    --output-dir "$CLIMB_SHARDS" --num-npu "$NUM_NPU"

if [ -f "$RANDOM_SHARDS/.done" ]; then
    echo "  Random baseline: already complete (.done), skip"
else
    # Paper App. C.1: equal uniform cluster weights (1/K), same token cap as
    # the CLIMB arm — NOT a doc-uniform draw (that would weight clusters by
    # their natural size). Shortfall policy mirrors the CLIMB arm's selector.
    python3 "$CLIMBMIX_DIR/scripts/prepare_random_baseline.py" \
        --data-dir "$DATA_DIR" --output-dir "$RANDOM_SHARDS" \
        --cluster-cache "$OUTPUT_DIR/cluster_cache.npz" \
        --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
        --target-tokens "$TARGET_TOKENS" \
        --seed 42 --num-npu "$NUM_NPU"
fi

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
mix_one "$RANDOM_SHARDS" "$OUTPUT_DIR/random_mixed" "Random"
CLIMB_DATA="$OUTPUT_DIR/climb_mixed"
RANDOM_DATA="$OUTPUT_DIR/random_mixed"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6: Target Training (d28 mid_train) — climb/random 独立 .done 标记
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6: Target Training (d${TARGET_DEPTH}) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_climb_${EXP_NAME}"
RANDOM_TAG="d${TARGET_DEPTH}_random_${EXP_NAME}"

run_mid_train() {
    local data_dir="$1" tag="$2" name="$3"
    local link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
    # Clean a stale/broken symlink from a previous crashed attempt BEFORE the
    # `[ -e ] || ln -s`: a broken link fails `[ -e ]` yet still blocks ln -s
    # (EEXIST), which kills the script under set -e.
    if [ -L "$link_dir" ] && [ ! -e "$link_dir" ]; then rm -f "$link_dir"; fi
    [ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"
    # Clear partial checkpoints from a crashed attempt (nanochat may otherwise
    # try to auto-resume from inconsistent state; whole-run atomicity instead)
    rm -rf "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag"
    # Step 6 = 单 8-rank torchrun (quadmix 验证过 env 块安全的唯一形态);
    # 并行搜索阶段绝不 source (2026-08-26 事故, 见 speedrun 头部注释).
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$TARGET_STEPS" \
        --lr-scale="$TARGET_LR_SCALE" --warmup-ratio="$TARGET_WARMUP" --warmdown-ratio="$TARGET_WARMDOWN" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --device-batch-size="$MID_DEVICE_BATCH_SIZE" \
        --loader="$MID_TRAIN_LOADER" \
        --sample-every=-1 \
        --eval-every=-1 \
        --run="${name}_mid" --model-tag="$tag" \
        --data-dir="$data_dir" 2>&1 | tee "$OUTPUT_DIR/mid_train_${name}.log"
    )
    # NOT `[ -L ] && rm` as the last statement: when link_dir is absent or not
    # a symlink the function would return 1, and under set -e the script dies
    # AFTER successful training with .done unwritten → retrain on every resume.
    if [ -L "$link_dir" ]; then rm -f "$link_dir"; fi
}

if [ -f "$OUTPUT_DIR/.done_mid_train_climb" ]; then
    echo "  mid_train climb: already done, skip"
else
    run_mid_train "$CLIMB_DATA" "$CLIMB_TAG" "climb"
    touch "$OUTPUT_DIR/.done_mid_train_climb"
fi

if [ -f "$OUTPUT_DIR/.done_mid_train_random" ]; then
    echo "  mid_train random: already done, skip"
else
    run_mid_train "$RANDOM_DATA" "$RANDOM_TAG" "random"
    touch "$OUTPUT_DIR/.done_mid_train_random"
fi

# ═══════════════════════════════════════════════════════════════════════
#  Step 7: Evaluation
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 7: Evaluation =====\n"

run_eval() {
    local tag="$1" name="$2"
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --max-per-task="$EVAL_MAX_PER_TASK" \
        --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
        --core-eval-batch-size="$EVAL_CORE_BATCH_SIZE" \
        --model-tag="$tag" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_${name}.log"
    )
    # base_eval writes a step-only CSV name (mid_model_{step}.csv) into the
    # shared base dir; both arms train the same step count, so the second
    # eval would overwrite the first. Evals are sequential here — archive
    # the newest CSV per arm right after its eval (Step 8 reads the LOGS,
    # this preserves the raw 4-column CSVs for the final analysis).
    local newest
    newest=$(ls -t "$NANOCHAT_BASE_DIR"/base_eval/mid_model_*.csv 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        cp -f "$newest" "$OUTPUT_DIR/eval_${name}.csv"
        echo "  Archived $(basename "$newest") -> eval_${name}.csv"
    else
        echo "  WARNING: no mid_model_*.csv found after eval ${name}"
    fi
}

if [ -f "$OUTPUT_DIR/.done_eval_climb" ]; then
    echo "  eval climb: already done, skip"
else
    run_eval "$CLIMB_TAG" "climb"
    touch "$OUTPUT_DIR/.done_eval_climb"
fi

if [ -f "$OUTPUT_DIR/.done_eval_random" ]; then
    echo "  eval random: already done, skip"
else
    run_eval "$RANDOM_TAG" "random"
    touch "$OUTPUT_DIR/.done_eval_random"
fi

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
