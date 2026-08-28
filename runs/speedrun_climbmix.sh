#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix Speedrun — 全流程端到端验证 (最小数据 + 最少步数)
#
#  目的: 验证代码正确性 (embedding → cluster → search → mid_train → eval)
#  不关注结果质量, 只确认所有步骤跑通无报错
#
#  设置:
#    - Step 0 从 DATA_DIR 流式切出 ${SPEED_SHARDS}×${SPEED_SHARD_DOCS} docs
#      (默认 10×100K = 1M), 其中 20K 条子采样做 embedding/聚类
#      (merge 超参校准用, EMBEDDING_SAMPLE_SIZE 可调)
#    - configs=6,6,6 (18 个; N≥10 解锁验证集切分+早停路径), proxy_steps=50, target_steps=50
#    - 8 NPU 做 embedding, 1 NPU/experiment 做 proxy search
#    - 70% STEM + 30% ClimbMix general data (含 proxy 实验内混合 + Step 5 混合;
#      首次运行会下载 3 个 ClimbMix 分片到 GENERAL_DATA_DIR, 之后复用缓存;
#      该缓存也供 full run 复用)。不做 random baseline。
#    - Proxy eval: 每实验私有 NANOCHAT_BASE_DIR (symlink farm) → eval 并行;
#      --max-per-task=$EVAL_MAX_PER_TASK 抽样 (固定种子, 各实验同一子集)
#
#  用法:  bash runs/speedrun_climbmix.sh
#
#  断点续跑: 直接重跑同一命令。阶段指纹匹配 → 自动续跑;
#    search 指纹变 (代码/参数) → 归档为 result/${EXP_NAME}_stale_search_<ts> 后全新开始
#    target 指纹变 → 仅归档 target 产物 (result/${EXP_NAME}_stale_target_<ts>,
#    Steps 4-8 重跑, 搜索结果保留)
#  (改代码后无需手动 rm -rf。强制全新: 换 EXP_NAME 或 rm -rf。)
#  生命周期: 活跃 = result/${EXP_NAME}_current; 正常跑完自动改名
#    result/${EXP_NAME}_<ts> (重跑同命令 → 自动恢复已完成 run, 全程跳过);
#    每个归档目录带 archive_meta.json (原因/时间/指纹/git HEAD)。
#  旧版单一 .fingerprint 目录: MIGRATE_LEGACY_FINGERPRINT=1 采纳(不校验),
#  否则归档。num_npu 不进指纹(并行形状可变, 见 runs/lib/stage_gate.sh)。
# ═══════════════════════════════════════════════════════════════════════
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

# ── Configuration ──
CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${CLIMBMIX_DIR}/src:${PYTHONPATH:-}"

EXP_NAME="${EXP_NAME:-speedrun}"
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
SPEED_DATA="/tmp/speedrun_data"
# Step-0 池形态: 从 DATA_DIR 流式切出 N 个精确 X docs 的 shard (旧做法 =
# 原样复制前 5 个 part 文件, 行数不受控)。1M docs 扩大 mixture 采样池;
# pool-keyed embedding 缓存自动换键 (重新 embed 20K 条, 一次性)。
# 改任一值 → search 指纹变 → stage_gate 归档整个目录重跑 (search 下游
# 全依赖数据形态, 语义正确)。.spec 标记防不同形态的陈旧池被复用。
SPEED_SHARDS="${SPEED_SHARDS:-10}"
SPEED_SHARD_DOCS="${SPEED_SHARD_DOCS:-100000}"
NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
GENERAL_DATA_DIR="${GENERAL_DATA_DIR:-$NANOCHAT_BASE_DIR/climbmix_shards}"

PROXY_DEPTH=20
TARGET_DEPTH=28
PROXY_NUM_ITERATIONS=50
TARGET_STEPS=50
# 6,6,6 (=18) 而非 2,3,2: N≥10 才触发验证集切分 + lgb.early_stopping
# (iterative_bootstrapper._fit_predictor_with_val), 让唯一一次重跑
# 覆盖生产全部路径; 追求最快冒烟可 CONFIGS_PER_ITER=2,3,2 覆盖。
CONFIGS_PER_ITER="${CONFIGS_PER_ITER:-6,6,6}"
TOTAL_CONFIGS="$(echo "$CONFIGS_PER_ITER" | awk -F, '{s=0;for(i=1;i<=NF;i++)s+=$i;print s}')"
SEARCH_NUM_ITERATIONS="${SEARCH_NUM_ITERATIONS:-3}"
K_ENHANCED="${K_ENHANCED:-3}"
K_CLUSTER_MAX="${K_CLUSTER_MAX:-15}"
K_INIT="${K_INIT:-100}"
FILTER_METHOD="${FILTER_METHOD:-none}"
PRUNE_THRESHOLD="${PRUNE_THRESHOLD:-3.0}"
MERGE_DISTANCE="${MERGE_DISTANCE:-0.9}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
# Stable pool-keyed cache for embeddings + K-means (survives fingerprint
# resets; K/merge knob changes reuse embeddings instead of re-embedding)
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$CLIMBMIX_DIR/cache/embeddings}"
# Training dynamics (semantic: change → fingerprint → fresh run)
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
# 生产若想升 dbs: 先看本 speedrun 日志的 "Peak memory usage" 实测余量。
MID_DEVICE_BATCH_SIZE="${MID_DEVICE_BATCH_SIZE:-1}"
# flat = 零裁剪文档打包 (DeepSeek V3 式), 与 proxy 搜索阶段及 quadmix 实验同口径
MID_TRAIN_LOADER="${MID_TRAIN_LOADER:-flat}"
# BPB-only 旋钮: base_eval 只在 bpb 分支读 --device-batch-size
# (base_eval.py:514/:522), 本流程 --eval=core 下是空操作; 32 是 8x910B3
# (64G HBM) 时代默认, 16 对齐 quadmix 同硬件 d28 实证值, 防将来开 bpb 踩坑。
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"
# core eval 的真实显存旋钮: --core-eval-batch-size (base_eval.py:417, 默认16)
# 把 chunk 内样本 pad 到最长序列一次 forward (峰值主体是 logits B×T×V)。
# 2026-08-28 Step-7 OOM 实证: max_per_task=100 → 每卡 13 条 → 默认 16 下
# 单块 forward 顶满 torch 池 (~24.5G), arc_easy 任务末尾 dist.barrier() 处
# HCCL 申请 401MiB allreduce 通信缓冲失败 (EL0004, allocator 记账之外,
# core_eval.py:412; 每任务后的 empty_cache 在 barrier 之后才跑)。
# 8x910B3(64G)→8x910B4(32G) 显存减半 → batch 同步减半 16→8。
EVAL_CORE_BATCH_SIZE="${EVAL_CORE_BATCH_SIZE:-8}"
CORE_METRIC_EVERY="${CORE_METRIC_EVERY:--1}"
NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
NUM_NPU=8
NPU_PER_EXP=1
# Embedding/聚类子采样池大小。τ (merge_distance) / floor / cap 的校准
# 依据是 merge_profile.json 里的质心距离树状图 (K vs 最近对距离) —
# 2000 条下 K_init=100 平均每簇 20 条文档, 距离分布失真, 树状图没有
# 校准价值; 20000 条 (每簇 ~200) 才能让 merge/prune 动态接近真实池。
# 改此值 → search 指纹变 (Steps 1-3 重跑) + embedding 缓存池键变
# (重新 embed 20000 条, 一次性, 之后复用)。
EMBEDDING_SAMPLE_SIZE="${EMBEDDING_SAMPLE_SIZE:-20000}"
EVAL_BENCHMARKS="stem"
# Eval subsample cap per task (speedrun keeps proxy evals cheap; the fixed
# shuffle seed 1337 inside base_eval keeps scores comparable across exps).
# Full sets = -1 (what the production run uses).
EVAL_MAX_PER_TASK="${EVAL_MAX_PER_TASK:-100}"
STEM_RATIO="${STEM_RATIO:-0.7}"
PROXY_TARGET_TOKENS=10M
TARGET_TOKENS=10M

OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/result/${EXP_NAME}_current}"
# 终态标记: 全部存在 => run 完整跑完, 末尾 mark_completed 把活跃目录
# 改名为已完成形态 result/${EXP_NAME}_<ts> (stage_gate.sh 生命周期)
COMPLETION_MARKERS=(".done_eval_climb")

# ── HF download endpoint ──
# The corporate proxy (proxy.modelarts.com) selectively rejects Python's bare
# CONNECT tunnels to huggingface.co (observed: 90+ consecutive 503s across two
# independent runs / 80 min, while curl to the same host AND Python to
# hf-mirror.com both succeeded). hf-mirror.com serves the same bytes (Range
# resume verified, 206). Covers ClimbMix shards + eval_stem.zip (nanochat
# reads HF_ENDPOINT at import time in dataset.py AND base_eval.py).
# Override to use the origin: HF_ENDPOINT=https://huggingface.co bash runs/...
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── NPU Environment (deliberately minimal — matches proven train_base_model.sh) ──
# The 2026-08-26 speedrun OOM'd with the allocator block copied from
# nanochat-npu/runs/speedrun.sh (TORCH_NPU_ALLOC_CONF=...,memory_pool:True,
# PYTORCH_NPU_ALLOC_MAX_SIZE=60G, ASCEND_ENABLE_CACHE, ASCEND_DISABLE_MEM_SWAP,
# PYTHONPRELOAD — proven there only at device-batch-size=2). Symptom: device 0
# full (29.4G/29.5G, 4 MiB free) while torch_npu stats showed just 3.3 GiB
# allocated/reserved → memory swallowed outside the allocator's accounting →
# kernel loads failed (aclnnMean 207001 / EL0004) in every exp AND Step 6;
# npu-smi clean after exit (live processes, not ghost memory). Do not re-add
# allocator overrides unless a specific need is proven on this hardware.
export OMP_NUM_THREADS=1 WANDB_MODE=offline NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
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
# search guards Steps 0-3 products (embedding/cluster/search_state/exp_*/
# sampled_dataset); target guards Steps 4-8 products (shards, mixes, .done_*).
# num_npu deliberately NOT fingerprinted — parallel shape only, see
# runs/lib/stage_gate.sh. Params are split by which stage consumes them;
# shared params (data mix, eval sets, dtype, data dirs) enter BOTH stages.
FP_SEARCH_PARAMS=(
    "proxy_depth=$PROXY_DEPTH"
    "proxy_num_iterations=$PROXY_NUM_ITERATIONS"
    "proxy_target_tokens=$PROXY_TARGET_TOKENS"
    "configs_per_iter=$CONFIGS_PER_ITER"
    "speed_shards=$SPEED_SHARDS"
    "speed_shard_docs=$SPEED_SHARD_DOCS"
    "search_num_iterations=$SEARCH_NUM_ITERATIONS"
    "K_enhanced=$K_ENHANCED"
    "K_cluster_max=$K_CLUSTER_MAX"
    "K_init=$K_INIT"
    "filter_method=$FILTER_METHOD"
    "prune_threshold=$PRUNE_THRESHOLD"
    "merge_distance=$MERGE_DISTANCE"
    "embedding_model=$EMBEDDING_MODEL"
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
echo -e "\n════════════════════════════════════════════════════════════"
echo "  ClimbMix Speedrun: d${PROXY_DEPTH} → d${TARGET_DEPTH}  |  $OUTPUT_DIR"
echo "  ${NUM_NPU}x910B4, ${NPU_PER_EXP} NPU/exp ($((NUM_NPU / NPU_PER_EXP)) parallel)"
echo "  Data: ${SPEED_SHARDS} shards × ${SPEED_SHARD_DOCS} docs | Configs: ${CONFIGS_PER_ITER} (${TOTAL_CONFIGS} total) | Steps: ${PROXY_NUM_ITERATIONS}"
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
#  Step 0: Prepare speedrun pool (${SPEED_SHARDS} shards × ${SPEED_SHARD_DOCS} docs)
#  流式读取 DATA_DIR/part-*.parquet, 按 20K 行 batch 累积, 每满
#  ${SPEED_SHARD_DOCS} 行写一个 shard (跨源文件无缝拼接, 峰值内存 ≈
#  一个 shard)。池不足 → 硬失败并列出可用行数。
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 0: Prepare speedrun pool (${SPEED_SHARDS} shards × ${SPEED_SHARD_DOCS} docs) =====\n"

cur_spec="$(cat "$SPEED_DATA/.spec" 2>/dev/null || true)"
if [ "$cur_spec" != "shards=${SPEED_SHARDS} docs=${SPEED_SHARD_DOCS}" ] \
   || [ "$(ls "$SPEED_DATA"/*.parquet 2>/dev/null | wc -l)" -lt "$SPEED_SHARDS" ]; then
    mkdir -p "$SPEED_DATA"
    rm -f "$SPEED_DATA"/*.parquet "$SPEED_DATA"/*.npz "$SPEED_DATA"/.spec 2>/dev/null
    python3 - "$DATA_DIR" "$SPEED_DATA" "$SPEED_SHARDS" "$SPEED_SHARD_DOCS" <<'PY'
import glob, os, sys
import pyarrow as pa
import pyarrow.parquet as pq

data_dir, out_dir = sys.argv[1], sys.argv[2]
n_shards, shard_docs = int(sys.argv[3]), int(sys.argv[4])

srcs = sorted(glob.glob(os.path.join(data_dir, "part-*.parquet")))
if not srcs:
    sys.exit(f"ERROR: no part-*.parquet under {data_dir}")

need = n_shards * shard_docs
schema = None
buf, buf_rows = [], 0
shard_i = 0
total_rows = 0

for f in srcs:
    if shard_i >= n_shards:
        break
    pf = pq.ParquetFile(f)
    total_rows += pf.metadata.num_rows
    for b in pf.iter_batches(batch_size=20000):
        if schema is None:
            schema = b.schema
        elif not b.schema.equals(schema, check_metadata=False):
            sys.exit(f"ERROR: schema mismatch in {f} vs earlier files")
        buf.append(b)
        buf_rows += b.num_rows
        while buf_rows >= shard_docs and shard_i < n_shards:
            table = pa.Table.from_batches(buf, schema=schema)
            out = os.path.join(out_dir, f"part-{shard_i:03d}.parquet")
            pq.write_table(table.slice(0, shard_docs), out, compression="zstd")
            print(f"  shard {shard_i:03d}/{n_shards}: {shard_docs} docs")
            tail = table.slice(shard_docs)
            buf, buf_rows = tail.to_batches(), tail.num_rows
            shard_i += 1
        if shard_i >= n_shards:
            break

if shard_i < n_shards:
    sys.exit(f"ERROR: pool under {data_dir} holds {total_rows} docs < "
             f"{need} required ({n_shards} x {shard_docs}); "
             f"lower SPEED_SHARDS/SPEED_SHARD_DOCS or point DATA_DIR elsewhere")
print(f"  pool ready: {shard_i} shards x {shard_docs} docs = {shard_i * shard_docs} docs")
PY
    echo "shards=${SPEED_SHARDS} docs=${SPEED_SHARD_DOCS}" > "$SPEED_DATA/.spec"
else
    echo "  Pool already prepared (spec match): $SPEED_DATA"
fi
echo "Speedrun data files:"
ls -lh "$SPEED_DATA"/*.parquet

# ═══════════════════════════════════════════════════════════════════════
#  Step 1-3: Embedding + Proxy Search + Data Selection
#  Tests: ProxyRunner._build_mid_train_cmd + _build_eval_cmd (fixed with --)
#  (search_state 迭代级续跑 + exp_*/meta.json 实验级复用 — 均自动)
# ═══════════════════════════════════════════════════════════════════════
if [ -f "$OUTPUT_DIR/sampled_dataset.parquet" ]; then
    echo -e "\n===== Step 1-3: Proxy Search — already complete, skip =====\n"
else
    echo -e "\n===== Step 1-3: Proxy Search (d${PROXY_DEPTH}, ${CONFIGS_PER_ITER} configs, ${PROXY_NUM_ITERATIONS} steps) =====\n"

    python3 "$CLIMBMIX_DIR/scripts/run_climb.py" \
        --data-dir "$SPEED_DATA" \
        --nanochat-dir "$NANOCHAT_DIR" \
        --nanochat-base-dir "$NANOCHAT_BASE_DIR" \
        --general-data-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" \
        --eval-benchmarks "$EVAL_BENCHMARKS" \
        --eval-max-per-task "$EVAL_MAX_PER_TASK" \
        --proxy-depth "$PROXY_DEPTH" \
        --proxy-num-iterations "$PROXY_NUM_ITERATIONS" \
        --proxy-lr-scale "$PROXY_LR_SCALE" --proxy-warmup "$PROXY_WARMUP" --proxy-warmdown "$PROXY_WARMDOWN" \
        --phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${PROXY_DEPTH}" \
        --target-depth "$TARGET_DEPTH" \
        --target-phase1-checkpoint-path "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" \
        --K-enhanced "$K_ENHANCED" \
        --K-max "$K_CLUSTER_MAX" \
        --K-init "$K_INIT" \
        --filter-method "$FILTER_METHOD" \
        --prune-threshold "$PRUNE_THRESHOLD" \
        --merge-distance "$MERGE_DISTANCE" \
        --embedding-model "$EMBEDDING_MODEL" \
        --num-iterations "$SEARCH_NUM_ITERATIONS" \
        --discovery-method embedding_cluster \
        --embedding-device npu \
        --embedding-sample-size "$EMBEDDING_SAMPLE_SIZE" \
        --configs-per-iter "$CONFIGS_PER_ITER" \
        --device-type npu --npu-devices "$NUM_NPU" --npu-per-exp "$NPU_PER_EXP" \
        --output-dir "$OUTPUT_DIR" \
        --exp-name "$EXP_NAME" \
        --cluster-cache-dir "$OUTPUT_DIR" \
        --embedding-cache-dir "$EMBEDDING_CACHE_DIR" \
        --resume-search \
        --proxy-target-tokens "$PROXY_TARGET_TOKENS" \
        --target-tokens "$TARGET_TOKENS" \
        --schema "$CLIMBMIX_DIR/config/schema_stem.yaml" \
        --skip-target
fi

[ ! -f "$OUTPUT_DIR/sampled_dataset.parquet" ] && { echo "✗ No sampled_dataset.parquet — Step 1-3 FAILED"; exit 1; }
echo -e "\n✓ Step 1-3 complete: sampled_dataset.parquet found"

# ═══════════════════════════════════════════════════════════════════════
#  Step 4: Prepare Target Data (shards only, skip random baseline)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 4: Prepare Target Data =====\n"

CLIMB_SHARDS="$OUTPUT_DIR/climb_shards"

python3 "$CLIMBMIX_DIR/scripts/prepare_shards.py" \
    --input "$OUTPUT_DIR/sampled_dataset.parquet" \
    --output-dir "$CLIMB_SHARDS" --num-npu "$NUM_NPU"

echo "✓ Shards prepared: $(ls "$CLIMB_SHARDS"/shard_*.parquet 2>/dev/null | wc -l) files"

# ═══════════════════════════════════════════════════════════════════════
#  Step 5: Mix STEM + General Data (anti-forgetting, ratio=$STEM_RATIO)
#  Tests: mix_general_data.py CLI (download cache + stream mixing + val 透传)
#  输出 shard 的 row group 按 NUM_NPU 自适应 (DDP 安全)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 5: Mix STEM + General Data (ratio=$STEM_RATIO) =====\n"

CLIMB_MIXED="$OUTPUT_DIR/climb_mixed"
if [ -f "$CLIMB_MIXED/.done" ]; then
    echo "  CLIMB mix: already complete (.done), skip"
else
    NANOCHAT_REPO="$NANOCHAT_DIR" python3 "$CLIMBMIX_DIR/scripts/mix_general_data.py" \
        --stem-dir "$CLIMB_SHARDS" --output-dir "$CLIMB_MIXED" \
        --climbmix-dir "$GENERAL_DATA_DIR" \
        --stem-ratio "$STEM_RATIO" --num-workers "$NUM_NPU" --num-npu "$NUM_NPU" \
        || { echo "✗ Mix failed"; exit 1; }
fi

CLIMB_DATA="$CLIMB_MIXED"

# ═══════════════════════════════════════════════════════════════════════
#  Step 6: Target Training (d28 mid_train, ${TARGET_STEPS} steps)
#  Tests: shell torchrun mid_train command (already has --)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 6: Target Training (d${TARGET_DEPTH}, ${TARGET_STEPS} steps) =====\n"

CLIMB_TAG="d${TARGET_DEPTH}_${EXP_NAME}"
link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$CLIMB_TAG"

if [ -f "$OUTPUT_DIR/.done_mid_train_climb" ]; then
    echo "  mid_train climb: already done, skip"
else
    # Clean a stale/broken symlink from a previous crashed attempt: a broken
    # link fails `[ -e ]` yet still blocks ln -s (EEXIST) under set -e.
    if [ -L "$link_dir" ] && [ ! -e "$link_dir" ]; then rm -f "$link_dir"; fi
    [ -e "$link_dir" ] || ln -s "$NANOCHAT_BASE_DIR/base_checkpoints/d${TARGET_DEPTH}" "$link_dir"
    # Clear partial checkpoints from a crashed attempt (whole-run atomicity)
    rm -rf "$NANOCHAT_BASE_DIR/mid_checkpoints/$CLIMB_TAG"

    # Step 6 = 单 8-rank torchrun (quadmix 验证过 env 块安全的唯一形态);
    # 并行搜索阶段绝不 source (2026-08-26 事故, 见文件头).
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
        --run="speedrun_climb" --model-tag="$CLIMB_TAG" \
        --data-dir="$CLIMB_DATA" 2>&1 | tee "$OUTPUT_DIR/mid_train_climb.log"
    ) || {
        echo "✗ Target mid_train FAILED"
        if [ -L "$link_dir" ]; then rm -f "$link_dir"; fi
        exit 1
    }
    # NOT `[ -L ] && rm`: a non-symlink link_dir would return 1 here, kill the
    # script under set -e, and leave .done unwritten → retrain on every resume.
    if [ -L "$link_dir" ]; then rm -f "$link_dir"; fi
    touch "$OUTPUT_DIR/.done_mid_train_climb"
fi
echo "✓ Target training complete"

# ═══════════════════════════════════════════════════════════════════════
#  Step 7: Evaluation
#  Tests: shell torchrun base_eval command (already has --)
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n===== Step 7: Evaluation =====\n"

if [ -f "$OUTPUT_DIR/.done_eval_climb" ]; then
    echo "  eval climb: already done, skip"
else
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --max-per-task="$EVAL_MAX_PER_TASK" \
        --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
        --core-eval-batch-size="$EVAL_CORE_BATCH_SIZE" \
        --model-tag="$CLIMB_TAG" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_climb.log"
    ) || {
        echo "✗ Eval FAILED"
        exit 1
    }
    touch "$OUTPUT_DIR/.done_eval_climb"
fi
echo "✓ Evaluation complete"

# ═══════════════════════════════════════════════════════════════════════
#  Summary
# ═══════════════════════════════════════════════════════════════════════
echo -e "\n════════════════════════════════════════════════════════════"
echo "  Speedrun Complete! All steps passed ✓"
echo "  Output: $OUTPUT_DIR"
echo ""
echo "  Verified code paths:"
echo "    ✓ Embedding (fallback path, 0% NaN)"
echo "    ✓ Clustering (FAISS K-means + merge)"
echo "    ✓ Proxy search (ProxyRunner: mix + mid_train + eval × ${TOTAL_CONFIGS} configs)"
echo "    ✓ Data selection (mixture weights → sampled_dataset)"
echo "    ✓ Shard preparation (prepare_shards.py)"
echo "    ✓ General data mixing (mix_general_data.py, stem_ratio=$STEM_RATIO)"
echo "    ✓ Target training (d${TARGET_DEPTH} mid_train, ${TARGET_STEPS} steps)"
echo "    ✓ Target evaluation (base_eval)"
echo ""
echo "  Output files:"
ls -lh "$OUTPUT_DIR"/*.parquet "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.log 2>/dev/null || echo "    (check output dir)"
echo "════════════════════════════════════════════════════════════"

# 正常跑完 → 活跃目录转已完成形态 (result/${EXP_NAME}_<ts>); 缺终态标记则保持活跃
mark_completed
