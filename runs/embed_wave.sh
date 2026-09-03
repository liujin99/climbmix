#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: 多节点全池嵌入 — 波次发射 (TODO E; scripts/embed_dispatch.py 的壳)
#
#  用法:
#    POOL_URI=obs://<池目录> MODEL_URI=obs://<stella目录> \
#        bash runs/embed_wave.sh   # 全量波 (池分片 ÷ UNIT_SHARDS 个单元)
#    SMOKE=2 bash runs/embed_wave.sh          # 烟雾: 前 2 片单单元 + 本地逐字节比对
#    MAX_JOBS=8 bash runs/embed_wave.sh       # 并发单元数 (8×8=64 卡)
#    FORCE=1 bash runs/embed_wave.sh          # 忽略 OBS 已有 partial 强制重发
#    SHARD_OFFSET=160 bash runs/embed_wave.sh # 续波: 跳过前 160 片 (10 单元)
#
#  POOL_URI/MODEL_URI 是 per-launch 直挂 (只 stage 进本波的 job, 不进
#  全局后端配置 — 池 197G 绝不能让别的 job 类也白拉)。
#
#  波次形状: 1000 分片 ÷ UNIT_SHARDS=16 = 63 单元; MAX_JOBS=6 × NPU_PER_JOB=8
#  = 48 卡在飞 (共享池, 别全占)。40h 单机嵌入 → ~40/J h。
#
#  失败语义: 单单元失败不杀波; 波末汇总失败清单并给出重试指引;
#  重跑同一命令只重试失败单元 (OBS 上已有 partial 的单元自动跳过)。
#  submit 被池满拒绝 → 按 RemoteConfig 的 submit_retry_* 退避重试。
#
#  波跑绿后 → bash runs/embed_merge.sh (把 partials 拼成 Step-1 缓存)。
#  语义旋钮 (EMBEDDING_MODEL/TRUNCATE_LEN/EMB_DIM) 必须与 run_climbmix.sh
#  一致 — 它们进 cache-key, 不一致 = 缓存 miss = 重新嵌入 40h。
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 后端 RemoteConfig JSON (必填 — 值含平台身份, 不在本仓)
REMOTE_CONFIG="${REMOTE_CONFIG:-}"
SHARD_INFO="${SHARD_INFO:-/home/ma-user/work/100B_stem_parquet_filtered/metadata_shard_info.json}"
OUTPUT_DIR="${OUTPUT_DIR:-$CLIMBMIX_DIR/cache/embed_wave}"

# ── 波次形状 ──
UNIT_SHARDS="${UNIT_SHARDS:-16}"       # 每单元分片数
MAX_JOBS="${MAX_JOBS:-6}"              # 并发单元数 (6×8=48 卡)
NPU_PER_JOB="${NPU_PER_JOB:-8}"
FLAVOR="${FLAVOR:-}"                   # 空 = 用后端配置的默认 flavor
JOB_TIMEOUT_S="${JOB_TIMEOUT_S:-7200}"

# ── 嵌入语义 (与 run_climbmix.sh 对齐) ──
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
TEXT_COL="${TEXT_COL:-text}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TRUNCATE_LEN="${TRUNCATE_LEN:-512}"
EMB_DIM="${EMB_DIM:-1024}"             # stella_en_400M_v5 输出维度

# ── 资产位置 (per-launch 直挂: embed 专属资产, 不进全局后端配置) ──
POOL_URI="${POOL_URI:-}"                # 池 parquet 目录 (obs://...; 必填)
MODEL_URI="${MODEL_URI:-}"              # stella 模型目录 (obs://...; 必填)

# ── 烟雾 / 本地比对 / 续波 ──
SMOKE="${SMOKE:-0}"
LOCAL_POOL_DIR="${LOCAL_POOL_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
LOCAL_MODEL_DIR="${LOCAL_MODEL_DIR:-/home/ma-user/work/stella_en_400M_v5}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
FORCE="${FORCE:-0}"

ARGS=(
  --remote-config "$REMOTE_CONFIG"
  --shard-info "$SHARD_INFO"
  --output-dir "$OUTPUT_DIR"
  --unit-shards "$UNIT_SHARDS"
  --max-jobs "$MAX_JOBS"
  --npu-per-job "$NPU_PER_JOB"
  --job-timeout-s "$JOB_TIMEOUT_S"
  --model "$EMBEDDING_MODEL"
  --text-col "$TEXT_COL"
  --batch-size "$BATCH_SIZE"
  --truncate-len "$TRUNCATE_LEN"
  --emb-dim "$EMB_DIM"
)
if [[ -z "$REMOTE_CONFIG" ]]; then
  echo "[embed_wave] FATAL: REMOTE_CONFIG 未设置 (后端 RemoteConfig JSON 路径)" >&2
  exit 2
fi
if [[ -n "$FLAVOR" ]]; then
  ARGS+=(--flavor "$FLAVOR")
fi
if [[ -n "$POOL_URI" ]]; then
  ARGS+=(--pool-uri "$POOL_URI")
else
  echo "[embed_wave] FATAL: POOL_URI 未设置 (obs://<池 parquet 目录>)" >&2
  exit 2
fi
if [[ -n "$MODEL_URI" ]]; then
  ARGS+=(--model-uri "$MODEL_URI")
fi

if [[ "$SMOKE" != "0" ]]; then
  ARGS+=(--smoke "$SMOKE")
  # 烟雾的判据就是本地逐字节比对 (--compare-local 用 LOCAL_POOL_DIR)
  ARGS+=(--compare-local "$LOCAL_POOL_DIR" --local-model-dir "$LOCAL_MODEL_DIR")
fi
if [[ "$SHARD_OFFSET" != "0" ]]; then
  ARGS+=(--shard-offset "$SHARD_OFFSET")
fi
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

echo "[embed_wave] python3 scripts/embed_dispatch.py ${ARGS[*]}"
exec python3 "$CLIMBMIX_DIR/scripts/embed_dispatch.py" "${ARGS[@]}"
