#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: 全池嵌入合并 — OBS 单元 partials → Step-1 canonical 缓存
#  (TODO E; scripts/embed_merge.py 的壳)
#
#  用法 (波次跑绿后):
#    bash runs/embed_merge.sh
#    FORCE=1 bash runs/embed_merge.sh                  # 覆盖已有缓存重拼
#    UPLOAD_BACKUP=obs://<bk>/<dir>/embedding_cache.npy bash runs/embed_merge.sh
#
#  语义: 校验分片覆盖 (每片恰好一次 + num_docs/全局偏移对账 + manifest
#  交叉核对 model/truncate_len) → 流式拼接 (单元在全局连续有序, 缓存体 =
#  header + 逐单元字节追加; 断点续传, ledger 记账, 断裂即整体重来) →
#  全池验证 → 落 EMBEDDING_CACHE_DIR/<key>/embedding_cache.npy (mmap 可读,
#  Step-1 命中时不整载进 RAM)。之后 run_climbmix.sh 的 Step 1 直接缓存命中,
#  跳过 ~40h 嵌入。
#
#  复用语义: cache-key 只含 池分片清单+模型+截断长度 — K/prune/lr/迭代数
#  等下游旋钮全部不影响 key, 所有 ClimbMix 实验共享同一个嵌入池 (Step 1
#  命中后从聚类继续)。池追加新分片 → 重跑 embed_wave (旧单元 resume-skip,
#  只嵌新数据) + 重跑本脚本 (新 key), 增量成本 = 嵌新数据 + 一次合并。
#
#  磁盘: 缓存所在文件系统 ≈ 池字节 (~475 GB) + 一个在飞单元 (~7.5 GB);
#  本地盘放不下时 EMBEDDING_CACHE_DIR 可指 OBS 挂载路径 (如 <挂载根>/...)
#  — 合并经挂载写, Step-1 经挂载 mmap (慢一点, 但本地零占用)。
#
#  注意:
#    - 语义旋钮 (EMBEDDING_MODEL/TRUNCATE_LEN/EMB_DIM/UNIT_SHARDS) 必须与
#      embed_wave.sh 一致; DATA_DIR 必须就是 run_climbmix.sh 的 DATA_DIR
#      (cache-key 从该目录的分片清单计算)。
#    - OBS 上的单元 partials 有意保留 (本地盘被清后 ~1-2h 重拼 vs 40h 重嵌)。
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# 后端 RemoteConfig JSON (必填 — 值含平台身份, 不在本仓)
REMOTE_CONFIG="${REMOTE_CONFIG:-}"
SHARD_INFO="${SHARD_INFO:-/home/ma-user/work/100B_stem_parquet_filtered/metadata_shard_info.json}"
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
# 与 run_climbmix.sh 的 EMBEDDING_CACHE_DIR 同名同默认 — 两边指同一个地方
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$CLIMBMIX_DIR/cache/embeddings}"

# ── 嵌入语义 (与 embed_wave.sh 对齐; 进 cache-key) ──
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
TRUNCATE_LEN="${TRUNCATE_LEN:-512}"
EMB_DIM="${EMB_DIM:-1024}"
UNIT_SHARDS="${UNIT_SHARDS:-16}"

SKIP_UNITS="${SKIP_UNITS:-}"
KEEP_DOWNLOADS="${KEEP_DOWNLOADS:-0}"
UPLOAD_BACKUP="${UPLOAD_BACKUP:-}"
FORCE="${FORCE:-0}"

if [[ -z "$REMOTE_CONFIG" ]]; then
  echo "[embed_merge] FATAL: REMOTE_CONFIG 未设置 (后端 RemoteConfig JSON 路径)" >&2
  exit 2
fi

ARGS=(
  --remote-config "$REMOTE_CONFIG"
  --shard-info "$SHARD_INFO"
  --data-dir "$DATA_DIR"
  --cache-dir "$EMBEDDING_CACHE_DIR"
  --model "$EMBEDDING_MODEL"
  --truncate-len "$TRUNCATE_LEN"
  --emb-dim "$EMB_DIM"
  --unit-shards "$UNIT_SHARDS"
)
if [[ -n "$SKIP_UNITS" ]]; then
  ARGS+=(--skip-units "$SKIP_UNITS")
fi
if [[ "$KEEP_DOWNLOADS" == "1" ]]; then
  ARGS+=(--keep-downloads)
fi
if [[ -n "$UPLOAD_BACKUP" ]]; then
  ARGS+=(--upload-backup "$UPLOAD_BACKUP")
fi
if [[ "$FORCE" == "1" ]]; then
  ARGS+=(--force)
fi

echo "[embed_merge] python3 scripts/embed_merge.py ${ARGS[*]}"
exec python3 "$CLIMBMIX_DIR/scripts/embed_merge.py" "${ARGS[@]}"
