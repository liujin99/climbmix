#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: 全池嵌入合并 — OBS 单元 partials → Step-1 canonical 缓存
#  (TODO E; scripts/embed_merge.py 的壳)
#
#  用法 (波次跑绿后):
#    bash runs/embed_merge.sh
#    FORCE=1 bash runs/embed_merge.sh                  # 覆盖已有缓存重拼
#    UPLOAD_BACKUP=obs://<bk>/<dir>/embedding_cache.npz bash runs/embed_merge.sh
#
#  语义: 校验分片覆盖 (每片恰好一次 + num_docs/全局偏移对账 + manifest
#  交叉核对 model/truncate_len) → memmap 汇装 (断点续传, ledger 记账) →
#  全池验证 → atomic_savez 落 EMBEDDING_CACHE_DIR/<key>/embedding_cache.npz。
#  之后 run_climbmix.sh 的 Step 1 直接缓存命中, 跳过 ~40h 嵌入。
#
#  注意:
#    - 语义旋钮 (EMBEDDING_MODEL/TRUNCATE_LEN/EMB_DIM/UNIT_SHARDS) 必须与
#      embed_wave.sh 一致; DATA_DIR 必须就是 run_climbmix.sh 的 DATA_DIR
#      (cache-key 从该目录的分片清单计算)。
#    - 缓存目录磁盘需求 ≈ 2× 池字节 (memmap + npz 临时) + ~7.5 GB/在飞单元。
#    - OBS 上的单元 partials 有意保留 (本地盘被清后 ~1-2h 重拼 vs 40h 重嵌)。
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "$0")/.." && pwd)"

REMOTE_CONFIG="${REMOTE_CONFIG:-$CLIMBMIX_DIR/../climbmix-ma/config/remote_config.ma.json}"
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
