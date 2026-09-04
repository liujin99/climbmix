#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  ClimbMix: 全池嵌入合并 — OBS 单元 partials → Step-1 分片缓存
#  (TODO E; scripts/embed_merge.py 的壳)
#
#  用法 (波次跑绿后):
#    bash runs/embed_merge.sh
#    FORCE=1 bash runs/embed_merge.sh                  # 覆盖已有缓存重拼
#    UPLOAD_BACKUP=obs://<bk>/<dir>/ bash runs/embed_merge.sh
#
#  产物 (分片格式 sharded-v1, 落 EMBEDDING_CACHE_DIR/<key>/):
#    manifest.json            — 发布闸门 (cache 存在 ⇔ 它存在且完整)
#    block_<unit_id>.npy      — 每单元一块 ~7.5 GB (全局连续行, 可独立 mmap)
#  为什么分片而非单个 .npy: 全池 ~443 GB 单文件会撞 FUSE 单文件上限
#  (2026-09-04 实测: 挂载在 ~191 GiB 处 EFBIG 拒写); 分块远低于任何上限,
#  可独立校验/搬运, 断点续跑按块进行。Step-1 经 ShardedEmbeddingCache
#  读取 — 切片语义与单文件等价, 聚类路径不变。
#
#  语义: 校验分片覆盖 (每片恰好一次 + num_docs/全局偏移对账 + manifest
#  交叉核对 model/truncate_len) → 逐单元下载→写块→sidecar 记账 (断点续跑
#  按块, 断裂块重新下载, sidecar 分片名漂移即大声失败) → 逐块全池验证 →
#  原子发布 manifest。之后 run_climbmix.sh 的 Step 1 直接缓存命中,
#  跳过 ~40h 嵌入。
#
#  复用语义: cache-key 只含 池分片清单+模型+截断长度 — K/prune/lr/迭代数
#  等下游旋钮全部不影响 key, 所有 ClimbMix 实验共享同一个嵌入池 (Step 1
#  命中后从聚类继续)。池追加新分片 → 重跑 embed_wave (旧单元 resume-skip,
#  只嵌新数据) + 重跑本脚本 (新 key), 增量成本 = 嵌新数据 + 一次合并。
#
#  磁盘: 缓存所在文件系统 ≈ 池字节 (~475 GB) + 一个在飞单元 (~7.5 GB);
#  本地盘放不下时 EMBEDDING_CACHE_DIR 可指 OBS 挂载路径 (如 <挂载根>/...)
#  — 分块经挂载写 (~7.5 GB/块, 无单文件 EFBIG 风险), Step-1 经挂载 mmap。
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
