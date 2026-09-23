#!/usr/bin/env bash
# ═════════════════════════════════════════════════════════════════════
#  preprocess_pool.sh — 池级 embedding 供给入口 (两入口设计, 2026-09-23)
#
#    preprocess_pool.sh = 供给: 缓存检查 → 磁盘守卫 → embed_merge
#                         (OBS 耐久层 → 本地性能层) → 探针验证
#    run_experiment.sh  = 消费: Stage 1 读缓存; 全池 miss → fail-loud
#                         指路本脚本 (内联嵌入守卫, embedding_cluster)
#
#  用法 (服务器; 全幂等 — 缓存已热时秒级退出):
#    bash runs/preprocess_pool.sh
#    DRY_RUN=1 bash runs/preprocess_pool.sh    # 预览 key/守卫/命令, 不执行
#
#  分层 (详见 docs/prod5_runbook.md §4):
#    耐久层 = {obs_prod_base}/embed_units/uXXXX (waves 产出, ~475GB,
#             设计上永不清删 — 本地被清盘后的恢复源)
#    性能层 = cache/embeddings/<key>/ (分片缓存, 引擎 Stage 1 直读)
#    key   = sha256(池分片清单 + 嵌入模型 + truncate) — 与引擎同公式
#            同默认 (覆盖 DATA_DIR/EMBEDDING_MODEL/EMBEDDING_TRUNCATE_LEN
#            时须与发射一致, 否则 merge 落在引擎不读的 key 上)
#
#  env 覆盖: DATA_DIR / EMBEDDING_CACHE_DIR / EMBEDDING_MODEL /
#            EMBEDDING_TRUNCATE_LEN / EMB_DIM / UNIT_SHARDS / DRY_RUN
# ═════════════════════════════════════════════════════════════════════
set -euo pipefail

CLIMBMIX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CLIMBMIX_DIR"

# 默认与 run_experiment.sh 逐键一致 (key 公式的全部输入)
DATA_DIR="${DATA_DIR:-/home/ma-user/work/100B_stem_parquet_filtered}"
EMBEDDING_CACHE_DIR="${EMBEDDING_CACHE_DIR:-$CLIMBMIX_DIR/cache/embeddings}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-NovaSearch/stella_en_400M_v5}"
EMBEDDING_TRUNCATE_LEN="${EMBEDDING_TRUNCATE_LEN:-512}"
EMB_DIM="${EMB_DIM:-1024}"
UNIT_SHARDS="${UNIT_SHARDS:-16}"
DRY_RUN="${DRY_RUN:-0}"
export DATA_DIR EMBEDDING_CACHE_DIR EMBEDDING_MODEL EMBEDDING_TRUNCATE_LEN \
       EMB_DIM UNIT_SHARDS DRY_RUN

python3 - <<'PYEOF'
import json
import os
import shutil
import subprocess
import sys

for _p in ("src", "climbmix-ma", os.path.join("..", "climbmix-ma")):
    _d = os.path.abspath(_p)
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)

data_dir = os.environ["DATA_DIR"]
cache_dir = os.environ["EMBEDDING_CACHE_DIR"]
model = os.environ["EMBEDDING_MODEL"]
trunc = int(os.environ["EMBEDDING_TRUNCATE_LEN"])
emb_dim = int(os.environ["EMB_DIM"])
unit_shards = int(os.environ["UNIT_SHARDS"])
dry = os.environ["DRY_RUN"] == "1"

# ── 1. key = 引擎同公式同默认 ──
from climbmix.utils.embed_cache import pool_embedding_cache_key

shards = sorted(f for f in os.listdir(data_dir) if f.endswith(".parquet"))
if not shards:
    sys.exit(f"✗ {data_dir} 下没有 parquet 分片")
key = pool_embedding_cache_key(
    ((n, os.path.getsize(os.path.join(data_dir, n))) for n in shards),
    model, trunc)
key_dir = os.path.join(cache_dir, key)
print(f"[pool]   {data_dir}: {len(shards)} shards, model={model}, "
      f"truncate={trunc}")
print(f"[cache]  key = {key}")

# ── 2. 已就绪? (幂等快路径) ──
if os.path.isfile(os.path.join(key_dir, "manifest.json")):
    from climbmix.core.embedding_cache import ShardedEmbeddingCache
    c = ShardedEmbeddingCache(key_dir)
    print(f"[ready]  sharded cache 在位: {c.n_rows:,} rows x {c.dim} dims, "
          f"{c.block_count} blocks — 无事可做")
    sys.exit(0)

# ── 3. 磁盘守卫 ──
shard_info_path = os.path.join(data_dir, "metadata_shard_info.json")
if not os.path.isfile(shard_info_path):
    sys.exit(f"✗ {shard_info_path} 不在 (waves 产物) — 先跑 embed_dispatch "
             f"(scripts/embed_dispatch.py, 集群并行嵌入)")
with open(shard_info_path) as f:
    infos = json.load(f)
    infos = infos.get("per_shard_info") or infos
if not isinstance(infos, list) or not infos:
    sys.exit(f"✗ {shard_info_path} 里没有 per_shard_info 列表")
n_docs = sum(int(i.get("num_docs") or 0) for i in infos)
if n_docs <= 0:
    sys.exit(f"✗ {shard_info_path} 的 num_docs 总和为 0 — 文件损坏?")
need = int(n_docs * emb_dim * 4 * 1.1)  # fp32 + 10% 余量
free = shutil.disk_usage(cache_dir).free
print(f"[guard]  估算需 ~{need / 2**30:.0f} GiB, 可用 {free / 2**30:.0f} GiB")
if free < need + max(50 * 2**30, int(0.15 * need)):
    sys.exit(f"✗ 磁盘余量不足: 需 ≥ {need / 2**30:.0f} GiB + 15%/50G 余量, "
             f"现有 {free / 2**30:.0f} GiB — 清理 cache/embeddings/ 下旧 key "
             f"或腾空间后重试")

# ── 4. root 前缀 + 耐久层探测 ──
try:
    from climbmix_ma.modelarts_job_api import load_ma_config
    ma_cfg = load_ma_config() or {}
except Exception as e:
    sys.exit(f"✗ climbmix_ma 平台配置读取失败 ({e}) — 检查 "
             f"~/.config/climbmix/remote_ma.json")

root = str(ma_cfg.get("obs_prod_base") or "").strip().rstrip("/")
if not root.startswith("obs://"):
    sys.exit("✗ ma 配置缺 obs_prod_base (~/.config/climbmix/remote_ma.json) "
             "— 与 run_experiment.sh 的 OBS 前缀自动派生同源")
rc_path = os.path.join("/tmp", "pool_preprocess_remote_config.json")
with open(rc_path, "w") as f:
    json.dump({"obs_prefix": root, "backend": "modelarts",
               "backend_module": "climbmix_ma:create_backend"}, f, indent=2)

from climbmix.remote.backends import resolve_backend
from climbmix.remote.remote_executor import RemoteConfig

rc = RemoteConfig.from_json_file(rc_path)
obs = resolve_backend(rc).make_obs_storage(rc)
if not obs.stat(f"{root}/embed_units/u0000"):
    sys.exit(f"✗ {root}/embed_units/u0000 不存在 — 耐久层缺失, 先跑 "
             f"embed_dispatch (waves; 池增长场景 = 对新分片补波)")

# ── 5. merge (耐久层 → 性能层; IO-bound ~1-2h, 按块可续) ──
cmd = [sys.executable, os.path.join("scripts", "embed_merge.py"),
       "--remote-config", rc_path,
       "--shard-info", shard_info_path,
       "--data-dir", data_dir,
       "--cache-dir", cache_dir,
       "--emb-dim", str(emb_dim),
       "--unit-shards", str(unit_shards)]
print(f"[merge]  {' '.join(cmd)}")
if dry:
    print("[dry]    DRY_RUN=1 — 只预览, 未执行")
    sys.exit(0)
r = subprocess.run(cmd)
if r.returncode != 0:
    sys.exit(r.returncode)

# ── 6. 探针 (行数对账 shard-info; 维度/块链 manifest 校验已内建) ──
from climbmix.core.embedding_cache import ShardedEmbeddingCache

c = ShardedEmbeddingCache(key_dir)
if c.n_rows != n_docs:
    sys.exit(f"✗ 探针对账失败: cache {c.n_rows:,} rows != shard-info "
             f"{n_docs:,} docs")
print(f"[ok]     {c.n_rows:,} rows x {c.dim} dims, {c.block_count} blocks "
      f"— 池缓存就绪, run_experiment 可直接发射 (no-seed)")
PYEOF
