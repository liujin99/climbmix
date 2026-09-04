#!/usr/bin/env python3
"""Embed unit merger (TODO E) — assemble OBS unit partials into the
canonical pool embedding cache that run_climbmix.sh Step 1 reads.

The dispatcher (scripts/embed_dispatch.py) banks per-unit
partial_block.npz files under {obs_prefix}/embed_units/ (~475 GB
durable tier). This tool turns them into the SHARDED cache format at
the Step-1 cache key:

  <cache-dir>/<key>/manifest.json          — the publish gate
  <cache-dir>/<key>/block_<unit_id>.npy    — one mmap-able block per
                                             unit (~7.5 GB, globally
                                             consecutive rows)

Why sharded and not one big .npy: a full-pool cache is ~443 GB, and a
single file that size hits FUSE single-file limits (live finding
2026-09-04: the OBS mount refused the append at ~191 GiB with EFBIG)
while being awkward to verify/move/back up. Blocks of one unit each
are far under any limit, independently mmap-able, and resume is
per-block instead of one fragile append chain. The reader
(climbmix.core.embedding_cache.ShardedEmbeddingCache) preserves the
single-file read semantics: chunked prescan/assign slice it unchanged,
faiss trains on a deterministic gathered sample.

The cache key (utils/embed_cache.pool_embedding_cache_key) is the EXACT
formula CLIMBPipeline._pool_embedding_cache_dir computes at cache-read
time — it hashes the pool's shard name+size manifest, the embedding
MODEL NAME, and the truncate length, so --model/--truncate-len MUST
match the run config's discovery.* values or the cache lands at a key
Step 1 never reads (a silent miss that re-embeds the pool; the
manifests carry the model name + truncate len and this tool
cross-checks them). The key inputs come from --data-dir — the SAME
local pool dir the pipeline will be pointed at (the pool identity is
its parquet set; sizes are not exposed through the obs protocol).

Usage (after the wave drains green):
  python3 scripts/embed_merge.py \
      --remote-config <backend repo>/config/remote_config.json \
      --shard-info <pool>/metadata_shard_info.json \
      --data-dir <LOCAL pool dir> \
      --cache-dir $EMBEDDING_CACHE_DIR

Semantics:
  - Units are DERIVED from --shard-info + --unit-shards (the real obs
    backends only list files, not dirs) and cross-checked against their
    manifests: every pool shard must be covered EXACTLY once, with
    matching num_docs and global offsets. Missing units → fail with the
    list (re-run the dispatcher; resume skips what succeeded). Stale
    partials from a different pool version fail the same check.
  - Resume is per-block: a banked unit is a block_<unit>.npy whose
    shape matches its sidecar block_<unit>.json (the unit's shard
    coverage recorded at write time — the per-unit replacement of the
    old merge_progress.json). A torn or sidecar-less block is
    re-downloaded; a block whose sidecar names shards the current pool
    does not recognize fails loudly (layout drift never splices a
    silent hole). manifest.json is written ONLY after every block is
    banked and validated — the cache exists iff it exists.
  - Idempotent: an existing sharded cache (manifest) or single-file
    .npy/.npz at the key with the right shape is left alone ("already
    present"); --force re-merges (deleting the old format's files —
    never the kmeans_*.npz caches that share the key dir).
  - The OBS unit partials are NEVER deleted — they are the tier that
    survives a wiped submit-host disk (re-merge in ~1-2h instead of a
    40h re-embed). --upload-backup optionally copies the manifest +
    blocks to OBS as redundancy.

Parallelism & staging (the merge is IO-bound, not compute):
  - --workers N fetches/validates units concurrently. Blocks are
    independent files written tmp+fsync+rename with per-block sidecar
    accounting, so parallelism needs no coordination: a banked block
    is sidecar-complete or it does not count. A failing unit cancels
    the pending queue, in-flight units finish (their blocks bank),
    and the run exits loudly.
  - The per-unit npz download is pure scratch (deleted after the
    block write), so it stages OUTSIDE the cache mount when possible:
    /dev/shm (RAM) > local /tmp > the cache dir itself. This keeps the
    FUSE mount at 7.5 GB of traffic per unit (the block write) instead
    of 22.5 GB (download + npz read-back + block write) — on the
    2026-09-04 merge that alone is a ~3x mount-IO cut. --tmp-dir
    overrides the auto-pick; explicit dirs are left in place after
    the run (auto-picked ones are removed).
  - Each worker process builds its OWN obs client from the same
    remote config (connection pools must not cross fork boundaries);
    the parent's client is never used by children.

Re-use semantics (why this cache is a one-time investment):
  - The cache key ignores every downstream knob (K_enhanced/K_max/
    merge_distance/prune_threshold/lr/iterations): all ClimbMix
    experiments on the SAME pool + model + truncate_len share ONE
    embedded pool — Step 1 cache-hits and the run continues from
    clustering onward.
  - Pool GREW by appending shards (the supported incremental case):
    re-run embed_dispatch (old units' partials are resume-skipped,
    only the new shards' units get embedded) + re-run this merge → a
    fresh cache at the new key, incremental cost = embed(new docs) +
    one merge. Structural changes (insert/delete/reorder) shift the
    unit layout and global offsets → loud failure; that is a new pool
    and a full re-embed.
  - Old-key caches stay on disk after pool growth (history is the
    price of stability); delete them once the new pool is trusted.
  - --cache-dir may point at a FUSE-mounted OBS path (e.g.
    <obs mount root>/...) when local disk can't hold ~475 GB: merge
    writes ~7.5 GB blocks through the mount, Step 1 mmaps them through
    it (slower, but zero local disk beyond one unit at a time).

Sample-mode caches (embedding_sample_size > 0) are a different code
path (subsampled rows) — this tool only merges FULL-pool units.
"""

import argparse
import fnmatch
import glob
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # source-tree fallback (no pip install needed)

DEFAULT_MODEL = "NovaSearch/stella_en_400M_v5"
MANIFEST_NAME = "manifest.json"


def _load_sibling(name):
    """Import a sibling script (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "_sib"), os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_block_sidecar(path):
    """{"shards": [names], "rows": N} from a banked block's sidecar, or
    None when absent/garbage (an untrusted block is re-downloaded — the
    safe direction)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            rec = json.load(f)
        shards = rec.get("shards")
        rows = int(rec.get("rows", -1))
        if not isinstance(shards, list) or rows <= 0:
            return None
        return {"shards": [str(n) for n in shards], "rows": rows}
    except (OSError, ValueError, TypeError):
        return None


def write_block_sidecar(path, unit_id, rows, emb_dim, shard_names):
    """Sidecar writer; tmp+rename (crash-safe like the block itself)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"unit_id": unit_id, "rows": int(rows),
                   "emb_dim": int(emb_dim), "shards": list(shard_names)},
                  f)
    os.replace(tmp, path)


def banked_block(key_dir, unit_id, rows, emb_dim):
    """A previously written block is trusted only when BOTH its .npy
    shape matches and its sidecar recorded the coverage AT WRITE TIME
    (mirroring the old ledger's torn-body + layout-drift guards, but
    self-describing per block). Returns the sidecar record — callers
    must use ITS shard names for coverage accounting, not the current
    layout's, or drift would go undetected."""
    npy = os.path.join(key_dir, f"block_{unit_id}.npy")
    side = os.path.join(key_dir, f"block_{unit_id}.json")
    rec = read_block_sidecar(side)
    if rec is None or rec["rows"] != rows:
        return None
    if not os.path.isfile(npy):
        return None
    if cache_shape(npy) != (rows, emb_dim):
        return None
    return rec


def cache_shape(path):
    """Header-only shape peek for .npy (mmap open — no data read) and
    legacy .npz (zip member header). None when unreadable."""
    try:
        import numpy as np
        if path.endswith(".npy"):
            return tuple(np.load(path, mmap_mode="r").shape)
        import zipfile
        from numpy.lib import format as npfmt
        with zipfile.ZipFile(path) as zf:
            # np.savez stores members as "<key>.npy"
            with zf.open("embeddings.npy") as f:
                version = npfmt.read_magic(f)
                if version == (1, 0):
                    shape, _, _ = npfmt.read_array_header_1_0(f)
                elif version == (2, 0):
                    shape, _, _ = npfmt.read_array_header_2_0(f)
                else:
                    return None
                return tuple(int(s) for s in shape)
    except Exception:
        return None


def pick_tmp_dir(key_dir, workers, unit_bytes):
    """npz staging preference: RAM (/dev/shm) > local /tmp > the cache
    dir itself. The npz round-trip is pure scratch — keeping it off the
    (often FUSE-mounted) cache dir cuts per-unit mount IO 22.5 -> 7.5 GB.
    Returns (path, kind) with kind in {"ram", "local", "cache-dir"}."""
    need = unit_bytes * max(1, workers) + (1 << 30)
    for cand, kind in (("/dev/shm", "ram"),
                       (tempfile.gettempdir(), "local")):
        if not cand or not os.path.isdir(cand) or not os.access(cand, os.W_OK):
            continue
        try:
            st = os.statvfs(cand)
            if st.f_bavail * st.f_frsize >= need:
                return os.path.join(cand, "climbmix_merge_tmp"), kind
        except OSError:
            continue
    return os.path.join(key_dir, "merge_downloads"), "cache-dir"


# ── worker-process side (parallel fetch + validate) ────────────────────────
# Module-level state built by the pool initializer; each child builds its
# OWN obs client from the remote config (forked connection pools must
# never be shared).

_W = {}


def _merge_worker_init(cfg):
    from climbmix.remote.remote_executor import RemoteConfig
    from climbmix.remote.backends import resolve_backend
    rc = RemoteConfig.from_json_file(cfg["remote_config"])
    bundle = resolve_backend(rc)
    _W["obs"] = bundle.make_obs_storage(rc)
    _W["cfg"] = cfg


def _fetch_unit(unit_id, man):
    """One unit: download the npz to the staging dir, verify its shape
    against the unit manifest, write the block + sidecar. The sidecar
    is written ONLY after the block is complete, so a crash window
    leaves an orphan .npy that the resume scan re-downloads. Returns
    (unit_id, rows, seconds)."""
    import numpy as np
    from climbmix.utils.io_utils import atomic_save_npy
    cfg = _W["cfg"]
    obs = _W["obs"]
    t0 = time.time()
    npz_path_u = os.path.join(cfg["tmp_dir"], f"{unit_id}.npz")
    try:
        obs.download_file(
            f"{cfg['obs_prefix'].rstrip('/')}/embed_units/{unit_id}"
            f"/result/partial_block.npz", npz_path_u)
        arr = np.load(npz_path_u)["embeddings"]
    except Exception as e:
        # name the unit — in a 63-unit parallel run a bare "File is not
        # a zip file" is unactionable
        raise RuntimeError(
            f"[merge] FATAL: {unit_id} partial download/read failed: "
            f"{e}") from None
    rows = int(man["total_rows"])
    if arr.shape != (rows, cfg["emb_dim"]):
        raise RuntimeError(
            f"[merge] FATAL: {unit_id} partial shape {arr.shape} != "
            f"manifest ({rows}, {cfg['emb_dim']})")
    atomic_save_npy(
        os.path.join(cfg["key_dir"], f"block_{unit_id}.npy"), arr)
    write_block_sidecar(
        os.path.join(cfg["key_dir"], f"block_{unit_id}.json"),
        unit_id, rows, cfg["emb_dim"],
        [s["path"] for s in man["shards"]])
    del arr
    if not cfg["keep_downloads"]:
        try:
            os.remove(npz_path_u)
        except OSError:
            pass
    return unit_id, rows, time.time() - t0


def _validate_block(task):
    """One block: mmap + full finite/shape scan (chunked views inside
    _validate_embeddings). task = (key_dir, block_meta, ranges, emb_dim)."""
    import numpy as np
    from climbmix.core.embedding_cluster import _validate_embeddings
    key_dir, b, ranges, emb_dim = task
    block_path = os.path.join(key_dir, b["file"])
    block = np.load(block_path, mmap_mode="r")
    if tuple(block.shape) != (b["rows"], emb_dim):
        raise RuntimeError(
            f"[merge] FATAL: {block_path} shape {tuple(block.shape)} "
            f"!= ({b['rows']}, {emb_dim})")
    _validate_embeddings(block, f"[Embed-Merge {b['unit_id']}]",
                         ranges=ranges)
    return b["unit_id"]


def load_and_check_units(ed, obs, shard_infos, args, done_units):
    """[(unit_id, manifest), ...] for the units still to merge, fully
    cross-checked against the global shard layout.

    done_units: {unit_id: {"shards": [...], "bytes": ...}} — units whose
    blocks are already banked (their OBS partials may even be gone);
    their recorded coverage still counts."""
    units = ed.build_units(shard_infos, args.unit_shards, smoke=0)
    skip = [p.strip() for p in args.skip_units.split(",") if p.strip()]
    by_name = {os.path.basename(s["path"]): s for s in shard_infos}

    manifests = {}
    missing = []
    for unit_id, shards in units:
        if any(fnmatch.fnmatch(unit_id, pat) for pat in skip):
            print(f"[merge] {unit_id}: skipped (--skip-units)")
            continue
        if unit_id in done_units:
            continue  # block already banked; partial not needed
        uri = (f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
               f"/result/partial_block.npz")
        if not obs.stat(uri):
            missing.append(unit_id)
            continue
        man_uri = (f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
                   f"/result/manifest.json")
        manifests[unit_id] = json.loads(obs.download_bytes(man_uri))

    if missing:
        raise SystemExit(
            f"[merge] FATAL: {len(missing)} unit(s) have no partial on "
            f"OBS: {', '.join(missing[:20])}"
            f"{' ...' if len(missing) > 20 else ''}\n"
            "  re-run embed_dispatch.py — completed units are "
            "resume-skipped, only these get dispatched")

    covered = {}
    # banked-block units: coverage from the RECORDED sidecar shard names
    # (a changed layout would mismatch here — loud, never silent)
    for unit_id, entry in done_units.items():
        for name in entry["shards"]:
            if name not in by_name:
                raise SystemExit(
                    f"[merge] FATAL: banked block {unit_id} covers "
                    f"unknown shard {name!r} — the pool or unit layout "
                    "changed since that merge attempt; delete its "
                    "block/sidecar files and re-merge")
            if name in covered:
                raise SystemExit(
                    f"[merge] FATAL: shard {name} covered by BOTH "
                    f"{covered[name]} and {unit_id} (banked block)")
            covered[name] = unit_id
    for unit_id, shards in units:
        man = manifests.get(unit_id)
        if man is None:
            continue
        if int(man.get("emb_dim", -1)) != args.emb_dim:
            raise SystemExit(
                f"[merge] FATAL: {unit_id} emb_dim {man.get('emb_dim')} "
                f"!= --emb-dim {args.emb_dim}")
        if int(man.get("truncate_len", -1)) != args.truncate_len:
            raise SystemExit(
                f"[merge] FATAL: {unit_id} was embedded with "
                f"truncate_len={man.get('truncate_len')}, this merge says "
                f"{args.truncate_len} — the key would not match the run "
                "config; pick the value the RUN uses")
        model = man.get("model")
        if model and model != args.model:
            raise SystemExit(
                f"[merge] FATAL: {unit_id} was embedded with model "
                f"{model!r}, this merge says {args.model!r} — pass the "
                "run config's discovery.embedding_model")
        if int(man.get("total_rows", -1)) != sum(
                int(s["num_docs"]) for s in man.get("shards", [])):
            raise SystemExit(f"[merge] FATAL: {unit_id} manifest "
                             "total_rows != sum(shard num_docs)")
        for s in man["shards"]:
            name = s["path"]
            g = by_name.get(name)
            if g is None:
                raise SystemExit(
                    f"[merge] FATAL: {unit_id} embeds unknown shard "
                    f"{name!r} — stale partial from another pool?")
            if int(s["num_docs"]) != int(g["num_docs"]):
                raise SystemExit(
                    f"[merge] FATAL: {name}: manifest num_docs "
                    f"{s['num_docs']} != pool {g['num_docs']}")
            if int(s["global_start"]) != int(g["start_idx"]):
                raise SystemExit(
                    f"[merge] FATAL: {name}: manifest global_start "
                    f"{s['global_start']} != pool start_idx "
                    f"{g['start_idx']}")
            if name in covered:
                raise SystemExit(
                    f"[merge] FATAL: shard {name} covered by BOTH "
                    f"{covered[name]} and {unit_id} (mixed unit "
                    "layouts? see --skip-units)")
            covered[name] = unit_id

    uncovered = [n for n in by_name if n not in covered]
    if uncovered:
        raise SystemExit(
            f"[merge] FATAL: {len(uncovered)} pool shard(s) not covered "
            f"by any unit partial: {', '.join(sorted(uncovered)[:20])}"
            f"{' ...' if len(uncovered) > 20 else ''}")
    return [(uid, manifests[uid]) for uid, _ in units
            if uid in manifests]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote-config", required=True,
                   help="RemoteConfig JSON (backend resolution + obs_prefix)")
    p.add_argument("--shard-info", required=True,
                   help="pool metadata_shard_info.json (per_shard_info)")
    p.add_argument("--data-dir", required=True,
                   help="LOCAL pool dir — the same dir the pipeline run "
                        "points at (source of the cache-key shard "
                        "manifest: names + sizes)")
    p.add_argument("--cache-dir", required=True,
                   help="EMBEDDING_CACHE_DIR root (the merge writes "
                        "<cache-dir>/<key>/{manifest.json, block_*.npy})")
    p.add_argument("--emb-dim", type=int, default=1024,
                   help="must match the dispatcher's --emb-dim")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="embedding model NAME — must equal the run "
                        "config's discovery.embedding_model")
    p.add_argument("--truncate-len", type=int, default=512,
                   help="must equal the run config's "
                        "discovery.embedding_truncate_len")
    p.add_argument("--unit-shards", type=int, default=16,
                   help="shards per unit — must match the wave's "
                        "--unit-shards")
    p.add_argument("--skip-units", default="",
                   help="comma-separated unit ids/globs to ignore (e.g. "
                        "stale mixed-layout units)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel unit fetches + block validations "
                        "(the merge is IO-bound; blocks are independent "
                        "tmp+rename files, so this is coordination-free)")
    p.add_argument("--tmp-dir", default="auto",
                   help="where per-unit npz downloads stage: 'auto' "
                        "picks /dev/shm (RAM) > local /tmp > the cache "
                        "dir — keeping the scratch round-trip off the "
                        "FUSE mount cuts its IO ~3x. An explicit dir is "
                        "left in place after the run")
    p.add_argument("--keep-downloads", action="store_true",
                   help="keep the per-unit npz files after copying "
                        "(default: deleted to save ~7.5 GB per unit)")
    p.add_argument("--upload-backup", default="",
                   help="obs:// URI to copy the manifest + blocks to "
                        "(redundancy)")
    p.add_argument("--force", action="store_true",
                   help="re-merge over an existing cache at the key")
    args = p.parse_args()

    from climbmix.remote.remote_executor import RemoteConfig
    from climbmix.remote.backends import resolve_backend
    from climbmix.utils.embed_cache import pool_embedding_cache_key

    remote_config = RemoteConfig.from_json_file(args.remote_config)
    args.obs_prefix = remote_config.obs_prefix
    bundle = resolve_backend(remote_config)
    obs = bundle.make_obs_storage(remote_config)

    ed = _load_sibling("embed_dispatch.py")
    shard_infos = ed.load_shard_info(args.shard_info)

    # global layout sanity: blocks land at global_start == start_idx;
    # a broken chain means stale metadata vs the real pool
    expect_start = 0
    for s in shard_infos:
        if int(s["start_idx"]) != expect_start:
            raise SystemExit(
                f"[merge] FATAL: shard_info start_idx chain broken at "
                f"{os.path.basename(s['path'])}: {s['start_idx']} != "
                f"expected {expect_start}")
        expect_start += int(s["num_docs"])
    total_docs = expect_start

    # the key inputs come from the SAME dir the pipeline will read
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"[merge] FATAL: --data-dir not a dir: "
                         f"{args.data_dir}")
    pool_names = sorted(
        f for f in os.listdir(args.data_dir) if f.endswith(".parquet"))
    by_name = {os.path.basename(s["path"]) for s in shard_infos}
    if set(pool_names) != by_name:
        only_dir = set(pool_names) - by_name
        only_info = by_name - set(pool_names)
        raise SystemExit(
            f"[merge] FATAL: --data-dir parquet set != shard_info "
            f"(dir-only: {sorted(only_dir)[:5]}, info-only: "
            f"{sorted(only_info)[:5]}) — wrong dir? The key would not "
            "match the pipeline's view of the pool")
    key = pool_embedding_cache_key(
        ((n, os.path.getsize(os.path.join(args.data_dir, n)))
         for n in pool_names),
        args.model, args.truncate_len)
    key_dir = os.path.join(args.cache_dir, key)
    man_path = os.path.join(key_dir, MANIFEST_NAME)
    npy_path = os.path.join(key_dir, "embedding_cache.npy")
    npz_path = os.path.join(key_dir, "embedding_cache.npz")
    print(f"[merge] pool: {len(shard_infos)} shards, {total_docs:,} docs")
    print(f"[merge] cache key: {key}")
    print(f"[merge] target: {key_dir} (sharded: manifest.json + "
          f"block_*.npy)")

    shape = (total_docs, args.emb_dim)

    def remove_embedding_cache_files():
        """Targeted removal at the key — never touches the kmeans_*.npz
        caches that share the key dir."""
        for path in (npy_path, npz_path, man_path):
            if os.path.exists(path):
                os.remove(path)
        for path in glob.glob(os.path.join(key_dir, "block_*.npy")) + \
                glob.glob(os.path.join(key_dir, "block_*.json")) + \
                glob.glob(os.path.join(key_dir, "block_*.npy.tmp.npy")):
            os.remove(path)

    if args.force:
        if os.path.isdir(key_dir):
            remove_embedding_cache_files()
            print("[merge] --force: removed the existing cache at the key")
    else:
        # single-file caches (single-machine runs / legacy): honored
        # as-is — no silent invalidation of old investments
        for path in (npy_path, npz_path):
            if os.path.exists(path):
                got = cache_shape(path)
                if got == shape:
                    print(f"[merge] cache already present ({path}, shape "
                          f"{got}) — nothing to do (use --force to "
                          "re-merge)")
                    return 0
                raise SystemExit(
                    f"[merge] FATAL: {path} exists with shape {got} "
                    f"(expected {shape}) — stale cache at this key; "
                    "delete it or pass --force")
        if os.path.isfile(man_path):
            from climbmix.core.embedding_cache import load_manifest
            man = load_manifest(key_dir)
            if (int(man["row_count"]), int(man["emb_dim"])) == shape:
                print(f"[merge] sharded cache already present "
                      f"({len(man['blocks'])} blocks, {shape[0]:,} rows) "
                      "— nothing to do (use --force to re-merge)")
                return 0
            raise SystemExit(
                f"[merge] FATAL: {man_path} exists but does not match "
                f"the pool ({man.get('row_count')}x{man.get('emb_dim')} "
                f"vs {shape[0]}x{shape[1]}) — stale cache at this key; "
                "delete it or pass --force")

    # ── assemble: per-unit blocks + sidecars, then publish ───────────
    os.makedirs(key_dir, exist_ok=True)
    dl_dir = os.path.join(key_dir, "merge_downloads")

    # v1-format crash residue (the single 443 GB body): junk under the
    # sharded writer — reclaim the mount space and start clean
    for junk in (os.path.join(key_dir, "embedding_cache.npy.partial"),
                 os.path.join(key_dir, "merge_progress.json")):
        if os.path.exists(junk):
            os.remove(junk)
            print(f"[merge] removed stale single-body artifact "
                  f"{os.path.basename(junk)}")
    # orphaned block tmp writes (killed mid-save): a banked block never
    # carries this suffix, so these are always crash residue
    for stale in glob.glob(os.path.join(key_dir, "block_*.npy.tmp.npy")):
        os.remove(stale)
        print(f"[merge] removed stale block tmp "
              f"{os.path.basename(stale)}")
    if os.path.isdir(dl_dir) and not args.keep_downloads:
        shutil.rmtree(dl_dir, ignore_errors=True)

    # resume: banked blocks count via their sidecars (whose recorded
    # shard names — not the current layout's — feed the coverage check)
    all_units = ed.build_units(shard_infos, args.unit_shards, smoke=0)
    done = {}
    for unit_id, shards in all_units:
        rows = sum(int(s["num_docs"]) for s in shards)
        rec = banked_block(key_dir, unit_id, rows, args.emb_dim)
        if rec is not None:
            done[unit_id] = {"shards": rec["shards"],
                             "bytes": rows * args.emb_dim * 4}

    # full coverage check BEFORE any file is written (a failing merge
    # must not leave new blocks behind — banked ones keep their value)
    unit_list = load_and_check_units(ed, obs, shard_infos, args, done)
    print(f"[merge] {len(unit_list)} unit partial(s) to fetch, "
          f"{len(done)} block(s) already banked, coverage complete")

    row_bytes = args.emb_dim * 4

    # staging dir for the scratch npz round-trips (RAM when available —
    # see the module docstring's parallelism notes)
    if args.tmp_dir and args.tmp_dir != "auto":
        tmp_dir, tmp_kind = os.path.abspath(args.tmp_dir), "explicit"
    else:
        unit_bytes = (total_docs * row_bytes) // max(1, len(all_units)) + 1
        tmp_dir, tmp_kind = pick_tmp_dir(
            key_dir, args.workers, unit_bytes)
    os.makedirs(tmp_dir, exist_ok=True)
    n_fetch = len(unit_list)
    workers = max(1, min(args.workers, n_fetch or 1))
    if n_fetch:
        print(f"[merge] staging npz in {tmp_dir} ({tmp_kind}), "
              f"{workers} worker(s) fetching in parallel", flush=True)
    else:
        print(f"[merge] staging: {tmp_dir} ({tmp_kind}) — all blocks "
              "already banked, nothing to fetch", flush=True)

    from concurrent.futures import (
        ProcessPoolExecutor, as_completed)
    cfg = {"remote_config": args.remote_config,
           "obs_prefix": args.obs_prefix,
           "key_dir": key_dir,
           "emb_dim": args.emb_dim,
           "keep_downloads": args.keep_downloads,
           "tmp_dir": tmp_dir}
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=workers,
                             initializer=_merge_worker_init,
                             initargs=(cfg,))
    futs = {}
    try:
        for unit_id, man in unit_list:
            futs[ex.submit(_fetch_unit, unit_id, man)] = unit_id
        banked_ct = 0
        for fut in as_completed(futs):
            unit_id, rows, secs = fut.result()
            banked_ct += 1
            rate = banked_ct / (time.time() - t0)
            eta = (n_fetch - banked_ct) / rate if rate > 0 else 0
            print(f"[merge] {unit_id} banked ({banked_ct}/{n_fetch}, "
                  f"{rows:,} rows, {secs:.0f}s, "
                  f"ETA {eta/60:.0f}m)", flush=True)
    except BaseException as e:
        # a failing unit fails the run loudly; in-flight units finish
        # (their banked blocks keep resume value), pending ones cancel
        for f in futs:
            f.cancel()
        ex.shutdown(wait=True)
        if isinstance(e, KeyboardInterrupt):
            print("[merge] interrupted — banked blocks are kept "
                  "(re-run resumes per block)", file=sys.stderr)
            return 130
        print(f"{e}", file=sys.stderr)
        return 1
    ex.shutdown(wait=True)

    # ── validate every block, then publish the manifest ──────────────
    blocks_meta = []
    for unit_id, shards in all_units:
        blocks_meta.append({
            "file": f"block_{unit_id}.npy",
            "unit_id": unit_id,
            "rows": sum(int(s["num_docs"]) for s in shards),
            "global_start": int(shards[0]["start_idx"]),
            "shards": [os.path.basename(s["path"]) for s in shards],
        })
    vtasks = []
    for b in blocks_meta:
        block_path = os.path.join(key_dir, b["file"])
        if not os.path.isfile(block_path):
            raise SystemExit(
                f"[merge] FATAL: block missing after assembly: "
                f"{block_path}")
        ranges = [(int(s["start_idx"]), int(s["start_idx"])
                   + int(s["num_docs"]), os.path.basename(s["path"]))
                  for s in shard_infos
                  if s["start_idx"] >= b["global_start"]
                  and s["start_idx"] < b["global_start"] + b["rows"]]
        vtasks.append((key_dir, b, ranges, args.emb_dim))
    vworkers = max(1, min(args.workers, len(vtasks) or 1))
    ex = ProcessPoolExecutor(max_workers=vworkers,
                             initializer=_merge_worker_init,
                             initargs=(cfg,))
    vfuts = {}
    try:
        for t in vtasks:
            vfuts[ex.submit(_validate_block, t)] = t[1]["unit_id"]
        vdone = 0
        for fut in as_completed(vfuts):
            print(f"[merge] {fut.result()} validated "
                  f"({vdone + 1}/{len(vfuts)})", flush=True)
            vdone += 1
    except BaseException as e:
        for f in vfuts:
            f.cancel()
        ex.shutdown(wait=True)
        if isinstance(e, KeyboardInterrupt):
            print("[merge] interrupted — banked blocks are kept "
                  "(re-run resumes per block)", file=sys.stderr)
            return 130
        print(f"{e}", file=sys.stderr)
        return 1
    ex.shutdown(wait=True)

    manifest = {
        "format": "sharded-v1",
        "row_count": total_docs,
        "emb_dim": args.emb_dim,
        "model": args.model,
        "truncate_len": args.truncate_len,
        "blocks": blocks_meta,
    }
    tmp = f"{man_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, man_path)
    total_bytes = total_docs * row_bytes
    print(f"[merge] published {man_path} ({len(blocks_meta)} blocks, "
          f"{total_bytes / (1024**3):.1f} GB of embeddings)")

    # cleanup partial-state files (the published cache is the product)
    for side in glob.glob(os.path.join(key_dir, "block_*.json")):
        os.remove(side)
    if not args.keep_downloads:
        if tmp_kind == "explicit":
            print(f"[merge] note: --tmp-dir {tmp_dir} left in place "
                  "(explicit dirs are the caller's to manage)")
        elif os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
    if os.path.isdir(dl_dir) and not args.keep_downloads:
        shutil.rmtree(dl_dir, ignore_errors=True)

    if args.upload_backup:
        print(f"[merge] uploading backup -> {args.upload_backup}")
        obs.upload_file(man_path, f"{args.upload_backup}/{MANIFEST_NAME}")
        for b in blocks_meta:
            obs.upload_file(os.path.join(key_dir, b["file"]),
                            f"{args.upload_backup}/{b['file']}")

    print(f"[merge] done: {total_docs:,} docs, dim={args.emb_dim}, "
          f"{len(blocks_meta)} blocks")
    print("[merge] unit partials on OBS are intentionally KEPT — they "
          "survive a wiped local disk (re-merge ~1-2h vs re-embed 40h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
