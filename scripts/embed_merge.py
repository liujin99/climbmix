#!/usr/bin/env python3
"""Embed unit merger (TODO E) — assemble OBS unit partials into the
canonical pool embedding cache that run_climbmix.sh Step 1 reads.

The dispatcher (scripts/embed_dispatch.py) banks per-unit
partial_block.npz files under {obs_prefix}/embed_units/ (~475 GB
durable tier). This tool downloads them, copies the rows into one
pool-shaped memmap at the Step-1 cache key, validates the whole pool
(same net the single-node run applies), and atomic_savez's the final
`embedding_cache.npz`. After it exits green, the next pipeline run
cache-hits and skips the ~40h embed.

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
      --remote-config climbmix-ma/config/remote_config.ma.json \
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
  - Resume: a crash mid-merge loses only the in-flight unit — completed
    units are recorded in merge_progress.json and re-running skips them
    (their downloaded npz is deleted after copying, so a resumed merge
    does not even re-download them).
  - Idempotent: an existing embedding_cache.npz with the right shape is
    left alone ("already merged"); --force re-merges over it.
  - The OBS unit partials are NEVER deleted — they are the tier that
    survives a wiped submit-host disk (re-merge in ~1-2h instead of a
    40h re-embed). --upload-backup optionally copies the final npz to
    OBS as redundancy.
  - Disk needed at the --cache-dir filesystem: pool_bytes (memmap) +
    pool_bytes (npz tmp during atomic_savez) + one unit (~7.5 GB).

Sample-mode caches (embedding_sample_size > 0) are a different code
path (subsampled rows) — this tool only merges FULL-pool units.
"""

import argparse
import fnmatch
import importlib.util
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # source-tree fallback (no pip install needed)

DEFAULT_MODEL = "NovaSearch/stella_en_400M_v5"


def _load_sibling(name):
    """Import a sibling script (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", "_sib"), os.path.join(_HERE, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def atomic_write_ledger(path, completed):
    """merge_progress.json writer; tmp+rename.

    completed maps unit_id -> [shard names it covered] — the recorded
    shard list (not just the id) makes resume airtight: if the unit
    layout (--unit-shards / shard-offset) changed between merge runs,
    the recorded coverage no longer matches the pool and merge fails
    loudly instead of leaving a silent hole in the memmap."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"completed": completed}, f)
    os.replace(tmp, path)


def load_ledger(path):
    """{unit_id: [shard names]} from a previous merge attempt.

    Garbage/legacy formats degrade to {} (memmap rows from an
    unaccounted attempt are re-copied — overwrite is idempotent)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        done = data.get("completed")
        if not isinstance(done, dict):
            return {}
        return {str(uid): [str(n) for n in names]
                for uid, names in done.items() if isinstance(names, list)}
    except (OSError, ValueError, TypeError):
        return {}


def npz_embeddings_shape(path):
    """Header-only peek at the 'embeddings' member's shape (a full
    np.load would materialize ~475 GB). None when unreadable."""
    try:
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


def load_and_check_units(ed, obs, shard_infos, args, done_units):
    """[(unit_id, manifest), ...] for the units still to merge, fully
    cross-checked against the global shard layout.

    done_units: {unit_id: [shard names]} from a previous attempt's
    ledger — those units' rows are already in the memmap (their OBS
    partials may even be gone); their recorded coverage still counts."""
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
            continue  # rows already in the memmap; partial not needed
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
    # ledger-done units: coverage from the RECORDED shard names (a
    # changed layout would mismatch here — loud, never silent)
    for unit_id, names in done_units.items():
        for name in names:
            if name not in by_name:
                raise SystemExit(
                    f"[merge] FATAL: ledger unit {unit_id} covers "
                    f"unknown shard {name!r} — the pool or unit layout "
                    "changed since that merge attempt; delete "
                    "merge_progress.json + the memmap and re-merge")
            if name in covered:
                raise SystemExit(
                    f"[merge] FATAL: shard {name} covered by BOTH "
                    f"{covered[name]} and {unit_id} (ledger)")
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
                f"config; pick the value the RUN uses")
        model = man.get("model")
        if model and model != args.model:
            raise SystemExit(
                f"[merge] FATAL: {unit_id} was embedded with model "
                f"{model!r}, this merge says {args.model!r} — pass the "
                f"run config's discovery.embedding_model")
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
                    f"layouts? see --skip-units)")
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
                        "<cache-dir>/<key>/embedding_cache.npz)")
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
    p.add_argument("--keep-downloads", action="store_true",
                   help="keep the per-unit npz files after copying "
                        "(default: deleted to save ~7.5 GB per unit)")
    p.add_argument("--upload-backup", default="",
                   help="obs:// URI to copy the final npz to (redundancy)")
    p.add_argument("--force", action="store_true",
                   help="re-merge over an existing embedding_cache.npz")
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

    # global layout sanity: the merge writes rows at global_start ==
    # start_idx; a broken chain means stale metadata vs the real pool
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
    cache_path = os.path.join(key_dir, "embedding_cache.npz")
    print(f"[merge] pool: {len(shard_infos)} shards, {total_docs:,} docs")
    print(f"[merge] cache key: {key}")
    print(f"[merge] target: {cache_path}")

    shape = (total_docs, args.emb_dim)
    if os.path.exists(cache_path) and not args.force:
        got = npz_embeddings_shape(cache_path)
        if got == shape:
            print(f"[merge] cache already present with shape {got} — "
                  "nothing to do (use --force to re-merge)")
            return 0
        raise SystemExit(
            f"[merge] FATAL: {cache_path} exists with shape {got} "
            f"(expected {shape}) — stale cache at this key; delete it or "
            "pass --force")

    # ── assemble: memmap + unit ledger (crash loses only the in-flight
    # unit — completed ones are not even re-downloaded) ────────────────
    import numpy as np
    os.makedirs(key_dir, exist_ok=True)
    dl_dir = os.path.join(key_dir, "merge_downloads")
    memmap_path = os.path.join(key_dir, "embedding_memmap.tmp")
    ledger_path = os.path.join(key_dir, "merge_progress.json")
    expected_bytes = total_docs * args.emb_dim * 4
    memmap_intact = (os.path.exists(memmap_path)
                     and os.path.getsize(memmap_path) == expected_bytes)
    # the ledger only counts when its memmap survived; a wrong-size
    # memmap means a different pool/dim — stale state, start over
    done = load_ledger(ledger_path) if memmap_intact else {}

    # full coverage check BEFORE any file is created (a failing merge
    # must not leave a 475 GB truncated memmap behind)
    unit_list = load_and_check_units(ed, obs, shard_infos, args, done)
    print(f"[merge] {len(unit_list)} unit partial(s) to merge, "
          f"{len(done)} already in the memmap, coverage complete")

    if not memmap_intact:
        for junk in (memmap_path, ledger_path):
            if os.path.exists(junk):
                os.remove(junk)
        with open(memmap_path, "wb") as f:
            f.truncate(expected_bytes)
        print(f"[merge] preallocated memmap {shape} "
              f"({expected_bytes / (1024**3):.1f} GB)")
    else:
        print(f"[merge] resuming: memmap intact")

    mm = np.memmap(memmap_path, dtype=np.float32, mode="r+", shape=shape)
    t0 = time.time()
    for i, (unit_id, man) in enumerate(unit_list):
        npz_path = os.path.join(dl_dir, f"{unit_id}.npz")
        os.makedirs(dl_dir, exist_ok=True)
        tu = time.time()
        obs.download_file(
            f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
            f"/result/partial_block.npz", npz_path)
        arr = np.load(npz_path)["embeddings"]
        if arr.shape != (int(man["total_rows"]), args.emb_dim):
            raise SystemExit(
                f"[merge] FATAL: {unit_id} partial shape {arr.shape} != "
                f"manifest ({man['total_rows']}, {args.emb_dim})")
        off = 0
        for s in man["shards"]:
            nd = int(s["num_docs"])
            gs = int(s["global_start"])
            mm[gs:gs + nd] = arr[off:off + nd]
            off += nd
        if off != arr.shape[0]:
            raise SystemExit(f"[merge] FATAL: {unit_id} row accounting")
        mm.flush()
        del arr
        if not args.keep_downloads:
            os.remove(npz_path)
        done[unit_id] = [s["path"] for s in man["shards"]]
        atomic_write_ledger(ledger_path, done)
        rate = (i + 1) / (time.time() - t0)
        eta = (len(unit_list) - i - 1) / rate if rate > 0 else 0
        print(f"[merge] {unit_id} merged ({i + 1}/{len(unit_list)}, "
              f"{(time.time() - tu):.0f}s, ETA {eta/60:.0f}m)", flush=True)
    del mm

    # ── validate the whole pool, then publish atomically ──────────────
    from climbmix.core.embedding_cluster import _validate_embeddings
    from climbmix.utils.io_utils import atomic_savez
    final = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=shape)
    ranges = [(int(s["start_idx"]), int(s["start_idx"]) + int(s["num_docs"]),
               os.path.basename(s["path"])) for s in shard_infos]
    _validate_embeddings(final, "[Embed-Merge]", ranges=ranges)
    atomic_savez(cache_path, embeddings=final)
    print(f"[merge] wrote {cache_path} "
          f"({os.path.getsize(cache_path) / (1024**3):.1f} GB)")

    # cleanup partial-state files (the cache is the product now)
    for junk in (memmap_path, ledger_path):
        if os.path.exists(junk):
            os.remove(junk)
    if os.path.isdir(dl_dir) and not args.keep_downloads:
        import shutil
        shutil.rmtree(dl_dir, ignore_errors=True)

    if args.upload_backup:
        print(f"[merge] uploading backup -> {args.upload_backup}")
        obs.upload_file(cache_path, args.upload_backup)

    print(f"[merge] done: {total_docs:,} docs, dim={args.emb_dim}")
    print("[merge] unit partials on OBS are intentionally KEPT — they "
          "survive a wiped local disk (re-merge ~1-2h vs re-embed 40h)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
