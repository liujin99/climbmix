#!/usr/bin/env python3
"""Embed unit merger (TODO E) — assemble OBS unit partials into the
canonical pool embedding cache that run_climbmix.sh Step 1 reads.

The dispatcher (scripts/embed_dispatch.py) banks per-unit
partial_block.npz files under {obs_prefix}/embed_units/ (~475 GB
durable tier). This tool streams them, one unit block after another,
into `embedding_cache.npy` at the Step-1 cache key (a pure byte-level
append — see below), validates the whole pool (the same net the
single-node run applies), and publishes it atomically. After it exits
green, the next pipeline run cache-hits and skips the ~40h embed.

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
  - The merged cache is a raw .npy written as a pure STREAM: units are
    consecutive slices of the pool, so unit k's partial is exactly the
    global row range it covers — the merge is a byte-level append of
    one unit block after the other (no pool-shaped memmap, no 2x disk:
    peak = final cache + ONE downloaded partial ~7.5 GB). .npy is also
    what Step 1 mmaps on cache hit (a pool-sized cache never
    materializes in RAM) and the only format that reads acceptably
    from a FUSE-mounted OBS path.
  - Resume: a crash mid-merge loses only the in-flight unit — the
    ledger records each unit's appended byte count and re-running
    verifies the partial file's size before continuing (a mismatch
    restarts the concat from scratch; the downloads are the slow part
    and completed units are NOT re-downloaded).
  - Idempotent: an existing embedding_cache.npy (or legacy .npz) with
    the right shape is left alone ("already merged"); --force
    re-merges over it.
  - The OBS unit partials are NEVER deleted — they are the tier that
    survives a wiped submit-host disk (re-merge in ~1-2h instead of a
    40h re-embed). --upload-backup optionally copies the final cache
    to OBS as redundancy.

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
    writes through the mount, Step 1 mmaps through it (slower, but
    zero local disk beyond one unit at a time).

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

    completed maps unit_id -> {"shards": [...], "bytes": N} where bytes
    is the raw block appended to the .npy body for that unit. The
    recorded shard list (not just the id) makes resume airtight: if the
    unit layout (--unit-shards / shard-offset) changed between merge
    runs, the recorded coverage no longer matches the pool and merge
    fails loudly instead of leaving a silent hole. The byte counts let
    a resumed run VERIFY the partial file's size before continuing —
    an append that crashed mid-unit restarts the concat from scratch,
    never splices a torn block."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"completed": completed}, f)
    os.replace(tmp, path)


def load_ledger(path):
    """{unit_id: {"shards": [names], "bytes": int}} from a previous
    merge attempt, in append order (dict insertion order survives the
    json round-trip).

    Garbage/legacy formats degrade to {} (bytes from an unaccounted
    attempt fail the size check → fresh restart — the safe direction)."""
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
        done = data.get("completed")
        if not isinstance(done, dict):
            return {}
        out = {}
        for uid, entry in done.items():
            if (isinstance(entry, dict)
                    and isinstance(entry.get("shards"), list)
                    and isinstance(entry.get("bytes"), int)):
                out[str(uid)] = {"shards": [str(n) for n in entry["shards"]],
                                 "bytes": entry["bytes"]}
        return out
    except (OSError, ValueError, TypeError):
        return {}


def npy_header_bytes(shape, dtype="<f4"):
    """The .npy header a fresh cache file starts with (magic + version
    + padded header dict) — written once, then unit blocks append."""
    import io
    from numpy.lib import format as npfmt
    buf = io.BytesIO()
    npfmt.write_array_header_1_0(buf, {
        "descr": dtype, "fortran_order": False,
        "shape": tuple(int(s) for s in shape)})
    return buf.getvalue()


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


def load_and_check_units(ed, obs, shard_infos, args, done_units):
    """[(unit_id, manifest), ...] for the units still to merge, fully
    cross-checked against the global shard layout.

    done_units: {unit_id: {"shards": ..., "bytes": ...}} from a previous
    attempt's ledger — those units' bytes are already in the cache body
    (their OBS partials may even be gone); their recorded coverage
    still counts."""
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
    for unit_id, entry in done_units.items():
        for name in entry["shards"]:
            if name not in by_name:
                raise SystemExit(
                    f"[merge] FATAL: ledger unit {unit_id} covers "
                    f"unknown shard {name!r} — the pool or unit layout "
                    "changed since that merge attempt; delete "
                    "merge_progress.json + the partial cache file and "
                    "re-merge")
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
                   help="obs:// URI to copy the final cache to (redundancy)")
    p.add_argument("--force", action="store_true",
                   help="re-merge over an existing embedding_cache.npy")
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
    cache_path = os.path.join(key_dir, "embedding_cache.npy")
    legacy_npz = os.path.join(key_dir, "embedding_cache.npz")
    print(f"[merge] pool: {len(shard_infos)} shards, {total_docs:,} docs")
    print(f"[merge] cache key: {key}")
    print(f"[merge] target: {cache_path}")

    shape = (total_docs, args.emb_dim)
    existing = ([p for p in (cache_path, legacy_npz) if os.path.exists(p)]
                if not args.force else [])
    for p in existing:
        got = cache_shape(p)
        if got == shape:
            print(f"[merge] cache already present ({p}, shape {got}) — "
                  "nothing to do (use --force to re-merge)")
            return 0
        raise SystemExit(
            f"[merge] FATAL: {p} exists with shape {got} "
            f"(expected {shape}) — stale cache at this key; delete it or "
            "pass --force")

    # ── assemble: streaming .npy concat + unit ledger ──────────────────
    # Units are consecutive slices of the pool in order, so unit k's
    # partial IS the global row range it covers: the cache body is the
    # byte-level append of one unit block after the other (the strict
    # coverage checks above are what guarantee this equals global row
    # order). Peak disk = final cache + ONE downloaded partial (~7.5 GB)
    # — no pool-shaped memmap, no second copy during the final write.
    import numpy as np
    os.makedirs(key_dir, exist_ok=True)
    dl_dir = os.path.join(key_dir, "merge_downloads")
    body_path = os.path.join(key_dir, "embedding_cache.npy.partial")
    ledger_path = os.path.join(key_dir, "merge_progress.json")
    header = npy_header_bytes(shape)
    row_bytes = args.emb_dim * 4

    # resume: ledger counts only when the body file survived AND its size
    # matches the recorded byte accounting (header + done units, in order)
    done = load_ledger(ledger_path)
    expected_size = header_len = len(header)
    for entry in done.values():
        expected_size += entry["bytes"]
    body_ok = (done and os.path.exists(body_path)
               and os.path.getsize(body_path) == expected_size)
    if not body_ok:
        done = {}
        expected_size = header_len

    # full coverage check BEFORE any file is touched (a failing merge
    # must not leave a half-written cache behind)
    unit_list = load_and_check_units(ed, obs, shard_infos, args, done)
    print(f"[merge] {len(unit_list)} unit partial(s) to append, "
          f"{len(done)} already in the cache body, coverage complete")

    if not done:
        for junk in (body_path, ledger_path):
            if os.path.exists(junk):
                os.remove(junk)
        with open(body_path, "wb") as f:
            f.write(header)
        print(f"[merge] started cache body {shape} "
              f"({total_docs * row_bytes / (1024**3):.1f} GB when done)")
    else:
        print(f"[merge] resuming: cache body intact at "
              f"{expected_size / (1024**3):.1f} GB")

    t0 = time.time()
    for i, (unit_id, man) in enumerate(unit_list):
        npz_path = os.path.join(dl_dir, f"{unit_id}.npz")
        os.makedirs(dl_dir, exist_ok=True)
        tu = time.time()
        obs.download_file(
            f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
            f"/result/partial_block.npz", npz_path)
        arr = np.load(npz_path)["embeddings"]
        rows = int(man["total_rows"])
        if arr.shape != (rows, args.emb_dim):
            raise SystemExit(
                f"[merge] FATAL: {unit_id} partial shape {arr.shape} != "
                f"manifest ({rows}, {args.emb_dim})")
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        unit_bytes = rows * row_bytes
        with open(body_path, "ab") as f:
            arr.tofile(f)          # raw fp32 block, no in-RAM copy
            f.flush()
            os.fsync(f.fileno())   # a crash never splices a torn block
        del arr
        if not args.keep_downloads:
            os.remove(npz_path)
        done[unit_id] = {"shards": [s["path"] for s in man["shards"]],
                         "bytes": unit_bytes}
        atomic_write_ledger(ledger_path, done)
        rate = (i + 1) / (time.time() - t0)
        eta = (len(unit_list) - i - 1) / rate if rate > 0 else 0
        print(f"[merge] {unit_id} appended ({i + 1}/{len(unit_list)}, "
              f"{(time.time() - tu):.0f}s, ETA {eta/60:.0f}m)", flush=True)

    # ── validate the whole pool, then publish atomically ──────────────
    from climbmix.core.embedding_cluster import _validate_embeddings
    final = np.load(body_path, mmap_mode="r")
    if tuple(final.shape) != shape:
        raise SystemExit(
            f"[merge] FATAL: cache body shape {final.shape} != {shape} "
            "— bug in the append accounting; NOT publishing")
    ranges = [(int(s["start_idx"]), int(s["start_idx"]) + int(s["num_docs"]),
               os.path.basename(s["path"])) for s in shard_infos]
    _validate_embeddings(final, "[Embed-Merge]", ranges=ranges)
    del final
    os.replace(body_path, cache_path)
    print(f"[merge] wrote {cache_path} "
          f"({os.path.getsize(cache_path) / (1024**3):.1f} GB)")

    # cleanup partial-state files (the cache is the product now)
    for junk in (ledger_path,):
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
