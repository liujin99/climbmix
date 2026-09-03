#!/usr/bin/env python3
"""Embed unit dispatcher (TODO E) — multi-node pool embedding over the
remote job fleet.

The pool (1000 parquet shards, ~116M docs) is split into units of
--unit-shards shards each. Every unit becomes one remote job: its spec
(kind == "embed") tells the worker which shards to embed; the job
uploads partial_block.npz + manifest.json back through its result
mount. Unit granularity = progress banked (a crashed job re-embeds
only its own in-flight shards) and retry cost stays bounded.

Layout (OBS, under the RemoteConfig obs_prefix — point the user config
at the PRODUCTION prefix via `ma_setup.py --obs-prefix` before a full
run; the smoke can stay on the calibration prefix):
  {prefix}/embed_units/{unit_id}/spec.json        — written here
  {prefix}/embed_units/{unit_id}/result/          — worker uploads
The dispatcher is fresh-prefix self-sufficient: it uploads the worker
bundle to {prefix}/assets and the assets_big placeholder itself.

Where the FINAL cache lives: the merged pool cache is a Step-1 cache
hit written at `<EMBEDDING_CACHE_DIR>/<content-key>/embedding_cache.npz`
(the key hashes shard names+sizes+model+truncate-len — see
CLIMBPipeline._pool_embedding_cache_dir). Set EMBEDDING_CACHE_DIR to
the production tree (e.g. <data-mix-run>/climbmix/cache/embeddings)
and the merge lands exactly where run_climbmix.sh Step 1 looks. The
OBS unit partials (~475 GB) are the durable tier: keep them — a wiped
local disk re-merges in ~1-2h instead of re-embedding 40h.

Container data plane (the resource package's DIRECT asset mounts —
declared once with ma_setup.py --resource-package, verified here):
  pool=<obs dir of the parquet pool>   -> /home/ma-user/modelarts/inputs/pool_0/
  stella=<obs dir of the model>        -> /home/ma-user/modelarts/inputs/stella_0/
(the pool is DATA, not a package asset — declare it separately with
--asset-mount pool=...)

Smoke (fast iteration — one 8-card job, 2 shards, ~10 min end to end):
  python3 scripts/embed_dispatch.py \
      --remote-config climbmix-ma/config/remote_config.ma.json \
      --shard-info /home/ma-user/work/100B_stem_parquet_filtered/metadata_shard_info.json \
      --smoke 2 --flavor modelarts.pool.visual.8xlarge --npu-per-job 8 \
      --output-dir /tmp/embed_smoke \
      --local-model-dir <server-local stella dir> \
      --compare-local /home/ma-user/work/100B_stem_parquet_filtered

  --compare-local re-embeds the SAME shards on the submit host through
  climbmix's own single-card path (_embed_streaming_worker) and demands
  byte-identical fp32 output — same model, same dtype, same math, so
  bitwise equality is the pass bar (a norm/max-diff report is printed
  alongside for diagnosis when it fails).

Resume: a unit whose {result}/partial_block.npz already exists on OBS
is SKIPPED (completed units are never re-embedded); --force re-dispatches
anyway. Smoke units are exp-id-safe: they live under embed_units/, never
touching the search experiment space.
"""

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # source-tree fallback (no pip install needed)

CONTAINER_INPUT_BASE = "/home/ma-user/modelarts/inputs"
DEFAULT_CONTAINER_WORK_ROOT = "/home/ma-user/work/climbmix_exp"
POOL_MOUNT = "pool"      # obs.asset_mounts key for the parquet pool
MODEL_MOUNT = "stella"   # obs.asset_mounts key for the model dir


def load_shard_info(path: str):
    """per_shard_info from the pool's metadata_shard_info.json."""
    with open(path) as f:
        data = json.load(f)
    infos = data.get("per_shard_info") or data
    if not isinstance(infos, list) or not infos:
        raise SystemExit(f"no per_shard_info list in {path}")
    return infos


def asset_mount_uris(remote_config, bundle):
    """{name: obs uri} the backend declares (direct mounts)."""
    hook = getattr(bundle, "asset_mounts", None)
    if hook is None:
        return {}
    return dict(hook(remote_config) or {})


def repo_root() -> str:
    """Source checkout root: scripts/.. when running from a checkout,
    else derived from the imported climbmix package (RemoteExecutor's
    staging uses the same trick)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isdir(os.path.join(here, "src", "climbmix")):
        return here
    import climbmix
    return os.path.normpath(
        os.path.join(os.path.dirname(climbmix.__file__), "..", ".."))


def ensure_worker_bundle(obs, prefix):
    """Stage + upload the worker bundle to {prefix}/assets.

    The backend's boot shell points the job's code dir at
    {prefix}/assets (the adapter's default code_dir), so a FRESH
    prefix needs the three files there BEFORE the first submit (the
    gateway also validates the dir exists, ModelArts.2802). Mirrors
    RemoteExecutor._stage_assets: local temp + atomic rename, then
    upload — concurrent dispatchers must never read a half-written
    file. The upload happens on every dispatcher run (idempotent
    refresh, same freshness semantics as the executor)."""
    import shutil
    root = repo_root()
    srcs = [os.path.join(root, "scripts", "remote_worker.py"),
            os.path.join(root, "scripts", "embed_worker.py"),
            os.path.join(root, "src", "climbmix", "pipeline",
                         "nanochat_cmds.py")]
    stage = os.path.join(root, "cache", "remote_assets")
    os.makedirs(stage, exist_ok=True)
    for src in srcs:
        if not os.path.isfile(src):
            raise SystemExit(f"worker bundle source missing: {src}")
        dst = os.path.join(stage, os.path.basename(src))
        tmp = f"{dst}.tmp.{os.getpid()}"
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
        obs.upload_file(dst, f"{prefix.rstrip('/')}/assets/"
                             f"{os.path.basename(src)}")
    print(f"[dispatch] worker bundle fresh -> {prefix.rstrip('/')}/assets")


def ensure_assets_big(obs, prefix):
    """The gateway validates EVERY input mount as an existing OBS dir
    (ModelArts.2791), and the adapter mounts {prefix}/assets_big — a
    fresh prefix needs the dir to exist. Embed jobs never read it (a
    placeholder file is inert: the boot shell skips unknown entries)."""
    marker = f"{prefix.rstrip('/')}/assets_big/.climbmix_placeholder"
    if not obs.stat(marker):
        obs.upload_bytes(b"placeholder - embed units mount no big assets",
                         marker)
        print(f"[dispatch] created placeholder assets_big under {prefix}")


def build_units(shard_infos, unit_shards, smoke):
    """[(unit_id, [shard_info, ...]), ...] in pool order."""
    if smoke:
        # smoke mode: exactly ONE unit of the first N shards (fast iteration)
        return [("smoke0000", shard_infos[:smoke])]
    units = []
    for ui in range(0, len(shard_infos), unit_shards):
        units.append((f"u{ui // unit_shards:04d}",
                      shard_infos[ui:ui + unit_shards]))
    return units


def build_spec(unit_id, shards, args, pool_uri, model_dir):
    """The kind == "embed" unit spec (see scripts/embed_worker.py)."""
    unit_start = 0
    spec_shards = []
    for s in shards:
        spec_shards.append({
            "path": f"{CONTAINER_INPUT_BASE}/{POOL_MOUNT}_0/"
                    f"{os.path.basename(s['path'])}",
            "num_docs": int(s["num_docs"]),
            "unit_start": unit_start,
            "global_start": int(s["start_idx"]),
        })
        unit_start += int(s["num_docs"])
    return {
        "spec_version": 1,
        "kind": "embed",
        "unit_id": unit_id,
        "text_col": args.text_col,
        "batch_size": args.batch_size,
        "truncate_len": args.truncate_len,
        "emb_dim": args.emb_dim,
        "model_dir": model_dir,
        "npu_count": args.npu_per_job,
        "shards": spec_shards,
        "work_dir": os.path.join(args.container_work_root,
                                 f"embed_{unit_id}"),
        "result_uri": f"{args.obs_prefix.rstrip('/')}/embed_units/"
                      f"{unit_id}/result",
        "log_stream_s": 30,
    }


def wait_job(job_api, job_id, timeout_s, unit_id, poll_s=30.0):
    """Terminal status; prints a log tail while waiting (mirrors the
    executor's _wait_job, minus the experiment coupling)."""
    from climbmix.remote.job_api import JobStatus
    t0 = time.time()
    last = 0.0
    while True:
        st = job_api.status(job_id)
        if st.is_terminal:
            return st
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            job_api.cancel(job_id)
            raise RuntimeError(
                f"unit {unit_id}: job {job_id} timed out after "
                f"{elapsed/60:.0f}m, cancelled")
        if time.time() - last >= 60.0:
            tail = (job_api.logs(job_id, 1) or "").strip()
            print(f"  [{unit_id}] job {job_id} {st.value} "
                  f"{elapsed/60:.0f}m | {tail[:120]}", flush=True)
            last = time.time()
        time.sleep(poll_s)


def compare_local(unit_id, shards, partial_path, args, pool_local_dir):
    """Re-embed the unit's shards on THIS host via climbmix's own
    single-card streaming worker; byte-compare against the remote block.

    The reference path is the exact function the single-node pool run
    uses (_embed_streaming_worker, worker 0) — so this checks the whole
    container pipeline (mount read, model load, fp16 math) against the
    production code path."""
    import numpy as np
    from climbmix.core.embedding_cluster import _embed_streaming_worker

    total_rows = sum(int(s["num_docs"]) for s in shards)
    emb_dim = args.emb_dim
    ref_dir = os.path.join(args.output_dir, f"{unit_id}_ref")
    os.makedirs(ref_dir, exist_ok=True)
    memmap_path = os.path.join(ref_dir, "ref_block.tmp")
    ref_block = np.memmap(memmap_path, dtype=np.float32, mode="w+",
                          shape=(total_rows, emb_dim))
    del ref_block

    # shard descriptors in _embed_streaming_worker's own terms: it writes
    # at GLOBAL start_idx, so give it a rebased start_idx (unit-local)
    shard_infos = [
        {"path": os.path.join(pool_local_dir, os.path.basename(s["path"])),
         "num_docs": int(s["num_docs"]),
         "start_idx": int(s["unit_start_global"])}
        for s in shards
    ]
    print(f"[compare] local reference embed: {total_rows:,} docs "
          f"via _embed_streaming_worker (NPU 0)")
    _embed_streaming_worker(
        0, list(range(len(shard_infos))), shard_infos, args.text_col,
        args.local_model_dir, args.batch_size, emb_dim, memmap_path,
        total_rows, [0], args.truncate_len, None)

    ref = np.memmap(memmap_path, dtype=np.float32, mode="r",
                    shape=(total_rows, emb_dim))
    remote = np.load(partial_path)["embeddings"]
    if remote.shape != ref.shape:
        print(f"[compare] SHAPE MISMATCH: remote {remote.shape} vs "
              f"ref {ref.shape}")
        return False
    equal = bool(np.array_equal(np.asarray(ref), remote))
    diff = np.asarray(ref, dtype=np.float64) - remote.astype(np.float64)
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    print(f"[compare] byte-identical: {equal} | max|diff|={max_abs:.3e} | "
          f"ref_norm={float(np.linalg.norm(np.asarray(ref)[:5])):.4f} "
          f"remote_norm={float(np.linalg.norm(remote[:5])):.4f}")
    return equal or max_abs == 0.0


def dispatch_unit(unit_id, shards, job_api, obs, args, model_dir):
    """One unit end to end: spec -> upload -> submit -> wait -> download.

    shards: the ORIGINAL shard_info dicts (with path/num_docs/start_idx)
    plus the derived unit-local layout. Returns True on success."""
    from climbmix.remote.job_api import JobStatus
    unit_prefix = f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
    result_uri = f"{unit_prefix}/result"
    out_dir = os.path.join(args.output_dir, unit_id)
    os.makedirs(out_dir, exist_ok=True)

    if not args.force and obs.stat(f"{result_uri}/partial_block.npz"):
        print(f"[{unit_id}] partial already on OBS — skipping (resume)")
        return True

    spec = build_spec(unit_id, shards, args, args.pool_uri, model_dir)
    spec_uri = f"{unit_prefix}/spec.json"
    obs.upload_bytes(json.dumps(spec, indent=2).encode("utf-8"), spec_uri)

    # command[1]: an ABSOLUTE worker path — the real backend's boot shell
    # replaces it with its code-dir copy anyway, and the mock backend
    # executes the argv verbatim (needs something runnable)
    command = ["python3",
               os.path.abspath(os.path.join(_HERE, "embed_worker.py")),
               "--spec-uri", spec_uri, "--storage", "moxing"]
    print(f"[{unit_id}] submitting ({len(shards)} shards, "
          f"{spec['shards'][0]['num_docs']:,}+ docs/shard, "
          f"{args.npu_per_job} cards)")
    job_id = job_api.submit(name=f"climbmix_embed_{unit_id}", command=command,
                            env={})
    print(f"[{unit_id}] job {job_id} submitted")

    st = wait_job(job_api, job_id, args.job_timeout_s, unit_id)
    if st != JobStatus.SUCCEEDED:
        print(f"[{unit_id}] job {job_id} -> {st.value}; logs (tail):\n"
              f"{job_api.logs(job_id, 40)}")
        return False

    ok = True
    for name in ("result.json", "manifest.json", "partial_block.npz",
                 "embed.log"):
        uri = f"{result_uri}/{name}"
        if obs.stat(uri):
            obs.download_file(uri, os.path.join(out_dir, name))
        elif name in ("result.json", "manifest.json", "partial_block.npz"):
            print(f"[{unit_id}] MISSING {name} at {uri}")
            ok = False
    if not ok:
        return False

    with open(os.path.join(out_dir, "result.json")) as f:
        res = json.load(f)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    print(f"[{unit_id}] embed_rc={res.get('embed_rc')} "
          f"docs={res.get('docs'):,} elapsed={res.get('elapsed_seconds')}s "
          f"validation={manifest.get('validation')}")
    return res.get("embed_rc") == 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote-config", required=True,
                   help="RemoteConfig JSON (backend resolution + obs_prefix)")
    p.add_argument("--shard-info", required=True,
                   help="pool metadata_shard_info.json (per_shard_info)")
    p.add_argument("--pool-uri", required=False,
                   help="OBS dir of the parquet pool (default: the pool "
                        "direct asset mount from the backend config)")
    p.add_argument("--model-dir", default="",
                   help="deprecated alias of --model-container-dir")
    p.add_argument("--model-container-dir", default="",
                   help="container path of the model dir for the spec "
                        "(default: the stella direct asset mount's "
                        "staged path; override for mock simulation or "
                        "non-standard layouts)")
    p.add_argument("--local-model-dir", default="",
                   help="LOCAL path of the same model for --compare-local")
    p.add_argument("--smoke", type=int, default=0,
                   help="smoke mode: first N shards as one unit, ignore "
                        "--unit-shards")
    p.add_argument("--unit-shards", type=int, default=16,
                   help="shards per unit (full-pool mode)")
    p.add_argument("--shard-offset", type=int, default=0,
                   help="skip the first N shards (partial waves)")
    p.add_argument("--npu-per-job", type=int, default=8)
    p.add_argument("--flavor", default="",
                   help="job flavor override (e.g. the 8-card "
                        "modelarts.pool.visual.8xlarge)")
    p.add_argument("--text-col", default="text")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--truncate-len", type=int, default=512)
    p.add_argument("--emb-dim", type=int, default=1024,
                   help="stella_en_400M_v5 output dim (the worker probes "
                        "the model and hard-fails on a mismatch)")
    p.add_argument("--job-timeout-s", type=float, default=2.0 * 3600)
    p.add_argument("--output-dir", default="./embed_out")
    p.add_argument("--force", action="store_true",
                   help="re-dispatch units that already have a partial")
    p.add_argument("--compare-local", nargs="?", const="", default=None,
                   metavar="POOL_LOCAL_DIR",
                   help="after a successful unit, re-embed locally via "
                        "climbmix's own path and byte-compare")
    args = p.parse_args()
    if args.model_dir and not args.model_container_dir:
        args.model_container_dir = args.model_dir

    from climbmix.remote.remote_executor import RemoteConfig
    from climbmix.remote.backends import resolve_backend
    remote_config = RemoteConfig.from_json_file(args.remote_config)
    if args.flavor:
        remote_config.flavor = args.flavor
    remote_config.npu_per_job = args.npu_per_job
    # the spec's work_dir uses the backend's container convention (the
    # mock simulation overrides it to a local path)
    args.container_work_root = (remote_config.container_work_root
                                 or DEFAULT_CONTAINER_WORK_ROOT)
    bundle = resolve_backend(remote_config)
    job_api = bundle.make_job_api(remote_config)
    obs = bundle.make_obs_storage(remote_config)

    args.obs_prefix = remote_config.obs_prefix

    mounts = asset_mount_uris(remote_config, bundle)
    args.pool_uri = args.pool_uri or mounts.get(POOL_MOUNT)
    if MODEL_MOUNT in mounts:
        model_dir = f"{CONTAINER_INPUT_BASE}/{MODEL_MOUNT}_0"
    elif args.model_container_dir:
        # explicit override (mock simulation / non-standard layouts)
        model_dir = args.model_container_dir
    else:
        print(f"FATAL: no {MODEL_MOUNT!r} asset mount. Declare it once:\n"
              "  python3 climbmix-ma/scripts/ma_setup.py "
              f"--asset-mount {MODEL_MOUNT}=obs://<bucket>/<model dir>")
        return 2
    if args.pool_uri:
        src = f"direct mount {mounts.get(POOL_MOUNT)}"
        print(f"[dispatch] pool: {args.pool_uri} ({src})")
    else:
        print("FATAL: no pool location. Pass --pool-uri or declare the "
              "direct mount once:\n  python3 climbmix-ma/scripts/ma_setup.py "
              f"--asset-mount {POOL_MOUNT}=obs://<bucket>/<pool dir>")
        return 2
    print(f"[dispatch] model dir (container view): {model_dir}")
    # the model dir must actually have content on OBS — a typo'd mount
    # path would burn a job at model-load time (~20 min round trip)
    if MODEL_MOUNT in mounts and not obs.list_objects(mounts[MODEL_MOUNT]):
        print(f"FATAL: stella direct mount {mounts[MODEL_MOUNT]} is EMPTY "
              f"on OBS — upload the model first")
        return 2

    # fresh-prefix readiness: worker bundle + assets_big placeholder
    ensure_worker_bundle(obs, args.obs_prefix)
    ensure_assets_big(obs, args.obs_prefix)

    shard_infos = load_shard_info(args.shard_info)
    if args.shard_offset:
        shard_infos = shard_infos[args.shard_offset:]
    units = build_units(shard_infos, args.unit_shards, args.smoke)
    print(f"[dispatch] {len(shard_infos)} shards in pool info; "
          f"{len(units)} unit(s) to dispatch")

    # pool sanity: every unit's shard files must exist on OBS (the pool
    # mount is what the container will read through)
    for unit_id, shards in units:
        for s in shards:
            uri = f"{args.pool_uri.rstrip('/')}/{os.path.basename(s['path'])}"
            if not obs.stat(uri):
                print(f"FATAL: pool object missing on OBS: {uri}")
                return 2

    pool_local_dir = None
    if args.compare_local is not None:
        pool_local_dir = args.compare_local or os.path.dirname(
            os.path.abspath(args.shard_info))
        if not args.local_model_dir:
            print("FATAL: --compare-local needs --local-model-dir")
            return 2
        print(f"[dispatch] local compare against pool dir: {pool_local_dir}")

    for unit_id, shards in units:
        if not dispatch_unit(unit_id, shards, job_api, obs, args, model_dir):
            print(f"[dispatch] unit {unit_id} FAILED — stopping")
            return 1
        if pool_local_dir is not None:
            enriched = []
            base = build_spec(unit_id, shards, args, args.pool_uri, model_dir)
            for s, spec_s in zip(shards, base["shards"]):
                d = dict(s)
                d["unit_start_global"] = spec_s["unit_start"]
                enriched.append(d)
            ok = compare_local(
                unit_id, enriched,
                os.path.join(args.output_dir, unit_id, "partial_block.npz"),
                args, pool_local_dir)
            if not ok:
                print(f"[dispatch] unit {unit_id} local compare MISMATCH")
                return 1
        print(f"[dispatch] unit {unit_id} OK\n")

    print("[dispatch] all units complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
