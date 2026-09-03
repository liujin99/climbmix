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
at the PRODUCTION prefix (the backend config's obs-prefix knob) before
a full run; the smoke can stay on the calibration prefix):
  {prefix}/embed_units/{unit_id}/spec.json        — written here
  {prefix}/embed_units/{unit_id}/result/          — worker uploads
The dispatcher is fresh-prefix self-sufficient: it uploads the worker
bundle to {prefix}/assets and the assets_big placeholder itself.

Where the FINAL cache lives: the merged pool cache is a Step-1 cache
hit written at `<EMBEDDING_CACHE_DIR>/<content-key>/embedding_cache.npy`
(the key hashes shard names+sizes+model+truncate-len — see
CLIMBPipeline._pool_embedding_cache_dir). Set EMBEDDING_CACHE_DIR to
the production tree (e.g. <data-mix-run>/climbmix/cache/embeddings)
and the merge lands exactly where run_climbmix.sh Step 1 looks. The
OBS unit partials (~475 GB) are the durable tier: keep them — a wiped
local disk re-merges in ~1-2h instead of re-embedding 40h.

Container data plane (per-launch DIRECT mounts — the embed wave's own
assets stage into ITS jobs only, never into the global backend config
where every job class would stage them too; the backend's submit()
treats them as a REPLACE of its global asset mounts):
  --pool-uri  <obs dir of the parquet pool>  -> inputs/pool_0/
  --model-uri <obs dir of the model>         -> inputs/stella_0/
(Unset flags fall back to same-named direct mounts declared in the
backend config — e.g. a mock simulation. --model-container-dir instead
points the spec at a NON-staged model path, for mock simulation or
non-standard layouts.)

Smoke (fast iteration — one 8-card job, 2 shards, ~10 min end to end):
  python3 scripts/embed_dispatch.py \
      --remote-config <backend repo>/config/remote_config.json \
      --shard-info /home/ma-user/work/100B_stem_parquet_filtered/metadata_shard_info.json \
      --smoke 2 --flavor <backend 8-card flavor> --npu-per-job 8 \
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

Wave mode (full pool): --max-jobs units run CONCURRENTLY (default 6 =
48 cards at --npu-per-job 8; the pool is shared, don't take it all).
Mirrors RemoteExecutor's concurrency discipline exactly:
  - every obs storage call goes through one coarse lock (the backend
    clients are not documented thread-safe);
  - submit() retries TransientSubmitError with exponential backoff
    (RemoteConfig.submit_retry_* knobs) — a full pool rejects now and
    frees minutes later;
  - job_api.status/logs/cancel are called unlocked (the search wave
    proved this live at 6 slots);
  - a FAILED unit never stops its siblings — the wave drains every
    unit, then prints a retry list (re-running the same command
    resume-skips everything that succeeded).
--compare-local re-embeds serially (one local NPU) while remote units
keep flying.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)  # source-tree fallback (no pip install needed)

DEFAULT_CONTAINER_WORK_ROOT = "/home/ma-user/work/climbmix_exp"
POOL_MOUNT = "pool"      # obs.asset_mounts key for the parquet pool
MODEL_MOUNT = "stella"   # obs.asset_mounts key for the model dir
DEFAULT_MODEL = "NovaSearch/stella_en_400M_v5"

# Wave-mode concurrency discipline (mirrors RemoteExecutor): one coarse
# lock around EVERY obs storage call, one for multi-line prints, one
# serializing --compare-local's local-NPU re-embeds.
_OBS_LOCK = threading.Lock()
_PRINT_LOCK = threading.Lock()
_COMPARE_LOCK = threading.Lock()


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
    gateway also validates the dir exists). Mirrors
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
    """The gateway validates EVERY input mount as an existing OBS dir,
    and the adapter mounts {prefix}/assets_big — a
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
            "path": f"{args.container_input_base}/{POOL_MOUNT}_0/"
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
        "model": getattr(args, "model", DEFAULT_MODEL),
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


def wait_job(job_api, job_id, timeout_s, unit_id, poll_s=30.0,
             print_s=300.0):
    """Terminal status; prints a log tail while waiting (mirrors the
    executor's _wait_job, minus the experiment coupling). poll_s stays
    tight (a cheap status call — fast completion detection and accurate
    timeouts), but the CHATTER follows the executor's
    status_print_interval_s: embed jobs run tens of minutes to hours,
    minute-level prints are noise."""
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
        if time.time() - last >= print_s:
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


def sampled_unit_rows(shards, n_samples, seed=0):
    """Deterministic sample of the unit's rows -> [(global_row, shard,
    in_shard_row), ...] in global-row order. Pure mapping logic (no
    model, no NPU) so the fast-compare criterion is unit-testable."""
    import random
    total = sum(int(s["num_docs"]) for s in shards)
    n = max(0, min(int(n_samples), total))
    if not n:
        return []
    rows = sorted(random.Random(seed).sample(range(total), n))
    out = []
    off = 0
    for s in shards:
        lo, hi = off, off + int(s["num_docs"])
        out.extend((g, s, g - lo) for g in rows if lo <= g < hi)
        off = hi
    return out


def compare_sampled(unit_id, shards, partial_path, args, pool_local_dir,
                    n_samples):
    """FAST smoke criterion: byte-compare a random SAMPLE of N docs.

    The fake-xformers attention path is per-document independent (the
    pad+bool-mask fallback runs the identical computation for every
    batch shape), so equality on sampled rows proves the same math
    produced the remote block — a 2048-doc sample turns the ~40-min
    single-card full re-embed into ~1 min while keeping the
    byte-identity bar. The reference math mirrors
    _embed_streaming_worker exactly: _load_model_stream, fp16,
    model(features), .float(), L2 normalize, fp32."""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from climbmix.core.embedding_cluster import (
        _load_model_stream, _repair_stella_buffers)

    picks = sampled_unit_rows(shards, n_samples)
    total_rows = sum(int(s["num_docs"]) for s in shards)
    if not picks:
        print(f"[compare] FATAL: sampled 0 of {total_rows} rows")
        return False

    # one parquet read per touched shard
    by_file = {}
    for g, s, r in picks:
        by_file.setdefault(os.path.basename(s["path"]), []).append((g, r))
    texts_by_row = {}
    for fname, grels in by_file.items():
        table = pq.read_table(os.path.join(pool_local_dir, fname),
                              columns=[args.text_col], use_threads=True)
        col = table.column(args.text_col).to_pylist()
        for g, r in grels:
            v = col[r]
            texts_by_row[g] = str(v) if v is not None else ""

    rows = [g for g, _, _ in picks]
    texts = [texts_by_row[g] for g in rows]

    print(f"[compare] local reference embed: {len(rows)}/{total_rows:,} "
          f"sampled docs via _load_model_stream (NPU 0)")
    model = _load_model_stream(args.local_model_dir, "npu")
    model.eval()
    _repair_stella_buffers(model)
    model.max_seq_length = args.truncate_len
    model.half()

    def _to_device(features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to("npu")
            elif isinstance(features[key], dict):
                for k2 in features[key]:
                    if isinstance(features[key][k2], torch.Tensor):
                        features[key][k2] = features[key][k2].to("npu")
        return features

    ref = np.zeros((len(rows), args.emb_dim), dtype=np.float32)
    bs = args.batch_size
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        features = _to_device(model.tokenize(chunk))
        with torch.no_grad():
            out = model(features)
        emb = out["sentence_embedding"].float()
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        ref[i:i + len(chunk)] = emb.cpu().numpy()

    remote = np.load(partial_path)["embeddings"]
    if remote.shape != (total_rows, args.emb_dim):
        print(f"[compare] SHAPE MISMATCH: remote {remote.shape} vs "
              f"({total_rows}, {args.emb_dim})")
        return False
    sel = np.array(rows)
    equal = bool(np.array_equal(remote[sel], ref))
    diff = remote[sel].astype(np.float64) - ref.astype(np.float64)
    max_abs = float(np.abs(diff).max()) if diff.size else 0.0
    print(f"[compare] sampled byte-identical: {equal} | "
          f"max|diff|={max_abs:.3e} | "
          f"ref_norm={float(np.linalg.norm(ref[:5])):.4f} "
          f"remote_norm={float(np.linalg.norm(remote[rows[0]:rows[0] + 5])):.4f}")
    return equal or max_abs == 0.0


def submit_with_retry(job_api, name, command, remote_config, unit_id,
                      asset_mounts=None):
    """submit() with exponential backoff on TransientSubmitError.

    Port of RemoteExecutor._submit_with_retry (same RemoteConfig knobs,
    same semantics): the shared pool fluctuates, so a rejection now often
    fits minutes later — a retrying thread holds a wave slot but no
    resources. Hard (non-transient) errors raise immediately.
    asset_mounts: per-launch direct mounts (replace semantics) —
    threaded through every retry attempt."""
    from climbmix.remote.job_api import TransientSubmitError
    backoff = remote_config.submit_retry_initial_s
    deadline = time.time() + remote_config.submit_retry_timeout_s
    attempt = 0
    last_warn = 0.0
    while True:
        attempt += 1
        try:
            return job_api.submit(name=name, command=command, env={},
                                  asset_mounts=asset_mounts)
        except TransientSubmitError as e:
            if time.time() >= deadline:
                raise RuntimeError(
                    f"unit {unit_id}: submit still rejected after "
                    f"{attempt} attempts over "
                    f"{remote_config.submit_retry_timeout_s/60:.0f}m; "
                    f"last error: {e}") from e
            now = time.time()
            if attempt == 1 or now - last_warn >= 300.0:
                print(f"  [{unit_id}] submit rejected (attempt "
                      f"{attempt}: {e}) — capacity full? backing off "
                      f"{backoff:.0f}s", flush=True)
                last_warn = now
            time.sleep(backoff)
            backoff = min(backoff * 2, remote_config.submit_retry_max_s)


def dispatch_unit(unit_id, shards, job_api, obs, args, model_dir,
                  remote_config):
    """One unit end to end: spec -> upload -> submit -> wait -> download.

    shards: the ORIGINAL shard_info dicts (with path/num_docs/start_idx)
    plus the derived unit-local layout. Returns "ok", "skipped" (partial
    already on OBS) or "failed"."""
    from climbmix.remote.job_api import JobStatus
    unit_prefix = f"{args.obs_prefix.rstrip('/')}/embed_units/{unit_id}"
    result_uri = f"{unit_prefix}/result"
    out_dir = os.path.join(args.output_dir, unit_id)
    os.makedirs(out_dir, exist_ok=True)

    with _OBS_LOCK:
        already = obs.stat(f"{result_uri}/partial_block.npz")
    if not args.force and already:
        print(f"[{unit_id}] partial already on OBS — skipping (resume)")
        return "skipped"

    spec = build_spec(unit_id, shards, args, args.pool_uri, model_dir)
    spec_uri = f"{unit_prefix}/spec.json"
    with _OBS_LOCK:
        obs.upload_bytes(json.dumps(spec, indent=2).encode("utf-8"),
                         spec_uri)

    # command[1]: an ABSOLUTE worker path — the real backend's boot shell
    # replaces it with its code-dir copy anyway, and the mock backend
    # executes the argv verbatim (needs something runnable)
    command = ["python3",
               os.path.abspath(os.path.join(_HERE, "embed_worker.py")),
               "--spec-uri", spec_uri, "--storage", "moxing"]
    print(f"[{unit_id}] submitting ({len(shards)} shards, "
          f"{spec['shards'][0]['num_docs']:,}+ docs/shard, "
          f"{args.npu_per_job} cards)")
    job_id = submit_with_retry(job_api, f"climbmix_embed_{unit_id}",
                               command, remote_config, unit_id,
                               getattr(args, "launch_mounts", None))
    print(f"[{unit_id}] job {job_id} submitted")

    st = wait_job(job_api, job_id, args.job_timeout_s, unit_id,
                  print_s=getattr(remote_config,
                                  "status_print_interval_s", 300.0))
    if st != JobStatus.SUCCEEDED:
        with _PRINT_LOCK:
            print(f"[{unit_id}] job {job_id} -> {st.value}; logs (tail):\n"
                  f"{job_api.logs(job_id, 40)}")
        return "failed"

    ok = True
    with _OBS_LOCK:
        for name in ("result.json", "manifest.json", "partial_block.npz",
                     "embed.log"):
            uri = f"{result_uri}/{name}"
            if obs.stat(uri):
                if name == "partial_block.npz" and args.compare_local is None:
                    # 7.5 GB per unit with NO consumer in wave mode (the
                    # merge re-downloads from OBS itself; only the smoke's
                    # --compare-local reads the local copy) — and the
                    # download would hold _OBS_LOCK, stalling every
                    # sibling unit's polls and submits for its duration.
                    # The stat above still gates its OBS presence.
                    continue
                obs.download_file(uri, os.path.join(out_dir, name))
            elif name in ("result.json", "manifest.json", "partial_block.npz"):
                print(f"[{unit_id}] MISSING {name} at {uri}")
                ok = False
    if not ok:
        return "failed"

    with open(os.path.join(out_dir, "result.json")) as f:
        res = json.load(f)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        manifest = json.load(f)
    print(f"[{unit_id}] embed_rc={res.get('embed_rc')} "
          f"docs={res.get('docs'):,} elapsed={res.get('elapsed_seconds')}s "
          f"validation={manifest.get('validation')}")
    return "ok" if res.get("embed_rc") == 0 else "failed"


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote-config", required=True,
                   help="RemoteConfig JSON (backend resolution + obs_prefix)")
    p.add_argument("--shard-info", required=True,
                   help="pool metadata_shard_info.json (per_shard_info)")
    p.add_argument("--pool-uri", required=False,
                   help="OBS dir of the parquet pool (per-launch direct "
                        "mount; default: the 'pool' direct mount from "
                        "the backend config)")
    p.add_argument("--model-uri", default="",
                   help="OBS dir of the embedding model (per-launch "
                        "direct mount; default: the 'stella' direct "
                        "mount from the backend config)")
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
    p.add_argument("--max-jobs", type=int, default=6,
                   help="wave mode: units dispatched CONCURRENTLY (default "
                        "6 = 48 cards at --npu-per-job 8; the pool is "
                        "shared — don't take all of it)")
    p.add_argument("--shard-offset", type=int, default=0,
                   help="skip the first N shards (partial waves)")
    p.add_argument("--npu-per-job", type=int, default=8)
    p.add_argument("--flavor", default="",
                   help="job flavor override (e.g. the backend's 8-card "
                        "flavor; empty = the backend config's default)")
    p.add_argument("--text-col", default="text")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="embedding model NAME (goes into the spec -> "
                        "manifest; embed_merge cross-checks it against "
                        "the run config's discovery.embedding_model)")
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
    p.add_argument("--compare-samples", type=int, default=0, metavar="N",
                   help="with --compare-local: byte-compare a random "
                        "SAMPLE of N docs instead of re-embedding the "
                        "whole unit (fast smoke criterion — the "
                        "per-document attention path makes sampled "
                        "equality as binding as full; 0 = full compare)")
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

    # where staged input mounts land in the container ({base}/{name}_0)
    # — a backend-declared convention (platform-specific value lives in
    # the backend repo, never here)
    args.container_input_base = getattr(bundle, "container_input_base",
                                        "") or ""
    if not args.container_input_base:
        print("FATAL: the backend bundle provides no container_input_base "
              "(the container dir staged mounts land under) — embed jobs "
              "read the pool/model through staged mounts")
        return 2

    mounts = asset_mount_uris(remote_config, bundle)
    args.pool_uri = args.pool_uri or mounts.get(POOL_MOUNT)
    model_uri = args.model_uri or mounts.get(MODEL_MOUNT)
    if not args.pool_uri:
        print("FATAL: no pool location. Pass --pool-uri (per-launch "
              "staging — the pool must NOT live in the global backend "
              "config: every job class would stage it)")
        return 2
    # per-launch direct mounts (replace semantics at the backend): the
    # embed wave's own assets stage into ITS jobs only
    launch_mounts = {POOL_MOUNT: args.pool_uri}
    if model_uri:
        model_dir = f"{args.container_input_base}/{MODEL_MOUNT}_0"
        launch_mounts[MODEL_MOUNT] = model_uri
    elif args.model_container_dir:
        # explicit override (mock simulation / non-standard layouts)
        model_dir = args.model_container_dir
    else:
        print("FATAL: no model location. Pass --model-uri (per-launch "
              "staging) or --model-container-dir (non-staged path)")
        return 2
    args.launch_mounts = launch_mounts
    print(f"[dispatch] pool: {args.pool_uri} "
          f"(per-launch mount {POOL_MOUNT!r})")
    print(f"[dispatch] model dir (container view): {model_dir}")
    # the model dir must actually have content on OBS — a typo'd mount
    # path would burn a job at model-load time (~20 min round trip)
    if model_uri and not obs.list_objects(model_uri):
        print(f"FATAL: model dir {model_uri} is EMPTY on OBS — "
              f"upload the model first")
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
        print(f"[dispatch] local compare against pool dir: "
              f"{pool_local_dir} "
              f"({'sampled ' + format(args.compare_samples, ',') + ' docs' if args.compare_samples else 'full re-embed'})")

    max_jobs = max(1, args.max_jobs)
    print(f"[dispatch] wave: {len(units)} unit(s), max {max_jobs} in flight")

    def run_unit(unit_id, shards):
        """One wave slot: dispatch (+ optional local compare).

        NEVER raises — a failed unit takes itself down, not the wave:
        siblings keep their slots and the end-of-wave summary lists the
        retry set."""
        try:
            status = dispatch_unit(unit_id, shards, job_api, obs, args,
                                   model_dir, remote_config)
        except Exception as e:
            print(f"[{unit_id}] EXCEPTION: {type(e).__name__}: {e}")
            status = "failed"
        if status != "ok":
            return status
        if pool_local_dir is not None:
            enriched = []
            base = build_spec(unit_id, shards, args, args.pool_uri, model_dir)
            for s, spec_s in zip(shards, base["shards"]):
                d = dict(s)
                d["unit_start_global"] = spec_s["unit_start"]
                enriched.append(d)
            partial = os.path.join(args.output_dir, unit_id,
                                   "partial_block.npz")
            with _COMPARE_LOCK:  # one local NPU — serialize re-embeds
                if args.compare_samples > 0:
                    ok = compare_sampled(unit_id, enriched, partial,
                                         args, pool_local_dir,
                                         args.compare_samples)
                else:
                    ok = compare_local(unit_id, enriched, partial,
                                       args, pool_local_dir)
            if not ok:
                print(f"[dispatch] unit {unit_id} local compare MISMATCH")
                return "failed"
        print(f"[dispatch] unit {unit_id} OK")
        return "ok"

    results = {}
    with ThreadPoolExecutor(max_workers=max_jobs) as pool:
        futs = {pool.submit(run_unit, uid, sh): uid for uid, sh in units}
        for fut, uid in futs.items():
            results[uid] = fut.result()

    ok_n = sum(1 for s in results.values() if s == "ok")
    skip_n = sum(1 for s in results.values() if s == "skipped")
    failed = sorted(u for u, s in results.items() if s == "failed")
    print(f"[dispatch] wave complete: {ok_n} ok, {skip_n} skipped, "
          f"{len(failed)} failed")
    if failed:
        print(f"[dispatch] failed units: {', '.join(failed)}")
        print("[dispatch] re-run the SAME command to retry them "
              "(completed units are resume-skipped)")
        return 1
    print("[dispatch] all units complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
