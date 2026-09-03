#!/usr/bin/env python3
"""Embed unit worker — runs INSIDE a remote job container (TODO E).

Invoked by remote_worker.py when the spec's kind == "embed" (the embed
dispatcher, scripts/embed_dispatch.py, builds the spec on the submit
host). Standalone by design: stdlib only — the container has NO climbmix
source. The embedding math is a faithful port of climbmix's
_embed_streaming_worker (core/embedding_cluster.py): fp16 inference,
fp32 L2-normalized output, NaN retry at batch size 1, tokenize/IO
prefetch pools, the stella buffer repair. Keep the two in sync when
touching either.

Unit spec (JSON, spec_version 1):
  kind: "embed"
  unit_id: "u0000"                  — unit label (logging + result files)
  text_col, batch_size, truncate_len, emb_dim
  model_dir: container path to the SentenceTransformer model directory
             (a direct OBS asset mount — no network in the container)
  npu_count: cards in this job; shards are split round-robin over cards
  shards: [{path, num_docs, unit_start, global_start}, ...]
      path         — container path of the pool parquet shard
      num_docs     — docs in the shard
      unit_start   — first row of this shard inside the unit block
      global_start — first row of this shard inside the full-pool cache
  result_uri: obs:// prefix for partial_block.npz + manifest.json

Child processes (one per card) are spawned as SUBPROCESSES of this
script (--child): fresh interpreters, no fork/spawn pickle coupling to
the parent. Each child embeds its shard slice into the unit-local
memmap (rows rebased to the unit block) and keeps a JSON ledger with
tmp+rename atomicity — re-running a crashed unit re-embeds only the
shards not in any ledger (the same resume semantics as the single-node
pool run, banked at unit granularity).

Exit code: 0 iff every child exits 0 and the post-embed validation
passes. partial_block.npz (float32, unit rows in shard order) and
manifest.json are uploaded to {result_uri} on success.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

SPEC_VERSION = 1


# ── ledger (ported from embedding_cluster._write_worker_progress) ─────────

def write_ledger(path, completed, shard_idx):
    """Merge shard_idx into the ledger file; tmp+rename, plain ints."""
    if not path:
        return
    entries = list(completed or [])
    entries.append(int(shard_idx))
    entries = sorted(set(int(e) for e in entries))
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"completed": entries}, f)
    os.replace(tmp, path)


def load_ledger(path):
    """Shard indices recorded in one ledger file (garbage-tolerant)."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return [int(e) for e in data.get("completed", [])]
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return []


# ── model loading + repair (ported from embedding_cluster) ────────────────

def repair_stella_buffers(model):
    """Re-init stella's non-persistent buffers corrupted at load on
    torch_npu (position_ids / inv_freq / cos_cached / sin_cached hold
    uninitialized heap memory on NPU -> OOB rope indexing -> NaN)."""
    import torch
    try:
        inner = model[0].model          # SentenceTransformer[0].model
        embeddings = inner.embeddings
        cfg = inner.config
        device = inner.device
    except (AttributeError, IndexError, TypeError):
        return  # not a stella model; nothing to do

    max_pos = cfg.max_position_embeddings
    embeddings.register_buffer(
        "position_ids", torch.arange(max_pos, device=device), persistent=False)
    embeddings._init_rope(cfg)
    embeddings.rotary_emb = embeddings.rotary_emb.to(device)
    assert torch.equal(embeddings.position_ids,
                       torch.arange(max_pos, device=device)), \
        "position_ids repair failed — still not arange"


def load_model(model_dir, device):
    """SentenceTransformer with xformers/SDPA fallback (same ladder as
    embedding_cluster._load_model_stream, but from a LOCAL model dir)."""
    from sentence_transformers import SentenceTransformer
    try:
        import xformers  # noqa: F401
        return SentenceTransformer(model_dir, device=device,
                                   trust_remote_code=True)
    except ImportError:
        pass
    for impl in ["sdpa", "eager"]:
        try:
            return SentenceTransformer(
                model_dir, device=device, trust_remote_code=True,
                model_kwargs={"attn_implementation": impl})
        except (KeyError, AssertionError, ValueError, TypeError) as e:
            print(f"[embed] attn_implementation={impl} failed: {e}")
    raise RuntimeError("Failed to load model: no compatible attention "
                       "implementation")


# ── child: embed one card's shard slice into the unit memmap ──────────────

def child_main(args) -> int:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.worker_id)
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_SILENT"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    with open(args.spec_path) as f:
        spec = json.load(f)
    shard_ids = [int(x) for x in args.shards.split(",") if x != ""]

    import numpy as np
    import torch
    import pyarrow.parquet as pq
    from concurrent.futures import ThreadPoolExecutor

    wid = args.worker_id
    text_col = spec["text_col"]
    truncate_len = int(spec.get("truncate_len", 512))
    batch_size = int(spec.get("batch_size", 512))
    emb_dim = int(spec["emb_dim"])
    shards = spec["shards"]

    print(f"[card {wid}] Loading model: {spec['model_dir']}", flush=True)
    model = load_model(spec["model_dir"], "npu")
    model.eval()
    repair_stella_buffers(model)
    model.max_seq_length = truncate_len
    model.half()
    # config sanity: a wrong --emb-dim would mis-shape the memmap and
    # burn the whole job; verify against the model's real output dim
    probe = model.encode(["dim probe"], show_progress_bar=False,
                         normalize_embeddings=True)
    if probe.shape[1] != emb_dim:
        print(f"[card {wid}] FATAL: model dim {probe.shape[1]} != spec "
              f"emb_dim {emb_dim}", flush=True)
        return 2
    print(f"[card {wid}] Model loaded (fp16, max_seq_len={truncate_len}, "
          f"dim={emb_dim})", flush=True)

    total_rows = sum(int(s["num_docs"]) for s in shards)
    all_emb = np.memmap(args.memmap, dtype=np.float32, mode="r+",
                        shape=(total_rows, emb_dim))

    def _to_device(features):
        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to("npu")
            elif isinstance(features[key], dict):
                _to_device(features[key])
        return features

    def _read_shard_texts(sinfo):
        table = pq.read_table(sinfo["path"], columns=[text_col],
                              use_threads=True)
        texts = [str(v) if v is not None else ""
                 for v in table.column(text_col).to_pylist()]
        del table
        return texts

    io_pool = ThreadPoolExecutor(max_workers=1)
    tok_pool = ThreadPoolExecutor(max_workers=1)

    ledger_path = args.ledger
    completed = load_ledger(ledger_path)
    todo = [si for si in shard_ids if si not in completed]
    if not todo:
        print(f"[card {wid}] All {len(shard_ids)} shards already in the "
              f"ledger, exiting", flush=True)
        return 0

    print(f"[card {wid}] Encoding {len(todo)}/{len(shard_ids)} shards "
          f"(batch_size={batch_size})", flush=True)
    next_future = io_pool.submit(_read_shard_texts, shards[todo[0]])

    docs_done = 0
    nan_count = 0
    t0 = time.time()

    for k, si in enumerate(todo):
        texts = next_future.result()
        if k + 1 < len(todo):
            next_future = io_pool.submit(_read_shard_texts, shards[todo[k + 1]])

        sinfo = shards[si]
        unit_start = int(sinfo["unit_start"])
        num_docs = int(sinfo["num_docs"])

        tok_future = tok_pool.submit(model.tokenize, texts[:batch_size])
        for j in range(0, len(texts), batch_size):
            batch_len = min(batch_size, len(texts) - j)
            features = tok_future.result()
            next_j = j + batch_size
            if next_j < len(texts):
                tok_future = tok_pool.submit(
                    model.tokenize, texts[next_j:next_j + batch_size])

            features = _to_device(features)
            with torch.no_grad():
                output = model(features)
            emb = output["sentence_embedding"].float()
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            emb = emb.cpu().numpy()

            nan_mask = np.isnan(emb).any(axis=1)
            if nan_mask.any():
                nan_count += int(nan_mask.sum())
                for r in np.where(nan_mask)[0]:
                    retry = _to_device(model.tokenize([texts[j + r]]))
                    with torch.no_grad():
                        out2 = model(retry)
                    e2 = out2["sentence_embedding"].float()
                    e2 = torch.nn.functional.normalize(e2, p=2, dim=1)
                    emb[r] = e2.cpu().numpy()[0]
                    del retry, out2
                if np.isnan(emb).any(axis=1).sum() > 0:
                    raise RuntimeError(
                        f"[card {wid}] NaN persists after bs=1 retry — "
                        f"aborting to prevent data loss")

            del features, output
            all_emb[unit_start + j:unit_start + j + batch_len] = emb

        docs_done += num_docs
        del texts
        write_ledger(ledger_path, load_ledger(ledger_path), si)
        print(f"[card {wid}] shard {si} done "
              f"({docs_done:,} docs, {docs_done/(time.time()-t0):.0f} docs/s)",
              flush=True)

    io_pool.shutdown()
    tok_pool.shutdown()
    all_emb.flush()
    nan_msg = f", {nan_count} NaN recovered" if nan_count else ""
    print(f"[card {wid}] Done: {docs_done:,} docs in "
          f"{time.time()-t0:.0f}s ({docs_done/(time.time()-t0):.0f} "
          f"docs/s{nan_msg})", flush=True)
    return 0


# ── parent: spawn children, validate, package, upload ─────────────────────

def validate_block(arr, ranges):
    """Chunked NaN/Inf/zero-norm/off-norm net (ported from
    _validate_embeddings, L2-normalized rows must sit at norm ~1)."""
    import numpy as np
    n, dim = arr.shape
    rows_per_chunk = max(1, min(65536, (1 << 28) // max(1, dim)))
    counts = {"nan": 0, "inf": 0, "zero": 0, "off_norm": 0}
    per_range = [0] * len(ranges)
    starts = [r[0] for r in ranges]
    for lo in range(0, n, rows_per_chunk):
        c = np.asarray(arr[lo:lo + rows_per_chunk])
        bad = np.isnan(c).any(axis=1) | np.isinf(c).any(axis=1)
        sq = np.einsum("ij,ij->i", c, c, dtype=np.float64)
        zero = (sq <= 1e-12) & ~bad
        off = (np.abs(sq - 1.0) > 0.01) & ~bad & ~zero
        flags = bad | zero | off
        if flags.any():
            for r in np.where(flags)[0]:
                row = lo + int(r)
                idx = max(i for i, s in enumerate(starts) if s <= row)
                per_range[idx] += 1
            counts["nan"] += int(np.isnan(c).any(axis=1).sum())
            counts["inf"] += int(np.isinf(c).any(axis=1).sum())
            counts["zero"] += int(zero.sum())
            counts["off_norm"] += int(off.sum())
    total = sum(counts.values())
    if total:
        for i, cnt in enumerate(per_range):
            if cnt:
                print(f"[embed] {cnt} anomalous rows in {ranges[i][2]}",
                      flush=True)
    return counts


def parent_main(args) -> int:
    with open(args.spec_path) as f:
        spec = json.load(f)

    if spec.get("spec_version") != SPEC_VERSION:
        print(f"[embed] FATAL: spec_version {spec.get('spec_version')!r} "
              f"!= {SPEC_VERSION}", flush=True)
        return 2

    work = args.work_dir or os.path.dirname(os.path.abspath(args.spec_path))
    os.makedirs(work, exist_ok=True)
    memmap_path = os.path.join(work, "unit_block.tmp")
    shards = spec["shards"]
    n_npus = int(spec.get("npu_count", 1))
    total_rows = sum(int(s["num_docs"]) for s in shards)
    emb_dim = int(spec["emb_dim"])
    unit_id = spec.get("unit_id", "unit")

    t0 = time.time()
    res = {
        "kind": "embed",
        "unit_id": unit_id,
        "embed_rc": -1,
        "docs": total_rows,
        "shards": len(shards),
        "elapsed_seconds": 0.0,
        "error": None,
    }

    def finish(rc):
        res["elapsed_seconds"] = round(time.time() - t0, 1)
        try:
            local = os.path.join(work, "result.json")
            with open(local, "w") as f:
                json.dump(res, f, indent=2)
        except Exception:
            traceback.print_exc()
        return rc

    try:
        # unit-local block; resume reuses it when shape matches
        expected_bytes = total_rows * emb_dim * 4
        if not (os.path.exists(memmap_path)
                and os.path.getsize(memmap_path) == expected_bytes):
            for junk in (memmap_path,):
                if os.path.exists(junk):
                    os.remove(junk)
            with open(memmap_path, "wb") as f:
                f.truncate(expected_bytes)
            print(f"[embed] Preallocated unit block "
                  f"({total_rows:,} x {emb_dim})", flush=True)
        else:
            print(f"[embed] Resuming: unit block intact "
                  f"({total_rows:,} rows)", flush=True)

        # round-robin shard->card; ledgers skip already-done shards
        todo = list(range(len(shards)))
        if n_npus > 1:
            groups = [todo[i::n_npus] for i in range(n_npus)]
        else:
            groups = [todo]
        env = dict(os.environ)
        procs = []
        for wid, group in enumerate(groups):
            ledger = os.path.join(work, f"embed_progress_w{wid}.json")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--child", "--worker-id", str(wid),
                   "--spec-path", os.path.abspath(args.spec_path),
                   "--memmap", memmap_path, "--ledger", ledger,
                   "--shards", ",".join(str(s) for s in group)]
            procs.append(subprocess.Popen(cmd, env=env))
        failed = []
        for wid, p in enumerate(procs):
            rc = p.wait()
            if rc != 0:
                failed.append((wid, rc))
        if failed:
            res["error"] = f"child failures: {failed}"
            return finish(1)

        import numpy as np
        block = np.memmap(memmap_path, dtype=np.float32, mode="r",
                          shape=(total_rows, emb_dim))
        ranges = [(int(s["unit_start"]), int(s["unit_start"])
                   + int(s["num_docs"]), os.path.basename(s["path"]))
                  for s in shards]
        counts = validate_block(block, ranges)
        if counts["nan"] or counts["inf"] or counts["zero"]:
            res["error"] = f"validation failed: {counts}"
            return finish(1)

        block_path = os.path.join(work, "partial_block.npz.tmp")
        np.savez(block_path, embeddings=block)
        os.replace(block_path, os.path.join(work, "partial_block.npz"))
        manifest = {
            "unit_id": unit_id,
            "spec_version": SPEC_VERSION,
            "emb_dim": emb_dim,
            "total_rows": total_rows,
            "model_dir": spec["model_dir"],
            "truncate_len": int(spec.get("truncate_len", 512)),
            "batch_size": int(spec.get("batch_size", 512)),
            "shards": [{"path": os.path.basename(s["path"]),
                        "num_docs": int(s["num_docs"]),
                        "global_start": int(s["global_start"])}
                       for s in shards],
            "validation": counts,
        }
        with open(os.path.join(work, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        res["embed_rc"] = 0
        return finish(0)

    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        return finish(3)


def main() -> int:
    p = argparse.ArgumentParser(description="climbmix embed unit worker")
    p.add_argument("--child", action="store_true",
                   help="internal: one card's shard slice")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--spec-path", required=True,
                   help="local path of the unit spec JSON")
    p.add_argument("--work-dir", default="",
                   help="unit work dir (block/ledger/outputs; default: "
                        "the spec's directory)")
    p.add_argument("--memmap", default="", help="child: unit block path")
    p.add_argument("--ledger", default="", help="child: ledger JSON path")
    p.add_argument("--shards", default="",
                   help="child: comma-separated shard indices")
    args = p.parse_args()
    if args.child:
        return child_main(args)
    return parent_main(args)


if __name__ == "__main__":
    sys.exit(main())
