#!/usr/bin/env python3
"""Backfill per-block sha256 + byte size into an existing sharded-cache
manifest.

embed_merge records each block's sha256 + bytes at publish time
(computed in the same pass as its per-block content validation), and
the run-side loader fast-trusts such caches (stat + 2M-row sample,
~1 min) instead of re-paying the ~55-min single-thread full scan on
every process start. Caches merged BEFORE that change lack the hash
fields and always take the legacy full-scan path; this script adds
the fields in place — pure local, no OBS access, no re-merge:

  python3 scripts/backfill_block_hashes.py cache/embeddings/<key>

What it does per block: shape check (mmap vs manifest) + sha256 +
size, then atomically rewrites manifest.json with the new fields.
Refuses when a block is missing or shape-mismatched — that cache
needs a re-merge, not a backfill. The hash freezes the CURRENT bytes:
if you want a content guarantee for a cache that was never fully
validated, run the loader once with CLIMBMIX_EMB_VERIFY=full first.

Do not run concurrently with embed_merge on the same key dir.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if os.path.isdir(os.path.join(_SRC, "climbmix")) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

MANIFEST_NAME = "manifest.json"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1 << 22), b""):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Add per-block sha256+bytes to a hashless sharded "
                    "cache manifest (enables the loader fast-verify path)")
    ap.add_argument("cache_dir",
                    help="the sharded cache key dir (holds manifest.json)")
    args = ap.parse_args()

    import numpy as np
    from climbmix.core.embedding_cache import load_manifest

    man = load_manifest(args.cache_dir)
    blocks = man["blocks"]
    man_path = os.path.join(args.cache_dir, MANIFEST_NAME)

    todo = [b for b in blocks
            if "sha256" not in b or "bytes" not in b]
    if not todo:
        print(f"[backfill] {man_path}: all {len(blocks)} blocks already "
              "carry sha256+bytes — nothing to do")
        return 0

    missing = [b["file"] for b in todo
               if not os.path.isfile(os.path.join(args.cache_dir, b["file"]))]
    if missing:
        print(f"[backfill] FATAL: {len(missing)} block file(s) missing: "
              f"{', '.join(missing[:10])}"
              f"{' ...' if len(missing) > 10 else ''} — re-merge instead",
              file=sys.stderr)
        return 1

    emb_dim = int(man["emb_dim"])
    for b in todo:
        p = os.path.join(args.cache_dir, b["file"])
        arr = np.load(p, mmap_mode="r")
        if tuple(arr.shape) != (int(b["rows"]), emb_dim):
            print(f"[backfill] FATAL: {p} shape {tuple(arr.shape)} != "
                  f"({b['rows']}, {emb_dim}) — re-merge instead",
                  file=sys.stderr)
            return 1

    with ThreadPoolExecutor(max_workers=min(24, len(todo))) as ex:
        paths = [os.path.join(args.cache_dir, b["file"]) for b in todo]
        for b, sha in zip(todo, ex.map(_sha256, paths)):
            b["sha256"] = sha
            b["bytes"] = os.path.getsize(
                os.path.join(args.cache_dir, b["file"]))

    tmp = f"{man_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, man_path)
    total = sum(int(b["bytes"]) for b in blocks)
    print(f"[backfill] {man_path}: {len(todo)}/{len(blocks)} blocks hashed "
          f"({total / (1024**3):.1f} GB) — loader fast-verify now active")
    return 0


if __name__ == "__main__":
    sys.exit(main())
