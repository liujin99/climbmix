"""Experiment fingerprint: detects code/param changes to auto-reset cached state.

The shell runners store a fingerprint of (repo source + semantic run params)
in ``$OUTPUT_DIR/.fingerprint``. On restart:

- matching fingerprint   -> resume (caches, search state, .done markers valid)
- mismatched fingerprint -> the runner archives the stale output dir and
  starts fresh, so old caches can never silently mask a code/param change.

``runs/*.sh`` themselves are deliberately NOT hashed (comment/echo-only edits
must not reset an experiment); every semantic knob is passed via ``--param``
instead. NOT covered (documented limitations): edits inside nanochat-npu, and
data files whose names/row-counts are unchanged.

CLI (PYTHONPATH must include ``src``):
    python3 -m climbmix.utils.fingerprint --base-dir . --param k=v [--param k2=v2]
"""

import argparse
import hashlib
import os


def _hash_file(path: str, hasher) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)


def compute_fingerprint(base_dir: str, params) -> str:
    """Hash repo sources (src/, scripts/, config/ *.py|*.yaml) + sorted params."""
    hasher = hashlib.sha256()

    paths = []
    for root in ("src", "scripts", "config"):
        root_path = os.path.join(base_dir, root)
        if not os.path.isdir(root_path):
            continue
        for dirpath, dirnames, filenames in os.walk(root_path):
            dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
            for fn in filenames:
                if fn.endswith((".py", ".yaml", ".yml")):
                    paths.append(os.path.join(dirpath, fn))

    for p in sorted(paths):
        hasher.update(os.path.relpath(p, base_dir).encode())
        _hash_file(p, hasher)

    for kv in sorted(params):
        hasher.update(kv.encode())

    return hasher.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description="Compute experiment fingerprint (code + params)")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--param", action="append", default=[])
    args = parser.parse_args()
    print(compute_fingerprint(args.base_dir, args.param))


if __name__ == "__main__":
    main()
