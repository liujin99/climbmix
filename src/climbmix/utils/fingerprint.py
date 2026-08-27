"""Experiment fingerprint: detects code/param changes to auto-reset cached state.

The shell runners store fingerprints of (repo source subsets + semantic run
params) in ``$OUTPUT_DIR``. On restart:

- matching fingerprints  -> resume (caches, search state, .done markers valid)
- mismatched fingerprint -> the runner archives the stale output and starts
  fresh, so old caches can never silently mask a code/param change.

Stage scoping (2026-08-27): the previous single global fingerprint meant a
target-stage knob (e.g. MID_DEVICE_BATCH_SIZE) also invalidated multi-day
search results. Runners now keep TWO fingerprints (see runs/lib/stage_gate.sh):

- ``.fingerprint_search``: code paths + params that decide Steps 1-3 products
  (embeddings, clusters, search_state, exp_*/, sampled_dataset.parquet)
- ``.fingerprint_target``: code paths + params that decide Steps 4-8 products
  (climb_shards/, *_mixed/, mid_train_*.log, .done_* markers)

File classification (``stage=None`` hashes everything = exact legacy behaviour):

- SEARCH_ONLY / TARGET_ONLY : the exact lists below
- scripts/diagnostics/, scripts/get_model_info.py : never hashed (dev tools)
- everything else           : BOTH stages (conservative — an unclassified new
  file resets both stages rather than silently keeping one)

``runs/*.sh`` themselves are deliberately NOT hashed (comment/echo-only edits
must not reset an experiment); every semantic knob is passed via ``--param``
instead. NOT covered (documented limitations): edits inside nanochat-npu, and
data files whose names/row-counts are unchanged.

CLI (PYTHONPATH must include ``src``):
    python3 -m climbmix.utils.fingerprint --base-dir . --stage search \
        --param k=v [--param k2=v2] [--write .fingerprint_search]
"""

import argparse
import hashlib
import os

# Files that only influence the search stage (Steps 1-3).
SEARCH_ONLY = {
    "src/climbmix/pipeline/climb_pipeline.py",
    "src/climbmix/pipeline/proxy_runner.py",
    "src/climbmix/core/embedding_cluster.py",
    "src/climbmix/core/discovery.py",
    "src/climbmix/core/predictor.py",
    "src/climbmix/core/dirichlet_sampler.py",
    "src/climbmix/core/iterative_bootstrapper.py",
    "src/climbmix/core/cluster_merge.py",
    "src/climbmix/core/quality_filter.py",
    "scripts/run_climb.py",
}

# Files that only influence the target stage (Steps 4-8).
TARGET_ONLY = {
    "src/climbmix/pipeline/target_runner.py",
    "src/climbmix/pipeline/report_generator.py",
    "scripts/prepare_shards.py",
    "scripts/prepare_random_baseline.py",
    "scripts/mix_general_data.py",
}

# Dev tools: never hashed by any stage.
GLOBAL_EXCLUDE = {
    "scripts/get_model_info.py",
}
GLOBAL_EXCLUDE_PREFIXES = ("scripts/diagnostics/",)


def _stages_for(rel_path: str):
    """Which fingerprint stages a repo file feeds. Empty set = never hashed."""
    if rel_path in GLOBAL_EXCLUDE:
        return set()
    if rel_path.startswith(GLOBAL_EXCLUDE_PREFIXES):
        return set()
    if rel_path in SEARCH_ONLY:
        return {"search"}
    if rel_path in TARGET_ONLY:
        return {"target"}
    # Conservative default: unclassified files affect BOTH stages, so an
    # unknown new file resets everything rather than silently keeping one.
    return {"search", "target"}


def _hash_file(path: str, hasher) -> None:
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)


def _repo_files(base_dir: str, stage):
    """Sorted repo .py/.yaml files (relative, forward-slash) for a stage.

    stage=None -> every file (legacy behaviour, no exclusions).
    """
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
    out = []
    for p in sorted(paths):
        rel = os.path.relpath(p, base_dir).replace(os.sep, "/")
        if stage is None or stage in _stages_for(rel):
            out.append((rel, p))
    return out


def compute_fingerprint(base_dir: str, params, stage=None) -> str:
    """Hash repo sources (stage-scoped subset) + sorted params."""
    hasher = hashlib.sha256()
    if stage is not None and stage not in ("search", "target"):
        raise ValueError(f"unknown stage: {stage!r} (expected search/target/None)")
    hasher.update(f"stage={stage}".encode())
    for rel, path in _repo_files(base_dir, stage):
        hasher.update(rel.encode())
        _hash_file(path, hasher)
    for kv in sorted(params):
        hasher.update(kv.encode())
    return hasher.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description="Compute experiment fingerprint (code + params, stage-scoped)")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--param", action="append", default=[])
    parser.add_argument("--stage", choices=["search", "target"], default=None,
                        help="scope: hash only files/params of this pipeline stage "
                             "(default: everything = legacy single fingerprint)")
    parser.add_argument("--write", default=None,
                        help="also write the hash to this path (atomic-ish: tmp+rename)")
    args = parser.parse_args()
    fp = compute_fingerprint(args.base_dir, args.param, stage=args.stage)
    print(fp)
    if args.write:
        tmp = args.write + ".tmp"
        with open(tmp, "w") as f:
            f.write(fp + "\n")
        os.replace(tmp, args.write)


if __name__ == "__main__":
    main()
