# -*- coding: utf-8 -*-
"""Shared local run-dir path helpers.

Layout contract (2026-10-10 state/ ruling — the uniform layout):

    run/
    ├── report.md + all charts    reader layer (humans)
    ├── state/                    ALL machine-layer state & result files
    ├── logs/                     process logs (write-only streams)
    ├── exps/                     per-config search artifacts
    └── (.done_* / .fingerprint_* / locks — invisible lifecycle markers,
         read at root by the idempotency machinery; zero ls cost)

Root is BORN as the publishing form — no post-hoc tidy. Writers write
state files ONLY via state_file(); post-hoc readers resolve via the
state/ → root → detail/ chain (root = pre-2026-10 flat archives,
detail/ = the 2026-10-09 deep-tidy generation) — cp4_report.
resolve_run_file is the canonical reader-side helper.

Runs started before this ruling keep their legacy layout: resume falls
back to the root path, and _restore_completed (stage_gate) upgrades a
reactivated archive into state/ form.
"""
import os

STATE_DIR = "state"


def state_file(output_dir: str, name: str) -> str:
    """Machine-layer file path (WRITERS + live machinery): the single
    canonical location output_dir/state/<name>. Pure path join —
    callers that create files must ensure the directory (atomic_write
    helpers already makedirs)."""
    return os.path.join(output_dir, STATE_DIR, name)


def exp_dir_for(output_dir: str, experiment_id: int) -> str:
    """Local per-config search artifact dir.

    New layout: output_dir/exps/exp_NNNN. Legacy fallback: a run started
    before the ruling still has output_dir/exp_NNNN — resume keeps
    reading it there (prefer the new path only when it exists).
    """
    legacy = os.path.join(output_dir, f"exp_{experiment_id:04d}")
    new = os.path.join(output_dir, "exps", f"exp_{experiment_id:04d}")
    if os.path.isdir(legacy) and not os.path.exists(new):
        return legacy
    return new


def stage1_pair_anywhere(output_dir):
    """⑬r 命名 × 三代布局: 在 state/ 根层 detail/ 中找第一个成对的
    stage-1 缓存 (macro_labels.npz+macro_info.json 或 legacy
    cluster_cache.npz+cluster_info_cache.json); 全缺时返回 state/ 的
    新写目标 (活跃 run 的 --cluster-cache-dir)。"""
    from climbmix.utils.io_utils import stage1_pair
    for d in (os.path.join(output_dir, STATE_DIR), output_dir,
              os.path.join(output_dir, "detail")):
        npz, jsn = stage1_pair(d)
        if os.path.isfile(npz) and os.path.isfile(jsn):
            return npz, jsn
    return stage1_pair(os.path.join(output_dir, STATE_DIR))
