# -*- coding: utf-8 -*-
"""Shared local run-dir path helpers (layout ruling 2026-10-09).

Prod5's run root held ~230 entries (115 exp_NNNN/ dirs + ~107 logs +
results). Writers now tuck process artifacts into subdirectories from
birth: search exp dirs under exps/, run-level logs under logs/. The run
root keeps only what humans and analysis scripts read (eval CSVs,
search_state.json, reports, charts, state JSONs).

Runs started before the ruling keep their root-level exp_NNNN/ dirs —
resume code paths fall back to the legacy location so an in-flight run
is never orphaned by a mid-run code upgrade.
"""
import os


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
