#!/usr/bin/env python3
"""derive_target_steps.py — TARGET_TOKENS → training steps (single source).

steps = TARGET_TOKENS // total_batch_size; total_batch_size is read from
the base ckpt's meta_*.json (never guessed — a wrong value silently
changes the arm's training length). Shared by runs/run_climbmix.sh (main
pipeline) and runs/run_arm_only.sh (arm reuse under a different budget)
so the formula lives in exactly one place:
climbmix.sampling.single_pass.derive_num_iterations.

Usage:
    python3 scripts/derive_target_steps.py \
        --target-tokens 2B \
        --ckpt-dir $NANOCHAT_BASE_DIR/base_checkpoints/d28

Prints the derived step count (exit 0); exit 2 = unresolvable
total_batch_size. Launcher/observability tool: fingerprint-excluded
(utils/fingerprint.py GLOBAL_EXCLUDE).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from climbmix.sampling.single_pass import (  # noqa: E402
    derive_num_iterations, read_total_batch_size)
from climbmix.utils.token_estimate import parse_token_count  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target-tokens", required=True,
                   help="training budget, e.g. 2B / 2097152000 / 400M")
    p.add_argument("--ckpt-dir", required=True,
                   help="base ckpt dir whose meta_*.json carries "
                        "total_batch_size (e.g. $NANOCHAT_BASE_DIR/"
                        "base_checkpoints/d28)")
    args = p.parse_args()

    tbs = read_total_batch_size(args.ckpt_dir)
    if not tbs:
        print(f"✗ no total_batch_size in meta_*.json under {args.ckpt_dir!r}",
              file=sys.stderr)
        return 2
    print(derive_num_iterations(parse_token_count(args.target_tokens), tbs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
