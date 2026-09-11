#!/usr/bin/env python3
"""check_single_pass.py — epoch<=1 guard for a target-arm mixture.

Measures the ACTUAL train-shard token pool of a mixed-data dir and fails
(exit 1) when TARGET_STEPS x total_batch_size would consume more than the
pool: nanochat's loader wraps silently on exhaustion, so an over-budget
config re-samples documents with no signal (see
src/climbmix/sampling/single_pass.py for the full rationale).

Usage (run_climbmix.sh run_arm / manual preflight):
    python3 scripts/check_single_pass.py \
        --data-dir $OUTPUT_DIR/climb_mixed \
        --num-iterations $TARGET_STEPS \
        --ckpt-dir $NANOCHAT_BASE_DIR/base_checkpoints/d$TARGET_DEPTH \
        --stem-ratio 0.7 --context "climb arm"

total_batch_size resolution: --total-batch-size > meta_*.json under
--ckpt-dir > hard error (never guessed — a wrong value silently weakens
the guard by the guess factor).

This is an observability tool: fingerprint-excluded (utils/fingerprint.py
GLOBAL_EXCLUDE), it must never reset run state.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from climbmix.sampling.single_pass import (  # noqa: E402
    check_single_pass, measure_train_tokens, read_total_batch_size)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", required=True,
                   help="mixed-data dir (shard_*.parquet + .done)")
    p.add_argument("--num-iterations", type=int, required=True,
                   help="planned TARGET_STEPS (mid_train --num-iterations)")
    p.add_argument("--total-batch-size", type=int, default=None,
                   help="override (default: read from --ckpt-dir meta_*.json)")
    p.add_argument("--ckpt-dir", default=None,
                   help="checkpoint dir with meta_*.json carrying "
                        "total_batch_size")
    p.add_argument("--stem-ratio", type=float, default=0.7,
                   help="STEM fraction of the mixture (fix-hint only)")
    p.add_argument("--max-epochs", type=float, default=1.02,
                   help="violation threshold (default 1.02; measurement "
                        "noise margin — ANY wrap breaks single-pass)")
    p.add_argument("--context", default="target arm",
                   help="label for the error message")
    args = p.parse_args()

    tbs = args.total_batch_size or read_total_batch_size(args.ckpt_dir)
    if not tbs:
        print(f"✗ cannot resolve total_batch_size: no --total-batch-size and "
              f"no usable meta_*.json under {args.ckpt_dir!r}", file=sys.stderr)
        return 2

    try:
        pool_tokens = measure_train_tokens(args.data_dir)
        info = check_single_pass(
            args.num_iterations, tbs, pool_tokens,
            stem_ratio=args.stem_ratio, max_epochs=args.max_epochs,
            context=args.context)
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    print(f"  [{args.context}] single-pass OK: "
          f"{info['epochs_x100'] / 100:.2f} epoch "
          f"(consume {info['consume_tokens']:,} <= pool {pool_tokens:,} "
          f"tokens, {args.num_iterations} steps x {tbs:,} tokens/step)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
