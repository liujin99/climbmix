#!/usr/bin/env python3
"""Auto-detect num_scaling_params and total_batch_size from checkpoint meta JSON.

3-tier fallback:
  1. meta_*.json → model_config → GPTConfig → num_scaling_params()
  2. Approximate formula from model_config dimensions
  3. Hardcoded default 730M (d24 model)

Output (stdout, for shell script capture):
  NUM_SCALING_PARAMS=<int>
  TOTAL_BATCH_SIZE=<int>
"""

import json
import glob
import os
import sys


DEFAULT_NUM_SCALING_PARAMS = 730000000
DEFAULT_TOTAL_BATCH_SIZE = 524288


def _compute_model_dim(depth, aspect_ratio=64, head_dim=128):
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    n_head = model_dim // head_dim
    return model_dim, n_head


def _approx_scaling_params(model_config):
    n_embd = model_config.get("n_embd")
    n_layer = model_config.get("n_layer", 24)
    n_head = model_config.get("n_head")
    vocab_size = model_config.get("vocab_size", 32768)
    head_dim = model_config.get("head_dim", 128)
    aspect_ratio = model_config.get("aspect_ratio", 64)

    if n_embd is None:
        n_embd, n_head_default = _compute_model_dim(n_layer, aspect_ratio, head_dim)
        if n_head is None:
            n_head = n_head_default
    else:
        if n_head is None:
            n_head = n_embd // head_dim

    n_kv_head = model_config.get("n_kv_head", n_head)

    pad_vocab_size_to = model_config.get("pad_vocab_size_to", 64)
    padded_vocab_size = ((vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to

    ve_gate_channels = model_config.get("ve_gate_channels", 12)
    has_ve_count = 0
    for layer_idx in range(n_layer):
        if layer_idx % 2 == (n_layer - 1) % 2:
            has_ve_count += 1

    transformer_matrices = 0
    for _ in range(n_layer):
        transformer_matrices += (
            n_embd * n_head * head_dim +
            n_embd * n_kv_head * head_dim +
            n_embd * n_kv_head * head_dim +
            n_embd * n_embd +
            n_embd * 4 * n_embd +
            4 * n_embd * n_embd
        )
    transformer_matrices += has_ve_count * ve_gate_channels * n_kv_head

    lm_head = padded_vocab_size * n_embd
    return transformer_matrices + lm_head


def _try_meta_json(ckpt_dir):
    meta_files = sorted(glob.glob(os.path.join(ckpt_dir, "meta_*.json")))
    if not meta_files:
        return None, None
    meta_path = meta_files[-1]
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None, None
    model_config = meta.get("model_config", {})
    total_batch_size = meta.get("total_batch_size", None)
    try:
        import torch
        from nanochat.gpt import GPT, GPTConfig
        config = GPTConfig(**model_config)
        with torch.device("meta"):
            model = GPT(config)
        params_counts = model.num_scaling_params()
        num_scaling_params = params_counts["transformer_matrices"] + params_counts["lm_head"]
        return num_scaling_params, total_batch_size
    except Exception:
        num_scaling_params = _approx_scaling_params(model_config)
        if num_scaling_params is None:
            return None, total_batch_size
        return num_scaling_params, total_batch_size


def get_model_info(ckpt_dir=None, num_scaling_params_override=None):
    if num_scaling_params_override is not None and num_scaling_params_override > 0:
        nsp = num_scaling_params_override
        tbs = DEFAULT_TOTAL_BATCH_SIZE
        if ckpt_dir:
            _, tbs = _try_meta_json(ckpt_dir)
        return nsp, tbs or DEFAULT_TOTAL_BATCH_SIZE
    if ckpt_dir:
        nsp, tbs = _try_meta_json(ckpt_dir)
        if nsp is not None:
            return nsp, tbs or DEFAULT_TOTAL_BATCH_SIZE
    return DEFAULT_NUM_SCALING_PARAMS, DEFAULT_TOTAL_BATCH_SIZE


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-detect model info from checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Checkpoint directory containing meta_*.json")
    parser.add_argument("--num-scaling-params", type=int, default=None,
                        help="Override num_scaling_params (skip auto-detection)")
    parser.add_argument("--nanochat-repo", type=str, default=None,
                        help="Nanochat repo path (for GPTConfig import). "
                             "If not provided, uses approximate formula.")
    args = parser.parse_args()

    if args.nanochat_repo and args.nanochat_repo not in sys.path:
        sys.path.insert(0, args.nanochat_repo)

    nsp, tbs = get_model_info(
        ckpt_dir=args.ckpt_dir,
        num_scaling_params_override=args.num_scaling_params,
    )
    print(f"NUM_SCALING_PARAMS={nsp}")
    print(f"TOTAL_BATCH_SIZE={tbs}")


if __name__ == "__main__":
    main()
