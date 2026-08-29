#!/usr/bin/env python3
"""Dispatch a single remote proxy experiment (one-off) — M3 validation tool.

Runs ONE experiment (by explicit weights or by re-running exp_0000's config)
through the full RemoteExecutor path against a THROWAWAY output dir, without
touching the search state or fingerprints of a real run. Validation protocol
(TODO M3): run the same weights locally (ProxyRunner) and remotely
(dispatch_remote), then compare the input shards byte-for-byte (sha256) and
the stem_metric (Δ < 0.002).

Also: --check-assets verifies the one-time OBS bootstrap (big assets:
nanochat-npu code, d20 checkpoint, tokenizer, eval_bundle/eval_stem — see
docs/remote_setup.md) and prints what is missing.

Examples:
  # asset check
  python3 scripts/dispatch_remote.py --remote-config remote.json \
      --check-assets --obs-prefix obs://bucket/climbmix

  # one experiment (weights over K clusters, comma-separated)
  python3 scripts/dispatch_remote.py --remote-config remote.json \
      --data-dir $DATA_DIR --schema config/schema_stem.yaml \
      --cluster-cache-dir result/prod_current \
      --nanochat-dir $NANOCHAT_DIR --nanochat-base-dir $NANOCHAT_BASE_DIR \
      --phase1-checkpoint-path $NANOCHAT_BASE_DIR/base_checkpoints/d20 \
      --output-dir /tmp/remote_validation --exp-id 0 \
      --weights 0.2,0.3,0.5
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from climbmix.core.types import CLIMBConfig, MixtureConfig, MixtureWeights
except ImportError:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from climbmix.core.types import CLIMBConfig, MixtureConfig, MixtureWeights


def build_config(args) -> CLIMBConfig:
    """Minimal CLIMBConfig mirroring run_climb.py's proxy-relevant subset."""
    from climbmix.core.types import (
        ClusterDiscoveryConfig, QualityFilterConfig, SearchConfig,
        ProxyConfig, TargetConfig, PredictorConfig, DeviceConfig,
        STEM_BENCHMARK_LABELS,
    )
    val_tasks = (STEM_BENCHMARK_LABELS.copy() if args.val_tasks is None
                 else [x.strip() for x in args.val_tasks.split(",")])
    return CLIMBConfig(
        discovery=ClusterDiscoveryConfig(
            K_enhanced=args.K_enhanced, K_max=args.K_max,
            merge_distance=args.merge_distance,
        ),
        filtering=QualityFilterConfig(),
        search=SearchConfig(),
        proxy=ProxyConfig(
            depth=args.proxy_depth,
            num_iterations=args.proxy_num_iterations,
            lr_scale=args.proxy_lr_scale,
            warmup=args.proxy_warmup,
            warmdown=args.proxy_warmdown,
            phase1_checkpoint_path=args.phase1_checkpoint_path,
            target_tokens=args.proxy_target_tokens,
        ),
        target=TargetConfig(),
        predictor=PredictorConfig(),
        device=DeviceConfig(device_type="npu", npu_devices=args.npu_devices),
        val_tasks=val_tasks,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        nanochat_dir=args.nanochat_dir,
        nanochat_base_dir=args.nanochat_base_dir,
        general_data_dir=args.general_data_dir,
        stem_ratio=args.stem_ratio,
        eval_benchmarks=args.eval_benchmarks,
        eval_max_per_task=args.eval_max_per_task,
        schema_path=args.schema,
        npu_per_exp=0,  # pure remote dispatch — no local execution
        experiment_name=args.exp_name,
    )


def attach_cluster_state(executor, args):
    """Mirror CLIMBPipeline's stage-0/1: load cluster cache + metadata manager
    onto the executor (the pieces _prepare_mixture_data needs)."""
    import numpy as np
    from climbmix.data.metadata_manager import ShardMetadataManager
    from climbmix.data.column_schema import DatasetSchema

    npz = os.path.join(args.cluster_cache_dir, "cluster_cache.npz")
    if not os.path.isfile(npz):
        raise FileNotFoundError(
            f"cluster cache not found: {npz} — run Steps 1-2 first (or point "
            f"--cluster-cache-dir at a completed run's output dir)")
    cache = np.load(npz, allow_pickle=False)
    executor.cluster_labels = cache["final_labels"]
    # NOTE: filtered labels (quality filter stage 2) — 'none' filter is the
    # production default, so final_labels == filtered_labels.
    schema = (DatasetSchema.from_yaml(args.schema) if args.schema
              else DatasetSchema.from_yaml("config/schema_stem.yaml"))
    mm = ShardMetadataManager(args.data_dir, schema=schema,
                              cache_dir=args.data_dir)
    executor.metadata_manager = mm
    executor.token_counts = mm.estimate_token_counts()
    print(f"[dispatch] cluster cache: {len(executor.cluster_labels):,} docs, "
          f"K={len(set(executor.cluster_labels.tolist()))}")


def check_assets(remote_config, args):
    """Verify the one-time OBS bootstrap (big assets, M1 — see the backend
    repo's README). The worker code bundle is uploaded automatically by
    RemoteExecutor; these are NOT."""
    from climbmix.remote.backends import resolve_backend
    obs = resolve_backend(remote_config).make_obs_storage(remote_config)

    prefix = (args.obs_prefix or remote_config.obs_prefix).rstrip("/")
    expected = {
        f"{prefix}/assets/remote_worker.py": "worker code (auto-uploaded)",
        f"{prefix}/assets/nanochat_cmds.py": "worker code (auto-uploaded)",
        f"{prefix}/assets_big/nanochat-npu.tar.gz": "nanochat-npu repo bundle",
        f"{prefix}/assets_big/d20/": "d20 base checkpoint",
        f"{prefix}/assets_big/tokenizer/": "tokenizer",
        f"{prefix}/assets_big/eval_bundle/": "eval datasets (bundle)",
        f"{prefix}/assets_big/eval_stem/": "eval datasets (stem)",
    }
    missing = []
    print(f"[check-assets] prefix: {prefix}")
    for uri, what in expected.items():
        if uri.endswith("/"):
            ok = bool(obs.list_objects(uri))
        else:
            ok = obs.stat(uri)
        print(f"  {'OK ' if ok else 'MISSING'}  {uri}  ({what})")
        if not ok:
            missing.append(uri)
    if missing:
        print(f"\n[check-assets] {len(missing)} missing — upload per "
              f"docs/remote_setup.md, then re-run.")
        return 1
    print("\n[check-assets] all assets present")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--remote-config", required=True,
                   help="RemoteConfig JSON (see RemoteConfig.from_json_file)")
    p.add_argument("--check-assets", action="store_true",
                   help="verify the one-time OBS bootstrap and exit")
    p.add_argument("--obs-prefix", default="",
                   help="override prefix for --check-assets (default: "
                        "remote-config's obs_prefix)")

    # Experiment definition
    p.add_argument("--exp-id", type=int, default=0)
    p.add_argument("--weights", required=False,
                   help="comma-separated mixture weights (K values)")
    p.add_argument("--exp-name", default="remoteval")

    # CLIMBConfig subset (mirror run_climb.py defaults)
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--schema", default="")
    p.add_argument("--cluster-cache-dir", required=False)
    p.add_argument("--nanochat-dir", default="")
    p.add_argument("--nanochat-base-dir", default="")
    p.add_argument("--phase1-checkpoint-path", default=None)
    p.add_argument("--general-data-dir", default="")
    p.add_argument("--stem-ratio", type=float, default=0.7)
    p.add_argument("--eval-benchmarks", default="stem")
    p.add_argument("--eval-max-per-task", type=int, default=-1)
    p.add_argument("--proxy-depth", type=int, default=20)
    p.add_argument("--proxy-num-iterations", type=int, default=None)
    p.add_argument("--proxy-lr-scale", type=float, default=1.0)
    p.add_argument("--proxy-warmup", type=float, default=0.0)
    p.add_argument("--proxy-warmdown", type=float, default=0.9)
    p.add_argument("--proxy-target-tokens", type=int, default=0)
    p.add_argument("--npu-devices", type=int, default=8)
    p.add_argument("--K-enhanced", type=int, default=3)
    p.add_argument("--K-max", type=int, default=15)
    p.add_argument("--merge-distance", type=float, default=0.9)
    p.add_argument("--val-tasks", default=None)
    p.add_argument("--output-dir", default="./remote_validation")

    args = p.parse_args()

    from climbmix.remote.remote_executor import RemoteConfig, RemoteExecutor
    remote_config = RemoteConfig.from_json_file(args.remote_config)

    if args.check_assets:
        sys.exit(check_assets(remote_config, args))

    if not args.weights:
        p.error("--weights is required (comma-separated, K values)")
    if not args.cluster_cache_dir:
        p.error("--cluster-cache-dir is required (a completed run's output "
                "dir with cluster_cache.npz)")
    if not args.nanochat_dir or not args.nanochat_base_dir:
        p.error("--nanochat-dir and --nanochat-base-dir are required")

    weights = [float(x) for x in args.weights.split(",")]
    config = build_config(args)

    executor = RemoteExecutor(config, remote_config)
    attach_cluster_state(executor, args)

    mixture = MixtureConfig.from_flattened(
        np.array(weights, dtype=np.float64), config_id=args.exp_id)
    result = executor._run_remote_experiment(mixture, args.exp_id,
                                             output_dir=args.output_dir)
    print(f"\n[dispatch] exp_{args.exp_id:04d}: "
          f"stem_metric={result.validation_accuracy:.4f} "
          f"stem_nll={result.validation_nll:.4f}")
    print(f"[dispatch] artifacts: {args.output_dir}/exp_{args.exp_id:04d}/")


if __name__ == "__main__":
    main()
