#!/usr/bin/env python3
"""
CLIMB CLI entry point — method A (subprocess nanochat).

Usage:
  python scripts/run_climb.py --data-dir /path/to/stem_data --dry-run
  python scripts/run_climb.py --data-dir /path/to/stem_data --proxy-depth 14 --target-depth 28
  python scripts/run_climb.py --proxy-depth 14 --phase1-checkpoint-path /path/to/d14_ckpt
"""

import argparse
import os
import sys

try:
    from climbmix.core.types import (
        CLIMBConfig, ClusterDiscoveryConfig, QualityFilterConfig,
        SearchConfig, ProxyConfig, TargetConfig, PredictorConfig,
        DeviceConfig, STEM_BENCHMARK_LABELS, DEFAULT_NANOCHAT_BASE_DIR,
    )
    from climbmix.pipeline.climb_pipeline import CLIMBPipeline
    from climbmix.utils.token_estimate import parse_token_count
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from climbmix.core.types import (
        CLIMBConfig, ClusterDiscoveryConfig, QualityFilterConfig,
        SearchConfig, ProxyConfig, TargetConfig, PredictorConfig,
        DeviceConfig, STEM_BENCHMARK_LABELS, DEFAULT_NANOCHAT_BASE_DIR,
    )
    from climbmix.pipeline.climb_pipeline import CLIMBPipeline
    from climbmix.utils.token_estimate import parse_token_count


def main():
    parser = argparse.ArgumentParser(description="Nemotron-CLIMB Pipeline (method A)")

    # ── Data ──
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--nanochat-dir", type=str, default="/home/liujin99/nanochat-npu")
    parser.add_argument("--nanochat-base-dir", type=str, default=DEFAULT_NANOCHAT_BASE_DIR,
                        help=f"Base directory for nanochat checkpoints (default: {DEFAULT_NANOCHAT_BASE_DIR})")
    parser.add_argument("--general-data-dir", type=str, default="",
                        help="Directory for cached ClimbMix general data shards (adaptive 3-50)")
    parser.add_argument("--stem-ratio", type=float, default=0.7,
                        help="STEM data ratio (default 0.7 = 70%% STEM + 30%% general)")
    parser.add_argument("--eval-benchmarks", type=str, default="stem",
                        help="Evaluation benchmarks: all, core, stem, or comma-separated labels")

    # ── Discovery ──
    parser.add_argument("--discovery-method", type=str, default="embedding_cluster",
                        choices=["embedding_cluster", "quality_cluster"])
    parser.add_argument("--K-enhanced", type=int, default=10)
    parser.add_argument("--K-init", type=int, default=1000,
                        help="Initial number of clusters before prune+merge")
    parser.add_argument("--embedding-model", type=str, default="NovaSearch/stella_en_400M_v5")
    parser.add_argument("--embedding-device", type=str, default="cpu",
                        help="Device for embedding: cpu, npu")
    parser.add_argument("--embedding-sample-size", type=int, default=0,
                        help="Subsample N docs for embedding (0 = all). Speeds up embedding_cluster on large datasets.")
    parser.add_argument("--prune-threshold", type=float, default=3.0)
    parser.add_argument("--merge-distance", type=float, default=1.5)

    # ── Filter ──
    parser.add_argument("--filter-method", type=str, default="none",
                        choices=["none", "doc_level", "cluster_level", "doc_and_cluster"])

    # ── Proxy ──
    parser.add_argument("--proxy-depth", type=int, default=20)
    parser.add_argument("--proxy-num-iterations", type=int, default=None)
    parser.add_argument("--proxy-ratio", type=float, default=None)
    parser.add_argument("--proxy-lr-scale", type=float, default=1.0)
    parser.add_argument("--proxy-warmup", type=float, default=0.0)
    parser.add_argument("--proxy-warmdown", type=float, default=0.9)
    parser.add_argument("--phase1-checkpoint-path", type=str, default=None)
    parser.add_argument("--validation-metric", type=str, default="accuracy",
                        choices=["accuracy", "loss"])
    parser.add_argument("--proxy-target-tokens", type=parse_token_count, default=0,
                        help="Cap data selection per proxy experiment (0 = all available). "
                             "Accepts human-readable suffixes: 2B, 10M, 500K, 1.5B")

    # ── Target ──
    parser.add_argument("--target-depth", type=int, default=28)
    parser.add_argument("--target-num-iterations", type=int, default=None)
    parser.add_argument("--target-ratio", type=float, default=None)
    parser.add_argument("--target-lr-scale", type=float, default=1.0)
    parser.add_argument("--target-warmup", type=float, default=0.0)
    parser.add_argument("--target-warmdown", type=float, default=0.9)
    parser.add_argument("--target-phase1-checkpoint-path", type=str, default=None)
    parser.add_argument("--skip-target", action="store_true",
                        help="Skip target training (only run proxy search)")
    parser.add_argument("--target-tokens", type=parse_token_count, default=0,
                        help="Cap final data selection for target training (0 = all available). "
                             "Accepts human-readable suffixes: 2B, 10M, 500K, 1.5B")

    # ── Search ──
    parser.add_argument("--num-iterations", type=int, default=3)
    parser.add_argument("--configs-per-iter", type=str, default="15,8,4")
    parser.add_argument("--dirichlet-alpha", type=float, default=None)

    # ── Predictor ──
    parser.add_argument("--predictor-method", type=str, default="lightgbm")

    # ── Device ──
    parser.add_argument("--device-type", type=str, default="npu",
                        choices=["cpu", "cuda", "npu"])
    parser.add_argument("--npu-devices", type=int, default=8)
    parser.add_argument("--npu-per-exp", type=int, default=0,
                        help="NPUs per proxy experiment (0=all NPUs, sequential. e.g. 4=2 parallel on 8 NPU)")

    # ── Validation tasks ──
    parser.add_argument("--val-tasks", type=str, default=None)

    # ── Dry run ──
    parser.add_argument("--dry-run", action="store_true")

    # ── Cache / Resume ──
    parser.add_argument("--cluster-cache-dir", type=str, default=None,
                        help="Directory with cached cluster info (skip embedding clustering if exists)")
    parser.add_argument("--resume-search", action="store_true",
                        help="Resume proxy search from saved state (search_state.json)")

    # ── Quality config ──
    parser.add_argument("--quality-config-path", type=str, default="",
                        help="(deprecated) YAML file with quality column names and prune_threshold")
    parser.add_argument("--schema", type=str, default="",
                        help="YAML schema file with column mappings (domain_col, quality_cols, text_col, etc.)")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="./climbmix_output")

    args = parser.parse_args()

    configs_per_iter = [int(x) for x in args.configs_per_iter.split(",")]
    val_tasks = STEM_BENCHMARK_LABELS.copy() if args.val_tasks is None else [x.strip() for x in args.val_tasks.split(",")]

    proxy_config = ProxyConfig(
        depth=args.proxy_depth,
        num_iterations=args.proxy_num_iterations,
        ratio=args.proxy_ratio,
        lr_scale=args.proxy_lr_scale,
        warmup=args.proxy_warmup,
        warmdown=args.proxy_warmdown,
        phase1_checkpoint_path=args.phase1_checkpoint_path,
        validation_metric=args.validation_metric,
        target_tokens=args.proxy_target_tokens,
    )

    target_config = TargetConfig(
        depth=args.target_depth,
        num_iterations=args.target_num_iterations,
        ratio=args.target_ratio,
        lr_scale=args.target_lr_scale,
        warmup=args.target_warmup,
        warmdown=args.target_warmdown,
        phase1_checkpoint_path=args.target_phase1_checkpoint_path,
        target_tokens=args.target_tokens,
    )

    config = CLIMBConfig(
        discovery=ClusterDiscoveryConfig(
            method=args.discovery_method,
            K_init=args.K_init,
            K_enhanced=args.K_enhanced,
            embedding_model=args.embedding_model,
            embedding_device=args.embedding_device,
            embedding_sample_size=args.embedding_sample_size,
            prune_threshold=args.prune_threshold,
            merge_distance=args.merge_distance,
        ),
        filtering=QualityFilterConfig(method=args.filter_method),
        search=SearchConfig(
            num_iterations=args.num_iterations,
            configs_per_iter=configs_per_iter,
            dirichlet_alpha=args.dirichlet_alpha,
        ),
        proxy=proxy_config,
        target=target_config,
        predictor=PredictorConfig(method=args.predictor_method),
        device=DeviceConfig(
            device_type=args.device_type,
            npu_devices=args.npu_devices,
        ),
        val_tasks=val_tasks,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        nanochat_dir=args.nanochat_dir,
        nanochat_base_dir=args.nanochat_base_dir,
        general_data_dir=args.general_data_dir,
        stem_ratio=args.stem_ratio,
        eval_benchmarks=args.eval_benchmarks,
        quality_config_path=args.quality_config_path,
        schema_path=args.schema,
        npu_per_exp=args.npu_per_exp,
    )

    print(f"\n{'=' * 70}")
    print("  CLIMB Configuration (method A)")
    print(f"  Proxy:      {config.proxy.model_tag} ({config.proxy.scaling_M:.1f}M scaling, {config.proxy.total_M:.0f}M total)")
    print(f"              {config.proxy.training_iterations} iterations, lr_scale={config.proxy.lr_scale}, warmup={config.proxy.warmup}, warmdown={config.proxy.warmdown}")
    print(f"              phase1={config.proxy.phase1_checkpoint_path or 'none'}")
    print(f"  Target:     {config.target.model_tag}")
    print(f"  Discovery:  {config.discovery.method} (K_enhanced={config.discovery.K_enhanced})")
    print(f"  Search:     {config.search.num_iterations} iterations, {configs_per_iter} = {sum(configs_per_iter)} configs")
    print(f"  Metric:     {config.val_tasks} ({config.metric_direction})")
    print(f"  Eval:       benchmarks={config.eval_benchmarks}")
    print(f"  Data mix:   {config.stem_ratio*100:.0f}% STEM + {(1-config.stem_ratio)*100:.0f}% general")
    print(f"  Device:     {config.device.device_type} ({config.device.npu_devices} devices)")
    print(f"  nanochat:   {config.nanochat_dir}")
    print(f"  base_dir:   {config.nanochat_base_dir}")
    print(f"{'=' * 70}\n")

    proxy_runner = None
    target_runner = None
    if not args.dry_run:
        from climbmix.pipeline.proxy_runner import ProxyRunner
        proxy_runner = ProxyRunner(config)

        if not args.skip_target:
            from climbmix.pipeline.target_runner import TargetRunner
            target_runner = TargetRunner(config)

    pipeline = CLIMBPipeline(config)
    results = pipeline.run(
        proxy_runner=proxy_runner,
        target_runner=target_runner,
        cluster_cache_dir=args.cluster_cache_dir,
        resume_search=args.resume_search,
    )

    print(f"\nDone! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
