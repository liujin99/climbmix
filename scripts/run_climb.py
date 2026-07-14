#!/usr/bin/env python3
"""
CLIMB CLI entry point.

All parameters are configurable via CLI flags, matching the
variables defined in runs/*.sh scripts.

Usage:
  python scripts/run_climb.py --data-dir ./data --discovery-method fdc_labels
  python scripts/run_climb.py --data-dir ./data --discovery-method embedding_cluster
  python scripts/run_climb.py --dry-run  # no training, random scores
  python scripts/run_climb.py --proxy-model-depth 10 --proxy-training-tokens 5e9
"""

import argparse
import os
import sys

try:
    from climbmix.core.types import CLIMBConfig, ClusterDiscoveryConfig, QualityFilterConfig, SearchConfig, ProxyConfig, PredictorConfig, DeviceConfig
    from climbmix.pipeline.climb_pipeline import CLIMBPipeline
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
    from climbmix.core.types import CLIMBConfig, ClusterDiscoveryConfig, QualityFilterConfig, SearchConfig, ProxyConfig, PredictorConfig, DeviceConfig
    from climbmix.pipeline.climb_pipeline import CLIMBPipeline


def main():
    parser = argparse.ArgumentParser(description="Nemotron-CLIMB Pipeline")

    # ── Data ──
    parser.add_argument("--data-dir", type=str, default="./data",
                        help="Data directory (auto-scans all parquet files)")

    # ── Discovery ──
    parser.add_argument("--discovery-method", type=str, default="fdc_labels",
                        choices=["fdc_labels", "embedding_cluster"],
                        help="Cluster discovery method")
    parser.add_argument("--K-init", type=int, default=1000,
                        help="Initial number of clusters (embedding_cluster)")
    parser.add_argument("--K-enhanced", type=int, default=21,
                        help="Final number of clusters after merge")
    parser.add_argument("--embedding-model", type=str, default="NovaSearch/stella_en_400M_v5",
                        help="Embedding model for clustering")
    parser.add_argument("--prune-threshold", type=float, default=3.0,
                        help="Cluster quality pruning threshold")
    parser.add_argument("--merge-distance", type=float, default=1.5,
                        help="Cluster merge distance threshold")

    # ── Filter ──
    parser.add_argument("--filter-method", type=str, default="none",
                        choices=["none", "doc_level", "cluster_level", "doc_and_cluster"],
                        help="Quality filtering method")
    parser.add_argument("--doc-english-min", type=float, default=0.3,
                        help="Min english score for doc-level filter")
    parser.add_argument("--doc-composite-min", type=float, default=0.5,
                        help="Min composite quality for doc-level filter")
    parser.add_argument("--cluster-avg-threshold", type=float, default=3.0,
                        help="Cluster avg quality threshold for cluster-level filter")

    # ── Proxy ──
    parser.add_argument("--proxy-model-depth", type=int, default=10,
                        help="nanochat model depth (5=~59M, 10=~196M, 24=~1.38B)")
    parser.add_argument("--proxy-training-tokens", type=int, default=5_000_000_000,
                        help="Training tokens per proxy experiment")
    parser.add_argument("--proxy-batch-tokens", type=int, default=500_000,
                        help="Global batch size in tokens")
    parser.add_argument("--proxy-lr", type=float, default=5e-5,
                        help="Stable phase learning rate")
    parser.add_argument("--proxy-decay-lr", type=float, default=1e-5,
                        help="Decay phase learning rate")
    parser.add_argument("--phase1-checkpoint-path", type=str, default=None,
                        help="Phase-1 pretrained checkpoint path (nanochat format)")
    parser.add_argument("--validation-metric", type=str, default="accuracy",
                        choices=["accuracy", "loss"],
                        help="Validation metric: accuracy (lm-eval) or loss")
    parser.add_argument("--lr-schedule", type=str, default="wsd",
                        choices=["wsd", "cosine", "linear"],
                        help="LR schedule for proxy training")

    # ── Search ──
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of bootstrapping iterations")
    parser.add_argument("--configs-per-iter", type=str, default="16,8,4",
                        help="Configs per iteration (comma-separated)")
    parser.add_argument("--dirichlet-alpha", type=float, default=None,
                        help="Dirichlet concentration (None=proportional)")

    # ── Predictor ──
    parser.add_argument("--predictor-method", type=str, default="lightgbm",
                        choices=["lightgbm"],
                        help="Predictor method")

    # ── Device ──
    parser.add_argument("--device-type", type=str, default="cpu",
                        choices=["cpu", "cuda", "npu"],
                        help="Device type for proxy training")
    parser.add_argument("--npu-devices", type=int, default=8,
                        help="Number of NPU devices")

    # ── Validation ──
    parser.add_argument("--val-tasks", type=str, default="piqa,arc_e,hellaswag",
                        help="Validation tasks (comma-separated)")

    # ── Dry run ──
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip proxy training (use random scores)")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="./climbmix_output")

    args = parser.parse_args()

    configs_per_iter = [int(x) for x in args.configs_per_iter.split(",")]
    val_tasks = [x.strip() for x in args.val_tasks.split(",")]

    DEPTH_PARAMS = {
        5:  {"params_approx": "59M",  "lr": 1e-4,  "decay_lr": 1e-5},
        10: {"params_approx": "196M", "lr": 5e-5,  "decay_lr": 1e-5},
        24: {"params_approx": "1.38B", "lr": 5e-5,  "decay_lr": 1e-5},
    }
    dp = DEPTH_PARAMS.get(args.proxy_model_depth, DEPTH_PARAMS[10])

    training_steps = max(1, args.proxy_training_tokens // args.proxy_batch_tokens)

    config = CLIMBConfig(
        discovery=ClusterDiscoveryConfig(
            method=args.discovery_method,
            K_init=args.K_init,
            K_enhanced=args.K_enhanced,
            embedding_model=args.embedding_model,
            prune_threshold=args.prune_threshold,
            merge_distance=args.merge_distance,
        ),
        filtering=QualityFilterConfig(
            method=args.filter_method,
            doc_english_min=args.doc_english_min,
            doc_composite_min=args.doc_composite_min,
            cluster_avg_threshold=args.cluster_avg_threshold,
        ),
        search=SearchConfig(
            num_iterations=args.num_iterations,
            configs_per_iter=configs_per_iter,
            dirichlet_alpha=args.dirichlet_alpha,
        ),
        proxy=ProxyConfig(
            model_size=f"depth{args.proxy_model_depth}",
            training_steps=training_steps,
            training_tokens=args.proxy_training_tokens,
            batch_tokens=args.proxy_batch_tokens,
            learning_rate=args.proxy_lr or dp["lr"],
            decay_learning_rate=args.proxy_decay_lr or dp["decay_lr"],
            lr_schedule=args.lr_schedule,
            phase1_checkpoint_path=args.phase1_checkpoint_path,
            validation_metric=args.validation_metric,
        ),
        predictor=PredictorConfig(
            method=args.predictor_method,
        ),
        device=DeviceConfig(
            device_type=args.device_type,
            npu_devices=args.npu_devices,
        ),
        val_tasks=val_tasks,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )

    print(f"\n{'=' * 70}")
    print("  CLIMB Configuration")
    print(f"  Discovery:  {config.discovery.method} (K_init={config.discovery.K_init}, K_enhanced={config.discovery.K_enhanced})")
    print(f"  Filter:     {config.filtering.method}")
    print(f"  Proxy:      depth={args.proxy_model_depth} (~{dp['params_approx']} params)")
    print(f"              {args.proxy_training_tokens/1e9:.1f}B tokens, {training_steps} steps")
    print(f"              batch={args.proxy_batch_tokens/1e6:.1f}M tokens, lr={config.proxy.learning_rate}, schedule={config.proxy.lr_schedule}")
    print(f"              validation={config.proxy.validation_metric}, phase1={config.proxy.phase1_checkpoint_path or 'none'}")
    print(f"  Search:     {config.search.num_iterations} iterations, {configs_per_iter} = {sum(configs_per_iter)} configs")
    print(f"  Data:       {config.data_dir}")
    print(f"  Device:     {config.device.device_type} ({config.device.npu_devices} devices)")
    print(f"  Metric:     {config.metric_direction}")
    print(f"{'=' * 70}\n")

    proxy_runner = None
    if not args.dry_run:
        from climbmix.pipeline.proxy_runner import ProxyRunner
        proxy_runner = ProxyRunner(config)

    pipeline = CLIMBPipeline(config)
    results = pipeline.run(proxy_runner=proxy_runner)

    print(f"\nDone! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
