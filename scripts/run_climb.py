#!/usr/bin/env python3
"""
CLIMB CLI entry point.

All parameters are configurable via CLI flags, matching the
variables defined in runs/*.sh scripts.

Usage:
  python scripts/run_climb.py --data-dir ./data --discovery-method fdc_labels
  python scripts/run_climb.py --data-dir ./data --discovery-method embedding_cluster --proxy-size 62M
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
    parser.add_argument("--filter-method", type=str, default="doc_and_cluster",
                        choices=["none", "doc_level", "cluster_level", "doc_and_cluster"],
                        help="Quality filtering method")
    parser.add_argument("--doc-english-min", type=float, default=0.3,
                        help="Min english score for doc-level filter")
    parser.add_argument("--doc-composite-min", type=float, default=0.5,
                        help="Min composite quality for doc-level filter")
    parser.add_argument("--cluster-avg-threshold", type=float, default=3.0,
                        help="Cluster avg quality threshold for cluster-level filter")

    # ── Proxy ──
    parser.add_argument("--proxy-size", type=str, default="62M",
                        choices=["1M", "5M", "20M", "62M", "132M", "350M"],
                        help="Proxy model size")
    parser.add_argument("--proxy-steps", type=int, default=1000,
                        help="Proxy training steps per experiment")
    parser.add_argument("--proxy-batch-size", type=int, default=64,
                        help="Proxy global batch size")
    parser.add_argument("--proxy-lr", type=float, default=4e-4,
                        help="Proxy learning rate")

    # ── Search ──
    parser.add_argument("--num-iterations", type=int, default=3,
                        help="Number of bootstrapping iterations")
    parser.add_argument("--configs-per-iter", type=str, default="64,32,16",
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
                        help="Skip proxy training (use random losses)")

    # ── Output ──
    parser.add_argument("--output-dir", type=str, default="./climbmix_output")

    args = parser.parse_args()

    configs_per_iter = [int(x) for x in args.configs_per_iter.split(",")]
    val_tasks = [x.strip() for x in args.val_tasks.split(",")]

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
            model_size=args.proxy_size,
            training_steps=args.proxy_steps,
            batch_size=args.proxy_batch_size,
            learning_rate=args.proxy_lr,
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
    print(f"  Discovery: {config.discovery.method} (K_init={config.discovery.K_init}, K_enhanced={config.discovery.K_enhanced})")
    print(f"  Filter:    {config.filtering.method} (english≥{config.filtering.doc_english_min}, composite≥{config.filtering.doc_composite_min}, cluster_avg≥{config.filtering.cluster_avg_threshold})")
    print(f"  Proxy:     {config.proxy.model_size} ({config.proxy.training_steps} steps, bs={config.proxy.batch_size}, lr={config.proxy.learning_rate})")
    print(f"  Search:    {config.search.num_iterations} iterations, {configs_per_iter}")
    print(f"  Data:      {config.data_dir}")
    print(f"  Device:    {config.device.device_type}")
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
