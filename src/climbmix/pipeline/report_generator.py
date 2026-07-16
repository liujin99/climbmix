"""
Pipeline output: Markdown report + matplotlib domain distribution comparison chart.

Two report modes:
  1. Pipeline report (called by climb_pipeline.py after search)
  2. Midtrain validation report (called by midtrain_validate.sh)
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Optional, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from climbmix.core.types import (
    ClusterInfo,
    MixtureConfig,
    IterationResult,
)


def generate_markdown_report(
    output_dir: str,
    config: Any,
    cluster_info: List[ClusterInfo],
    optimal_weights: Any,
    iter_results: List[IterationResult],
    stats: dict,
    stage_times: Dict[str, float],
    elapsed_seconds: float,
) -> str:
    K = len(cluster_info)
    labels = [c.label for c in cluster_info]
    orig_dist = stats["original_distribution"]
    sel_dist = stats["selected_distribution"]
    weights = stats["mixture_weights"]

    lines: List[str] = []

    lines.append("# Nemotron-CLIMB Pipeline Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Elapsed:** {elapsed_seconds:.1f}s")
    lines.append("")

    lines.append("## Configuration")
    lines.append("")
    lines.append(f"| Parameter | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Discovery method | `{config.discovery.method}` |")
    lines.append(f"| Quality filter | `{config.filtering.method}` |")
    lines.append(f"| Proxy | d{config.proxy.depth} ({config.proxy.scaling_M:.1f}M scaling, {config.proxy.total_M:.0f}M total) |")
    lines.append(f"| K_enhanced | {config.discovery.K_enhanced} |")
    lines.append(f"| Num iterations | {config.search.num_iterations} |")
    lines.append(f"| Configs per iter | {config.search.configs_per_iter} |")
    lines.append(f"| Dirichlet α | {config.search.dirichlet_alpha or 'proportional'} |")
    lines.append("")

    lines.append("## Clusters")
    lines.append("")
    lines.append("| # | Label | Docs (orig) | Docs (sel) | Tokens | Weight α | Ratio |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, c in enumerate(cluster_info):
        if i >= K:
            break
        o = orig_dist[i] if i < len(orig_dist) else 0
        s = sel_dist[i] if i < len(sel_dist) else 0
        ratio = s / max(1, o)
        lines.append(f"| {i} | {c.label} | {o:,} | {s:,} | {c.num_tokens:,} | {weights[i]:.4f} | {ratio:.2f} |")
    lines.append("")

    lines.append("## Iteration Summary")
    lines.append("")
    lines.append("| Iter | Configs | Trained | Best Score | Predictor R\u00b2 |")
    lines.append("|---|---|---|---|---|")
    for r in iter_results:
        score_str = f"{r.best_score:.4f}" if r.best_score is not None else "N/A"
        r2_str = f"{r.predictor_r2:.4f}" if r.predictor_r2 is not None else "N/A"
        lines.append(f"| {r.iteration} | {r.n_configs} | {r.n_trained} | {score_str} | {r2_str} |")
    lines.append("")

    lines.append("## Stage Timing")
    lines.append("")
    lines.append("| Stage | Time (s) |")
    lines.append("|---|---|")
    for stage, t in stage_times.items():
        lines.append(f"| {stage} | {t:.1f} |")
    lines.append("")

    lines.append("## Domain Distribution Chart")
    lines.append("")
    chart_path = os.path.join(output_dir, "domain_distribution.png")
    lines.append(f"![Original vs Final Domain Distribution]({os.path.basename(chart_path)})")
    lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    return report_path


def generate_distribution_chart(
    output_dir: str,
    cluster_info: List[ClusterInfo],
    stats: dict,
) -> str:
    K = len(cluster_info)
    labels = [c.label for c in cluster_info]
    orig_dist = np.array(stats["original_distribution"][:K], dtype=np.float64)
    sel_dist = np.array(stats["selected_distribution"][:K], dtype=np.float64)

    orig_frac = orig_dist / orig_dist.sum() if orig_dist.sum() > 0 else orig_dist
    sel_frac = sel_dist / sel_dist.sum() if sel_dist.sum() > 0 else sel_dist

    x = np.arange(K)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, K * 0.6), 6))
    bars_orig = ax.bar(x - width / 2, orig_frac, width, label="Original", color="#4C72B0", alpha=0.85)
    bars_sel = ax.bar(x + width / 2, sel_frac, width, label="Selected (CLIMB)", color="#DD8452", alpha=0.85)

    ax.set_xlabel("Domain")
    ax.set_ylabel("Proportion")
    ax.set_title("Domain Distribution: Original vs CLIMB Selected")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    chart_path = os.path.join(output_dir, "domain_distribution.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return chart_path


def parse_eval_log(path: str) -> dict:
    info = {"core_metric": None, "tasks": {}}
    if not path or not os.path.exists(path):
        return info
    task_pat = re.compile(
        r"Evaluating:\s+(.+?)\s+\(.*?\)\.\.\.\s+accuracy:\s+([\d.]+)\s+\|\s+centered:\s+([\d.-]+)\s+\|\s+time:\s+([\d.]+)s"
    )
    core_pat = re.compile(r"CORE metric:\s+([\d.]+)")
    for line in open(path):
        m = task_pat.search(line)
        if m:
            info["tasks"][m.group(1)] = {
                "accuracy": float(m.group(2)),
                "centered": float(m.group(3)),
                "time": float(m.group(4)),
            }
        m = core_pat.search(line)
        if m:
            info["core_metric"] = float(m.group(1))
    return info


def generate_midtrain_report(
    result_dir: str,
    climb_eval_log: str,
    random_eval_log: str,
    base_model_tag: str,
    climb_model_tag: str,
    random_model_tag: str,
) -> str:
    climb_eval = parse_eval_log(climb_eval_log)
    random_eval = parse_eval_log(random_eval_log)

    lines: List[str] = []
    lines.append("# CLIMB Mid-Training Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Base model:** `{base_model_tag}`")
    lines.append(f"**CLIMB model:** `{climb_model_tag}`")
    lines.append(f"**Random model:** `{random_model_tag}`")
    lines.append("")

    lines.append("## CORE Metric Comparison")
    lines.append("")
    climb_core = climb_eval["core_metric"]
    random_core = random_eval["core_metric"]
    lines.append(f"| Method | CORE metric |")
    lines.append(f"|---|---|")
    lines.append(f"| CLIMB optimal | {climb_core:.4f} |" if climb_core else "| CLIMB optimal | N/A |")
    lines.append(f"| Random baseline | {random_core:.4f} |" if random_core else "| Random baseline | N/A |")
    if climb_core and random_core:
        delta = climb_core - random_core
        lines.append(f"| **Delta** | **{delta:+.4f}** |")
    lines.append("")

    all_tasks = sorted(set(climb_eval["tasks"].keys()) | set(random_eval["tasks"].keys()))
    if all_tasks:
        lines.append("## Per-Task Breakdown")
        lines.append("")
        lines.append("| Task | CLIMB accuracy | Random accuracy | Delta |")
        lines.append("|---|---|---|---|")
        for task in all_tasks:
            ca = climb_eval["tasks"].get(task, {}).get("accuracy", None)
            ra = random_eval["tasks"].get(task, {}).get("accuracy", None)
            ca_str = f"{ca:.4f}" if ca else "N/A"
            ra_str = f"{ra:.4f}" if ra else "N/A"
            if ca and ra:
                delta_str = f"{ca - ra:+.4f}"
            else:
                delta_str = "N/A"
            lines.append(f"| {task} | {ca_str} | {ra_str} | {delta_str} |")
        lines.append("")

    report_text = "\n".join(lines)
    report_path = os.path.join(result_dir, "validation_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    return report_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CLIMB validation report")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--climb-train-log", default="")
    parser.add_argument("--random-train-log", default="")
    parser.add_argument("--climb-eval-log", required=True)
    parser.add_argument("--random-eval-log", required=True)
    parser.add_argument("--base-model-tag", required=True)
    parser.add_argument("--climb-model-tag", required=True)
    parser.add_argument("--random-model-tag", required=True)
    args = parser.parse_args()

    generate_midtrain_report(
        result_dir=args.result_dir,
        climb_eval_log=args.climb_eval_log,
        random_eval_log=args.random_eval_log,
        base_model_tag=args.base_model_tag,
        climb_model_tag=args.climb_model_tag,
        random_model_tag=args.random_model_tag,
    )
