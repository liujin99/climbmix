"""
Pipeline output: Markdown report + matplotlib domain distribution comparison chart.
"""

import json
import os
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
    lines.append(f"| Proxy size | `{config.proxy.model_size}` |")
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
