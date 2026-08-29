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
from climbmix.core.predictor import LightGBMPredictor


def generate_predictor_scatter(
    output_dir: str,
    predictor_eval: List[Dict[str, Any]],
    maximize: bool,
) -> Optional[str]:
    """Paper Fig.9-style scatter: predictor predictions vs ground truth on
    held-out pairs (pooled across rounds). Returns None when no pairs exist."""
    preds: List[float] = []
    targets: List[float] = []
    for e in predictor_eval:
        preds.extend(e.get("val_preds") or [])
        targets.extend(e.get("val_targets") or [])
    if len(preds) < 2:
        return None

    # Predictor targets are loss-like (lower = better); flip to score space
    # (higher = better) when the metric is accuracy/maximize so the chart
    # reads the same way as the iteration table.
    sign = -1.0 if maximize else 1.0
    p = sign * np.array(preds, dtype=np.float64)
    t = sign * np.array(targets, dtype=np.float64)
    better = "higher" if maximize else "lower"

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(t, p, s=30, alpha=0.75, color="#4C72B0", edgecolors="none")
    lims = [float(min(t.min(), p.min())), float(max(t.max(), p.max()))]
    pad = 0.05 * (lims[1] - lims[0] + 1e-9)
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="perfect prediction")
    ax.set_xlabel(f"actual held-out score ({better} = better)")
    ax.set_ylabel(f"predicted score ({better} = better)")
    ax.set_title(f"Predictor quality: predicted vs actual (n={len(p)}, pooled)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "predictor_scatter.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_markdown_report(
    output_dir: str,
    config: Any,
    cluster_info: List[ClusterInfo],
    optimal_weights: Any,
    iter_results: List[IterationResult],
    stats: dict,
    stage_times: Dict[str, float],
    elapsed_seconds: float,
    search_state: Optional[dict] = None,
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
    lines.append("| Iter | Configs | Trained | Best Score | val R\u00b2 | val \u03c1 | online \u03c1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in iter_results:
        score_str = f"{r.best_score:.4f}" if r.best_score is not None else "N/A"
        r2_str = f"{r.predictor_r2:.4f}" if r.predictor_r2 is not None else "N/A"
        rho_str = (f"{r.predictor_spearman:.4f}"
                   if r.predictor_spearman is not None else "N/A")
        online_str = (f"{r.online_spearman:.4f}"
                      if r.online_spearman is not None else "N/A")
        lines.append(
            f"| {r.iteration} | {r.n_configs} | {r.n_trained} | {score_str} "
            f"| {r2_str} | {rho_str} | {online_str} |")
    lines.append("")

    # Predictor / pruning visibility: prefer the persisted search state
    # (canonical + carries the held-out pairs); fall back to the in-memory
    # iteration results (e.g. no-resume runs, old state files).
    predictor_eval = (search_state or {}).get("predictor_eval") or []
    online_eval = (search_state or {}).get("online_eval") or []
    pruning_history = (search_state or {}).get("pruning_history") or []
    if not predictor_eval:
        predictor_eval = [
            {"iteration": r.iteration, "val_r2": r.predictor_r2,
             "val_spearman": r.predictor_spearman}
            for r in iter_results if r.predictor_r2 is not None
        ]
    if not online_eval:
        online_eval = [
            {"iteration": r.iteration, "n": None, "spearman": r.online_spearman}
            for r in iter_results if r.online_spearman is not None
        ]
    if not pruning_history:
        pruning_history = [
            dict(r.pruning, iteration=r.iteration)
            for r in iter_results if r.pruning
        ]
    online_n_by_iter = {e.get("iteration"): e.get("n") for e in online_eval}

    lines.append("## Predictor Quality (LightGBM, paper D.10)")
    lines.append("")
    lines.append(
        "Per-round predictor metrics. val = held-out split of the accumulated\n"
        "configs (exists from accumulated N\u226510); online = rank agreement\n"
        "between predictions made **before** training a round's configs and\n"
        "their actual scores (no split needed). Paper D.10 reports 94%\n"
        "held-out Spearman at 112 configs / 350M proxy \u2014 a smaller N here\n"
        "necessarily reads lower; that is a budget artifact, not a bug.")
    lines.append("")
    if predictor_eval or online_eval:
        lines.append("| Iter | Val n | val R\u00b2 | val Spearman | online \u03c1 (n) |")
        lines.append("|---|---|---|---|---|")
        iters = sorted({e.get("iteration") for e in predictor_eval} |
                       {e.get("iteration") for e in online_eval})
        pe_by_iter = {e.get("iteration"): e for e in predictor_eval}
        oe_by_iter = {e.get("iteration"): e for e in online_eval}
        for it in iters:
            pe = pe_by_iter.get(it) or {}
            oe = oe_by_iter.get(it) or {}
            n_val = pe.get("n_val")
            n_val_str = str(n_val) if n_val is not None else "N/A"
            v_r2 = pe.get("val_r2")
            v_r2_str = f"{v_r2:.4f}" if v_r2 is not None else "N/A"
            v_rho = pe.get("val_spearman")
            v_rho_str = f"{v_rho:.4f}" if v_rho is not None else "N/A"
            o_rho = oe.get("spearman")
            o_n = oe.get("n")
            if o_rho is not None:
                o_str = f"{o_rho:.4f}" + (f" ({o_n})" if o_n is not None else "")
            else:
                o_str = "N/A"
            lines.append(f"| {it} | {n_val_str} | {v_r2_str} | {v_rho_str} | {o_str} |")
        lines.append("")

        pooled_preds: List[float] = []
        pooled_targets: List[float] = []
        for e in predictor_eval:
            pooled_preds.extend(e.get("val_preds") or [])
            pooled_targets.extend(e.get("val_targets") or [])
        if len(pooled_preds) >= 2:
            pooled_rho = LightGBMPredictor._spearman(
                np.array(pooled_preds, dtype=np.float64),
                np.array(pooled_targets, dtype=np.float64),
            )
            if np.isfinite(pooled_rho):
                lines.append(
                    f"**Pooled held-out Spearman: \u03c1 = {pooled_rho:.4f}** "
                    f"({len(pooled_preds)} pairs across rounds; each predicted "
                    f"by that round's predictor)")
                lines.append("")
    else:
        lines.append("_No predictor-eval data (accumulated N < 10 throughout \u2014 "
                     "e.g. speedrun budget \u2014 and no online backtest records)._")
        lines.append("")

    maximize = getattr(config, "metric_direction", "minimize") == "maximize"
    scatter_path = generate_predictor_scatter(output_dir, predictor_eval, maximize)
    if scatter_path:
        lines.append("![Predictor: predicted vs actual]"
                     f"({os.path.basename(scatter_path)})")
        lines.append("")

    # Feature importance of the FINAL predictor (the one that drove the
    # full-design-space selection): which clusters' weights the LightGBM
    # actually split on. Split counts; Share = percent of all splits.
    # Rendered only when in-memory iteration results carry a fitted model
    # (fresh post-search report, or a resume which refits the predictor);
    # skipped silently otherwise (e.g. report from a state file alone).
    final_model = None
    for r in reversed(iter_results):
        m = getattr(getattr(r, "predictor", None), "_model", None)
        if m is not None and hasattr(m, "feature_importances_"):
            final_model = m
            break
    if final_model is not None:
        imp = np.asarray(final_model.feature_importances_, dtype=np.float64)
        if len(imp) == K and float(imp.sum()) > 0:
            share = 100.0 * imp / float(imp.sum())
            order = np.argsort(-imp)
            lines.append("## Predictor Feature Importance")
            lines.append("")
            lines.append(
                "Split counts of the final LightGBM predictor (the model that\n"
                "drove the full-design-space selection): which cluster weights\n"
                "the predictor actually used to rank mixtures.")
            lines.append("")
            lines.append("| Rank | Cluster | Splits | Share |")
            lines.append("|---|---|---|---|")
            for rank, k in enumerate(order, 1):
                lines.append(f"| {rank} | {labels[k]} | {int(imp[k])} "
                             f"| {share[k]:.1f}% |")
            lines.append("")

    lines.append("## Search Pruning (top-N narrowing)")
    lines.append("")
    lines.append(
        "Paper \u00a72.2 pruning is selection-level: each guided round ranks the\n"
        "candidate pool with the predictor and only the top-N band is\n"
        "eligible for verbatim sampling \u2014 candidates below the cutoff are\n"
        "never proxy-trained (in addition to the per-round budget decay).\n"
        "Iteration 1 samples from the token-count Dirichlet prior (no\n"
        "predictor yet, nothing pruned).")
    lines.append("")
    if pruning_history:
        lines.append("| Iter | Pool | Novel | Top-N | Top % | Sampled | "
                     "Excluded (never trained) |")
        lines.append("|---|---|---|---|---|---|---|")
        total_excluded = 0
        for e in pruning_history:
            pool = e.get("pool")
            novel = e.get("novel", pool)
            top_n = e.get("top_n")
            sampled = e.get("sampled")
            excluded = (novel - top_n) if (novel is not None
                                           and top_n is not None) else None
            if excluded is not None:
                total_excluded += excluded
            top_pct = (f"{100.0 * top_n / max(1, novel):.1f}%"
                       if top_n is not None and novel else "N/A")
            lines.append(
                f"| {e.get('iteration')} | {pool} | {novel} | {top_n} "
                f"| {top_pct} | {sampled} | {excluded} |")
        lines.append(f"| | | | | | **total** | **{total_excluded}** |")
        lines.append("")
    else:
        lines.append("_No guided rounds recorded (single-iteration search or "
                     "old state file)._")
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
    info = {"core_metric": None, "stem_metric": None, "stem_nll": None, "tasks": {}}
    if not path or not os.path.exists(path):
        return info
    # base_eval.py prints (same line, print end=''):
    #   "Evaluating: <task> (<n>-shot, type: <t>)... "
    #   "accuracy: A | centered: C | nll: N | time: Ts"
    # nll was added to the line later; treat it (and time) as optional so the
    # pattern parses both old and new logs. A missing optional group is None.
    task_pat = re.compile(
        r"Evaluating:\s+(.+?)\s+\(.*?\)\.\.\.\s+accuracy:\s+([\d.]+)\s+\|\s+centered:\s+(-?[\d.]+)"
        r"(?:\s+\|\s+nll:\s+(-?[\d.]+))?"
        r"(?:\s+\|\s+time:\s+([\d.]+)s)?"
    )
    core_pat = re.compile(r"CORE metric:\s+(-?[\d.]+)")
    # Printed for --eval-benchmarks=stem (CORE only exists when all 22 DCLM
    # core tasks ran): "STEM metric: X" / "STEM NLL: Y" — without these the
    # whole Step 8 comparison showed N/A for every STEM-only eval.
    stem_pat = re.compile(r"STEM metric:\s+(-?[\d.]+)")
    stem_nll_pat = re.compile(r"STEM NLL:\s+(-?[\d.]+)")
    for line in open(path):
        m = task_pat.search(line)
        if m:
            info["tasks"][m.group(1)] = {
                "accuracy": float(m.group(2)),
                "centered": float(m.group(3)),
                "nll": float(m.group(4)) if m.group(4) is not None else None,
                "time": float(m.group(5)) if m.group(5) is not None else None,
            }
        m = core_pat.search(line)
        if m:
            info["core_metric"] = float(m.group(1))
        m = stem_pat.search(line)
        if m:
            info["stem_metric"] = float(m.group(1))
        m = stem_nll_pat.search(line)
        if m:
            info["stem_nll"] = float(m.group(1))
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

    # Aggregated metric: CORE only exists when all 22 DCLM core tasks ran;
    # STEM-only evals (this pipeline's default) print "STEM metric" instead.
    # Prefer CORE, fall back to STEM — never show N/A when we have a number.
    if climb_eval["core_metric"] is not None or random_eval["core_metric"] is not None:
        metric_name = "CORE"
    else:
        metric_name = "STEM (centered acc)"
    climb_m = climb_eval["core_metric"] if climb_eval["core_metric"] is not None \
        else climb_eval["stem_metric"]
    random_m = random_eval["core_metric"] if random_eval["core_metric"] is not None \
        else random_eval["stem_metric"]

    lines: List[str] = []
    lines.append("# CLIMB Mid-Training Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Base model:** `{base_model_tag}`")
    lines.append(f"**CLIMB model:** `{climb_model_tag}`")
    lines.append(f"**Random model:** `{random_model_tag}`")
    lines.append("")

    lines.append(f"## {metric_name} Comparison")
    lines.append("")
    lines.append(f"| Method | {metric_name} |")
    lines.append(f"|---|---|")
    lines.append(f"| CLIMB optimal | {climb_m:.4f} |" if climb_m is not None else "| CLIMB optimal | N/A |")
    lines.append(f"| Random baseline | {random_m:.4f} |" if random_m is not None else "| Random baseline | N/A |")
    if climb_m is not None and random_m is not None:
        delta = climb_m - random_m
        lines.append(f"| **Delta** | **{delta:+.4f}** |")
    if metric_name.startswith("STEM") and (climb_eval["stem_nll"] is not None
                                            or random_eval["stem_nll"] is not None):
        lines.append("")
        lines.append("| Method | STEM NLL |")
        lines.append("|---|---|")
        cn, rn = climb_eval["stem_nll"], random_eval["stem_nll"]
        lines.append(f"| CLIMB optimal | {cn:.4f} |" if cn is not None else "| CLIMB optimal | N/A |")
        lines.append(f"| Random baseline | {rn:.4f} |" if rn is not None else "| Random baseline | N/A |")
        if cn is not None and rn is not None:
            lines.append(f"| **Delta** | **{cn - rn:+.4f}** |")
    lines.append("")

    all_tasks = sorted(set(climb_eval["tasks"].keys()) | set(random_eval["tasks"].keys()))
    if all_tasks:
        lines.append("## Per-Task Breakdown")
        lines.append("")
        # Centered accuracy is the paper's comparison metric (raw accuracy
        # sits near chance on multiple-choice STEM tasks and hides differences).
        lines.append("| Task | CLIMB acc | CLIMB centered | Random acc | Random centered | Δ centered |")
        lines.append("|---|---|---|---|---|---|")
        for task in all_tasks:
            ct = climb_eval["tasks"].get(task, {})
            rt = random_eval["tasks"].get(task, {})
            ca, ra = ct.get("accuracy"), rt.get("accuracy")
            cc, rc = ct.get("centered"), rt.get("centered")
            ca_str = f"{ca:.4f}" if ca is not None else "N/A"
            ra_str = f"{ra:.4f}" if ra is not None else "N/A"
            cc_str = f"{cc:.4f}" if cc is not None else "N/A"
            rc_str = f"{rc:.4f}" if rc is not None else "N/A"
            if cc is not None and rc is not None:
                delta_str = f"{cc - rc:+.4f}"
            else:
                delta_str = "N/A"
            lines.append(f"| {task} | {ca_str} | {cc_str} | {ra_str} | {rc_str} | {delta_str} |")
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
