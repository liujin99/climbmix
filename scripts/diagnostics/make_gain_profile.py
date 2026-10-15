#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  make_gain_profile.py — per-task 增益剖面图 (README prod5 节配图)
#
#  用法:
#    python3 scripts/diagnostics/make_gain_profile.py            # 出图到
#                                          # docs/assets/gain_profile.png
#    python3 scripts/diagnostics/make_gain_profile.py --out /tmp/x.png
#
#  设计裁决 (2026-10-15 与用户多轮迭代定稿):
#    - 主对照 = uniform-cluster (论文 App. C.1 预注册对照): 同池/同簇/
#      同预算/同配方, 唯一差异是权重 — 隔离"CLIMB 学到了什么";
#    - 每任务一个点 + 95% CI 胡须 (两臂二项 SE), MC 蓝色 / 生成式橙色;
#    - 多基线稳健性不进画面, 收进脚注一行 (对每任务最强基线:
#      gsm8k z=+11.1 / math z=+2.3 / MC 全 |z|<1 — 数值见下方 stdout);
#    - 已否决的备选: 三基线全画 (18 根胡须, 信息密度不成比例) / 对基线
#      均值 (零线不对应任何可运行方法, 且 gpqa 仍为负 −.017, 两维度皆输)。
#
#  数据与验证链 (零内网依赖 — 全部来自已发布文档):
#    - 24 个原始 acc 取自 docs/experiment_prod5.md §4b 表 (列序 climb/
#      cfg90/cfg100/domainfix/natural/uniform/base, 其中 gsm8k 行与
#      README prod5 表逐位一致锚定列映射);
#    - 交叉核验: centered 复合分 (MC (acc-.25)/.75, 生成式原值) 从本表
#      逐位复现 README 的 climb .2086 / uniform .1657 (+.0429);
#    - 单种子边界: 胡须仅含评测抽样噪声, 训练种子带 (±1.6~3.2pt STEM,
#      来自 prod4 种子对) 已在图注声明。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforcement_charts import verify_layout

ROWS = [  # task, N, climb, uniform — docs/experiment_prod5.md §4b
    ("arc_easy",      2376, 0.7294, 0.7269),
    ("arc_challenge", 1172, 0.4360, 0.4334),
    ("mmlu_stem",     3545, 0.2920, 0.2872),
    ("gpqa_diamond",   198, 0.2121, 0.2071),
    ("gsm8k_cot",     1319, 0.3010, 0.1054),
    ("math_cot_500",   500, 0.0580, 0.0160),
]
GEN = {"gsm8k_cot", "math_cot_500"}
MC_C, GEN_C = "#4C72B0", "#DD8452"
CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "mmlu_stem": 0.25,
          "gpqa_diamond": 0.25, "gsm8k_cot": 0.0, "math_cot_500": 0.0}


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    default_out = os.path.join(repo_root, "docs", "assets",
                                "gain_profile.png")
    ap = argparse.ArgumentParser(
        description="per-task 增益剖面图 (climb vs uniform, prod5)")
    ap.add_argument("--out", default=default_out, help="输出 PNG 路径")
    args = ap.parse_args()

    print(f"{'task':<14} {'delta':>8} {'diff SE':>8} {'z':>7}")
    centered = {"climb": [], "uniform": []}
    ys, deltas, ci95s, zs, colors, labels = [], [], [], [], [], []
    for i, (task, n, pc, pu) in enumerate(ROWS):
        se = np.sqrt(pc * (1 - pc) / n + pu * (1 - pu) / n)
        d = pc - pu
        print(f"{task:<14} {d:>+8.4f} {se:>8.4f} {d / se:>+7.1f}")
        ch = CHANCE[task]
        centered["climb"].append((pc - ch) / (1 - ch))
        centered["uniform"].append((pu - ch) / (1 - ch))
        ys.append(i)
        deltas.append(d)
        ci95s.append(1.96 * se)
        zs.append(d / se)
        colors.append(GEN_C if task in GEN else MC_C)
        labels.append(f"{task}  (N={n})")
    stem_c = float(np.mean(centered["climb"]))
    stem_u = float(np.mean(centered["uniform"]))
    print(f"STEM cross-check (centered mean): climb {stem_c:.4f}  "
          f"uniform {stem_u:.4f}  delta {stem_c - stem_u:+.4f}")
    print("  (README prod5 table: 0.2086 / 0.1657 / +0.0429)")
    assert abs(stem_c - 0.2086) < 5e-4 and abs(stem_u - 0.1657) < 5e-4, \
        "STEM composite no longer matches README prod5 table — check ROWS"

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.axvline(0, color="#555555", lw=1.0)
    for y, d, c, col in zip(ys, deltas, ci95s, colors):
        ax.hlines(y, d - c, d + c, color=col, lw=1.8, zorder=3)
        for xe in (d - c, d + c):
            ax.vlines(xe, y - 0.09, y + 0.09, color=col, lw=1.8, zorder=3)
    ax.scatter(deltas, ys, s=72, color=colors, zorder=5,
               edgecolors="white", linewidths=1.1)
    for y, d, c, z in zip(ys, deltas, ci95s, zs):
        ax.annotate(f"{d:+.3f}  (z={z:+.1f})", (d + c, y),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=8.5, color="#333333")
    ax.axhline(3.5, color="#999999", lw=0.8, ls=":")
    ax.text(-0.055, 1.5, "Multiple choice\n(knowledge recall)",
            fontsize=8.5, color="#666666", ha="left", va="center",
            style="italic")
    ax.text(-0.055, 4.5, "Generative CoT\n(write-out math)",
            fontsize=8.5, color="#666666", ha="left", va="center",
            style="italic")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(-0.06, 0.25)
    ax.set_xticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("Δ raw accuracy  (CLIMB − uniform-cluster baseline)")
    ax.set_title("Where the gain lives: flat on multiple choice, "
                 "decisive on write-out math", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    fig.text(0.5, 0.005,
             "prod5, d28 target, 3B-token mid-training; CLIMB = final "
             "no-claim selection; baseline = equal 1/K cluster weights "
             "(paper App. C.1 control). Whiskers = 95% CI\n(two-arm "
             "binomial SE; single-seed arms — eval sampling noise only). "
             "Same pattern vs the strongest baseline per task: gsm8k "
             "z=+11.1, math z=+2.3, all MC |z|<1.\n"
             "Data: docs/experiment_prod5.md §4b.",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    if not verify_layout(fig, "gain_profile"):
        print("✗ layout check FAILED")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"✓ layout check PASS — saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
