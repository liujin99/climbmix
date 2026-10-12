#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  reinforcement_charts.py — 『三处加固』举证图 (deck v8, 2026-10-12)
#
#  用法:
#    python3 scripts/diagnostics/reinforcement_charts.py result/prod5_20260929_201108
#
#  输出 (RUN_DIR 根, 与 report_charts.py 的 PNG 同层; make_deck.py 自动引用):
#    reinforcement_d14_cluster_balance.png   ① 两张环形图: 论文距离合并
#                                            (1 巨簇) vs 均衡划分 (15 簇真实份额)
#    reinforcement_d8_scoring_snr.png        ② 左: 逐基准 信号 vs 噪声 (SNR);
#                                            右: prod3 误加权事故 逐基准 Δ
#    reinforcement_d19_final_selection.png   ③ prod4 同轮对决: 实测点 vs
#                                            从未训练的外推『冠军』+ uniform 家族带
#
#  数据 = 实验记录已核实数字 (逐数注明出处; 跨轮历史数字无法从 prod5
#  run 目录读取, 为常量; 15 簇真实份额读 run 目录):
#    D14 塌缩诊断   paper_deviations.md D14 (prod1 事后 A6/A7):
#                   巨簇 99.05% / 13 孤岛 0.95%; K=14/21/24/32 ->
#                   97.5~99.0%; 去 floor -> K=3 (99.9%); cap <= 8.8%
#    D8  信号/噪声  experiment_prod2_k15bal.md §4.2 (30 候选实测):
#                   sd_cfg / SE_binom / SNR 逐基准表
#        prod3 事故 TODO.md 2026-09-15 P0 记录 (run4 双臂验证):
#                   gsm8k 0.119->0.052 (z=-6.2) / math 0.018->0.004 /
#                   arc_challenge +0.0296 / 总分 0.166 vs 0.178
#    D19 prod4 对决 experiment_prod4.md §3d/§4d (d28, 旧协议口径):
#                   cfg72 0.2142 / cfg25 0.2066 / 外推 0.1972 / -rep 0.1990;
#                   uniform 家族带 .1647-.1833
#    15 簇份额      RUN_DIR/macro_info.json (⑬r) 或 cluster_info_cache.json
#                   (legacy), schema 同 recipe_report.py (label/num_tokens);
#                   全缺 -> 等分示意 (图内注明)
#
#  原则与 report_charts.py 一致: 图内标签 ASCII (服务器无 CJK 字体),
#  mfu_probe 同教训; matplotlib 缺失则报错退出; 只读 RUN_DIR。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import json
import math
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("✗ 缺 matplotlib — pip install matplotlib (PEP 668 环境加 "
             "--break-system-packages), 装好后重跑本脚本")

TEAL = "#55A868"      # 信号 / 实测
RED = "#C44E52"       # 问题 / 失败
BLUE = "#4C72B0"      # 基线 / 均衡簇
ORANGE = "#DD8452"    # C10 / 赢家 (report_charts CLIMB_COLOR)
GRAY = "#9A9A9A"
DGRAY = "#555555"

# ── 数据 (出处见文件头) ────────────────────────────────────────────────
D14_KSWEEP_NOTE = ("K=14/21/24/32: giant keeps 97.5-99.0%\n"
                   "remove floor -> swallows to 3 clusters (99.9%)")
D8_SNR = [  # bench, signal sd_cfg, noise SE_binom, SNR (prod2 §4.2)
    ("arc_easy", 0.0225, 0.0127, 1.77),
    ("mmlu_stem", 0.0099, 0.0100, 1.00),
    ("arc_challenge", 0.0157, 0.0191, 0.82),
    ("gpqa_diamond", 0.0283, 0.0411, 0.69),
    ("gsm8k", 0.0048, 0.0163, 0.29),
    ("math", 0.0024, 0.0259, 0.09),
]
D8_PROD3 = [  # bench, delta winner-control, label (TODO.md prod3 记录)
    ("gsm8k", -0.067, "-56% (z=-6.2)"),
    ("math", -0.014, "-78%"),
    ("arc_challenge", 0.0296, "+3% (MC flat)"),
]
D19_RACE = [  # label, score, color, value note (experiment_prod4.md §3d)
    ("cfg72 - measured", 0.2142, BLUE, "0.2142 - best of run"),
    ("cfg25 - measured", 0.2066, BLUE, "0.2066"),
    ("extrapolated-rep - never trained", 0.1990, GRAY, "0.1990"),
    ("extrapolated - paper's final pick", 0.1972, RED, "0.1972 - hot-zone bottom"),
]
D19_BAND = (0.1647, 0.1833)  # uniform 家族带 (random3b x2 + natural + domainfix)


def load_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def cluster_token_shares(run_dir):
    """macro_info.json / cluster_info_cache.json → (labels, shares%)。
    schema 同 recipe_report.py: label / cluster_id / num_tokens。"""
    for name in ("macro_info.json", "cluster_info_cache.json"):
        for sub in ("", "state/", "detail/"):
            ci = load_json(os.path.join(run_dir, sub, name))
            if isinstance(ci, list) and ci:
                labels = [str(c.get("label") or f"C{c.get('cluster_id', i)}")
                          for i, c in enumerate(ci)]
                toks = np.array([float(c.get("num_tokens") or 0) for c in ci])
                if toks.sum() > 0:
                    return labels, toks / toks.sum() * 100.0
                return labels, np.full(len(ci), 100.0 / len(ci))
    return None, None


def chart_d14(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    real = shares is not None
    if not real:
        labels = [f"C{i}" for i in range(15)]
        shares = np.full(15, 100.0 / 15)
    hi = labels.index("C10") if "C10" in labels else 9
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # 左: 论文距离合并 — 1 巨簇 + 13 孤岛
    ax = axes[0]
    ax.pie([99.05, 0.95], colors=[RED, GRAY], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "1\nusable\nknob", ha="center", va="center",
            fontsize=16, fontweight="bold", color=RED)
    ax.text(0, -0.70, "99.05%", ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")
    ax.annotate("13 island clusters: 0.95%", xy=(0.06, 1.0),
                xytext=(0.42, 1.22), fontsize=10, color=DGRAY,
                arrowprops=dict(arrowstyle="->", color=DGRAY))
    ax.set_title("Paper's distance merge (measured on our pool)",
                 fontsize=12, fontweight="bold")
    ax.text(0, -1.52, D14_KSWEEP_NOTE + "\n-> no K/tau works; all 35 configs "
            "pinned to the giant cluster", ha="center", fontsize=9.5, color=DGRAY)

    # 右: 均衡划分 — 15 簇真实份额, C10 高亮
    ax = axes[1]
    colors = [ORANGE if i == hi else BLUE for i in range(len(shares))]
    ax.pie(shares, colors=colors, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "15\nusable\nknobs", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#1F4E79")
    cum = float(np.sum(shares[:hi]))
    mid = 90.0 - (cum + float(shares[hi]) / 2) * 3.6
    th = math.radians(mid)
    ax.annotate(f"C10: {shares[hi]:.1f}% of pool\n-> winner later bets 80% (14.6x)",
                xy=(0.80 * math.cos(th), 0.80 * math.sin(th)),
                xytext=(0.10, 1.30), fontsize=10, color=DGRAY,
                arrowprops=dict(arrowstyle="->", color=ORANGE))
    ax.set_title("Balanced split - our fix (D14), real token shares"
                 + ("" if real else ", schematic"),
                 fontsize=12, fontweight="bold")
    ax.text(0, -1.52, "cap <= 8.8% by construction (max token share)\n"
            "single-cluster > 50% -> pipeline aborts (structural gate)"
            + ("" if real else "\n(macro_info.json not found: equal shares shown)"),
            ha="center", fontsize=9.5, color=DGRAY)

    fig.suptitle("One pool, two partitions - what each cluster gets (token share)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, "reinforcement_d14_cluster_balance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}" + ("  [real shares from macro_info.json]"
                           if real else "  [schematic: no macro_info.json]"))


def chart_d8(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    # 左: 逐基准 信号 vs 噪声
    ax = axes[0]
    y = np.arange(len(D8_SNR))[::-1]
    sig = [r[1] for r in D8_SNR]
    noi = [r[2] for r in D8_SNR]
    ax.barh(y + 0.19, sig, height=0.34, color=TEAL, label="signal: cfg-to-cfg spread")
    ax.barh(y - 0.19, noi, height=0.34, color=RED, label="noise: binomial SE")
    ax.set_yticks(y, [r[0] for r in D8_SNR])
    for yi, s, n, r in zip(y, sig, noi, [x[3] for x in D8_SNR]):
        ax.text(max(s, n) + 0.0009, yi, f"SNR {r:.2f}", va="center",
                fontsize=9.5, fontweight="bold",
                color=TEAL if r >= 1.0 else RED)
    ax.set_xlim(0, 0.055)
    ax.set_xlabel("score spread (accuracy units)")
    ax.set_title("Why weight by SNR: real signal vs sampling noise\n"
                 "(prod2, 30 measured configs)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.text(0.0275, 4.35, "signal < noise -> accuracy is a coin flip",
            fontsize=9.5, style="italic", color=DGRAY)

    # 右: prod3 误加权事故
    ax = axes[1]
    y = np.arange(len(D8_PROD3))[::-1]
    for yi, (b, d, lab) in zip(y, D8_PROD3):
        ax.barh(yi, d, height=0.5, color=RED if d < 0 else TEAL)
        ax.text(d + (0.002 if d >= 0 else -0.002), yi, lab, va="center",
                ha="left" if d >= 0 else "right", fontsize=10, fontweight="bold",
                color=RED if d < 0 else TEAL)
    ax.axvline(0, color=DGRAY, lw=1)
    ax.set_yticks(y, [r[0] for r in D8_PROD3])
    ax.set_xlim(-0.085, 0.055)
    ax.set_xlabel("delta (winner - control arm)")
    ax.set_title("prod3, mis-weighted scoring: what the search picked\n"
                 "(per-benchmark delta, winner vs control)", fontsize=12,
                 fontweight="bold")
    ax.text(0, -0.95, "headline STEM 0.166 vs 0.178 (noise-level) - the average "
            "hid it;\nwinner diet: encyclopedia/prose clusters, reasoning "
            "clusters zeroed", ha="center", fontsize=9.5, color=DGRAY)

    fig.suptitle("Scoring by SNR: quiet benchmarks must not drown the loud ones",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    path = os.path.join(out_dir, "reinforcement_d8_scoring_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def chart_d19(out_dir):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    xmin, xmax = 0.16, 0.225
    ax.axvspan(D19_BAND[0], D19_BAND[1], color=GRAY, alpha=0.18)
    ax.text((D19_BAND[0] + D19_BAND[1]) / 2, 3.62,
            "uniform family band\n.1647 - .1833", ha="center", fontsize=9,
            color=DGRAY)
    for i, (lab, v, c, note) in enumerate(D19_RACE):
        yy = 3 - i
        ax.barh(yy, v - xmin, left=xmin, height=0.55, color=c)
        ax.text(v + 0.0007, yy, note, va="center", fontsize=10.5,
                fontweight="bold", color=c if c != GRAY else "#777777")
    ax.set_yticks([3, 2, 1, 0], [r[0] for r in D19_RACE])
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.75, 4.15)
    ax.set_xlabel("d28 STEM (centered; axis starts at 0.16)")
    ax.annotate("", xy=(0.2142, 0.45), xytext=(0.1972, 0.45),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    ax.text(0.2060, 0.60, "gap 0.017 (z~2.0)", ha="center", fontsize=10,
            fontweight="bold", color=RED)
    ax.set_title("prod4, same search run: measured points beat the "
                 "never-trained 'champion'", fontsize=13, fontweight="bold")
    ax.text(0.1925, -0.68, "both extrapolated points sit above the uniform "
            "band, but bottom of the hot zone;\nargmin lead below the noise "
            "floor = fitted noise, not signal", ha="center", fontsize=9.5,
            color=DGRAY)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = os.path.join(out_dir, "reinforcement_d19_final_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")


def main():
    ap = argparse.ArgumentParser(
        description="『三处加固』举证图 (D14/D8/D19) → run 目录根层 3 张 PNG")
    ap.add_argument("run_dir", help="run 目录 (prod5 归档; 图输出到其根层)")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="输出目录 (默认 = run_dir)")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        sys.exit(f"✗ 目录不存在: {args.run_dir}")
    out_dir = args.out_dir or args.run_dir
    print(f"『三处加固』举证图 → {out_dir}")
    print("  数据出处: paper_deviations.md D14 / experiment_prod2_k15bal.md "
          "§4.2 / TODO.md prod3 记录 / experiment_prod4.md §3d")
    chart_d14(out_dir, args.run_dir)
    chart_d8(out_dir)
    chart_d19(out_dir)
    print("下一步: python3 scripts/diagnostics/make_deck.py <run_dir>")


if __name__ == "__main__":
    main()
