#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  make_search_convergence.py — 搜索收敛图 (README prod5 节第二张配图)
#
#  用法:
#    python3 scripts/diagnostics/make_search_convergence.py   # 出图到
#                                  # docs/assets/search_convergence.png
#    python3 scripts/diagnostics/make_search_convergence.py --out /tmp/x.png
#
#  复刻 deck chart A (report_charts.py chart_convergence) 的设计:
#    - 115 实测点按轮散布 (x = 时序 config 序号, y = 终态 utility),
#      轮界虚线; 第 1 轮随机带 (mean ± 1σ) 灰底 — 后两轮整体抬出
#      该带 = "搜索在学" 的一眼结论;
#    - 每轮最佳黑色连线 + 数值标注; 末轮最佳 = 舰队最佳 cfg#101 =
#      d28 验证的赢家臂 (README prod5 表 climb 行) — 搜索最好的点
#      在目标规模上兑现;
#    - lift 文本框: top-3/top-10 均值 vs 第 1 轮均值 (σ 单位)。
#
#  数据与口径 (治理链 — 全部数字可溯源到已提交工件):
#    - 逐点数据 = scripts/diagnostics/data/prod5_search_dump.csv
#      (prod5 search_state 舰队终态口径, 与 prod4_search_dump.csv 同
#      一入档先例; 经文本通道从服务器 RUN_DIR 压出, deck v1.0.79 机制);
#    - **双口径**: §3b 表"轮最佳"(0.9742/1.1967/1.4524)为当轮收官实时
#      口径; 本图为终态口径(全部 115 点在收官舰队统计下的重评分, 轮最佳
#      0.8560/1.0324/1.4524)。同一配比的复合分随舰队统计漂移 — prod4
#      §3b 注[1] 已记载该现象; 两口径单调性一致、末轮(=收官轮)重合。
#    - 轮边界按文档口径 66/33/16 (§3a/§3c online-ρ n=33 佐证); 原始
#      state 的 realized 重建为 67/32/16, 差一个 R1/R2 边界点 (id 66,
#      +0.626, 非任一轮最佳) — 两边界下各轮最佳与单调性逐位相同。
#
#  断言链 (render 前全跑, 数据漂移即 fail-loud):
#    n=115; 轮计数 66/33/16; 终态轮最佳 0.8560/1.0324/1.4524;
#    全场最佳在 cfg#101 = §3b 末轮"轮最佳"1.4524 (两口径唯一公共数)。
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

CSV_NAME = "prod5_search_dump.csv"
BEST_EXPECTED = {1: 0.8560, 2: 1.0324, 3: 1.4524}   # 终态口径轮最佳
COUNTS_EXPECTED = {1: 66, 2: 33, 3: 16}              # 文档口径 §3a
FLEET_BEST_ID, FLEET_BEST_SCORE = 101, 1.4524         # cfg#101, 两口径公共
ROUND_COLORS = {1: "#4C72B0", 2: "#55A868", 3: "#C44E52"}  # deck 同款
ROUND_LABELS = {1: "round 1 (Dirichlet random, n=66)",
                2: "round 2 (predictor-guided, n=33)",
                3: "round 3 (predictor-guided, n=16)"}


def load_points():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "data", CSV_NAME)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    it, ids, scores = data[:, 0].astype(int), data[:, 1].astype(int), data[:, 2]
    # ── 断言链 ──
    assert len(scores) == 115, f"n={len(scores)} != 115"
    counts = {r: int((it == r).sum()) for r in (1, 2, 3)}
    assert counts == COUNTS_EXPECTED, f"round counts {counts}"
    bests = {r: float(scores[it == r].max()) for r in (1, 2, 3)}
    for r, v in BEST_EXPECTED.items():
        assert abs(bests[r] - v) < 5e-4, f"round-{r} best {bests[r]} != {v}"
    gi = int(np.argmax(scores))
    assert ids[gi] == FLEET_BEST_ID, f"fleet best at id {ids[gi]} != 101"
    assert abs(scores[gi] - FLEET_BEST_SCORE) < 5e-4, "fleet best != 1.4524"
    assert abs(bests[3] - FLEET_BEST_SCORE) < 5e-4, "round-3 best != fleet best"
    assert np.all(np.diff(it) >= 0), "round blocks not contiguous"
    return it, ids, scores


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    default_out = os.path.join(repo_root, "docs", "assets",
                                "search_convergence.png")
    ap = argparse.ArgumentParser(
        description="搜索收敛图 (prod5, 115 点 × 3 轮, 终态口径)")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()

    it, ids, scores = load_points()
    r1 = scores[it == 1]
    mu, sd = float(r1.mean()), float(r1.std(ddof=1))
    lifts = {}
    for k in (3, 10):
        top = np.sort(scores)[-k:]
        lifts[k] = (float(top.mean()) - mu) / sd

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(1, len(scores) + 1)
    for r in (1, 2, 3):
        m = it == r
        ax.scatter(x[m], scores[m], s=22, alpha=0.65,
                   color=ROUND_COLORS[r], edgecolors="none",
                   label=ROUND_LABELS[r], zorder=3)
    for r, b in ((2, 66.5), (3, 99.5)):
        ax.axvline(b, color="gray", linestyle=":", linewidth=0.8)
    ax.axhspan(mu - sd, mu + sd, color="#4C72B0", alpha=0.10, zorder=1,
               label="round-1 random level (mean ± 1σ)")
    # best 连线锚定各轮最佳的**真实位置** (deck chart A 同款, 非轮末)
    bx, by = [], []
    for r in (1, 2, 3):
        m = np.where(it == r)[0]
        gi = int(m[np.argmax(scores[m])])
        bx.append(gi + 1)
        by.append(float(scores[gi]))
    assert ids[bx[2] - 1] == FLEET_BEST_ID, "round-3 best is not cfg#101"
    ax.plot(bx, by, "o-", color="#222222", linewidth=1.4, markersize=6,
            label="best per round", zorder=5)
    for xi, yi in zip(bx, by):
        ax.annotate(f"{yi:+.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    lift_txt = " | ".join(f"top-{k}: {v:+.2f} σ" for k, v in lifts.items())
    ax.text(0.02, 0.97, f"search lift vs round-1:  {lift_txt}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    ax.set_xlabel("measured mixture (chronological)")
    ax.set_ylabel("proxy utility (z-scale, higher = better)")
    ax.set_xlim(-1, 118)
    ax.set_ylim(float(scores.min()) - 0.12, max(by) + 0.24)
    ax.set_title("Search convergence: 115 measured mixtures over "
                 "3 guided rounds")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.005,
             "proxy utility = SNR-weighted composite of each mixture's "
             "d20-proxy scores (6 tasks, z-scale); rounds 2-3 leave the "
             "round-1 random band — the search is learning.\n"
             "the round-3 best (cfg#101) became the d28-winning arm; "
             "final-fleet-statistics caliber — closing-snapshot bests "
             "0.974 → 1.197 → 1.452 (experiment_prod5.md §3b note)",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    if not verify_layout(fig, "search_convergence"):
        sys.exit("LAYOUT FAIL: search_convergence")
    fig.savefig(args.out, dpi=200)
    print(f"saved: {args.out}")
    print(f"round bests (final caliber): "
          f"{by[0]:+.4f} / {by[1]:+.4f} / {by[2]:+.4f}  "
          f"(fleet best cfg#101, d28 winner)")
    print(f"lifts: {lift_txt}")


if __name__ == "__main__":
    main()
