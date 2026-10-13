#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_flow_icons.py — 流程页意象小图 (deck_assets/, 生成后随仓库分发)。

make_deck.py 的『怎么做的』流程页四张卡片顶部用的小插图: 无文字纯意象
(分簇散点 / 小模型替身 / 学习器收敛 / 正式规模对决), 底色逐卡匹配
(①-③ 浅青 #E8F5F6, ④ navy #1F4E79) — pptx 里与卡片无缝融合。
改图重跑本脚本并提交 4 个 PNG (make_deck 缺件自动回退矢量图标,
所以只改 PNG 不改逻辑也安全)。"""
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle
except ImportError:
    raise SystemExit("✗ 缺 matplotlib")

HERE = os.path.dirname(os.path.abspath(__file__))

TEAL, NAVY, ORANGE, BLUE, GRAY = ("#55A868", "#1F4E79", "#DD8452",
                                  "#4C72B0", "#9A9A9A")
CARD_BG, NAVY_BG = "#E8F5F6", "#1F4E79"
W, H = 3.2, 1.0          # inch; 卡片图标区 ≈ 2.1×0.66in, 比例 3.2:1


def new_ax(bg):
    fig = plt.figure(figsize=(W, H), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    return fig, ax


def save(fig, name):
    p = os.path.join(HERE, name)
    fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✓ {p}")


# ① 语义分簇: 文档按内容自动聚成主题簇
def icon_cluster():
    fig, ax = new_ax(CARD_BG)
    rng = np.random.default_rng(7)
    blobs = [  # (cx, cy, sx, sy, n, color)
        (0.55, 0.55, 0.13, 0.15, 38, TEAL),
        (1.45, 0.68, 0.09, 0.11, 30, NAVY),
        (2.05, 0.35, 0.11, 0.13, 32, ORANGE),
        (2.78, 0.68, 0.08, 0.10, 26, BLUE),
        (1.05, 0.26, 0.07, 0.08, 20, GRAY),
    ]
    for cx, cy, sx, sy, n, c in blobs:
        x = rng.normal(cx, sx, n)
        y = rng.normal(cy, sy, n)
        ax.scatter(x, y, s=13, color=c, alpha=0.9, edgecolors="none",
                   zorder=3, clip_on=True)
        ax.add_patch(Rectangle((cx - 2.4 * sx, cy - 2.4 * sy),
                               4.8 * sx, 4.8 * sy, fill=False, ec=c,
                               ls=(0, (3, 2)), lw=1.0, alpha=0.45, zorder=2))
    save(fig, "flow_cluster.png")


# ② 配比采样: 候选配比 = 权重堆叠条, 逐轮向高分集中 (底部 = 本轮选中)
def icon_sample():
    fig, ax = new_ax(CARD_BG)
    cols = [TEAL, NAVY, ORANGE, BLUE, GRAY]
    rows = [  # (y, 段占比, 整体透明度 — 越下越"聚焦")
        (0.72, [0.30, 0.22, 0.18, 0.14, 0.16], 0.45),
        (0.47, [0.20, 0.11, 0.41, 0.15, 0.13], 0.70),
        (0.22, [0.08, 0.05, 0.68, 0.09, 0.10], 1.00),
    ]
    for y, fracs, al in rows:
        x = 0.28
        for f, c in zip(fracs, cols):
            w = 2.20 * f
            ax.add_patch(Rectangle((x, y), w, 0.17, fc=c, ec="white",
                                   lw=0.8, alpha=al, zorder=3))
            x += w
    ax.add_patch(Rectangle((0.24, 0.18), 2.28, 0.25, fill=False, ec=ORANGE,
                           ls=(0, (4, 2)), lw=1.2, zorder=5))
    save(fig, "flow_sample.png")


# ③ 小模型当替身: 小方块快速试错 → 大方块(虚线)最终复验
def icon_proxy():
    fig, ax = new_ax(CARD_BG)

    def cube(x, y, s, faces, ec, ls="-"):
        dx = dy = 0.30 * s
        front, top, side = faces
        ax.add_patch(Polygon([[x, y], [x + s, y], [x + s, y + s], [x, y + s]],
                             fc=front, ec=ec, lw=1.3, ls=ls, zorder=3))
        ax.add_patch(Polygon([[x, y + s], [x + s, y + s],
                              [x + s + dx, y + s + dy], [x + dx, y + s + dy]],
                             fc=top, ec=ec, lw=1.3, ls=ls, zorder=3))
        ax.add_patch(Polygon([[x + s, y], [x + s + dx, y + dy],
                              [x + s + dx, y + s + dy], [x + s, y + s]],
                             fc=side, ec=ec, lw=1.3, ls=ls, zorder=3))

    for k, dy in ((0.10, 0.14), (0.05, 0.26), (0.10, 0.38)):  # 速度线
        ax.plot([0.12, 0.12 + k + 0.16], [dy, dy], color=TEAL, lw=1.4,
                solid_capstyle="round", alpha=0.55, zorder=2)
    cube(0.42, 0.26, 0.30, (TEAL, "#7CC49B", "#468A57"), "#2E6B4B")
    ax.add_patch(FancyArrowPatch((0.92, 0.44), (1.82, 0.44),
                                 arrowstyle="-|>", mutation_scale=16,
                                 color=TEAL, lw=2.0, zorder=4))
    cube(2.00, 0.14, 0.58, ("none", "#00000000", "#00000000"), NAVY,
         ls=(0, (4, 3)))
    save(fig, "flow_proxy.png")


# ③ 学习器指路: 候选逐轮向高分区域收拢
def icon_learner():
    fig, ax = new_ax(CARD_BG)
    x = np.linspace(0, W, 200)
    y = np.linspace(0, H, 80)
    X, Y = np.meshgrid(x, y)
    Z = (np.exp(-((X - 2.45) ** 2 / 0.55 + (Y - 0.55) ** 2 / 0.10))
         + 0.55 * np.exp(-((X - 0.95) ** 2 / 0.22 + (Y - 0.30) ** 2 / 0.05)))
    ax.imshow(Z, extent=[0, W, 0, H], origin="lower", cmap="Blues",
              alpha=0.32, zorder=1, aspect="auto")
    rng = np.random.default_rng(11)
    g1 = np.column_stack([rng.uniform(0.15, 3.0, 9),
                          rng.uniform(0.12, 0.9, 9)])
    ax.scatter(*g1.T, s=17, color="#B7C4CC", edgecolors="white",
               linewidths=0.5, zorder=3)
    g2 = np.column_stack([2.45 + rng.normal(0, 0.42, 7),
                          0.55 + rng.normal(0, 0.16, 7)])
    ax.scatter(*g2.T, s=20, color=BLUE, edgecolors="white", linewidths=0.5,
               zorder=4)
    g3 = np.column_stack([2.45 + rng.normal(0, 0.16, 5),
                          0.55 + rng.normal(0, 0.07, 5)])
    ax.scatter(*g3.T, s=24, color=NAVY, edgecolors="white", linewidths=0.6,
               zorder=5)
    ax.scatter([2.45], [0.55], marker="*", s=200, color=ORANGE,
               edgecolors="white", linewidths=0.8, zorder=6)
    save(fig, "flow_learner.png")


# ④ 正式规模复验: 搜索臂(亮)全线高于传统臂(暗), 白虚线 = base 锚点
def icon_final():
    fig, ax = new_ax(NAVY_BG)
    bars = [  # (x, 高, 颜色)
        (0.35, 0.34, "#5A7D9E"), (0.85, 0.29, "#5A7D9E"),
        (1.35, 0.38, "#5A7D9E"),
        (2.00, 0.78, "#6CD9DE"), (2.50, 0.72, "#6CD9DE"),
        (3.00, 0.66, "#6CD9DE"),
    ]
    for x, h, c in bars:
        ax.bar(x, h, width=0.32, color=c, zorder=3)
        ax.plot([x, x], [max(0.02, h - 0.05), min(0.95, h + 0.05)],
                color="white", lw=1.2, alpha=0.75, zorder=4)
    ax.axhline(0.42, color="white", lw=1.0, ls=(0, (4, 3)), alpha=0.55,
               zorder=2)
    save(fig, "flow_final.png")


if __name__ == "__main__":
    icon_cluster()
    icon_sample()
    icon_proxy()
    icon_learner()
    icon_final()
    print("done — 提交: git add scripts/diagnostics/deck_assets/")
