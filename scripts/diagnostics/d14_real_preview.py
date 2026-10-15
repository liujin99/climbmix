#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
#  d14_real_preview.py — p7『簇结构示意』的真实数据版草稿(2026-10-15
#  评审要求:能不能用真实数据 + 两种真实簇合并方法生成;先出草稿过目,
#  再定是否替换 deck 里的手绘示意图)。
#
#  两种划分都是生产函数实跑, 不是手绘:
#    As-is = merge_clusters_by_distance  (论文式距离合并, 固定 K 语义)
#    To-be = balanced_macro_clusters     (容量约束均衡, prod2 起生产路径)
#  点 = 细簇(K_init=1000 kmeans 的簇), 大小 ∝ token, 位置 = 归一化质心的
#  PCA 二维投影, 颜色 = 宏观簇归属 — 左图整条流形一色(塌缩), 右图 15 色
#  均衡分片。
#
#  用法:
#    服务器(真实池; 注意 -o 放 run 目录根层, 别放 deck/ — make_deck
#    会清掉 CHARTS_USED 之外的 png):
#      python3 scripts/diagnostics/d14_real_preview.py \
#        --kmeans '/l00916525/prod/climbmix/cache/embeddings/*/kmeans_K1000.npz' \
#        -o result/prod5_20260929_201108/d14_real_preview.png
#      # 需要真实 token 权重(而非文档数)时加:
#      #   --data-dir /home/ma-user/work/100B_stem_parquet_filtered
#      # (元数据加载约几分钟; 两种权重下结构结论相同)
#    本地(合成细簇, 看样式; 结构机理相同, 数字非真实):
#      python3 scripts/diagnostics/d14_real_preview.py --fixture \
#        -o /tmp/opencode/d14_real_preview_fixture.png
#
#  注: 预览跳过质量剪枝(prune) — 两种方法跑在同一批细簇上, 公平对比;
#  生产路径的剪枝是质量侧通道, 不影响"塌缩 vs 均衡"的结构对比。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from climbmix.core.cluster_merge import (  # noqa: E402
    balanced_macro_clusters,
    merge_clusters_by_distance,
)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, HERE)
from reinforcement_charts import (GRAY, NAVY, RED, TEAL,  # noqa: E402
                                  setup_cjk_font, verify_layout)

K_TARGET = 15


def make_fixture():
    """合成细簇: 一团致密连续流形(整体直径 < tau, 距离合并会链式吞并成
    1 簇 — 复刻真实池"dendrogram 全程 < 0.9"的塌缩条件)+ 14 个远离
    孤岛(与流形及彼此距离 > tau, 永不合并)。质心不归一(与真实 kmeans
    缓存一致), 距离合并跑原始 L2。"""
    rng = np.random.default_rng(7)
    F_cont, F_isl, D = 150, 14, 64
    e0 = np.zeros(D)
    e0[0] = 1.0
    u1, u2 = np.zeros(D), np.zeros(D)
    u1[1], u2[2] = 1.0, 1.0
    t = np.linspace(0.0, 1.0, F_cont)
    cen = np.zeros((F_cont + F_isl, D))
    cen[:F_cont] = (e0[None, :] + 0.55 * t[:, None] * u1[None, :]
                    + 0.15 * np.sin(3 * np.pi * t)[:, None] * u2[None, :]
                    + rng.normal(0, 0.02, (F_cont, D)))
    for i in range(F_isl):
        v = rng.normal(0, 0.28, D)
        v[0] = rng.normal(0, 0.9)
        cen[F_cont + i] = v
    docs = np.concatenate([
        rng.normal(80_000, 5_000, F_cont).astype(np.int64),
        np.full(F_isl, 8_500, dtype=np.int64)])
    labels = np.repeat(np.arange(F_cont + F_isl), docs)
    return labels, cen.astype(np.float32), None, \
        "本图 = 合成细簇(本地样式预览 — 塌缩/均衡机理相同, 数字非真实)"


def load_real(kmeans_glob, data_dir):
    matches = sorted(glob.glob(kmeans_glob))
    if not matches:
        sys.exit(f"✗ 无 kmeans 缓存匹配: {kmeans_glob}")
    path = matches[-1]
    print(f"[1/3] 读取 kmeans 缓存: {path}", flush=True)
    z = np.load(path)
    labels, cen = z["labels"], z["centroids"].astype(np.float32)
    tok = None
    src = f"细簇 = {os.path.basename(path)}(K_init={len(cen)});权重 = 文档数"
    if data_dir:
        print(f"[2/3] 加载池元数据(token 权重, 分钟级): {data_dir}", flush=True)
        from climbmix.data.metadata_manager import ShardMetadataManager
        mm = ShardMetadataManager(data_dir, cache_dir=data_dir)
        tok = mm.estimate_token_counts()
        src = (f"细簇 = {os.path.basename(path)}(K_init={len(cen)});"
               f"权重 = token")
    print(f"      labels={len(labels):,} docs, F={len(cen)}", flush=True)
    return labels, cen, tok, src


def main():
    ap = argparse.ArgumentParser(
        description="p7『簇结构示意』真实数据版草稿 — 两种划分 = 生产函数实跑"
                    "(距离合并 vs 容量约束均衡),点 = 细簇,颜色 = 宏观归属")
    ap.add_argument("--kmeans", default="",
                    help="kmeans_K*.npz 的 glob(服务器真实池)")
    ap.add_argument("--data-dir", default="",
                    help="池 parquet 目录 — 提供 token 权重(可选)")
    ap.add_argument("--fixture", action="store_true",
                    help="本地合成细簇(看样式)")
    ap.add_argument("--K", type=int, default=K_TARGET)
    ap.add_argument("--slack", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=0.9,
                    help="距离合并的 merge_distance(生产默认 0.9)")
    ap.add_argument("-o", "--out", default="d14_real_preview.png")
    args = ap.parse_args()

    if not args.fixture and not args.kmeans:
        sys.exit("✗ 需要 --kmeans <glob> 或 --fixture 之一")
    if args.fixture:
        labels, cen, tok, src_note = make_fixture()
    else:
        labels, cen, tok, src_note = load_real(args.kmeans, args.data_dir)

    # 细簇量(token 或文档数) — 点大小与份额统计的权重
    w = tok if tok is not None else np.ones(len(labels), dtype=np.int64)
    fine_tok = np.bincount(np.asarray(labels), weights=w,
                           minlength=len(cen)).astype(np.float64)

    print(f"[3/3] 跑两种划分(生产函数, K={args.K})…", flush=True)
    _, _, dist_map = merge_clusters_by_distance(
        labels, cen, merge_distance=args.tau, target_K=args.K)
    _, _, bal_map = balanced_macro_clusters(
        labels, cen, token_counts=tok, K=args.K, slack=args.slack)

    dm = np.array([dist_map[i] for i in range(len(cen))])
    bm = np.array([bal_map[i] for i in range(len(cen))])
    tot = fine_tok.sum()
    d_sh = np.array([fine_tok[dm == k].sum() / tot for k in np.unique(dm)])
    b_sh = np.array([fine_tok[bm == k].sum() / tot for k in np.unique(bm)])
    print(f"    距离合并: K={len(np.unique(dm))}, 最大簇 {d_sh.max():.1%}")
    print(f"    均衡划分: K={len(np.unique(bm))}, 最大簇 {b_sh.max():.1%}"
          f"(上限 {(1 + args.slack) / args.K:.1%})")

    # ── 出图: 归一化质心 PCA 二维, 点 = 细簇, 颜色 = 宏观归属 ──
    setup_cjk_font()
    nrm = cen / np.clip(np.linalg.norm(cen, axis=1, keepdims=True), 1e-9, None)
    X = nrm - nrm.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = X @ Vt[:2].T
    sizes = 4 + 70 * fine_tok / fine_tok.max()
    fig = plt.figure(figsize=(13, 4.05))

    # 左 As-is: 巨簇一色, 其余灰 — 塌缩直接可见
    ax = fig.add_axes([0.035, 0.13, 0.42, 0.70])
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7FBFB")
    giant = int(np.unique(dm)[np.argmax(d_sh)])
    cols = np.where(dm == giant, RED, GRAY)
    ax.scatter(P[:, 0], P[:, 1], s=sizes, c=cols, alpha=0.55, lw=0)
    ax.text(0.5, 0.985, f"As-is:论文式距离合并 — 最大簇 {d_sh.max():.1%}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=RED)
    ax.text(0.5, 0.03, f"红 = 巨簇({len(np.flatnonzero(dm == giant))} 个细簇"
            f" 并成 1 簇)· 灰 = 孤岛 — 可用旋钮 ≈ 1",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=9, color=RED)

    # 右 To-be: 15 色均衡分片
    ax = fig.add_axes([0.545, 0.13, 0.42, 0.70])
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7FBFB")
    cmap = plt.get_cmap("tab20")
    ax.scatter(P[:, 0], P[:, 1], s=sizes,
               c=[cmap(int(b) % 20) for b in bm], alpha=0.65, lw=0)
    ax.text(0.5, 0.985, f"To-be:容量约束均衡 — 最大簇 {b_sh.max():.1%}",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=11.5, fontweight="bold", color=NAVY)
    ax.text(0.5, 0.03, f"同一批细簇切成大小相近的 {len(np.unique(bm))} 簇"
            f"(上限 {(1 + args.slack) / args.K:.1%})— 每簇一个旋钮",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=9, color=NAVY)

    fig.text(0.5, 0.965, "同一个池,两种簇划分 — 细簇与合并均为真实运行结果",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, "点 = 细簇(大小 ∝ token),位置 = 归一化质心的 PCA "
             "二维投影,颜色 = 宏观簇归属;" + src_note,
             ha="center", va="center", fontsize=9, color=GRAY)
    ok = verify_layout(fig, "d14_real")
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"  {'✓' if ok else '✗'} {args.out}"
          + ("" if ok else "  (标签自检未过 — 贴回给维护者)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
