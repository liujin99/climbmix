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
#  t-SNE 二维嵌入(孤岛外推到流形外, 图内脚注声明), 分组 = 宏观簇边界
#  — 左图整条流形一圈(塌缩)+ 孤岛蓝圈, 右图 15 个均衡分片环(切片非
#  天然聚类)。中缝 = 改进动作箭头; 各栏右上角嵌旋钮配比小饼(同一运行
#  的真实 token 份额, 辅助; 左 1 巨 + 14 孤岛 = 1 个旋钮, 右 15 均分 =
#  15 个旋钮, 右饼分片与右图环同色同序)。
#  伴随输出 d14_merge_diag.png = 合并诊断曲线(merge_profile.json 数据,
#  无投影无调整 — "最近邻距离全程未够到 τ"的硬证据, deck 附录用)。
#
#  用法:
#    服务器(真实池; 两图 = deck 副产物, -o 直接写进汇报包目录 —
#    make_deck 已把两图纳入 CHARTS_USED 不会清掉, 取图 deck 目录优先、
#    run 根层副本仅兜底旧布局):
#      python3 scripts/diagnostics/d14_real_preview.py \
#        --kmeans '/l00916525/prod/climbmix/cache/embeddings/*/kmeans_K1000.npz' \
#        -o result/prod5_20260929_201108/deck/d14_real_preview.png
#      # (伴随输出 d14_merge_diag.png 自动落在同目录)
#      # 需要真实 token 权重(而非文档数)时加:
#      #   --data-dir /home/ma-user/work/100B_stem_parquet_filtered
#      # (元数据加载约几分钟; 两种权重下结构结论相同)
#    本地(合成细簇, 看样式; 结构机理相同, 数字非真实):
#      python3 scripts/diagnostics/d14_real_preview.py --fixture \
#        -o /tmp/opencode/rc2/deck/d14_real_preview.png
#
#  注: 预览跳过质量剪枝(prune) — 两种方法跑在同一批细簇上, 公平对比;
#  生产路径的剪枝是质量侧通道, 不影响"塌缩 vs 均衡"的结构对比。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob
import json
import os
import sys
import tempfile

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
from matplotlib.patches import (Ellipse,  # noqa: E402
                                FancyArrowPatch, Polygon as MplPolygon)
from matplotlib.path import Path as MplPath  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

sys.path.insert(0, HERE)
from reinforcement_charts import (BLUE, GRAY, NAVY, RED, TEAL,  # noqa: E402
                                  setup_cjk_font, verify_layout)

K_TARGET = 15


def blob_ellipse(ax, pts, edge, data_span, lw=1.1, ls="--", fc=None,
                 n_sig=2.2, z=4):
    """宏观簇边界 = 2.2σ 协方差椭圆(评审:"像论文那张图,能看出不同的
    聚类簇,最好圈起来;纯颜色区分太眼花")。主轴方向 = 协方差特征向量;
    少于 3 点退化为小圆。fc = (r,g,b,a) 元组时带浅色底。"""
    n = len(pts)
    if n == 0:
        return
    c = pts.mean(0)
    if n < 3:
        r = 0.02 * data_span if n == 1 else max(
            0.02 * data_span, 1.4 * float(np.hypot(*(pts[1] - pts[0]))) / 2)
        ax.add_patch(Ellipse(c, 2 * r, 2 * r, facecolor=fc or "none",
                             edgecolor=edge, lw=lw, ls=ls, zorder=z))
        return
    cov = np.cov((pts - c).T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, (0.01 * data_span) ** 2, None)   # 退化方向兜底
    ang = float(np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1])))
    ax.add_patch(Ellipse(
        c, 2 * n_sig * float(np.sqrt(vals[1])),
        2 * n_sig * float(np.sqrt(vals[0])), angle=ang,
        facecolor=fc or "none", edgecolor=edge, lw=lw, ls=ls, zorder=z))


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
        "本图 = 合成细簇(本地样式预览,数字非真实)"


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


def embed_2d(nrm):
    """细簇质心 → 二维。优先 t-SNE:只保邻域结构,连续流形会摊成一条
    "蛇"、容量切片成一串珠子;PCA 会把高维上分开的切片压成一团
    (评审反馈"左图一个大圈、右图一堆重叠小圈"的根源 — 真实池在
    PCA 前两维上是团圆的连续流形)。sklearn 缺席时退回 PCA。"""
    try:
        from sklearn.manifold import TSNE
        emb = TSNE(n_components=2, init="pca", perplexity=30.0,
                   random_state=42).fit_transform(nrm)
        return emb, ("位置 = t-SNE 二维(保邻域,距离无尺度;"
                     "孤岛沿方向外推到流形外)")
    except ImportError:
        X = nrm - nrm.mean(0)
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        return X @ Vt[:2].T, ("位置 = 归一化质心的 PCA 二维投影(降级);"
                             "孤岛沿方向外推到流形外")


def separate_islands(P, man_mask, span):
    """孤岛外推(画圈的前提):t-SNE 只保邻域、不保全局距离 — 高维中
    距流形 > τ 的孤岛常被投影进主簇空隙或贴在主簇边界(历轮评审:"孤岛
    在大簇里面"/"单点贴着大簇")。把落在流形凸包内、或距凸包边界不足
    gap 的孤岛,沿(质心→孤岛)方向推到凸包外 gap 处;挤在同一空隙的
    孤岛互相推开。图内脚注声明此处理。返回 (P 修正副本, 外推个数)。"""
    man = P[man_mask]
    try:
        hull = ConvexHull(man)
    except Exception:
        return P, 0
    verts = man[hull.vertices]
    poly = MplPath(verts)
    c = man.mean(0)
    gap = 0.07 * span

    def _place_outside(p):
        d = p - c
        r0 = float(np.hypot(*d))
        u = d / r0 if r0 > 1e-9 else np.array([1.0, 0.0])
        lo, hi = 1e-6, 4.0
        while poly.contains_point(c + u * (r0 * hi)) and hi < 64:
            hi *= 2.0
        for _ in range(26):
            mid = 0.5 * (lo + hi)
            if poly.contains_point(c + u * (r0 * mid)):
                lo = mid
            else:
                hi = mid
        return c + u * (r0 * hi + gap)

    out = P.copy()
    idx = np.flatnonzero(~man_mask)
    n = 0
    for i in idx:
        p = out[i]
        near = float(np.min(np.linalg.norm(verts - p, axis=1))) < gap
        if poly.contains_point(p) or near:
            out[i] = _place_outside(p)
            n += 1
    # 碰撞松弛: 挤在一起的孤岛沿连线分开, 推回凸包内则再次外推
    if len(idx) > 1:
        sep = 0.055 * span
        for _ in range(60):
            Q = out[idx]
            dd = np.linalg.norm(Q[:, None] - Q[None], axis=2)
            np.fill_diagonal(dd, np.inf)
            i, j = np.unravel_index(np.argmin(dd), dd.shape)
            if dd[i, j] >= sep:
                break
            a, b = int(idx[i]), int(idx[j])
            v = out[b] - out[a]
            L = float(np.hypot(*v))
            v = v / L if L > 1e-9 else np.array([1.0, 0.0])
            push = 0.5 * (sep - dd[i, j]) + 0.01 * span
            out[a] = out[a] - v * push
            out[b] = out[b] + v * push
            for q in (a, b):
                if poly.contains_point(out[q]):
                    out[q] = _place_outside(out[q])
    return out, n


def chart_merge_diag(prof, tau, out, src_note):
    """附录硬证据:距离合并全程的最近邻距离曲线(数据 = merge_profile
    .json,合并器自己的运行记录 — 无投影、无调整)。τ 线悬在曲线上方 =
    合并从未被阈值叫停;一路并到簇数下限才停 → 塌缩是结构性的。"""
    K = np.array([p["K"] for p in prof["profile"]], dtype=float)
    d = np.array([p["closest_pair_dist"] for p in prof["profile"]])
    kf = int(prof["K_final"])
    fig = plt.figure(figsize=(13, 3.9))
    ax = fig.add_axes([0.055, 0.175, 0.895, 0.60])
    ax.plot(K, d, color=NAVY, lw=1.9, zorder=3)
    ax.axhline(tau, color=RED, ls="--", lw=1.3, zorder=2)
    ax.axhline(0.7, color=GRAY, ls=":", lw=1.0, zorder=2)
    ax.axvline(kf, color=NAVY, lw=1.0, alpha=0.55, zorder=2)
    ax.set_xlim(K.max() * 1.03, K.min() * 0.90)   # 左多右少 = 合并推进
    ax.set_ylim(0, tau * 1.12)
    ax.set_xlabel("簇数 K(左 → 右 = 合并推进)", fontsize=10.5, color=NAVY)
    ax.set_ylabel("最近一对簇的距离", fontsize=10.5, color=NAVY)
    ax.tick_params(labelsize=9.5, colors=NAVY)
    ax.text(0.015, tau - 0.05, f"τ = {tau:g}(合并阈值)— 全程未触发",
            transform=ax.get_yaxis_transform(), ha="left", va="top",
            fontsize=10, color=RED)
    ax.text(0.985, 0.06, f"K = {kf}:撞簇数下限才停(stop = "
            f"{prof.get('stop_reason', 'floor')})", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color=NAVY)
    nk = prof.get("natural_K", {}).get("0.7")
    if isinstance(nk, int):
        ax.axvline(nk, color=GRAY, ls=":", lw=1.0, zorder=2)
        ax.text(nk, tau * 1.06, f" natural_K(0.7) = {nk}",
                ha="left", va="top", fontsize=9.5, color=GRAY)
    fig.text(0.5, 0.955, "距离合并全程:最近邻距离从未够到 τ — 塌缩是结构性的",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.035, f"数据 = merge_profile.json(合并器运行记录,无投影、"
             f"无调整);{src_note}", ha="center", va="center", fontsize=9,
             color=GRAY)
    ok = verify_layout(fig, "d14_diag")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {'✓' if ok else '✗'} {out}"
          + ("" if ok else "  (标签自检未过 — 贴回给维护者)"))
    return ok


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
    fd, prof_path = tempfile.mkstemp(suffix=".json", prefix="d14prof_")
    os.close(fd)
    _, _, dist_map = merge_clusters_by_distance(
        labels, cen, merge_distance=args.tau, target_K=args.K,
        profile_path=prof_path)
    with open(prof_path) as f:
        prof = json.load(f)
    os.unlink(prof_path)
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

    # ── 出图: 二维嵌入(t-SNE 优先; 孤岛外推到流形外), 点 = 细簇珠(白描
    #    边互相分开 — 否则 986 个点糊成一片, 读者看不出"框里有许多
    #    细簇"), 分组靠圈/环; -o 可指向尚不存在的 deck/ 目录 ──
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    setup_cjk_font()
    nrm = cen / np.clip(np.linalg.norm(cen, axis=1, keepdims=True), 1e-9, None)
    P, embed_note = embed_2d(nrm)
    giant = int(np.unique(dm)[np.argmax(d_sh)])
    span = float(max(np.ptp(P[:, 0]), np.ptp(P[:, 1]), 1e-9))
    P, n_push = separate_islands(P, dm == giant, span)
    if n_push:
        print(f"    孤岛外推: {n_push} 个细簇移到流形外(t-SNE 不保全局"
              f"距离; 高维中孤岛距流形 > τ)", flush=True)
    sizes = 6 + 24 * np.sqrt(fine_tok / fine_tok.max())
    # 统一留边: 圈/环不会顶到面板边缘, 更不会压到面板外的文字带
    xlim = (float(P[:, 0].min() - 0.05 * span), float(P[:, 0].max() + 0.05 * span))
    ylim = (float(P[:, 1].min() - 0.06 * span), float(P[:, 1].max() + 0.06 * span))
    # 13×3.60(宽高比 ≈ 3.6)= 贴满 p7 图带(约 12.1×3.35in)的比例 —
    # 整图放置宽度 10.76 → ~12.1in; 旋钮饼嵌在各栏右上角(评审: 中缝
    # 两饼夹箭头"有点丑", 改角落小图 + 缩小 — 毕竟是辅助说明)
    fig = plt.figure(figsize=(13, 3.60))
    isl_n = len(np.unique(dm)) - 1
    isl_sh = 1.0 - float(d_sh.max())

    def _panel(x0):
        ax = fig.add_axes([x0, 0.155, 0.42, 0.645])
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F7FBFB")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.scatter(P[:, 0], P[:, 1], s=sizes, color=GRAY, alpha=0.55,
                   lw=0.4, edgecolors="#F7FBFB", zorder=3)
        return ax

    # 左 As-is: 巨簇边界 = 流形凸包; 孤岛画蓝圈(外推到流形外 — 左图要
    #   显示"距离合并也产出 15 簇: 1 巨 + 14 孤岛", 但 14 个孤岛合计仅
    #   1.1%, 旋钮救不回); 点 = 细簇珠
    ax = _panel(0.035)
    man = P[dm == giant]
    ax.add_patch(MplPolygon(man[ConvexHull(man).vertices], closed=True,
                            facecolor=(0.99, 0.93, 0.92, 0.55),
                            edgecolor=RED, lw=1.6, zorder=2))
    for k in np.unique(dm):
        if k != giant:
            blob_ellipse(ax, P[dm == k], BLUE, span, lw=1.2,
                         fc=(0.92, 0.95, 1.0, 0.50), z=4)
    fig.text(0.245, 0.862, f"As-is:论文式距离合并 — 最大簇 {d_sh.max():.1%}",
             ha="center", va="center", fontsize=11.5, fontweight="bold",
             color=RED)
    fig.text(0.245, 0.075, f"点 = 细簇:红框 = {len(np.flatnonzero(dm == giant))}"
             f" 个并成 1 簇({d_sh.max():.1%}),蓝圈 = 孤岛 {isl_n} 簇"
             f"({isl_sh:.1%})— 旋钮只剩 1 个",
             ha="center", va="center", fontsize=9, color=RED)

    # 右 To-be: 同一批细簇切成 15 个大小相近的簇 — 每簇一圈(细虚线环,
    #   不填色, 交叉相接时仍可辨) + 簇中心菱形锚点; 环必然沿流形相接:
    #   切片是连续流形上的容量切分, 不是天然主题团(见发现③)。孤岛不画
    #   圈(在均衡划分里是分片成员, 不是簇)— 归属由右脚注一句说明(评审:
    #   连接线方案否决 — 14 条细线解释 1.1% 的点, 喧宾夺主)
    ax = _panel(0.545)
    cmap = plt.get_cmap("tab20")
    cents = []
    for i, k in enumerate(np.unique(bm)):
        col = cmap(i % 20)
        pts = P[bm == k]
        if len(pts) > 25:  # 稳健环: 剔除离切片中位最远 10%(离群细簇)
            dd = np.hypot(*(pts - np.median(pts, axis=0)).T)
            pts = pts[dd <= np.quantile(dd, 0.90)]
        blob_ellipse(ax, pts, col, span, lw=1.2, n_sig=1.9)
        cents.append(pts.mean(0))
    cents = np.array(cents)
    ax.scatter(cents[:, 0], cents[:, 1], marker="D", s=46,
               c=[cmap(i % 20) for i in range(len(cents))],
               edgecolors="white", lw=0.7, zorder=6)
    fig.text(0.755, 0.862, f"To-be:容量约束均衡 — 最大簇 {b_sh.max():.1%}",
             ha="center", va="center", fontsize=11.5, fontweight="bold",
             color=NAVY)
    fig.text(0.755, 0.075, f"同批细簇切成 {len(np.unique(bm))} 簇(切片 ≠ 天然"
             f"聚类);圈外灰点 = 孤岛,已并入所属分片;每簇一个旋钮",
             ha="center", va="center", fontsize=9, color=NAVY)

    # 旋钮饼 = 各栏右上角的辅助小图(评审两轮: 先"把两个饼状旋钮图放进
    #   来, 可以缩小一点", 再"中缝夹箭头的排法有点丑, 放到聚类簇图的
    #   右上/右下角, 整体再小一点")— 饼 = 同一次运行的真实 token 份额,
    #   环状样式沿用旧手绘图(reinforcement_d14_cluster_balance); 左饼
    #   红巨簇 + 蓝孤岛(与左图圈色一致, 白字标巨簇份额), 右饼 15 片与
    #   右图环同色同序; 直径 ≈ 1.07in(放置尺寸), 底衬圆 = 面板底色,
    #   隔开底下的散点
    def _pie(x0, vals, colors, center, ccol, wlab=None):
        ax = fig.add_axes([x0, 0.4686, 0.0885, 0.3194])
        ax.add_patch(plt.Circle((0, 0), 1.14, facecolor="#F7FBFB",
                                edgecolor="#C4D2D2", lw=0.9, zorder=0))
        wedges, _ = ax.pie(vals, colors=colors, startangle=90,
                           counterclock=False,
                           wedgeprops=dict(width=0.42, edgecolor="white"))
        ax.text(0, 0, center, ha="center", va="center", fontsize=9,
                fontweight="bold", color=ccol)
        if wlab is not None:
            w = wedges[0]
            th = float(np.radians((w.theta1 + w.theta2) / 2))
            ax.text(0.78 * np.cos(th), 0.78 * np.sin(th), wlab,
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", color="white")
        ax.set_axis_off()
        return ax

    gi = int(np.argmax(d_sh))
    _pie(0.3545, [d_sh[gi]] + [s for i, s in enumerate(d_sh) if i != gi],
         [RED] + [BLUE] * (len(d_sh) - 1),
         "1 个\n旋钮", RED, wlab=f"{d_sh.max():.1%}")
    _pie(0.8645, b_sh, [cmap(i % 20) for i in range(len(b_sh))],
         f"{len(b_sh)} 个\n旋钮", NAVY)

    # 中缝: 改进动作箭头(评审: "as is 和 to be 中间加个箭头, 显示前后
    #   关系")— 饼移去角落后中缝只剩箭头, 加长加粗
    fig.add_artist(FancyArrowPatch(
        (0.458, 0.48), (0.542, 0.48), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=30, lw=2.4, color=TEAL, zorder=5))
    fig.text(0.5, 0.565, "容量重切", ha="center", va="center",
             fontsize=10, fontweight="bold", color=TEAL)

    fig.text(0.5, 0.952, "同一个池,两种簇划分 — 细簇与合并均为真实运行结果",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.02, "点 = 细簇(大小 ∝ token),圈 = 宏观簇边界"
             "(◆ = 簇中心),饼 = 簇配比(token 份额);" + embed_note + ";"
             + src_note,
             ha="center", va="center", fontsize=9, color=GRAY)
    ok = verify_layout(fig, "d14_real")
    fig.savefig(args.out, dpi=150)
    plt.close(fig)
    print(f"  {'✓' if ok else '✗'} {args.out}"
          + ("" if ok else "  (标签自检未过 — 贴回给维护者)"))
    diag_out = os.path.join(os.path.dirname(os.path.abspath(args.out)),
                            "d14_merge_diag.png")
    ok2 = chart_merge_diag(prof, args.tau, diag_out, src_note)
    return 0 if (ok and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
