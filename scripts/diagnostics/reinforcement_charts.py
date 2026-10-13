#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
#  reinforcement_charts.py — 『三处加固』举证图 (deck v10, 2026-10-13)
#
#  用法:
#    python3 scripts/diagnostics/reinforcement_charts.py result/prod5_20260929_201108
#    # 服务器若无中文字体: --font /path/to/cjk.ttf, 或安装 fonts-noto-cjk
#
#  输出 (RUN_DIR 根, 与 report_charts.py 的 PNG 同层; make_deck.py 自动引用):
#    reinforcement_d14_cluster_balance.png  ① 左 as-is(论文距离合并, 为什么
#                                           翻车)vs 右 to-be(强制均衡, 改进
#                                           后), 双环形图 + 两侧实验佐证
#    reinforcement_d8_scoring_snr.png       ② 左=原因(逐基准 真信号 vs 抽样
#                                           噪声), 右=结果(gsm8k: prod3 误
#                                           加权 vs prod5 修复后), 下=做法
#    reinforcement_d19_extrapolation.png    ③ 外推问题示意图(非真实数据):
#                                           拟合模型只在实测区域内可信,
#                                           『外推最优点』= 拟合噪声
#
#  v10 (用户反馈重设计): ① d14 改 as-is/to-be 左右对照并各挂实验佐证;
#  ② d8 按『原因 → 结果 → 做法』重组, 右联改 gsm8k 修复前后对比
#  (gsm8k 跨协议逐位可比, D17); ③ d19 赛跑图弃用(不易读), 改外推概念
#  示意图, prod4 数据表格移入 deck 本体页; ④ 图内标签中文 (仅 arm 名/
#  基准名保留原文); ⑤ 无指向式箭头; ⑥ 保存前自检文字重叠/越界 (本文件
#  的 verify_layout 也供 report_charts.py 复用)。
#
#  数据 = 实验记录已核实数字 (逐数注明出处; 跨轮历史数字无法从 prod5
#  run 目录读取, 为常量; 15 簇真实份额读 run 目录):
#    D14 塌缩诊断   paper_deviations.md D14 (prod1 事后 A6/A7):
#                   巨簇 99.05% / 13 孤岛 0.95%; K=14/21/24/32 → 巨簇
#                   ≥97.5%; cap ≤ 8.8%; prod1 -0.0036 (0.1623 vs 0.1659);
#                   prod2 +0.0102 (0.1752 vs 0.1650); prod5 C10 5.5% → 80%
#    D8  信号/噪声  experiment_prod2_k15bal.md §4.2 (30 候选实测):
#                   sd_cfg / SE_binom / SNR 逐基准表 (基准名以该表为准, gpqa)
#        prod3 事故 TODO.md 2026-09-15 P0: gsm8k 赢家 0.052 vs 对照 0.119
#                   (-56%, z=-6.2); math 0.018→0.004; 修复 = SNR 定权 +
#                   实测噪声地板 + w_floor=0.5
#        prod5 结果 experiment_prod5.md §3b: gsm8k climb 0.3010 /
#                   uniform 0.1054 / base 0.0288 (2.9x / 10.4x)
#    D19 prod4 对决 experiment_prod4.md §3d/§4d — 数字在 make_deck.py
#                   的表格页; 本脚本只出概念示意图
#    15 簇份额      RUN_DIR/macro_info.json (⑬r) 或 cluster_info_cache.json
#                   (legacy), schema 同 recipe_report.py (label/num_tokens);
#                   全缺 -> 等分示意 (图内注明)
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("✗ 缺 matplotlib — pip install matplotlib (PEP 668 环境加 "
             "--break-system-packages), 装好后重跑本脚本")

TEAL = "#55A868"      # 真信号 / 变好 / 修复后
RED = "#C44E52"       # 噪声 / 变差 / 外推终选
BLUE = "#4C72B0"      # 均衡簇 / 预测器拟合
NAVY = "#1F4E79"      # 实测点
ORANGE = "#DD8452"    # C10 / 赢家 (report_charts CLIMB_COLOR)
GRAY = "#9A9A9A"
DGRAY = "#555555"

CJK_KEYS = ("cjk", "zenhei", "wqy", "hei", "song", "kai", "noto sans sc",
            "source han", "pingfang", "droid sans fallback", "arial unicode",
            "msyh", "simhei", "simsun")

# ── 数据 (出处见文件头) ────────────────────────────────────────────────
D14_GIANT, D14_ISLANDS = 99.05, 0.95
D8_SNR = [  # bench, 真信号 sd_cfg, 抽样噪声 SE_binom, SNR (prod2 §4.2)
    ("arc_easy", 0.0225, 0.0127, 1.77),
    ("mmlu_stem", 0.0099, 0.0100, 1.00),
    ("arc_challenge", 0.0157, 0.0191, 0.82),
    ("gpqa", 0.0283, 0.0411, 0.69),
    ("gsm8k", 0.0048, 0.0163, 0.29),
    ("math", 0.0024, 0.0259, 0.09),
]
D8_GSM8K = [  # (标签, 赢家分, 对照 uniform 分, 赢家颜色) — gsm8k 跨协议可比
    ("prod3 · 误加权", 0.052, 0.119, RED),
    ("prod5 · 修复后", 0.3010, 0.1054, TEAL),
]
D8_HOW = ("做法:各基准按信噪比(SNR)定话语权 w;噪声地板按任务实测比例计算"
          "(prod3 曾把 CoT 按 4 选 1 硬算,高估约 20 倍);w_floor = 0.5,NLL 只补位")
D8_AFTER = ("修复后(prod5):赢家 gsm8k 0.301 = uniform 的 2.9×、base 的 10.4×;"
            "三臂全胜 uniform(+0.036 ~ +0.043)")


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


# ── 中文字体 (report_charts.py 复用) ─────────────────────────────────
def _apply_font(name):
    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def setup_cjk_font(font_path=None):
    """返回可用中文字体名并写入 rcParams; 找不到返回 None。"""
    if font_path:
        if not os.path.isfile(font_path):
            sys.exit(f"✗ --font 指定的文件不存在: {font_path}")
        try:
            fm.fontManager.addfont(font_path)
            name = fm.FontProperties(fname=font_path).get_name()
        except Exception as e:
            sys.exit(f"✗ 注册字体失败 {font_path}: {e}")
        _apply_font(name)
        return name
    names = [f.name for f in fm.fontManager.ttflist
             if any(k in f.name.lower() or k in f.fname.lower()
                    for k in CJK_KEYS)]
    if not names:
        pats = ["/usr/share/fonts/**/*.tt[fc]", "/usr/share/fonts/**/*.otf",
                "/usr/local/share/fonts/**/*.tt[fc]",
                "/usr/local/share/fonts/**/*.otf",
                os.path.expanduser("~/.fonts/**/*.tt[fc]"),
                os.path.expanduser("~/.fonts/**/*.otf"),
                os.path.expanduser("~/.local/share/fonts/**/*.tt[fc]"),
                os.path.expanduser("~/.local/share/fonts/**/*.otf"),
                os.path.join(sys.prefix, "fonts/**/*.tt[fc]")]
        seen = set()
        for pat in pats:
            for p in sorted(glob.glob(pat, recursive=True)):
                if p in seen or not any(k in os.path.basename(p).lower()
                                        for k in CJK_KEYS):
                    continue
                seen.add(p)
                try:
                    fm.fontManager.addfont(p)
                    names.append(fm.FontProperties(fname=p).get_name())
                except Exception:
                    pass
    if not names:
        return None
    _apply_font(names[0])
    return names[0]


# ── 自检: 文字两两不重叠、不越出画布 (report_charts.py 复用) ─────────
def verify_layout(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []

    def _add(t):
        s = t.get_text().strip()
        if not s:
            return
        bb = t.get_window_extent(r)
        # 退化 bbox (≈1px): colorbar 等特殊轴经 get_major_ticks 会拿到
        # 未绘制的新 tick, 位置未初始化 — 无法验证, 跳过
        if bb.x1 - bb.x0 < 2 or bb.y1 - bb.y0 < 2:
            return
        items.append((s, bb))

    for t in fig.texts:
        _add(t)
    for ax in fig.axes:
        _add(ax.title)
        _add(ax.xaxis.label)
        _add(ax.yaxis.label)
        for t in ax.texts:
            _add(t)
        # 刻度标签只收视窗内的: matplotlib 会为视窗外位置创建
        # visible=True 但不绘制的『幻影刻度』(如 ylim (0,0.35) 时的 0.40),
        # 一并检查会误报重叠/越界。locs 与标签逐位对齐; 对不上时宁可不
        # 过滤 (degenerate 过滤仍兜底)
        for axis, labels, (lo, hi) in (
                (ax.xaxis, ax.get_xticklabels(), ax.get_xlim()),
                (ax.yaxis, ax.get_yticklabels(), ax.get_ylim())):
            locs = axis.get_majorticklocs()
            if len(locs) != len(labels):
                locs = [None] * len(labels)
            for loc, t in zip(locs, labels):
                if loc is not None and (loc < lo or loc > hi):
                    continue
                _add(t)
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                _add(t)
    w, h = fig.canvas.get_width_height()
    bad = []
    for i, (s1, b1) in enumerate(items):
        if b1.x0 < -2 or b1.y0 < -2 or b1.x1 > w + 2 or b1.y1 > h + 2:
            bad.append(f"越界:『{s1[:14]}』")
        for s2, b2 in items[i + 1:]:
            if b1.overlaps(b2):
                bad.append(f"重叠:『{s1[:10]}』×『{s2[:10]}』")
    if bad:
        print(f"  ✗ {name} 自检失败 — " + "; ".join(bad[:6]))
        return False
    print(f"  ✓ {name}: 标签无重叠、无越界")
    return True


# ── ① D14: as-is vs to-be 双环形图 + 实验佐证 ────────────────────────
def chart_d14(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    real = shares is not None
    if not real:
        labels = [f"C{i}" for i in range(15)]
        shares = np.full(15, 100.0 / 15)
    hi = labels.index("C10") if "C10" in labels else 9
    fig = plt.figure(figsize=(13, 5.0))

    # 左 as-is: 论文距离合并 — 1 巨簇 + 13 孤岛 (为什么翻车)
    ax = fig.add_axes([0.045, 0.37, 0.42, 0.46])
    wedges, _ = ax.pie([D14_GIANT, D14_ISLANDS], colors=[RED, GRAY],
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "1 个\n可用旋钮", ha="center", va="center",
            fontsize=15, fontweight="bold", color=RED)
    w = wedges[0]
    th = math.radians((w.theta1 + w.theta2) / 2)
    ax.text(0.79 * math.cos(th), 0.79 * math.sin(th), "巨簇\n99.05%",
            ha="center", va="center", fontsize=10.5, fontweight="bold",
            color="white")
    ax.set_title("As-is:论文式距离合并(为什么翻车)", fontsize=12,
                 fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # 右 to-be: 均衡划分 — 15 簇真实份额, C10 高亮 (改进后)
    ax = fig.add_axes([0.545, 0.37, 0.42, 0.46])
    colors = [ORANGE if i == hi else BLUE for i in range(len(shares))]
    explode = [0.05 if i == hi else 0.0 for i in range(len(shares))]
    wedges, _ = ax.pie(shares, colors=colors, explode=explode,
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    for i, w in enumerate(wedges):
        if shares[i] < 3.0:
            continue
        th = math.radians((w.theta1 + w.theta2) / 2)
        rr = 0.79 + (0.05 if i == hi else 0.0)
        ax.text(rr * math.cos(th), rr * math.sin(th), labels[i],
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white")
    ax.text(0, 0, "15 个\n可用旋钮", ha="center", va="center",
            fontsize=15, fontweight="bold", color=NAVY)
    ax.set_title("To-be:强制簇大小均衡(改进后)", fontsize=12,
                 fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.5, 0.96, "同一个数据池,两种簇划分:各簇分到的 token 份额",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.045, 0.30,
             "实验佐证(prod1):\n"
             "· K=14/21/24/32:巨簇都 ≥97.5%,怎么调都无解\n"
             "· 其余 13 个孤岛簇合计 0.95%(图中灰缝)\n"
             "· 35 个候选配方全被钉死 → prod1 输 uniform 0.0036",
             ha="left", va="top", fontsize=9.5, color=DGRAY, linespacing=1.55)
    if real:
        right_note = ("实验佐证(prod2 / prod5):\n"
                      "· 单簇 ≤ 8.8%,15 个旋钮全部可用\n"
                      "· prod2 首次转正:+0.0102(0.1752 vs 0.1650)\n"
                      "· prod5 赢家把 C10 押到 80%(池占比 5.5% 的 14.6 倍)")
    else:
        right_note = ("实验佐证(prod2):\n"
                      "· 单簇 ≤ 8.8%,15 个旋钮全部可用\n"
                      "· prod2 首次转正:+0.0102(0.1752 vs 0.1650)\n"
                      "· 未找到 macro_info.json — 图为等分示意")
    fig.text(0.545, 0.30, right_note, ha="left", va="top", fontsize=9.5,
             color=DGRAY, linespacing=1.55)
    ok = verify_layout(fig, "d14")
    path = os.path.join(out_dir, "reinforcement_d14_cluster_balance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}" + ("  [真实份额: macro_info.json]"
                           if real else "  [示意: 无 macro_info.json]"))
    return ok


# ── ② D8: 原因(左)· 结果(右)· 做法(下) ─────────────────────────────
def chart_d8(out_dir):
    fig = plt.figure(figsize=(13, 5.0))

    # 左 原因: 逐基准 真信号 vs 抽样噪声 (SNR 降序, 上=响)
    ax = fig.add_axes([0.075, 0.40, 0.36, 0.43])
    n = len(D8_SNR)
    for i, (bench, sig, noi, snr) in enumerate(D8_SNR):
        y = n - 1 - i
        ax.barh(y + 0.20, sig, height=0.34, color=TEAL, zorder=3)
        ax.barh(y - 0.20, noi, height=0.34, color=RED, zorder=3)
        ax.text(sig + 0.0006, y + 0.20, f"{sig:.4f}", va="center",
                fontsize=8, color=DGRAY, zorder=4)
        ax.text(noi + 0.0006, y - 0.20, f"{noi:.4f}", va="center",
                fontsize=8, color=DGRAY, zorder=4)
        ax.text(0.0508, y, f"{snr:.2f}", va="center", ha="left",
                fontsize=9.5, fontweight="bold",
                color=TEAL if snr >= 1.0 else RED, zorder=4)
    ax.axvline(0.0495, color=GRAY, lw=0.7, ls="--", zorder=1)
    ax.text(0.0508, n - 1 + 0.66, "信噪比", ha="left", va="center",
            fontsize=9, fontweight="bold", color=DGRAY, zorder=4)
    ax.text(0.033, -0.33, "信号 < 噪声 = 瞎猜", ha="left", va="center",
            fontsize=9, style="italic", color=DGRAY, zorder=4)
    ax.set_xlim(0, 0.062)
    ax.set_ylim(-0.55, n - 1 + 0.95)
    ax.set_yticks(range(n - 1, -1, -1), [r[0] for r in D8_SNR], fontsize=10)
    ax.set_xticks([])
    ax.set_title("原因:各基准的真信号 vs 抽样噪声", fontsize=11.5,
                 fontweight="bold")
    ax.legend(handles=[Patch(color=TEAL, label="真信号:候选间真实差异"),
                       Patch(color=RED, label="抽样噪声:题量决定的标准误")],
              loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=2,
              fontsize=8.5, frameon=False)

    # 右 结果: gsm8k — prod3 误加权 vs prod5 修复后 (赢家 vs uniform)
    ax = fig.add_axes([0.565, 0.40, 0.36, 0.43])
    for gi, (glab, win, uni, wc) in enumerate(D8_GSM8K):
        gx = 1.0 + 1.55 * gi
        ax.bar(gx - 0.17, win, width=0.30, color=wc, zorder=3)
        ax.bar(gx + 0.17, uni, width=0.30, color=GRAY, zorder=3)
        tag = " (-56%)" if gi == 0 else " (2.9×)"
        ax.text(gx - 0.17, win + 0.012, f"{win:.3f}{tag}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=wc,
                zorder=4)
        ax.text(gx + 0.17, uni + 0.012, f"{uni:.3f}", ha="center",
                va="bottom", fontsize=9, fontweight="bold", color=DGRAY,
                zorder=4)
    ax.set_xticks([1.0, 2.55], [r[0] for r in D8_GSM8K], fontsize=10)
    ax.set_xlim(0.4, 3.15)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("结果:gsm8k — 赢家 vs uniform", fontsize=11.5,
                 fontweight="bold")
    ax.legend(handles=[Patch(color=RED, label="prod3 赢家(误加权)"),
                       Patch(color=TEAL, label="prod5 赢家(修复后)"),
                       Patch(color=GRAY, label="uniform 对照")],
              loc="upper center", bbox_to_anchor=(0.5, -0.135), ncol=3,
              fontsize=8.5, frameon=False)

    fig.text(0.5, 0.955, "评分按信噪比(SNR)加权:原因(左)、结果(右)、做法(下)",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.185, D8_HOW, ha="center", va="center", fontsize=10,
             color=DGRAY)
    fig.text(0.5, 0.105, D8_AFTER, ha="center", va="center", fontsize=9.5,
             color=DGRAY)
    ok = verify_layout(fig, "d8")
    path = os.path.join(out_dir, "reinforcement_d8_scoring_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── ③ D19: 外推问题示意图 (非真实数据) ───────────────────────────────
def chart_d19(out_dir):
    fig = plt.figure(figsize=(13, 4.4))
    ax = fig.add_axes([0.06, 0.15, 0.90, 0.66])

    def true_trend(x):
        return 0.30 + 0.10 * ((x - 5.5) / 4.5) ** 2

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predictor(x):
        # 实测区内贴合真趋势 + 小幅拟合摆动; 8.4 之后被末端数据带偏,
        # 急速俯冲 — 高阶拟合外推的典型行为 (示意)
        return true_trend(x) + 0.018 * np.sin(2.6 * x) \
            - 0.32 * sigmoid((x - 8.4) * 3.2)

    xg = np.linspace(0.8, 10.4, 400)
    xs = np.array([1.4, 2.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3])
    noise = np.array([0.025, -0.030, 0.018, -0.020, 0.028, 0.010,
                      0.022, -0.008, 0.018, 0.012, 0.020])
    ys = true_trend(xs) + noise
    bi = int(np.argmin(ys))
    xstar = float(xg[np.argmin(predictor(xg))])

    ax.axvspan(0.8, 7.8, color=TEAL, alpha=0.06, zorder=1)
    ax.axvspan(7.8, 11.0, color=RED, alpha=0.07, zorder=1)
    ax.axvline(7.8, color=DGRAY, lw=0.9, ls="--", zorder=2)
    ax.plot(xg, true_trend(xg), "--", color=GRAY, lw=1.6, zorder=3,
            label="真实趋势(训练前不可见)")
    ax.plot(xg, predictor(xg), "-", color=BLUE, lw=1.8, zorder=4,
            label="预测器(在实测点上拟合)")
    ax.scatter(xs, ys, s=36, color=NAVY, zorder=5,
               label="实测过的配比(搜索阶段)")
    ax.scatter([xs[bi]], [ys[bi]], s=170, color=TEAL,
               edgecolors="white", linewidth=1.5, zorder=6)
    ax.scatter([xstar], [predictor(xstar)], s=170, color=RED,
               edgecolors="white", linewidth=1.5, zorder=6)
    ax.text(xs[bi], 0.235, "实测最优(可信)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEAL, zorder=7)
    ax.text(xstar, 0.155, "外推『最优点』\n= 拟合噪声,不是信号",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=RED, linespacing=1.4, zorder=7)
    ax.text(4.3, 0.60, "实测过的区域(预测可信)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=DGRAY, zorder=7)
    ax.text(9.4, 0.60, "从未实测 = 外推(预测不可信)", ha="center",
            va="center", fontsize=10.5, fontweight="bold", color=RED,
            zorder=7)
    ax.text(0.9, 0.05, "prod4 实况:外推『终选』0.1972,输给实测最优 "
            "cfg72 的 0.2142(差 0.017,z~2.0)— 全部数字见下一页",
            ha="left", va="center", fontsize=9.5, color=DGRAY, zorder=7)
    ax.set_xlim(0.8, 11.0)
    ax.set_ylim(0.0, 0.65)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("配比空间(一维示意;真实为 15 维)", fontsize=9.5)
    ax.set_ylabel("得分", fontsize=9.5)
    ax.set_title("为什么外推不可信:拟合模型只对见过的区域负责(示意,非真实数据)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ok = verify_layout(fig, "d19")
    path = os.path.join(out_dir, "reinforcement_d19_extrapolation.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="『三处加固』举证图 (D14/D8/D19) → run 目录根层 3 张 PNG")
    ap.add_argument("run_dir", help="run 目录 (prod5 归档; 图输出到其根层)")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="输出目录 (默认 = run_dir)")
    ap.add_argument("--font", default=None,
                    help="显式指定中文字体文件 (.ttf/.ttc/.otf), 覆盖自动检测")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        sys.exit(f"✗ 目录不存在: {args.run_dir}")
    name = setup_cjk_font(args.font)
    if name is None:
        sys.exit(
            "✗ 未找到中文字体, 无法生成中文标签图。任选其一后重跑:\n"
            "  1) sudo apt-get install -y fonts-noto-cjk\n"
            "  2) 找服务器现成字体并显式指定:\n"
            "       find /usr/share/fonts /opt ~/.fonts \\\n"
            "         \\( -name '*.ttf' -o -name '*.ttc' -o -name '*.otf' \\) 2>/dev/null\n"
            "     python3 scripts/diagnostics/reinforcement_charts.py <run_dir> \\\n"
            "       --font /path/to/font.ttf\n"
            "  3) 把任一 CJK 字体文件拷到 ~/.fonts/ (自动扫描)")
    print(f"✓ 中文字体: {name}")
    out_dir = args.out_dir or args.run_dir
    print(f"『三处加固』举证图 → {out_dir}")
    print("  数据出处: paper_deviations.md D14 / experiment_prod2_k15bal.md "
          "§4.2 / TODO.md prod3 记录 / experiment_prod5.md §3b / "
          "experiment_prod4.md §3d")
    oks = [chart_d14(out_dir, args.run_dir),
           chart_d8(out_dir),
           chart_d19(out_dir)]
    if not all(oks):
        sys.exit("✗ 有图未通过标签重叠自检 — 请把上面的『重叠/越界』行贴回来")
    print("下一步: python3 scripts/diagnostics/make_deck.py <run_dir>")


if __name__ == "__main__":
    main()
