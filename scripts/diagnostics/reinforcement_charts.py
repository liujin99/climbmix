#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
#  reinforcement_charts.py — 『三处加固』举证图 (deck v9, 2026-10-12)
#
#  用法:
#    python3 scripts/diagnostics/reinforcement_charts.py result/prod5_20260929_201108
#    # 服务器若无中文字体: --font /path/to/cjk.ttf, 或安装 fonts-noto-cjk
#
#  输出 (RUN_DIR 根, 与 report_charts.py 的 PNG 同层; make_deck.py 自动引用):
#    reinforcement_d14_cluster_balance.png   ① 双环形图: 论文距离合并 (1 巨簇)
#                                            vs 均衡划分 (15 簇真实份额, C10 高亮)
#    reinforcement_d8_scoring_snr.png        ② 左: 逐基准 真信号 vs 抽样噪声 (SNR);
#                                            右: prod3 误加权事故 逐基准 Δ
#    reinforcement_d19_final_selection.png   ③ prod4 同轮对决: 实测点 vs 外推终选
#                                            + uniform 家族带 + 差距括号
#
#  v9 (用户反馈重设计): 图内标签改中文 (仅 arm 名/基准名保留原文);
#  去掉指向式箭头标注 (唯一箭头 = d19 差距括号, 水平 + 虚线导轨);
#  文字只进预留空白区; 保存前自检文字重叠/越界, 未过退出码 1。
#  CJK 字体自动检测 (matplotlib 注册表 → 文件系统扫描 + addfont),
#  找不到则报错并给安装指引。report_charts.py 的 ASCII 约定仍适用于
#  无中文字体的环境。
#
#  数据 = 实验记录已核实数字 (逐数注明出处; 跨轮历史数字无法从 prod5
#  run 目录读取, 为常量; 15 簇真实份额读 run 目录):
#    D14 塌缩诊断   paper_deviations.md D14 (prod1 事后 A6/A7):
#                   巨簇 99.05% / 13 孤岛 0.95%
#    D8  信号/噪声  experiment_prod2_k15bal.md §4.2 (30 候选实测):
#                   sd_cfg / SE_binom / SNR 逐基准表 (基准名以该表为准, gpqa)
#        prod3 事故 TODO.md 2026-09-15 P0 记录 (run4 双臂验证,
#                   Δ = 赢家 - 对照臂): gsm8k 0.119→0.052 (z=-6.2) /
#                   math 0.018→0.004 / arc_challenge +0.0296 /
#                   总分 0.166 vs 0.178
#    D19 prod4 对决 experiment_prod4.md §3d/§4d (d28, 旧协议口径):
#                   climb-cfg72 0.2142 / climb-cfg25 0.2066 /
#                   climb-终选 0.1972 / climb-终选-rep 0.1990;
#                   uniform 家族带 .1647-.1833 (random3b x2 + natural + domainfix)
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

TEAL = "#55A868"      # 真信号 / 变好 / 实测
RED = "#C44E52"       # 噪声 / 变差 / 外推终选
BLUE = "#4C72B0"      # 均衡簇 / 同轮实测点
ORANGE = "#DD8452"    # C10 / 赢家 (report_charts CLIMB_COLOR)
GRAY = "#9A9A9A"
DGRAY = "#555555"

CJK_KEYS = ("cjk", "zenhei", "wqy", "hei", "song", "kai", "noto sans sc",
            "source han", "pingfang", "droid sans fallback", "arial unicode",
            "msyh", "simhei", "simsun")

# ── 数据 (出处见文件头) ────────────────────────────────────────────────
D14_GIANT, D14_ISLANDS = 99.05, 0.95
D14_C10_BET = "赢家最终押注 80%(14.6 倍)"   # prod5 赢家 α=0.8036 vs 池占比 5.5%
D8_SNR = [  # bench, 真信号 sd_cfg, 抽样噪声 SE_binom, SNR (prod2 §4.2)
    ("arc_easy", 0.0225, 0.0127, 1.77),
    ("mmlu_stem", 0.0099, 0.0100, 1.00),
    ("arc_challenge", 0.0157, 0.0191, 0.82),
    ("gpqa", 0.0283, 0.0411, 0.69),
    ("gsm8k", 0.0048, 0.0163, 0.29),
    ("math", 0.0024, 0.0259, 0.09),
]
D8_PROD3 = [  # bench, Δ(赢家-对照臂), 标签, 放置 (TODO.md 2026-09-15 P0)
    ("gsm8k", -0.067, "-0.067 (-56%, z=-6.2)", "in"),
    ("math", -0.014, "-0.014 (-78%)", "left"),
    ("arc_challenge", 0.0296, "+0.030 (+3%)", "right"),
]
D19_RACE = [  # arm 名(原文), 分数, 颜色 (experiment_prod4.md §3d/§4d)
    ("climb-cfg72", 0.2142, BLUE),
    ("climb-cfg25", 0.2066, BLUE),
    ("climb-终选-rep", 0.1990, GRAY),
    ("climb-终选", 0.1972, RED),
]
D19_BAND = (0.1647, 0.1833)  # uniform 家族带


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


# ── 中文字体 ──────────────────────────────────────────────────────────
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


# ── 自检: 文字两两不重叠、不越出画布 ─────────────────────────────────
def _verify(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    items = []
    for t in fig.texts:
        if t.get_text().strip():
            items.append((t.get_text(), t.get_window_extent(r)))
    for ax in fig.axes:
        cands = [ax.title, ax.xaxis.label, ax.yaxis.label]
        cands += list(ax.texts)
        cands += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
        leg = ax.get_legend()
        if leg is not None:
            cands += list(leg.get_texts())
        for t in cands:
            if t.get_text().strip():
                items.append((t.get_text(), t.get_window_extent(r)))
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


# ── ① D14: 双环形图 ──────────────────────────────────────────────────
def chart_d14(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    real = shares is not None
    if not real:
        labels = [f"C{i}" for i in range(15)]
        shares = np.full(15, 100.0 / 15)
    hi = labels.index("C10") if "C10" in labels else 9
    fig = plt.figure(figsize=(13, 4.6))

    # 左: 论文距离合并 — 1 巨簇 + 13 孤岛
    ax = fig.add_axes([0.05, 0.13, 0.40, 0.70])
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
    ax.set_title("论文式距离合并(实测构成)", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    # 右: 均衡划分 — 15 簇真实份额, C10 高亮
    ax = fig.add_axes([0.55, 0.13, 0.40, 0.70])
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
            fontsize=15, fontweight="bold", color="#1F4E79")
    ax.set_title("均衡划分(我们的修复,D14)", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.text(0.5, 0.955, "同一个数据池,两种簇划分:各簇分到的 token 份额",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.25, 0.05, "其余 13 个孤岛簇合计 0.95%(图中灰缝,几乎不可见)",
             ha="center", va="center", fontsize=9.5, color=DGRAY)
    if real:
        note = f"{labels[hi]} 只占数据池 {shares[hi]:.1f}% — {D14_C10_BET}"
    else:
        note = "未找到 macro_info.json — 图为等分示意"
    fig.text(0.75, 0.05, note, ha="center", va="center", fontsize=9.5,
             color=DGRAY)
    ok = _verify(fig, "d14")
    path = os.path.join(out_dir, "reinforcement_d14_cluster_balance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}" + ("  [真实份额: macro_info.json]"
                           if real else "  [示意: 无 macro_info.json]"))
    return ok


# ── ② D8: 信号 vs 噪声 + prod3 事故 ─────────────────────────────────
def chart_d8(out_dir):
    fig = plt.figure(figsize=(13, 4.4))

    # 左: 逐基准 真信号 vs 抽样噪声 (SNR 降序, 上=响)
    ax = fig.add_axes([0.085, 0.21, 0.37, 0.62])
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
    ax.set_title("各基准:真信号 vs 抽样噪声(prod2,30 个实测候选)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(handles=[Patch(color=TEAL, label="真信号:候选间真实差异"),
                       Patch(color=RED, label="抽样噪声:题量决定的标准误")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=9, frameon=False)

    # 右: prod3 误加权事故 — Δ = 赢家 - 对照臂
    ax = fig.add_axes([0.565, 0.21, 0.37, 0.62])
    for y, (bench, d, lab, place) in zip((2, 1, 0), D8_PROD3):
        c = RED if d < 0 else TEAL
        ax.barh(y, d, height=0.5, color=c, zorder=3)
        if place == "in":      # 标签放长条内部 (白字)
            ax.text(-0.004, y, lab, va="center", ha="right", fontsize=10,
                    fontweight="bold", color="white", zorder=4)
        elif place == "left":
            ax.text(d - 0.002, y, lab, va="center", ha="right", fontsize=10,
                    fontweight="bold", color=c, zorder=4)
        else:
            ax.text(d + 0.002, y, lab, va="center", ha="left", fontsize=10,
                    fontweight="bold", color=c, zorder=4)
    ax.axvline(0, color=DGRAY, lw=1, zorder=2)
    ax.text(-0.102, -0.55, "总分:赢家 0.166 vs 对照 0.178\n"
            "平均掩盖了方向性塌方", ha="left", va="center", fontsize=9,
            color=DGRAY, linespacing=1.5, zorder=4)
    ax.set_xlim(-0.105, 0.072)
    ax.set_ylim(-0.85, 2.75)
    ax.set_yticks((2, 1, 0), [r[0] for r in D8_PROD3], fontsize=10)
    ax.set_xticks([])
    ax.set_title("prod3 事故:搜索赢家 vs 对照臂(逐基准 Δ)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(handles=[Patch(color=RED, label="变差"),
                       Patch(color=TEAL, label="变好")],
              loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=9, frameon=False)

    fig.text(0.5, 0.955, "评分按 SNR 加权:先看『为什么』,再看『不加权会怎样』",
             ha="center", va="center", fontsize=13, fontweight="bold")
    ok = _verify(fig, "d8")
    path = os.path.join(out_dir, "reinforcement_d8_scoring_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── ③ D19: prod4 实测 vs 外推对决 ────────────────────────────────────
def chart_d19(out_dir):
    fig = plt.figure(figsize=(13, 4.2))
    ax = fig.add_axes([0.14, 0.19, 0.62, 0.66])
    ax.axvspan(D19_BAND[0], D19_BAND[1], color=GRAY, alpha=0.18, zorder=1)
    ax.text(0.1653, -0.68, "uniform 家族 .1647-.1833", ha="left", va="center",
            fontsize=8.5, color=DGRAY, zorder=4)
    for i, (name, v, c) in enumerate(D19_RACE):
        y = 3 - i
        ax.barh(y, v - 0.16, left=0.16, height=0.55, color=c, zorder=3)
        ax.text(v + 0.0006, y, f"{v:.4f}", va="center", ha="left",
                fontsize=11, fontweight="bold",
                color=c if c != GRAY else "#777777", zorder=4)
    ax.vlines([0.2142, 0.1972], -0.15, 3.85, colors=DGRAY, linestyles="--",
              lw=0.9, zorder=2)
    ax.annotate("", xy=(0.2142, 3.85), xytext=(0.1972, 3.85),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=1.4))
    ax.text(0.2057, 4.05, "差距 0.017(z~2.0)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=RED, zorder=4)
    ax.text(0.196, -0.68, "两个外推点高于 uniform 带但热区垫底;\n"
            "外推的领先低于噪声地板 = 拟合噪声", ha="left", va="center",
            fontsize=8.5, color=DGRAY, linespacing=1.5, zorder=4)
    ax.set_xlim(0.16, 0.225)
    ax.set_ylim(-1.05, 4.4)
    ax.set_yticks((3, 2, 1, 0), [r[0] for r in D19_RACE], fontsize=10.5)
    ax.set_xticks([0.16, 0.18, 0.20, 0.22])
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("prod4 同轮对决(d28,轴从 0.16 起):实测点 vs 从未训练的外推『终选』",
                 fontsize=12.5, fontweight="bold")
    ax.legend(handles=[Patch(color=BLUE, label="同轮实测点"),
                       Patch(color=GRAY, label="外推·换种子复刻(未训练)"),
                       Patch(color=RED, label="外推·论文式终选(未训练)"),
                       Patch(color=GRAY, alpha=0.18, label="uniform 家族带")],
              loc="upper center", bbox_to_anchor=(0.5, -0.155), ncol=4,
              fontsize=8.5, frameon=False)
    ok = _verify(fig, "d19")
    path = os.path.join(out_dir, "reinforcement_d19_final_selection.png")
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
          "§4.2 / TODO.md prod3 记录 / experiment_prod4.md §3d")
    oks = [chart_d14(out_dir, args.run_dir),
           chart_d8(out_dir),
           chart_d19(out_dir)]
    if not all(oks):
        sys.exit("✗ 有图未通过标签重叠自检 — 请把上面的『重叠/越界』行贴回来")
    print("下一步: python3 scripts/diagnostics/make_deck.py <run_dir>")


if __name__ == "__main__":
    main()
