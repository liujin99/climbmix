#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ═══════════════════════════════════════════════════════════════════════
#  reinforcement_charts.py — 『四处机制改进优化』举证图 (deck v10, 2026-10-13)
#
#  用法:
#    python3 scripts/diagnostics/reinforcement_charts.py result/prod5_20260929_201108
#    # 服务器若无中文字体: --font /path/to/cjk.ttf, 或安装 fonts-noto-cjk
#
#  输出 (v10.2: 汇报包目录 RUN_DIR/deck/ 子目录, 与实验结果分离;
#  make_deck.py 在同目录引用; -o 可覆盖):
#    reinforcement_d14_cluster_balance.png  ① as-is vs to-be: 上排 = 簇结构
#                                           示意(左 整条流形并成一簇 /
#                                           右 切成大小相近的 15 簇),
#                                           下排 = token 份额环形图,
#                                           中间箭头 = 改进动作
#    reinforcement_d8_dual_signal.png      ② 双信号三联: 左/中=为什么(相变
#                                           区左侧得分不动; 精度不动 vs
#                                           损失缓降), 右=怎么做(混合公式
#                                           独立成卡)
#    finding_quality_value.png             发现① 质量分 × 赢家权重 排序倒挂
#                                           (左=质量分 降序, 右=climb 权重)
#    finding_pool_manifold.png             发现③ 池结构机理示意
#                                           (通用池成团 vs 窄域池流形+孤岛)
#    reinforcement_d8_scoring_snr.png       ③ 左=问题(各基准信号强度悬殊,
#                                           第3轮重放), 中=做法(公式卡,
#                                           mathtext), 右=结果(gsm8k:
#                                           第3轮误加权 vs 第5轮改进后)
#    reinforcement_d19_extrapolation.png    ③ 外推问题(示意图, 非真实
#                                           数据; 真实数字在 deck 表格与
#                                           p10 备注): 双区域 + 真实趋势/
#                                           预测器双曲线 + 实测点(示意)+
#                                           外推红点 + 正式实测落差
#    recipe_distribution.png       ④ 配比全景对比(v10.4): 赢家/cfg90/cfg100/
#                                  uniform/池自然占比 在 15 语义簇上的 100%
#                                  堆叠 + domainfix 域标签配比单列(域空间,
#                                  无簇 α 向量); 配方读 run 目录 optimal_
#                                  mixture_weights / topk / search_state
#    cluster_content_map.png       ⑤ 簇内容分析(v10.5): 15 个语义簇逐簇
#                                  池 token 占比(按域配色)+ 赢家 α 列;
#                                  内容速写(域构成/中位长度)解析 run 目录
#                                  cluster_semantics.md (recipe_report §2b
#                                  生成), 缺件回退域分组常量
#
#  v10.8 (评审反馈): ① d8 图内标签 → 『第3轮实验/第5轮实验』+『改进后』
#    (deck 全 deck 术语: 实验轮次 vs 单轮内搜索迭代); ② 簇内容图标签
#    第 2 行 = 关键词 + 样本开头 (cluster_peek 新特性: 对比 IDF 关键词,
#    §2b『每簇关键词』节; prod5 试点), 摘录按显示宽度预算截断 (拉丁半
#    宽, 英文样本不再只露 3 个单词); 图加宽 14.4×6.2, 左标签区扩到
#    0.445。旧版 cluster_semantics.md 无关键词节 → 退化为纯摘录, 兼容。
#
#  v10.7 (评审反馈): ① d8 图内标签 prod3/prod5 → 第3轮/第5轮(与 deck
#    弃用内部代号一致); ② 簇内容图每簇标签附『样本开头摘录』(§2b 的
#    每簇 1 篇抽样, 两行标签), 摘录逐字对子集字体 cmap 校验防豆腐块;
#    图幅 13×6.4 略加高(两行标签行距)。
#
#  v10.6 (评审反馈): ① d19 『正式实测』→『中等规模实测』(与 deck 术语
#    统一); ② 簇内容图标签差异化 — 细鉴命名簇(§4c)用鉴定名, 其余 = 域
#    纯度%(混合簇双域)+ 中位长度, 不再重复『××簇 · ××%』; ③ 簇内容图
#    加赢家下注/地板分隔线 + 注记(地板结构不成立时不画, 防误标)。
#
#  v10.5 (评审反馈): ① d14/d8/d19 底部文字佐证块删除(数字改由 deck 页
#  表格承载, 图只管机理) + 图幅压扁适配『表 + 图』同页; ② d8 删底部小
#  结行与口语注释; ③ d19 删 prod4 数字行(进表格), 『打回原形』改专业
#  表述; ④ 新增簇内容分析图 (用户: 缺一页表示每个簇分别是什么)。
#
#  v10.4 (评审反馈): ① d8 三联重排为 问题(左)· 做法(中, 公式卡)· 结果
#  (右), 与 deck 加固页的『问题→做法→结果』三短点同构; ② 新增配比全景
#  对比图 (用户: 最终配比的分布对比漏了; domainfix 用域标签, 不能直接用
#  语义簇表示 — 单列右图)。
#
#  v10.3 (评审反馈): ① d19 极性修复 — 原图把『实测最优』『外推最优点』
#  都标在曲线最低点(得分极性反了), 改为: 实测最优 = 最高实测点, 外推
#  最优点 = 预测器在无数据区的虚假最高峰(论文式终选挑的就是它), 并加
#  『正式实测:打回原形』空心点; ② 图内得分统一百分点口径 (xx.x% /
#  x.xxpt), 与 deck 表格 (v10.3) 一致。
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
#    D8  信号/噪声  scoring_metric_design.md §6.2 "prod3 实测 vs 预测"
#                   (54 实测点, 修复后公式重放; v10.27 前用 prod2 §4.2
#                   30 点表 — 其 gsm8k/math 噪声列带先验 bug, 已弃用)
#        prod3 事故 TODO.md 2026-09-15 P0: gsm8k 赢家 0.052 vs 对照 0.119
#                   (-56%, z=-6.2); math 0.018→0.004; 修复 = SNR 定权 +
#                   实测噪声地板 + w_floor=0.5
#        prod5 结果 experiment_prod5.md §3b: gsm8k climb 0.3010 /
#                   uniform 0.1054 / base 0.0288 (2.9x / 10.4x)
#    D19 prod4 对决 experiment_prod4.md §3d/§4d + predictor_audit.py 同款
#                   重放(data/prod4_search_dump.csv 111 点舰队)— 图 =
#                   真实数据: 残差带 + L1a 四臂实测
#    15 簇份额      RUN_DIR/macro_info.json (⑬r) 或 cluster_info_cache.json
#                   (legacy), schema 同 recipe_report.py (label/num_tokens);
#                   全缺 -> 等分示意 (图内注明)
# ═══════════════════════════════════════════════════════════════════════
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    from matplotlib.patches import (Ellipse, FancyArrowPatch, FancyBboxPatch,
                                    Patch)
except ImportError:
    sys.exit("✗ 缺 matplotlib — pip install matplotlib (PEP 668 环境加 "
             "--break-system-packages), 装好后重跑本脚本")

TEAL = "#55A868"      # 真信号 / 变好 / 改进后
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
D8_SNR = [  # bench, 信号 σ_between, 噪声 σ_noise, f = 1 − σ²n/σ²b —
    #  v10.27: 换第3轮 54 实测点、修复后公式的重放值(原 prod2 §4.2 的
    #  30 点表早于噪声修复, gsm8k/math 的噪声列带着先验 bug, 与本页
    #  "gsm8k 最强信号 w=0.93" 的结果自相矛盾 — 评审正是因此觉得
    #  "讲的是另一个问题")。出处 scoring_metric_design.md §6.2 "prod3
    #  实测 vs 预测"表: σ²_between 取实测列, f 与 σ²_noise 由该表反解
    #  (f = 2w−1, σ²_noise = (1−f)·σ²_between); σ 单位 = 去蒙对校正分。
    ("gsm8k", 0.0107, 0.0041, +0.85),   # σ²b 1.14e-4 / σ²n 1.7e-5, 最强信号
    ("arc_challenge", 0.0176, 0.0188, -0.14),
    ("math", 0.0019, 0.0021, -0.22),
    ("arc_easy", 0.0112, 0.0127, -0.30),
    ("mmlu_stem", 0.0080, 0.0100, -0.56),
    ("gpqa", 0.0302, 0.0405, -0.80),
]
D8_GSM8K = [  # (标签, 赢家分, 对照 uniform 分, 赢家颜色) — gsm8k 跨协议可比
    ("第3轮实验 · 误加权", 0.052, 0.119, RED),
    ("第5轮实验 · 改进后", 0.3010, 0.1054, TEAL),
]

# 配比全景对比 (v10.4) — 语义名/域配比出处:
CLUSTER_NAMES = {  # 全簇命名 (cluster_semantics.md §2b, n=25 抽样 + 对比 IDF
    #  关键词, 2026-10-14; 沿革见 experiment_prod5.md 记录沿革 v1.0.4)。
    #  勘误: C10 旧名"统计练习题"(2026-09-30, n=1 抽样)系误读 — n=25 关键词
    #  divide·percentage·multiply + 样本"22% of 12 餐厅小费" = 算术应用题,
    #  与 gsm8k 机制闭环更贴合。C6/C2: 格式名 → 内容名。
    "C0": "形式数学解题", "C1": "教学与实验长文", "C2": "SMILES 分子性质问答",
    "C3": "海洋与生态生物学", "C4": "材料化学综述", "C5": "粒子物理与宇宙学",
    "C6": "代数与数论长文", "C7": "化学常识概述", "C8": "高等数学论文",
    "C9": "分数运算练习", "C10": "算术应用题", "C11": "论坛问答与任务文本",
    "C12": "凝聚态与量子光学", "C13": "材料计算问答", "C14": "蛋白质与膜生化",
}
DOMAINFIX_DOMAINS = [  # experiment_prod4.md §4e — 域标签空间, 无簇 α 向量
    ("数学", 60.0), ("物理", 15.0), ("化学", 12.5), ("生物", 12.5),
]

# 簇内容分析 (v10.5) — 域分组出处 experiment_prod5.md §4c v1.0.3
# (§2b 全簇速写读数: 数学 ×5 / 生物 ×3 / 化学 ×2 / 物理 ×2 / 混合 ×3)
CLUSTER_DOMAIN = {
    "C0": "数学", "C6": "数学", "C8": "数学", "C9": "数学", "C10": "数学",
    "C1": "生物", "C3": "生物", "C14": "生物",
    "C4": "化学", "C7": "化学",
    "C5": "物理", "C12": "物理",
    "C2": "混合", "C11": "混合", "C13": "混合",
}
DOMAIN_COLORS = {"数学": BLUE, "生物": "#7FB77E", "化学": "#C9A66B",
                 "物理": "#A88BC4", "混合": "#B9C6D1"}


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


# ── 中文字体 (v10.1: 仅本脚本三张 deck 专用图用中文; report_charts 的
#    实验结果图保持英文, 不经此处) ────────────────────────────────────
# 仓库内置子集字体优先 (服务器无 root / 无系统 CJK 字体也能出图);
# fonts/wqy-zenhei-deck.ttf = 文泉驿正黑子集 (GPL + 字体例外), 覆盖
# 本脚本全部图内文案 — 改文案后跑 fonts/make_font_subset.py
BUNDLED_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "fonts")


def _apply_font(name):
    matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False


def _register(path):
    try:
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    except Exception:
        return None


def setup_cjk_font(font_path=None):
    """返回可用中文字体名并写入 rcParams; 找不到返回 None。

    优先级: --font 显式指定 > 仓库内置子集字体 (fonts/) > matplotlib
    注册表 (系统已装) > 文件系统扫描 (~/.fonts/ 等)。"""
    if font_path:
        if not os.path.isfile(font_path):
            sys.exit(f"✗ --font 指定的文件不存在: {font_path}")
        name = _register(font_path)
        if name is None:
            sys.exit(f"✗ 注册字体失败: {font_path}")
        _apply_font(name)
        return name
    for pat in ("*.ttf", "*.ttc", "*.otf"):
        for p in sorted(glob.glob(os.path.join(BUNDLED_FONT_DIR, pat))):
            name = _register(p)
            if name:
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
                name = _register(p)
                if name:
                    names.append(name)
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
        edges = []
        if b1.x0 < -2:
            edges.append("左")
        if b1.y0 < -2:
            edges.append("下")
        if b1.x1 > w + 2:
            edges.append("右")
        if b1.y1 > h + 2:
            edges.append("上")
        if edges:
            bad.append(f"越界[{'+'.join(edges)}]:『{s1[:14]}』")
        for s2, b2 in items[i + 1:]:
            if b1.overlaps(b2):
                bad.append(f"重叠:『{s1[:10]}』×『{s2[:10]}』")
    if bad:
        print(f"  ✗ {name} 自检失败 — " + "; ".join(bad[:6]))
        return False
    print(f"  ✓ {name}: 标签无重叠、无越界")
    return True


# ── ① D14: as-is vs to-be (v10.27 评审反馈重设计: ① 两饼图中间加箭头
#    表示前后改进关系; ② 链式塌缩不再只靠文字 — 上下各配 p15 同款聚类
#    簇示意图, 问题(整条流形并成一簇)与改进后(切成大小相近的 15 簇)
#    直接可见。v10.5: 佐证数字移入 deck 表格, 图只管机理) ─────────────
def chart_d14(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    real = shares is not None
    if not real:
        labels = [f"C{i}" for i in range(15)]
        shares = np.full(15, 100.0 / 15)
    hi = labels.index("C10") if "C10" in labels else 9
    rng = np.random.default_rng(14)
    fig = plt.figure(figsize=(13, 4.05))

    fig.text(0.5, 0.965, "同一个数据池,两种簇划分", ha="center", va="center",
             fontsize=13, fontweight="bold")
    fig.text(0.245, 0.905, "As-is:论文式距离合并 — 链式塌缩", ha="center",
             va="center", fontsize=11.5, fontweight="bold", color=RED)
    fig.text(0.755, 0.905, "To-be:容量约束均衡(K=15)", ha="center",
             va="center", fontsize=11.5, fontweight="bold", color=NAVY)

    def cloud(ax):
        """p15 同款窄域池示意: 一条连续流形 + 3 个格式孤岛(无标签,
        命名住在 p15 / §2b)。返回流形中心线, 供均衡切分定位。"""
        ax.set_axis_off()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("#F7FBFB")
        t = np.linspace(0.06, 0.94, 420)
        ym = 0.50 + 0.11 * np.sin(2 * np.pi * t * 0.9)
        ax.fill_between(t, ym - 0.17, ym + 0.17, color=TEAL, alpha=0.13,
                        zorder=2)
        ax.scatter(t, ym + rng.normal(0, 0.052, 420), s=4, color=TEAL,
                   alpha=0.35, zorder=3)
        for ix, iy, k in ((0.13, 0.87, 9), (0.55, 0.91, 6), (0.88, 0.85, 8)):
            pts = rng.normal(0, 0.018, (k, 2)) + [ix, iy]
            ax.scatter(pts[:, 0], pts[:, 1], s=6, color=GRAY, alpha=0.8,
                       edgecolors="none", zorder=3)
        # 固定 0-1 视窗: 自动缩放会把底部说明文字挤到轴外(与幻影刻度
        # 相撞 — verify_layout 连 axis-off 轴的刻度标签一并检查)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return t, ym

    # 左上 as-is 簇结构: 整条流形被一个红圈吞并(链式塌缩的结局)
    ax = fig.add_axes([0.035, 0.50, 0.40, 0.335])
    cloud(ax)
    ax.add_patch(Ellipse((0.5, 0.50), 1.02, 0.58, fill=False, ls="--",
                         color=RED, lw=1.4, zorder=4))
    ax.text(0.5, 0.50, "并成 1 簇", ha="center", va="center", fontsize=10,
            fontweight="bold", color=RED, zorder=5,
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
    ax.text(0.5, 0.035, "链式塌缩:整条流形并成一簇(99.05% tokens)",
            ha="center", va="bottom", fontsize=9.5, color=RED, zorder=5)

    # 右上 to-be 簇结构: 同一条流形, 按容量切成大小相近的簇(示意 7 段)
    ax = fig.add_axes([0.565, 0.50, 0.40, 0.335])
    t, ym = cloud(ax)
    for xc in (0.13, 0.255, 0.38, 0.505, 0.63, 0.755, 0.88):
        yc = float(np.interp(xc, t, ym))
        ax.add_patch(Ellipse((xc, yc), 0.135, 0.30, fill=False, ls="--",
                             color=TEAL, lw=1.1, zorder=4))
    for ix, iy in ((0.13, 0.87), (0.55, 0.91), (0.88, 0.85)):
        ax.add_patch(Ellipse((ix, iy), 0.075, 0.10, fill=False, ls=":",
                             color=GRAY, lw=0.9, zorder=4))
    ax.text(0.5, 0.035, "容量约束:切成大小相近的 15 簇 — 每簇一个旋钮",
            ha="center", va="bottom", fontsize=9.5, color=NAVY, zorder=5)

    # 中间箭头: 改进动作(前后关系, 评审要求)
    fig.add_artist(FancyArrowPatch(
        (0.452, 0.47), (0.548, 0.47), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=42, lw=2.5, color=TEAL, zorder=5))
    fig.text(0.5, 0.55, "容量约束均衡", ha="center", va="center",
             fontsize=10.5, fontweight="bold", color=TEAL)

    # 左下 as-is 份额: 1 巨簇 + 13 孤岛 → 1 个可用旋钮
    ax = fig.add_axes([0.135, 0.045, 0.20, 0.40])
    wedges, _ = ax.pie([D14_GIANT, D14_ISLANDS], colors=[RED, GRAY],
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, "1 个\n可用旋钮", ha="center", va="center",
            fontsize=12, fontweight="bold", color=RED)
    w = wedges[0]
    th = math.radians((w.theta1 + w.theta2) / 2)
    ax.text(0.78 * math.cos(th), 0.78 * math.sin(th), "巨簇\n99.05%",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="white")
    ax.set_xticks([])
    ax.set_yticks([])

    # 右下 to-be 份额: 15 簇真实份额, C10 高亮(改进后)
    ax = fig.add_axes([0.665, 0.045, 0.20, 0.40])
    colors = [ORANGE if i == hi else BLUE for i in range(len(shares))]
    explode = [0.05 if i == hi else 0.0 for i in range(len(shares))]
    wedges, _ = ax.pie(shares, colors=colors, explode=explode,
                       startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white"))
    # v10.27: 饼图缩小后相邻扇区标签互撞 — 只标高亮的赢家簇 C10
    #   (其余簇的身份住 p11/p12; 均衡份额数字住页面表格)
    w = wedges[hi]
    th = math.radians((w.theta1 + w.theta2) / 2)
    ax.text(0.83 * math.cos(th), 0.83 * math.sin(th), labels[hi],
            ha="center", va="center", fontsize=8, fontweight="bold",
            color="white")
    ax.text(0, 0, "15 个\n可用旋钮", ha="center", va="center",
            fontsize=12, fontweight="bold", color=NAVY)
    ax.set_xticks([])
    ax.set_yticks([])

    ok = verify_layout(fig, "d14")
    path = os.path.join(out_dir, "reinforcement_d14_cluster_balance.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}" + ("  [真实份额: macro_info.json]"
                           if real else "  [示意: 无 macro_info.json]"))
    return ok


# ── ② D8(双信号): 问题(左, 精度单信号分辨不开)· 做法(中, 混合公式)·
#    反证(右, w 下限) (v10.14 评审反馈: 改进②纯文字页太干 — 补机理图;
#    左图 33%/31% 为示意值, SE 公式为实算, 右图数字全部实测) ──────────
def chart_d8_dual(out_dir):
    # v10.18: 两联放大重讲(评审反馈: 相变故事应讲"区左侧" — 代理小 +
    #   中训练量少 → 潜在能力缓慢积累但基准得分不动 → 不同配比得分
    #   重叠、分辨不出好坏; 原右板"两种信号选出的赢家"表达不清, 撤下,
    #   结果数字由页面表格承载(均实测)) —
    #   左=为什么(相变区左侧, 得分看不出配比好坏; Phase Transitions,
    #   NeurIPS'25 Spotlight)
    #   右=怎么做(精度轨迹区分度小 vs 损失轨迹连续平稳 → 混合, w ≥ 0.5)
    # v10.19: 右板修正(评审反馈) — 损失曲线改缓慢下降(原正弦的上升段
    #   占面过大, 与"配比变好 → 损失下降"不符); 标题/标签补"相变区之前";
    #   底注补 w 定法指向(按逐基准实测信噪比定, 见下页改进③)。
    # v10.20: 右板再修(评审反馈"像两条不稳定的准确率曲线") — 损失改
    #   NAVY 加粗、噪声 σ 0.045→0.012 的平滑稳定缓降, 精度细线抖动
    #   (σ 0.14), 平滑度对比一目了然; 底注精简(评审反馈文字太多)。
    # v10.21: 损失曲线形态再修(评审反馈"不是一条直线") — 改指数型平滑
    #   下降(先快后慢), 贴近真实 loss 曲线形态。
    # v10.24: 公式与 p9 公式卡同步改正体(\mathrm 全覆盖) — 两页同一套
    #   符号, 风格一致。
    # v10.26: 与 chart_d8 同套四级字号(评审"字一会大一会小")— 总标题
    #   13.5→13、面板标题 12→11.5、公式 14→11.5(= p9 公式卡)、刻度
    #   9→10、精度/损失标签 9.5→10、小注 9.5→9、图注 8→9。
    # v10.27: 评审反馈 — ① 删总标题"评分双信号:为什么、怎么做"(页标题
    #   已表达, 图内三个面板标题自足); ② 评审读法: 右板本质也在讲
    #   "为什么"(为什么需要损失补位), "怎么做"只是混合公式 — 公式
    #   独立成右侧公式卡(p9 公式卡同款 FancyBbox), 轨迹板标题改
    #   "为什么补位", 让位给两信号对比本身。
    # v10.28: 评审反馈 — ① 面板标题去口语("为什么/为什么补位")→
    #   "根因1/根因2"; ② 右侧公式: 不换行(单行 11.5pt 实测 2.35in),
    #   撤掉 FancyBbox 框, 区域与左右面板同高同底(y 0.20 / h 0.62),
    #   下方注解从两段多行砍成一行。
    rng = np.random.default_rng(11)
    fig = plt.figure(figsize=(13, 3.4))

    # 左 根因1: 相变曲线 — 我们在跳变左侧, 不同配比的得分几乎重叠
    ax = fig.add_axes([0.04, 0.20, 0.355, 0.62])
    x = np.linspace(0, 1, 300)
    y = 0.62 / (1 + np.exp(-(x - 0.62) * 16))
    ax.plot(x, y, color=NAVY, lw=2.4, zorder=3)
    ax.axvspan(0.55, 1.0, color="#FDECEA", zorder=1)
    xs = rng.uniform(0.30, 0.50, 22)
    ys = 0.62 / (1 + np.exp(-(xs - 0.62) * 16)) + rng.normal(0, 0.018, 22)
    ax.scatter(xs, np.clip(ys, -0.02, 0.70), s=16, color=RED, alpha=0.75,
               zorder=4)
    ax.text(0.775, 0.665, "相变跳变区\n(代理 + 少量中训练还没到)",
            ha="center", va="top", fontsize=9, color=RED,
            linespacing=1.5, zorder=4)
    ax.annotate("不同配比的代理得分:几乎重叠\n→ 分辨不出配比好坏",
                xy=(0.40, 0.075), xytext=(0.60, 0.30), fontsize=10,
                color=RED, fontweight="bold", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))
    ax.text(0.12, 0.16, "得分平缓区\n潜在能力在积累", ha="center",
            fontsize=9, color=DGRAY, linespacing=1.5, zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 0.72)
    ax.set_xticks([])
    ax.set_yticks([0, 0.3, 0.6], ["0", "30%", "60%"], fontsize=10)
    ax.set_xlabel("能力驱动量(模型规模 × 数学配比 × 数据量)", fontsize=10)
    ax.set_title("根因1:相变区左侧,基准得分看不出配比好坏", fontsize=11.5,
                 fontweight="bold")

    # 中 根因2: 相变区之前 — 精度几乎不动, 损失缓慢下降
    ax = fig.add_axes([0.43, 0.20, 0.31, 0.62])
    xs = np.arange(24)
    acc = rng.normal(0, 0.14, 24)                        # 精度: 原地抖动
    nll = 0.75 * np.exp(-xs / 8.0) - 0.45 + rng.normal(0, 0.012, 24)  # 损失: 先快后慢平滑下降
    ax.axhline(0, color="#CCCCCC", lw=0.7, zorder=1)
    ax.plot(xs, acc, color=TEAL, lw=1.5, zorder=3)
    ax.plot(xs, nll, color=NAVY, lw=2.6, zorder=4)
    ax.set_xlim(-0.5, 42)
    ax.set_ylim(-1.5, 1.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(24.3, 0.42, "准确率:几乎不动\n(相变区之前,区分度小)", fontsize=10,
            color=TEAL, fontweight="bold", va="center", linespacing=1.4,
            zorder=4)
    ax.text(24.3, -0.52, "损失:平滑下降\n(先快后慢,连续可分)", fontsize=10,
            color=NAVY, fontweight="bold", va="center", linespacing=1.4,
            zorder=4)
    ax.set_xlabel("候选配比(按序)", fontsize=10)
    ax.set_title("根因2:准确率几乎不动、损失平滑下降", fontsize=11.5,
                 fontweight="bold")

    # 右 做法: 双信号混合公式 — 独立成条(评审: 公式只是示意; 不框框,
    #   区域与左右面板同底同高, 公式单行)
    ax = fig.add_axes([0.775, 0.20, 0.205, 0.62])
    ax.set_axis_off()
    ax.set_title("做法:双信号混合", fontsize=11.5, fontweight="bold")
    ax.text(0.5, 0.56,
            r"$\mathrm{score} = \mathrm{w} \cdot \mathrm{acc}_z"
            r" + (1-\mathrm{w}) \cdot \mathrm{nll}_z$",
            ha="center", va="center", fontsize=11.5, color=NAVY,
            transform=ax.transAxes, zorder=3)
    ax.text(0.5, 0.38, "w ≥ 0.5,定法见下页", ha="center", va="center",
            fontsize=9, color=DGRAY, transform=ax.transAxes, zorder=3)

    fig.text(0.5, 0.035, "左 = 相变示意(Phase Transitions, NeurIPS\'25 "
              "Spotlight — 阈值左侧能力缓慢积累、得分不动);中 = 信号轨迹示意"
              "(acc_z、nll_z = 准确率、损失的标准分;nll_z 取负 — 损失降 → "
              "nll_z 升);100 题 speedrun:SE ≈ ±4.6pt(实算)",
              ha="center", va="center", fontsize=9, color=DGRAY)
    ok = verify_layout(fig, "d8_dual")
    path = os.path.join(out_dir, "reinforcement_d8_dual_signal.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── ③ D8: 问题(左)· 做法(中, 公式)· 结果(右) (v10.5: 删底部小结行,
#    数字进 deck 表格; 图幅压扁) ────────────────────────────────────
def chart_d8(out_dir):
    # v10.22: 图幅 4.2→3.8 压扁 + 页表 row_h 0.56→0.50 — 上页显示宽
    #   7.5→9.1in(评审"稍微放大一点"); 中部公式卡改正式数学公式
    #   (mathtext, 评审"用一行来写不好看"), 行③ 的下划线式公式移入图内。
    # v10.23: 评审反馈 — 图再放大(3.8→3.4 压扁, 页表显示宽 9.1→10.2in),
    #   三联面板拉开间隔(0.33/0.58in → 0.81/0.98in, 左板 y 刻度 10→9pt
    #   给基准名让位); 图底加符号脚注(评审"公式符号需要标注":
    #   p/K_b/c_b/σ²_noise/σ²_between/f)。
    # v10.24: 评审反馈 — ① 公式卡字体统一(评审"字体不一致"): 四条公式
    #   一律 12pt/NAVY/正体(\mathrm 全覆盖 — 原斜体变量与正体函数名混
    #   排, 且 13/12pt、NAVY/灰混用); ② 补 σ²_between 的计算式(评审:
    #   只给了 σ²_noise 的); ③ 公式设计原因((1+f)/2 与 0.5 下限、
    #   1−noise/between 的信号占比含义、二项方差与 (1−c_b)² 的来历)
    #   写入 p9 PPT 备注(make_deck notes 字段)。
    # v10.25: 评审反馈 — ① 公式字号仍不齐(评审"字的大小怪怪的,Var(acc)
    #   有点大"): 根因 \frac 的分子/分母自动缩 ~70% — f1/f3 的核心内容
    #   偏小,全尺寸的 f4 显大;改 \dfrac(全尺寸分数),卡片加高让位
    #   (axes 0.30/0.50 → 0.28/0.54) — 四条公式的核心字一律 12pt 全
    #   尺寸,仅真上下标缩小。② 备注补"为什么两个 σ 的算法不一样"
    #   (σ²_noise 二项闭式解 vs σ²_between 无模型只能实测,见 make_deck)。
    # v10.26: 评审反馈(评审"字一会大一会小,难受")— 全图改严格四级字号,
    #   同角色同字号,左右图/公式卡/脚注对齐: 13 粗=总标题; 11.5 粗=三个
    #   面板标题(卡标题 12→11.5), 公式 12→11.5(与 d8_dual 公式一致);
    #   10=正文数据(基准名 9→10、信噪比数字 9.5→10、全部刻度、柱值
    #   标签 9→10); 9=注释(柱内 pt 值 8→9、图例 8.5→9、符号行 8→9)。
    #   左板 x 0.078→0.085(基准名放大让位); 公式卡五元素重排。
    # v10.27: 评审反馈 — 页的主线改为"不同基准信号强度不同 → 按强度分配
    #   话语权": ① 左板数据换第3轮 54 实测点的修复后重放(原 prod2 30 点
    #   表的 gsm8k/math 噪声带先验 bug, 左板"gsm8k 弱"与结果板"w=0.93
    #   最强"自相矛盾 — 评审"讲的是另一个问题"的根源); 第三列改 f
    #   (信号占比, 可为负), 排序按 f 降序; ② 总标题改"话语权按信噪比
    #   分配"; 脚注补数据出处与 f 释义。
    fig = plt.figure(figsize=(13, 3.4))

    # 左 问题: 逐基准 信号 vs 噪声 (f 降序, 上=强) — v10.26: 柱内 pt 值
    #   改右侧三列(信号/噪声/f) — 9pt 柱内标签在信号≈噪声的行垂直
    #   重叠, 列式排版永不打架; 列头彩色(信号 TEAL/噪声 RED)替代图例。
    ax = fig.add_axes([0.085, 0.30, 0.235, 0.50])
    n = len(D8_SNR)
    for i, (bench, sig, noi, f) in enumerate(D8_SNR):
        y = n - 1 - i
        ax.barh(y + 0.20, sig * 100, height=0.34, color=TEAL, zorder=3)
        ax.barh(y - 0.20, noi * 100, height=0.34, color=RED, zorder=3)
        ax.text(4.72, y, f"{sig * 100:.2f}", va="center", ha="right",
                fontsize=9, color=DGRAY, zorder=4)
        ax.text(5.50, y, f"{noi * 100:.2f}", va="center", ha="right",
                fontsize=9, color=DGRAY, zorder=4)
        ax.text(5.58, y, f"{f:+.2f}", va="center", ha="left",
                fontsize=10, fontweight="bold",
                color=TEAL if f >= 0 else RED, zorder=4)
    ax.text(4.72, n - 1 + 0.66, "信号", ha="right", va="center",
            fontsize=9, fontweight="bold", color=TEAL, zorder=4)
    ax.text(5.50, n - 1 + 0.66, "噪声", ha="right", va="center",
            fontsize=9, fontweight="bold", color=RED, zorder=4)
    ax.text(5.58, n - 1 + 0.66, "f", ha="left", va="center",
            fontsize=9, fontweight="bold", color=DGRAY, zorder=4)
    ax.set_xlim(0, 6.3)
    ax.set_ylim(-0.55, n - 1 + 0.95)
    ax.set_yticks(range(n - 1, -1, -1), [r[0] for r in D8_SNR], fontsize=10)
    ax.set_xticks([])
    ax.set_title("问题:基准间信号强度悬殊", fontsize=11.5,
                 fontweight="bold")

    # 中 做法: 公式卡 (话语权按信噪比分配) — v10.24: 四条公式统一
    #   12pt/NAVY/正体(评审"字体不一致" — 斜体变量与正体函数名混排、
    #   13/12pt 与 NAVY/灰混用), 补 σ²_between 定义(评审: 只给了
    #   σ²_noise 的计算式), 公式设计原因进 p9 PPT 备注。
    ax = fig.add_axes([0.365, 0.27, 0.175, 0.56])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch(
        (0.02, 0.03), 0.96, 0.94, transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor="#EDF5F6", edgecolor="#9DC8CC", lw=1.0, zorder=1))
    ax.text(0.5, 0.895, "做法:话语权按信噪比", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=NAVY, zorder=3)
    ax.text(0.5, 0.73,
            r"$\mathrm{w} = \mathrm{clip}\left(\dfrac{1+\mathrm{f}}{2},\; 0.5,\; 1\right)$",
            ha="center", va="center", fontsize=11.5, color=NAVY, zorder=3)
    ax.text(0.5, 0.54,
            r"$\mathrm{f} = 1 - \mathrm{\sigma}^2_{\mathrm{noise}} \,/\, \mathrm{\sigma}^2_{\mathrm{between}}$",
            ha="center", va="center", fontsize=11.5, color=NAVY, zorder=3)
    ax.text(0.5, 0.33,
            r"$\mathrm{\sigma}^2_{\mathrm{noise}} = \dfrac{\mathrm{p}\,(1-\mathrm{p})}{\mathrm{K}_b\,(1-\mathrm{c}_b)^2}$",
            ha="center", va="center", fontsize=11.5, color=NAVY, zorder=3)
    ax.text(0.5, 0.12,
            r"$\mathrm{\sigma}^2_{\mathrm{between}} = \mathrm{Var}_b(\mathrm{acc})$",
            ha="center", va="center", fontsize=11.5, color=NAVY, zorder=3)

    # 右 结果: gsm8k — 第3轮实验 误加权 vs 第5轮实验 改进后 (赢家 vs uniform)
    ax = fig.add_axes([0.615, 0.30, 0.355, 0.50])
    for gi, (glab, win, uni, wc) in enumerate(D8_GSM8K):
        gx = 1.0 + 1.55 * gi
        ax.bar(gx - 0.17, win, width=0.30, color=wc, zorder=3)
        ax.bar(gx + 0.17, uni, width=0.30, color=GRAY, zorder=3)
        tag = " (-56%)" if gi == 0 else " (2.9×)"
        ax.text(gx - 0.17, win + 0.012, f"{win * 100:.1f}%{tag}", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color=wc,
                zorder=4)
        ax.text(gx + 0.17, uni + 0.012, f"{uni * 100:.1f}%", ha="center",
                va="bottom", fontsize=10, fontweight="bold", color=DGRAY,
                zorder=4)
    ax.set_xticks([1.0, 2.55], [r[0] for r in D8_GSM8K], fontsize=10)
    ax.set_xlim(0.4, 3.15)
    ax.set_ylim(0, 0.42)
    ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4],
                  ["0", "10%", "20%", "30%", "40%"], fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("结果:gsm8k — 赢家 vs uniform", fontsize=11.5,
                 fontweight="bold")
    ax.legend(handles=[Patch(color=RED, label="第3轮实验赢家(误加权)"),
                       Patch(color=TEAL, label="第5轮实验赢家(改进后)"),
                       Patch(color=GRAY, label="uniform 对照")],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3,
              fontsize=9, frameon=False)

    fig.text(0.5, 0.95, "话语权按实测信噪比分配:问题(左)· 做法(中)· 结果(右)",
              ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.085,
              "符号:$\\mathrm{p}$ = 实测平均准确率 · $\\mathrm{K}_b$ = 题量 · "
              "$\\mathrm{c}_b$ = 蒙对率 · $\\mathrm{acc}$ = 候选实测分"
              "(去蒙对校正) · $\\mathrm{f}$ = 信号占比(> 0 = 信号盖过噪声)",
              ha="center", va="center", fontsize=9, color=DGRAY)
    fig.text(0.5, 0.028,
              "$\\mathrm{\\sigma}^2_{\\mathrm{noise}}$ = 同一配比重复测量的分数"
              "波动(题量越少,波动越大) · $\\mathrm{\\sigma}^2_{\\mathrm{between}}$ = "
              "候选间实测分差的方差;左图 = 第3轮 54 实测点重放,单位 pt",
              ha="center", va="center", fontsize=9, color=DGRAY)
    ok = verify_layout(fig, "d8")
    path = os.path.join(out_dir, "reinforcement_d8_scoring_snr.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── ③ D19: 外推问题示意图 (非真实数据; v10.5: prod4 数字移入 deck
#    表格, 图幅压扁, 口语标注改专业表述; v1.0.46: 四轮真实数据改造均被
#    评审否决后回退 — 真实数据画不出该形状: 15 维空间太稀疏, 任何一维
#    切片抓不到点云; 本地重放终轮模型外推无上翘(终选 +0.79 仅排第
#    6/112); 服务器拟合曲线未存档, 只留『排第一』的选择记录 — 真实数
#    字全部在 deck 表格与 p10 备注; 画幅 13×3.4/轴高 0.78(原 3.9/0.72:
#    p10 表四行后图带仅 2.43in, 3.9 会被压到 8.1in 宽 < 评审下限 10in;
#    配 deck row_h 0.50 → 图宽 10.2in, 落差保持原图 94%))
#    ────────────────────────────────────────────────────────────────
def chart_d19(out_dir):
    fig = plt.figure(figsize=(13, 3.4))
    ax = fig.add_axes([0.055, 0.10, 0.91, 0.78])

    def true_trend(x):
        # 真实得分: 缓升、见顶、回落 (示意; 纵轴得分, 越高越好)
        return 0.34 + 0.13 * np.exp(-0.5 * ((x - 6.0) / 3.6) ** 2)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def predictor(x):
        # 实测区内贴合真趋势 + 小幅拟合摆动; 实测区右端之外被末端数据
        # 带偏, 冲上虚假高峰 — 高阶拟合外推的典型行为 (示意)
        return true_trend(x) + 0.016 * np.sin(2.6 * x) \
            + 0.21 * sigmoid((x - 8.6) * 3.0)

    xg = np.linspace(0.8, 10.4, 400)
    xs = np.array([1.4, 2.0, 2.6, 3.2, 3.8, 4.4, 5.0, 5.6, 6.2, 6.8, 7.3])
    noise = np.array([0.025, -0.030, 0.018, -0.020, 0.028, 0.010,
                      0.022, -0.008, 0.018, 0.012, 0.020])
    ys = true_trend(xs) + noise
    bi = int(np.argmax(ys))                      # 实测最优 = 最高实测点
    xstar = float(xg[np.argmax(predictor(xg))])  # 论文式终选 = 预测最大值
    ystar = float(predictor(xstar))              # → 落在无数据区 = 虚假高峰
    yreal = float(true_trend(xstar))             # 该点正式实测落在真趋势上

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
    ax.scatter([xstar], [ystar], s=170, color=RED,
               edgecolors="white", linewidth=1.5, zorder=6)
    # 外推点正式实测: 预测虚高的落差 (prod4 式失败)
    ax.plot([xstar, xstar], [yreal, ystar], ":", color=RED, lw=1.6,
            zorder=5)
    ax.scatter([xstar], [yreal], s=150, facecolors="none", edgecolors=RED,
               linewidth=2.0, zorder=6)
    ax.text(xs[bi], 0.55, "实测最优(可信)", ha="center", va="center",
            fontsize=10, fontweight="bold", color=TEAL, zorder=7)
    ax.text(xstar + 0.15, 0.687, "外推『最优点』= 拟合噪声", ha="center",
            va="center", fontsize=10, fontweight="bold", color=RED,
            zorder=7)
    ax.text(xstar, 0.335, "中等规模实测:大幅低于预测", ha="center",
            va="center", fontsize=10, fontweight="bold", color=RED, zorder=7)
    ax.text(4.3, 0.62, "实测过的区域(预测可信)", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=DGRAY, zorder=7)
    ax.text(9.0, 0.22, "从未实测 = 外推(预测不可信)", ha="center",
            va="center", fontsize=10.5, fontweight="bold", color=RED,
            zorder=7)
    ax.set_xlim(0.8, 11.0)
    ax.set_ylim(0.0, 0.72)
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
    print(f"  ✓ {path}  [示意图(非真实数据)| 真实数字在 p10 表格与备注]")
    return ok


# ── ④ 配比全景对比: 赢家 vs 传统方法 vs 池 (v10.4) ───────────────────
def find_run_file(run_dir, name):
    for sub in ("", "state/", "detail/"):
        p = os.path.join(run_dir, sub, name)
        if os.path.isfile(p):
            return p
    return None


def _weights_vec(payload, labels):
    """权重载荷 (标签键 dict | list | {"weights": …}) → 按 labels 的向量。"""
    if isinstance(payload, dict):
        payload = payload.get("weights", payload)
    if isinstance(payload, dict):
        try:
            return np.array([float(payload[l]) for l in labels])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(payload, list) and len(payload) == len(labels):
        return np.array(payload, dtype=float)
    return None


def load_recipes(run_dir, labels):
    """臂 → 15 簇权重向量: climb(终选) ← optimal_mixture_weights.json
    (兜底 topk rank-1);cfg90 / cfg100 ← topk_mixture_candidates.json
    (config_id 90/100, 兜底 search_state.accumulated_configs)。"""
    out = {}
    topk = load_json(find_run_file(run_dir,
                                   "topk_mixture_candidates.json")) or {}
    by_id = {int(c["config_id"]): c for c in topk.get("candidates") or []
             if c.get("config_id") is not None}
    v = _weights_vec(load_json(find_run_file(
        run_dir, "optimal_mixture_weights.json")), labels)
    if v is None:
        r1 = next((c for c in topk.get("candidates") or []
                   if c.get("rank") == 1), None)
        v = _weights_vec((r1 or {}).get("weights"), labels)
    if v is not None and v.sum() > 0:
        out["climb(终选)"] = v / v.sum()
    state_by_id = {}
    state = load_json(find_run_file(run_dir, "search_state.json")) or {}
    for c in state.get("accumulated_configs") or []:
        if c.get("config_id") is not None:
            state_by_id[int(c["config_id"])] = c.get("weights")
    for cid, name in ((90, "climb-cfg90"), (100, "climb-cfg100")):
        v = _weights_vec((by_id.get(cid) or {}).get("weights"), labels)
        if v is None:
            v = _weights_vec(state_by_id.get(cid), labels)
        if v is not None and v.sum() > 0:
            out[name] = v / v.sum()
    return out


def chart_recipe(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    if labels is None:
        print("  [!] 无 macro_info.json — 配比全景图跳过")
        return False
    K = len(labels)
    recipes = load_recipes(run_dir, labels)
    if not recipes:
        print("  [!] optimal_mixture_weights / topk / search_state 均无配方"
              " — 配比全景图跳过")
        return False
    climb_w = recipes.get("climb(终选)")
    order = sorted(range(K),
                   key=lambda i: -(climb_w[i] if climb_w is not None
                                   else shares[i] / 100.0))
    # 语义簇高亮色, 其余统一浅灰蓝 (小份额 floor, 同色不抢焦点)
    hi = {"C10": ORANGE, "C0": TEAL, "C6": BLUE}
    colors = [hi.get(labels[i], "#B9C6D1") for i in range(K)]

    rows = [(name, recipes[name]) for name in
            ("climb(终选)", "climb-cfg90", "climb-cfg100")
            if name in recipes]
    rows.append(("uniform(等权)", np.full(K, 1.0 / K)))
    rows.append(("池自然占比", shares / 100.0))
    n = len(rows)

    fig = plt.figure(figsize=(13, 4.8))
    ax = fig.add_axes([0.105, 0.30, 0.545, 0.52])
    for r, (name, w) in enumerate(rows):
        left = 0.0
        y = n - 1 - r
        for i in order:
            seg = w[i] * 100.0
            if seg <= 0:
                continue
            ax.barh(y, seg, left=left, height=0.62, color=colors[i],
                    edgecolor="white", linewidth=0.4, zorder=3)
            if seg >= 10.0:
                lbl = f"{CLUSTER_NAMES.get(labels[i], labels[i])} {seg:.0f}%"
            elif seg >= 8.0:
                lbl = f"{seg:.0f}%"
            else:
                lbl = None
            if lbl:
                ax.text(left + seg / 2.0, y, lbl, ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color="white",
                        zorder=4)
            left += seg
    ax.set_yticks(range(n - 1, -1, -1), [r[0] for r in rows], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xticks([0, 25, 50, 75, 100], ["0", "25", "50", "75", "100%"],
                  fontsize=8.5)
    ax.set_title("各策略 token 配比(15 个语义簇,按赢家 α 降序)",
                 fontsize=11, fontweight="bold")

    # 右: domainfix — 域标签空间, 不在语义簇空间 (无法映射, 单列)
    ax = fig.add_axes([0.735, 0.30, 0.20, 0.52])
    dc = [ORANGE, BLUE, TEAL, GRAY]
    for j, ((dom, pct), c) in enumerate(zip(DOMAINFIX_DOMAINS, dc)):
        y = len(DOMAINFIX_DOMAINS) - 1 - j
        ax.barh(y, pct, height=0.55, color=c, zorder=3)
        ax.text(pct + 1.5, y, f"{pct:.1f}%".rstrip("0").rstrip("."),
                va="center", ha="left", fontsize=9, fontweight="bold",
                color=DGRAY, zorder=4)
    ax.set_yticks(range(len(DOMAINFIX_DOMAINS) - 1, -1, -1),
                  [d[0] for d in DOMAINFIX_DOMAINS], fontsize=9.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(DOMAINFIX_DOMAINS) - 0.4)
    ax.set_xticks([])
    ax.set_title("domainfix:域标签配比\n(不在语义簇空间)", fontsize=9.5,
                 fontweight="bold")

    fig.text(0.5, 0.955, "最终配比:赢家 vs 传统方法 vs 池自然分布",
              ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.10,
             "赢家把 C10『算术应用题』押到 80%(池占比 5.5%,14.6× 浓缩);"
             "top-3 配方全部 C10-heavy(0.55~0.80)\n"
             "domainfix 定义在域标签空间(数学 0.60),无簇空间 α 向量,"
             "不能与语义簇直接对比 — 右图单列;natural 臂 ≡ 池自然占比",
             ha="center", va="center", fontsize=9.5, color=DGRAY,
             linespacing=1.7)
    ok = verify_layout(fig, "recipe")
    path = os.path.join(out_dir, "recipe_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── ⑤ 簇内容分析: 每个簇分别是什么 (v10.5; v10.6 标签差异化 + 地板分隔;
#    v10.7 每簇标签附 1 篇样本摘录) ───────────────────────────────────
# 数据面: 池份额 = macro_info.json; 内容速写 = cluster_semantics.md
# (recipe_report §2b 生成, cluster_peek 抽样: 域构成前2 / 中位 ch /
# 每簇 1 篇样本开头), 三代解析, 缺件回退域分组常量; 赢家 α =
# load_recipes 的 climb(终选)。
# 标签两行 (v10.8): 第 1 行 = 细鉴名(§4c)或『域 纯度%』+ 中位长度;
# 第 2 行 = 簇关键词(§2b 新节, cluster_peek 对比 IDF 提取)+ 样本开头摘
# 录 — 域名相同的簇靠真实内容词区分。整行按显示宽度预算截断 (CJK 1 单
# 位 / 拉丁 0.5, fontsize 8 下 1 单位 ≈ 0.111in, 总预算 44 单位); 摘录
# 逐字对内置子集字体 cmap 校验 (子集外字符截断, 防豆腐块; 字体子集已扩
# 到全 CJK 基本块, 通常不会截); 用系统全量字体时无校验必要。旧版
# cluster_semantics.md 无关键词节 → 第 2 行退化为纯摘录, 兼容旧归档。
SEM_ROW = re.compile(
    r"^\|\s*(C\d+)\s*\|\s*[\d,]+\s*\|[^|]*\|\s*([^|]*?)\s*\|"
    r"\s*(\d+)\s*\|")
SNIPPET_ROW = re.compile(
    r"^-\s*\*\*(C\d+)\*\*\s*\[[^\]]*\]\s*\([^)]*\):\s*(.+)$")
KEYWORD_ROW = re.compile(
    r"^-\s*\*\*(C\d+)\*\*\s*关键词[:：]\s*(.+)$")


def parse_cluster_semantics(run_dir):
    """cluster_semantics.md (render_brief 表) → {label: {dom1, dom2, med}}。
    dom1/dom2 = 域构成前两名(如『数学 95.7%』/『化学 2.1%』); 容错。"""
    path = find_run_file(run_dir, "cluster_semantics.md")
    if not path:
        return {}
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = SEM_ROW.match(line.strip())
                if not m:
                    continue
                doms = [d.strip() for d in m.group(2).split("/") if d.strip()]
                out[m.group(1)] = {"dom1": doms[0] if doms else "",
                                   "dom2": doms[1] if len(doms) > 1 else "",
                                   "med": int(m.group(3))}
    except (OSError, ValueError):
        return {}
    return out


def parse_cluster_snippets(run_dir):
    """cluster_semantics.md 尾部『每簇 1 篇抽样开头』→ {label: 摘录文本}。"""
    path = find_run_file(run_dir, "cluster_semantics.md")
    if not path:
        return {}
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = SNIPPET_ROW.match(line.strip())
                if m:
                    out[m.group(1)] = m.group(2).strip()
    except OSError:
        return {}
    return out


def parse_cluster_keywords(run_dir):
    """cluster_semantics.md『每簇关键词』节 (v10.8 新) → {label: 词串}。

    格式 = cluster_peek.render_brief 的『- **C0** 关键词: a · b · c』;
    空值 (—) 与旧版无该节的归档都返回空 dict (标签退化, 不报错)。"""
    path = find_run_file(run_dir, "cluster_semantics.md")
    if not path:
        return {}
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = KEYWORD_ROW.match(line.strip())
                if m and m.group(2).strip() not in ("", "—"):
                    out[m.group(1)] = m.group(2).strip()
    except OSError:
        return {}
    return out


def _subset_cmap():
    """内置子集字体的码位覆盖; None = 无子集文件(用系统全量字体, 免校验),
    空 set = 有子集但读不出(保守: 摘录全部弃用)。"""
    p = os.path.join(BUNDLED_FONT_DIR, "wqy-zenhei-deck.ttf")
    if not os.path.isfile(p):
        return None
    try:
        from fontTools.ttLib import TTFont
        have = set()
        for t in TTFont(p)["cmap"].tables:
            if t.isUnicode():
                have.update(t.cmap.keys())
        return have
    except Exception:
        return set()


def _disp_units(s):
    """显示宽度单位: CJK 记 1, 其余 (拉丁/数字/符号) 记 0.5。"""
    return sum(1.0 if ord(ch) >= 0x2E80 else 0.5 for ch in s)


def _excerpt(snips, lab, cmap, budget=40.0):
    """样本摘录 → 『…』; 按显示宽度预算截断 (v10.8: 单位制, 拉丁字符
    半宽 — 英文样本能放下约 2× 字符, 不再只露 3 个单词); 子集字体缺字
    的位置截断; 太短则弃用。"""
    s = snips.get(lab)
    if not s:
        return ""
    if cmap is not None:
        i = 0
        while i < len(s) and (s[i] == " " or ord(s[i]) in cmap):
            i += 1
        s = s[:i].rstrip()
    out, u = "", 0.0
    for ch in s:
        cu = 1.0 if ord(ch) >= 0x2E80 else 0.5
        if u + cu > budget:
            break
        out += ch
        u += cu
    out = out.rstrip()
    return f"『{out}…』" if u >= 5.0 else ""


def _dom_pct(s):
    """『数学 95.7%』→ (『数学』, 95.7); 解析失败 → (原串, None)。"""
    m = re.match(r"^(.*?)\s*([\d.]+)\s*%$", s.strip())
    return (m.group(1).strip(), float(m.group(2))) if m else (s.strip(), None)


# ── 关键发现①: 质量分 × 赢家权重 排序倒挂 (v10.15 评审反馈: 发现块补
#    机理图。prod5 本轮 recipe 质量列全 0(E1 缓存继承, filter=none),
#    数字取自实验档案已核实三簇 — 与 D8_GSM8K 同款硬编码房样) ────────
F1_ROWS = [("C6 代数与数论长文(质量分最高)", 4.45, 0.012, "#8A8A8A"),
           ("C0 形式数学解题", 4.33, 0.081, "#6E9EC4"),
           ("C10 算术应用题(赢家重仓)", 3.88, 0.804, ORANGE)]


def chart_finding_quality(out_dir):
    fig = plt.figure(figsize=(13, 3.8))

    # 左: 质量分(stem_relevance) — 降序, "像不像好文本"
    ax = fig.add_axes([0.10, 0.30, 0.34, 0.52])
    n = len(F1_ROWS)
    for i, (lab, q, _, c) in enumerate(F1_ROWS):
        y = n - 1 - i
        ax.barh(y, q - 3.5, left=3.5, height=0.52, color=c, zorder=3)
        ax.text(q + 0.015, y, f"{q:.2f}", va="center", fontsize=10,
                fontweight="bold", color=DGRAY, zorder=4)
    ax.set_yticks(range(n - 1, -1, -1),
                  ["C6 代数与数论长文\n(质量分最高)", "C0 形式数学解题",
                   "C10 算术应用题\n(赢家重仓)"], fontsize=9.5)
    ax.set_xlim(3.5, 4.7)
    ax.set_ylim(-0.55, n - 1 + 0.6)
    ax.set_xticks([3.5, 4.0, 4.5], ["3.5", "4.0", "4.5"], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("质量分:『像不像好文本』", fontsize=11.5, fontweight="bold")

    # 右: 赢家配方权重 α — 与质量分倒挂, "喂了有没有用"
    ax = fig.add_axes([0.565, 0.30, 0.35, 0.52])
    for i, (lab, q, w, c) in enumerate(F1_ROWS):
        y = n - 1 - i
        ax.barh(y, w, height=0.52, color=c, zorder=3)
    ax.get_yaxis().set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.55, n - 1 + 0.6)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0],
                  ["0", "0.25", "0.5", "0.75", "1"], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title("climb 赢家权重:『喂了有没有用』", fontsize=11.5,
                 fontweight="bold")
    ax.text(0.012 + 0.012, 2, "0.1× 地板", va="center", fontsize=9,
            color=DGRAY, zorder=4)
    ax.annotate("cfg100 反例:另押 C6 0.16\n→ math 反而减半(2.60% vs 5.80%)",
                xy=(0.175, 1.75), xytext=(0.42, 2.12), fontsize=8.5,
                color=RED, ha="left", va="center", linespacing=1.5,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.0))
    ax.text(0.081 + 0.015, 1, "0.08", va="center", fontsize=9.5,
            fontweight="bold", color=DGRAY, zorder=4)
    ax.text(0.804 + 0.015, 0, "0.80(池占比的 14.6×)", va="center",
            fontsize=10, fontweight="bold", color=ORANGE, zorder=4)

    fig.text(0.5, 0.94, "发现①:质量分与训练价值 — 排序完全倒挂",
             ha="center", va="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.05, "质量分:experiment_prod5.md §4c 档案三簇(stem_relevance;"
             "本 run recipe 质量列全 0,E1 缓存继承);权重:optimal_mixture_weights.json",
             ha="center", va="center", fontsize=8, color=DGRAY)
    ok = verify_layout(fig, "finding_quality")
    path = os.path.join(out_dir, "finding_quality_value.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


# ── 关键发现③: 池结构机理示意 (通用池天然成团 vs 窄域池一条流形;
#    v10.15 评审反馈: 发现块补机理图。纯示意, 无数据主张) ────────────
def chart_finding_manifold(out_dir):
    rng = np.random.default_rng(7)
    # v10.16 评审反馈: 原图小、字小难看 — 图幅压扁(13×3.0, 上页更大幅面)、
    # 全字号上调、流形加软带底、面板微底色; 纯示意, 无数据主张
    fig = plt.figure(figsize=(13, 3.0))

    # 左: 通用池(论文场景) — 多域天然成团
    ax = fig.add_axes([0.05, 0.15, 0.42, 0.64])
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7FBFB")
    blobs = [(0.20, 0.70, TEAL), (0.55, 0.78, NAVY),
             (0.80, 0.60, "#8A8A8A"), (0.38, 0.24, "#6CD9DE")]
    for cx, cy, c in blobs:
        pts = rng.normal(0, 0.042, (120, 2)) + [cx, cy]
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=c, alpha=0.55, zorder=3)
        el = plt.matplotlib.patches.Ellipse(
            (cx, cy), 0.25, 0.23, fill=False, ls="--", color=GRAY, lw=1.2,
            zorder=2)
        ax.add_patch(el)
    ax.text(0.5, 0.98, "通用池(论文):多域天然成团", ha="center", va="top",
            fontsize=12.5, fontweight="bold", color=NAVY, zorder=4)
    ax.text(0.5, 0.03, "域间有间隙 → 距离合并正好切在域边界上",
            ha="center", va="bottom", fontsize=10.5, color=DGRAY, zorder=4)

    # 右: 窄域池(我们,116M 篇 STEM) — 一条连续流形 + 格式孤岛
    ax = fig.add_axes([0.53, 0.15, 0.42, 0.64])
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F7FBFB")
    t = np.linspace(0.05, 0.95, 800)
    ym = 0.5 + 0.15 * np.sin(2 * np.pi * t * 0.9)
    ax.fill_between(t, ym - 0.11, ym + 0.11, color=TEAL, alpha=0.14,
                    zorder=2)
    ax.scatter(t, ym + rng.normal(0, 0.032, 800), s=5, color=TEAL,
               alpha=0.35, zorder=3)
    islands = [(0.10, 0.82, 14), (0.30, 0.90, 8), (0.86, 0.80, 10),
               (0.80, 0.14, 12), (0.38, 0.10, 7)]
    for ix, iy, k in islands:
        pts = rng.normal(0, 0.014, (k, 2)) + [ix, iy]
        ax.scatter(pts[:, 0], pts[:, 1], s=7, color=GRAY, alpha=0.8,
                   edgecolors="none", zorder=3)
    ax.text(0.045, 0.87, "SMILES", fontsize=9, color=DGRAY, zorder=4)
    ax.text(0.875, 0.845, "代码", fontsize=9, color=DGRAY, zorder=4)
    ax.text(0.835, 0.185, "表格", fontsize=9, color=DGRAY, zorder=4)
    ax.annotate("一条连续流形(99% tokens)", xy=(0.45, 0.60), xytext=(0.5, 0.99),
                ha="center", va="top", fontsize=12.5, fontweight="bold",
                color=NAVY, zorder=4,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.2))
    el = plt.matplotlib.patches.Ellipse(
        (0.5, 0.5), 1.04, 0.74, fill=False, ls="--", color=RED, lw=1.4,
        zorder=2)
    ax.add_patch(el)
    ax.text(0.5, 0.02, "+ 13 个格式孤岛(标注 3 个)\n"
            "距离合并链式塌缩:任意 K 都并成一簇",
            ha="center", va="bottom", fontsize=10, color=RED, zorder=4,
            linespacing=1.5)

    fig.text(0.5, 0.95, "池结构:通用池天然成团 vs 窄域池一条流形",
             ha="center", va="center", fontsize=13.5, fontweight="bold")
    fig.text(0.5, 0.02, "示意(点云为概念示意,非嵌入数据);实证:"
             "paper_deviations.md D14(K=14/21/24/32 → 最大簇 99.0/98.5/98.3/97.5%)",
             ha="center", va="center", fontsize=9, color=DGRAY)
    ok = verify_layout(fig, "finding_manifold")
    path = os.path.join(out_dir, "finding_pool_manifold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}")
    return ok


def chart_cluster_content(out_dir, run_dir):
    labels, shares = cluster_token_shares(run_dir)
    if labels is None:
        print("  [!] 无 macro_info.json — 簇内容图跳过")
        return False
    K = len(labels)
    sem = parse_cluster_semantics(run_dir)
    alpha = load_recipes(run_dir, labels).get("climb(终选)")
    # 地板结构判定 (v10.6): 多数簇权重相同且 ≪ max → 赢家下注簇/地板簇
    # 分隔线才成立; 结构不成立(如无 floor 的 fixture)不画, 防误标
    above, show_sep = [], False
    if alpha is not None:
        floor_v = float(np.median(alpha))
        n_floor = int(np.sum(np.isclose(alpha, floor_v, rtol=1e-6)))
        above = [int(i) for i in np.where(alpha > 3.0 * floor_v)[0]]
        show_sep = (len(above) >= 1 and n_floor >= 3
                    and len(above) + n_floor >= K
                    and floor_v < 0.25 * float(alpha.max()))
    # 行序: 赢家 α 降序 (无 α 按池份额) — 搜索判决权重高的簇排上面
    order = sorted(range(K),
                   key=lambda i: (-(alpha[i] if alpha is not None else 0.0),
                                  -shares[i]))
    xmax = max(float(shares.max()) * 1.22, 4.0)
    # v10.14: 关键词/摘录不再上图 (parse_cluster_* 与 _excerpt 留给
    # cluster_peek/旧格式回退路径, 此处不再调用)
    # v10.8: 图加宽 (13→14.4) + 左标签区扩到 0.445 — 第 2 行 = 关键词 +
    # 样本摘录 (总预算 44 单位 ≈ 4.9in), 无关键词时整行预算给摘录
    # v10.10: 摘录移出图面 (评审反馈: 只有开头读者看不出所以然) —
    # 第 2 行只留关键词, 标签区缩到 0.20、条形区放宽到 0.62;
    # 完整摘录仍在 §2b (cluster_semantics.md), 旧格式文件无关键词时
    # 退回摘录 (优雅降级)
    # v10.14: 关键词也移出图面 (评审反馈: 只留簇的文字描述更易读) —
    # 标签单行 = 簇名 + 中位长度, 标签区 0.17、字号 8→9;关键词与样本
    # 摘录全部住在 §2b 档案
    fig = plt.figure(figsize=(14.4, 6.2))
    ax = fig.add_axes([0.17, 0.08, 0.65, 0.76])
    ylabels = []
    for r, i in enumerate(order):
        y = K - 1 - r
        lab = labels[i]
        dom = CLUSTER_DOMAIN.get(lab)
        si = sem.get(lab)
        name = CLUSTER_NAMES.get(lab)
        if si and si["dom1"] not in ("", "—"):
            # 命名簇用鉴定名; 其余 = 域纯度(混合簇给双域) + 中位长度
            d1, p1 = _dom_pct(si["dom1"])
            d2, p2 = _dom_pct(si["dom2"]) if si["dom2"] else ("", None)
            mix = f"{d1} {p1:.0f}%" if p1 is not None else d1
            if p2 is not None and p2 >= 20.0:
                mix += f"·{d2} {p2:.0f}%"
            med = f" · 中位 {si['med']:,} ch"
            line1 = f"{lab} {name}{med}" if name else f"{lab} {mix}{med}"
        else:
            line1 = f"{lab} {name or (f'{dom}簇' if dom else '')}"
        ylabels.append(line1)
        ax.barh(y, shares[i], height=0.62,
                color=DOMAIN_COLORS.get(dom, GRAY), zorder=3)
        ax.text(shares[i] + xmax * 0.012, y, f"{shares[i]:.1f}%",
                va="center", fontsize=8.5, color=DGRAY, zorder=4)
    if show_sep:
        y_sep = K - 1 - len(above) + 0.5
        ax.axhline(y_sep, ls=(0, (4, 3)), color=GRAY, lw=0.9, zorder=2)
    ax.set_yticks(range(K - 1, -1, -1), ylabels, fontsize=9)
    # 摘录含真实文档文本: $..$ 会被 matplotlib 当 mathtext 解析, 摘录里
    # 的 LaTeX 命令 (\spn 等) 直接触发 ParseFatal 崩溃 (服务器 2026-10-14
    # 实测) — 逐标签关闭数学解析, 摘录原样展示; 老 matplotlib 无该 API
    # 时退回 rcParam 全局关 (本脚本各图均不用 mathtext, 全局关无副作用)
    _tick_labels = ax.get_yticklabels()
    if all(hasattr(t, "set_parse_math") for t in _tick_labels):
        for t in _tick_labels:
            t.set_parse_math(False)
    else:
        try:
            plt.rcParams["text.parse_math"] = False
        except KeyError:
            pass
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.7, K - 0.1)
    ax.set_xticks([])
    ax.set_xlabel("池 token 占比(%)", fontsize=9.5)
    ax.set_title(f"搜索空间:{K} 个语义簇 — 内容构成与体量", fontsize=12.5,
                 fontweight="bold", pad=30)
    ax.legend(handles=[Patch(color=c, label=f"{d}簇")
                       for d, c in DOMAIN_COLORS.items()],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=5,
              fontsize=9, frameon=False)
    # 右列: 赢家配方权重 α (= 搜索对每个簇的判决); 下注簇高亮, 地板簇置灰
    if alpha is not None:
        tr = ax.get_yaxis_transform()
        ax.text(1.045, K - 0.42, "赢家 α", transform=tr, ha="left",
                va="center", fontsize=9.5, fontweight="bold", color=NAVY)
        amax = int(np.argmax(alpha))
        for r, i in enumerate(order):
            y = K - 1 - r
            hot = i in above
            ax.text(1.045, y, f"{alpha[i]:.3f}", transform=tr, ha="left",
                    va="center", fontsize=9.5 if hot else 8,
                    fontweight="bold" if hot else "normal",
                    color=ORANGE if i == amax else (TEAL if hot else GRAY))
        if show_sep:
            ax.text(1.045, K - 1 - len(above) + 0.5,
                    f"以下 {K - len(above)} 簇全部压到 0.1× 地板",
                    transform=tr, ha="left", va="center", fontsize=7.5,
                    color=DGRAY, zorder=5)
    src = ("内容速写:cluster_semantics.md(§2b,cluster_peek 抽样鉴定;"
           "簇名 = 人工鉴定;关键词/样本摘录亦存 §2b)" if sem
           else "未找到 cluster_semantics.md — 域构成按 §4c 常量回退")
    fig.text(0.5, 0.008,
              f"{src};α = climb 终选配方权重(optimal_mixture_weights.json);"
              "域分组出处 experiment_prod5.md §4c",
              ha="center", va="bottom", fontsize=8.5, color=DGRAY)
    ok = verify_layout(fig, "cluster_content")
    path = os.path.join(out_dir, "cluster_content_map.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✓ {path}"
          + (f"  [内容速写: {len(sem)}/{K} 簇]"
             if sem else "  [回退: 无 cluster_semantics.md]"))
    return ok


def deck_dir_for(run_dir):
    """汇报包目录 = run 目录下的 deck/ 子目录 (v10.2 分离实验结果与汇报
    产物; 用户裁决: deck 跟着 run 走)。make_deck.py 有同名同实现副本 —
    不互相 import, 免得把 numpy/matplotlib/pptx 依赖拉进
    对方的裸跑路径。"""
    return os.path.join(os.path.normpath(run_dir), "deck")


def main():
    ap = argparse.ArgumentParser(
        description="『四处机制改进优化』举证图 (D14/D8×2/D19) + 配比全景 + 簇内容"
                    " → 汇报包目录 <run>/deck")
    ap.add_argument("run_dir", help="run 目录 (prod5 归档; 读取其中的 "
                    "macro_info.json 等, 图输出到其 deck/ 子目录)")
    ap.add_argument("-o", "--out-dir", default=None,
                    help="输出目录 (默认 = <run_dir>/deck)")
    ap.add_argument("--font", default=None,
                    help="显式指定中文字体文件 (.ttf/.ttc/.otf), 覆盖自动检测")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        sys.exit(f"✗ 目录不存在: {args.run_dir}")
    name = setup_cjk_font(args.font)
    if name is None:
        sys.exit(
            "✗ 未找到中文字体 — 仓库内置字体 (scripts/diagnostics/fonts/) "
            "也没找到, 无法生成中文标签图。任选其一后重跑:\n"
            "  1) git pull 更新代码 — 内置字体随仓库分发, 通常 pull 一次即好\n"
            "     (若 pull 后仍报此错, 把 git pull 的输出贴回)\n"
            "  2) 显式指定: python3 scripts/diagnostics/reinforcement_charts.py "
            "<run_dir> --font /path/to/font.ttf\n"
            "     (找字体: find /usr/share/fonts /opt ~/.fonts -name '*.tt[fc]' "
            "-o -name '*.otf' 2>/dev/null)\n"
            "  3) 把任一 CJK 字体文件拷到 ~/.fonts/ (自动扫描)\n"
            "  4) 有 root 的话: sudo apt-get install -y fonts-noto-cjk")
    print(f"✓ 中文字体: {name}")
    out_dir = args.out_dir or deck_dir_for(args.run_dir)
    os.makedirs(out_dir, exist_ok=True)
    print(f"『四处机制改进优化』举证图 + 配比全景 + 簇内容 → {out_dir} "
          "(汇报包, 与实验结果目录分离)")
    print("  数据出处: paper_deviations.md D14 / experiment_prod2_k15bal.md "
          "§4.2 / TODO.md prod3 记录 / experiment_prod5.md §3b·§4c / "
          "experiment_prod4.md §3d·§4e / cluster_semantics.md (§2b)")
    oks = [chart_d14(out_dir, args.run_dir),
           chart_d8_dual(out_dir),
           chart_d8(out_dir),
           chart_d19(out_dir),
           chart_recipe(out_dir, args.run_dir),
           chart_cluster_content(out_dir, args.run_dir),
           chart_finding_quality(out_dir),
           chart_finding_manifold(out_dir)]
    if not all(oks):
        sys.exit("✗ 有图未通过标签重叠自检 — 请把上面的『重叠/越界』行贴回来")
    print("下一步: python3 scripts/diagnostics/make_deck.py <run_dir>")


if __name__ == "__main__":
    main()
