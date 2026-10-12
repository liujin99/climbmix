#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_deck.py — prod5 → 数据团队汇报 PPTX 生成器 (2026-10-12)。

用法 (服务器, run 目录含 report_charts 的图):
    python3 scripts/diagnostics/make_deck.py result/prod5_20260929_201108
    python3 scripts/diagnostics/make_deck.py <run_dir> --check    # 只验图
    python3 scripts/diagnostics/make_deck.py <run_dir> -o out.pptx

依赖: python-pptx (pip install python-pptx; 若报 PEP 668 加
--break-system-packages 或用 venv)。图表按三代布局解析 (root → state/
→ detail/); 文案 = 本文件 SLIDES 常量 (审稿在对话里, 落地在这里)。
输出默认 <run_dir>/<basename>_deck.pptx, 16:9, 中文字体 Microsoft YaHei。
"""
import argparse
import os
import struct
import sys

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
    from pptx.util import Emu, Inches, Pt
    from pptx.oxml.ns import qn
except ImportError:
    sys.exit("✗ 缺 python-pptx — pip install python-pptx (PEP 668 环境加 "
             "--break-system-packages), 装好后重跑本脚本")

FONT = "Microsoft YaHei"
NAVY = RGBColor(0x1F, 0x4E, 0x79)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
DARK = RGBColor(0x33, 0x33, 0x33)
GRAY = RGBColor(0x8A, 0x8A, 0x8A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
PAGE_W, PAGE_H = Inches(13.333), Inches(7.5)
ML = Inches(0.6)                       # 左右页边距
CW = PAGE_W - 2 * ML                   # 内容宽

CHARTS_USED = ["search_convergence.png", "arms_main_results.png",
               "proxy_target_consistency.png", "cluster_alpha_vs_score.png",
               "reinforcement_d14_cluster_balance.png",
               "reinforcement_d8_scoring_snr.png",
               "reinforcement_d19_final_selection.png"]

# ── 幻灯片文案 (v9: 『三处加固』三页 = 真·数据图 (reinforcement_charts.py
#    生成, 图内标签中文 + 标签重叠自检): ① 簇份额双环形图(巨簇 vs 15 簇
#    真实份额) ② 信号 vs 噪声条形 + prod3 事故逐基准 Δ ③ prod4 实测 vs
#    外推对决 + uniform 家族带; 页面 = bullets(讲机理)+ 图(举证)+ caption;
#    arm 名保留原文, 其余图内文字中文;
#    **…** = 段内加粗; 数字已对 experiment_prod2/prod4/prod5.md、
#    scoring_metric_design.md、paper_deviations.md D8/D14/D19) ──
SLIDES = [
    dict(kind="title",
         title="训练数据怎么配?让算法自己搜",
         subtitle="CLIMB 数据配比搜索 · prod5 实验结果",
         meta="数据团队汇报 · 2026-10",
         stats=[("+0.043", "STEM 平均分提升(6 项基准全胜)"),
                ("10.4×", "数学应用题 gsm8k:0.029 → 0.301"),
                ("88.5%", "赢家训练量集中在 2 个数学簇")]),
    dict(kind="table", title="从手调到搜索",
         takeaway="quadmix 已证明配比重要;系统搜索第一次给出可复现、可解释的答案",
         table=(["配比策略", "STEM", "Δ vs 不训练", "结论(vs uniform)"],
                [["base(不继续训练)", "0.1752", "—", "对照锚点"],
                 ["uniform(15 簇等权)", "0.1657", "−0.0095", "参考基准"],
                 ["domainfix(手调域配比)", "0.1773", "+0.0021", "无显著差异(z=+1.0)"],
                 ["climb(系统搜索)", "0.2086", "+0.0334", "显著胜出(z=+3.8)"]]),
         emph=[3], y=2.0, row_h=0.52,
         widths=[0.34, 0.12, 0.16, 0.38],
         note="同数据池 · 同底座 · 同预算(每臂 3B tokens)— **增益全部来自『喂哪些数据』,不来自多训练**"),
    dict(kind="table", title="结论先行",
         takeaway="三个搜索配方全部显著胜出 — 只多训练、不挑数据,一分不涨",
         table=(["配方", "STEM", "Δ vs uniform", "z", "结论"],
                [["climb(搜索终选)", "0.2086", "+0.0429", "+3.80", "显著胜出"],
                 ["climb-cfg90", "0.2078", "+0.0421", "+3.67", "**显著胜出**"],
                 ["climb-cfg100", "0.2013", "+0.0356", "+3.16", "**显著胜出**"],
                 ["domainfix", "0.1773", "+0.0116", "+1.03", "无显著差异"],
                 ["natural", "0.1723", "+0.0067", "+0.59", "无显著差异"],
                 ["uniform(参考)", "0.1657", "(基准)", "—", "—"]]),
         emph=[0], y=1.9, row_h=0.46,
         widths=[0.26, 0.14, 0.2, 0.12, 0.28],
         note="climb = 搜索终选配方(cfg#101);cfg90 / cfg100 = 搜索实测 top-3 的另两个配方;"
              "domainfix = 手调域配比(数学 0.60);natural = 池自然占比;uniform = 15 簇等权 — 各页同此;"
              "climb 对 uniform **6 项基准全胜**(符号检验 p=0.016,相当于连掷 6 次硬币全是正面);"
              "三个非搜索臂都没显著超过 base(0.1752)"),
    dict(kind="flow", title="怎么做的 — CLIMB:把『选数据』变成『搜参数』",
         takeaway="配比不再靠人拍:15 个簇权重当作可搜索的参数,小模型快速试错,学习器指路,逐轮逼近最优",
         steps=[
             ("① 语义分簇", "data",
              "1.16 亿篇 STEM 文档按内容自动聚成 15 个主题簇(统计练习题 / 形式数学 / 综述长文…)",
              "『选哪些文档』→ 调 15 个数字"),
             ("② 小模型当替身", "d20",
              "小模型(约为正式模型的 1/3)按候选配比短训,在 6 项基准上打分",
              "每配比 640M tokens ≈ 全量 1/5 · ~4h"),
             ("③ 学习器指路", "lgbm",
              "LightGBM 学『配比 → 得分』规律,下一轮集中测有希望的配比,混入随机点防走偏",
              "3 轮 64/32/16 ≈ 115 个实测配比"),
             ("④ 正式规模复验", "d28",
              "最优配方交正式模型全量训练,与全部基线同台对比",
              "每臂 3B tokens · 9,110 题"),
         ],
         loop_label="得分反馈给下一轮采样",
          note="CLIMB = NVIDIA 论文方法(arXiv:2504.13161,聚类 + 迭代搜索数据配比);"
               "我们在其机制不设防的三处各加一道闸 — 见后文『三处加固』三页"),
    dict(kind="image", title="搜索过程:便宜试错 + 昂贵裁决",
         takeaway="两层设计:小模型海量试错(搜索),正式模型只对决赛配方一锤定音(验证)",
         bullets=[
             ("搜索轨迹", "第 1 轮随机撒网,第 2 轮起学习器引导 — **候选逐步向高分区域收拢,3 轮收敛**"),
             ("验证规模", "正式规模只跑 **6 个训练臂(共 18B tokens)+ 1 个不训练锚点** — "
                       "海量试错不烧大模型的钱"),
         ],
         image="search_convergence.png",
         caption="115 个候选配比的得分轨迹(各轮最佳连成折线)"),
    dict(kind="image", title="主结果",
         takeaway="三个 CLIMB 配方全部显著胜出,最高比 uniform 高 26%(0.209 vs 0.166)",
         image="arms_main_results.png",
         caption="同预算对比(每臂 3B tokens);误差棒逐题实算(9,110 题)— 臂间间距远大于误差棒"),
    dict(kind="table", title="全基准得分总表",
         takeaway="三个 CLIMB 配方包揽生成式两项前三;选择题各臂打平,最高分反而是 base(不训练)— 净差距全部来自『要写过程』的题",
         table=(["基准(题量)", "climb", "climb-cfg90", "climb-cfg100", "domainfix", "natural", "uniform", "base"],
                [["arc_easy(2376·选择)", "0.7294", "0.7247", "0.7214", "0.7273", "0.7210", "0.7269", "**0.7424**"],
                 ["arc_challenge(1172·选择)", "0.4360", "0.4403", "0.4471", "0.4283", "0.4334", "0.4334", "**0.4608**"],
                 ["mmlu_stem(3545·选择)", "0.2920", "0.2905", "**0.3016**", "0.2928", "0.2905", "0.2872", "0.2979"],
                 ["gpqa_diamond(198·选择)", "0.2121", "0.2475", "0.2172", "0.2323", "0.2475", "0.2071", "**0.2626**"],
                 ["gsm8k(1319·写过程)", "**0.3010**", "0.2593", "0.2654", "0.1281", "0.0948", "0.1054", "0.0288"],
                 ["math(500·写过程)", "**0.0580**", "0.0500", "0.0260", "0.0280", "0.0160", "0.0160", "0.0040"],
                 ["STEM 平均(centered)", "**0.2086**", "0.2078", "0.2013", "0.1773", "0.1723", "0.1657", "0.1752"]]),
         emph=[6], y=1.85, row_h=0.42, size=11,
         widths=[0.26] + [0.1057] * 7,
         note="加粗 = 行最优;数字与 report.md §3 完全同源(逐题 ±SE 明细见 report);"
              "选择题 4 行的行最优都是 base — 3B 继续训练的共性回退,全部臂同向,与配方无关"),
    dict(kind="table", title="提升来自哪里 — 对最强传统方法",
         takeaway="逐基准跟最强传统方法比:大赢的全是『要写过程的数学题』(2×+),选择题全部打平 — 净增益干净落在推理能力",
         table=(["基准(题量)", "climb", "最强传统方法", "Δ", "结论"],
                [["gsm8k 应用数学(1319·写过程)", "0.3010", "0.1281(domainfix)", "+0.173", "2.3× 大赢"],
                 ["math 竞赛数学(500·写过程)", "0.0580", "0.0280(domainfix)", "+0.030", "2.1× 大赢"],
                 ["mmlu_stem(3545·选择)", "0.2920", "0.2928(domainfix)", "−0.001", "打平"],
                 ["arc_easy(2376·选择)", "0.7294", "0.7273(domainfix)", "+0.002", "打平"],
                 ["arc_challenge(1172·选择)", "0.4360", "0.4334(uniform)", "+0.003", "打平"],
                 ["gpqa_diamond(198·选择)", "0.2121", "0.2475(natural)", "−0.035", "打平(噪声内)"]]),
         emph=[0, 1], y=1.9, row_h=0.46,
         widths=[0.33, 0.12, 0.24, 0.12, 0.19],
         note="传统方法 = 非搜索配比:domainfix(手调域配比)/ natural(池自然占比)/ uniform(15 簇等权),"
              "逐基准取最强;选择题 Δ 全部 ≪ SE(逐题 SE 见 report §3);vs base:gsm8k **10.4×**、math **14.5×**"),
    dict(kind="image", title="小模型替身靠谱吗",
         takeaway="搜索在替身上选中的配方,正式模型全部兑现 — 替身没有骗我们",
         bullets=[
             ("选中 → 兑现", "替身分挑出的 3 个决赛配方,正式训练**全部显著胜出**"),
             ("重搜一遍,结果一样", "冷启动(115 个配比全新采样)重搜:0.209 vs 上代热启动 0.214,"
                                "落在种子噪声带(±0.016~0.032)内 — **两次独立搜索抵达同一水平**"),
         ],
         image="proxy_target_consistency.png",
         caption="climb 配方:替身分 vs 正式实测"),
    dict(kind="image", title="三处加固 ①:簇大小强制均衡",
         takeaway="论文的合并把 99% 数据塌进一个巨簇,配比只剩 1 个真旋钮 — 强制均衡后 15 个旋钮全部可用",
         bullets=[
             ("为什么翻车", "我们的池是一整块连续流形,论文的『相近合并』链式塌缩 — "
                        "**巨簇吃掉 99.05%,全部 35 个候选配方被钉死,配比只剩 1 个真旋钮**"
                        "(调 K=14/21/24/32 巨簇都 ≥97.5%,怎么调都无解)"),
             ("怎么修", "细簇 k-means++ 锚点 + 容量约束:恰好 15 簇、单簇 ≤8.8% — "
                     "**搜索从 prod1 −0.0036(输)翻到 prod2 +0.0102(首次转正)**;"
                     "prod5 赢家更把 C10 重仓到 80%(池占比的 14.6×)"),
         ],
         image="reinforcement_d14_cluster_balance.png",
         caption="左:论文距离合并的实测构成(巨簇 99.05% + 13 个孤岛 0.95%);"
                 "右:均衡划分后的真实簇份额(读自 run 目录);决策留档 D14"),
    dict(kind="image", title="三处加固 ②:评分按信噪比加权",
         takeaway="评分错一位,方向错一片:搜索会押『好预测的文本』而非『会做题』 — 信噪比定话语权,NLL 只补位",
         bullets=[
             ("为什么加权", "题量与难度不同 → 抽样噪声不同:math 的候选间真信号只有噪声的 1/11 — "
                        "**信号小于噪声的基准,accuracy 纯属瞎猜;按信噪比给每个基准定话语权**"),
             ("不加权会怎样", "prod3 评分 bug 把最强真信号的权重归零,搜索转而讨好『好预测的文本』 — "
                        "**赢家把 gsm8k 砍半(−56%,z=−6.2)、math −78%,选择题却持平:平均分掩盖了方向性塌方**;"
                        "修复 = 实测噪声地板 + w_floor=0.5(NLL 只补位)→ gsm8k 做到 0.301"),
         ],
         image="reinforcement_d8_scoring_snr.png",
         caption="左:逐基准『真信号 vs 抽样噪声』(prod2 · 30 候选实测,SNR = 信号/噪声);"
                 "右:prod3 搜索赢家 vs 对照臂的逐基准变化;决策留档 D8"),
    dict(kind="image", title="三处加固 ③:终选守卫",
         takeaway="外推的冠军不可信:上版实测反输自家实测点 0.017 — 终选必须是实测第一,外推须大幅领先才许上场",
         bullets=[
             ("为什么翻车", "论文让预测器外推选冠军 — prod4 同轮实测:"
                        "**从未训练过的『外推冠军』(0.1972)输给自家实测点 cfg72(0.2142)0.017(z≈2.0)**,热区垫底"),
             ("怎么修", "外推照跑留审计,但预测领先 ≤ max(0.30, 留出残差 σ) 就自动降级实测第一 — "
                     "**prod5:climb = cfg#101 = 实测第一,0.2086、6/6 全胜,top-3 全部显著胜出,"
                     "prod4 式反例构造性不可复现**"),
         ],
         image="reinforcement_d19_final_selection.png",
         caption="prod4 同轮对决(d28 正式规模):两个实测点 vs 两个从未训练的外推点;"
                 "灰带 = uniform 家族(.1647~.1833);决策留档 D19"),
    dict(kind="image", title="赢家是谁",
         takeaway="搜索把 80% 训练量押给一个簇:统计练习题 — 全池仅占 5.5%,相当于 14.6 倍浓缩",
         bullets=[
             ("赢家配方", "80% 给『统计练习题』簇 + 8% 给『形式数学解题』簇,其余 13 簇全压保底"
                       "(每簇 ≈ 0.9%)— **两个簇占 88.5%**"),
             ("结论稳健", "排名前 3 的配方全都重仓同一簇(权重 0.55~0.80)— **不是单点运气**"),
         ],
         image="cluster_alpha_vs_score.png",
         caption="赢家配方的簇权重分布(逐簇 α 与得分)"),
    dict(kind="table", title="尖锐发现:文本质量分 ≠ 训练价值",
         takeaway="赢家重仓的簇质量分反而最低 — 质量分衡量『像不像好文本』,不衡量『喂了有没有用』",
         table=(["簇", "质量分(相关性 / 知识性)", "在配方中的角色", "实测效果"],
                [["统计练习题(赢家重仓)", "3.88 / 3.65 — 三簇最低", "权重 0.80(池占比的 14.6×)",
                  "gsm8k 2.9× / math 3.6×"],
                 ["形式数学解题", "4.33 / 4.12", "权重 0.08", "与赢家并列的机制解释"],
                 ["综述长文(质量分最高)", "4.45 / 4.07", "climb-cfg100 押注(0.16)",
                  "cfg100 的 math 直接减半(0.026 vs 0.058)"]]),
         emph=[0], y=1.95, row_h=0.62, size=12, widths=[0.24, 0.25, 0.24, 0.27],
         note="同样是短练习题:**化学练习没用(权重 0.1×),统计练习大赚(14.6×)— "
              "押对领域,比押『高分文本』重要**"),
    dict(kind="table", title="怎么保证没骗自己",
         takeaway="规则先于结果、误差逐题实算、结论独立复现 — 四道检查全部量化、全部在案",
         table=(["检查", "数值", "判定"],
                [["评测对账(换环境复评)", "同一底座复评:0.1752 vs 基准 0.1738(差 0.0014)",
                  "**PASS**(阈值 0.002)"],
                 ["误差逐题实算", "全部 6 臂 × 9,110 题,逐题二项分布算误差,不用默认值",
                  "**所有结论带 z / p**"],
                 ["独立复现", "两次独立搜索:0.209 vs 0.214(种子噪声带 ±0.016~0.032)",
                  "**同一水平**"],
                 ["规则先于结果", "判决标准(V1–V5,含加冕规则)实验前写死,留档可查",
                  "**按规则执行**"]]),
         y=1.95, row_h=0.62, size=12.5, widths=[0.24, 0.5, 0.26],
         note="四者齐备,结论才出手;原始数据与判决记录全部在案,可复查"),
    dict(kind="stats", title="这意味着什么",
         takeaway="配比是一个可搜索、可复现、可解释的旋钮 — 这次把它拧对了",
         stats=[("+0.043", "STEM 平均分提升(6 项基准全胜)"),
                ("2 次", "独立搜索同水平复现(0.209 / 0.214)"),
                ("≈3,900 NPU·h", "全部实验总开销(约 163 张卡 · 天)")],
         note="赢家配方可直接复用到下一版训练;自然延伸:更大规模验证 + 质量信号协作 — 值得一起做"),
]


def resolve_chart(run_dir, name):
    """三代布局: root → state/ → detail/ (图表历史上一直住根层, 兼容为主)。"""
    for sub in ("", "state/", "detail/"):
        p = os.path.join(run_dir, sub, name)
        if os.path.isfile(p):
            return p
    return None


def png_size(path):
    """PNG IHDR 宽高 (零依赖, 不赌 PIL)。"""
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return struct.unpack(">II", head[16:24])


def _run(p, text, size, bold=False, color=DARK):
    from lxml import etree
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = FONT
    rPr = r._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rPr, qn("a:ea"))
    ea.set("typeface", FONT)
    return r


def _md(p, text, size, color, base_bold=False, bold_color=None):
    """**加粗** 分段: 奇数段加粗(bold_color 给强调色), 其余原样。"""
    for i, seg in enumerate(str(text).split("**")):
        if seg:
            _run(p, seg, size, bold=base_bold or i % 2 == 1,
                 color=bold_color if (bold_color and i % 2 == 1) else color)


def _box(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _rect(slide, x, y, w, h, color):
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = color
    sp.line.fill.background()
    sp.shadow.inherit = False
    return sp


def _header(slide, s):
    tf = _box(slide, ML, Inches(0.32), CW, Inches(0.62))
    _run(tf.paragraphs[0], s["title"], 25, bold=True, color=NAVY)
    _rect(slide, ML, Inches(1.0), Inches(2.0), Inches(0.035), TEAL)
    tf = _box(slide, ML, Inches(1.08), CW, Inches(0.5))
    _run(tf.paragraphs[0], s["takeaway"], 15.5, bold=True, color=TEAL)


def _footer(slide, idx, total, run_name):
    tf = _box(slide, ML, Inches(7.08), Inches(8.0), Inches(0.3))
    _run(tf.paragraphs[0], f"prod5 · 数据配比系统搜索 · {run_name}", 9, color=GRAY)
    tf = _box(slide, PAGE_W - Inches(1.4), Inches(7.08), Inches(0.8), Inches(0.3))
    p = tf.paragraphs[0]
    p.alignment = 2
    _run(p, f"{idx} / {total}", 9, color=GRAY)


def _bullets(slide, s, y=Inches(1.72), size=15.5, gap=14):
    tf = _box(slide, ML, y, CW, Inches(7.0) - y)
    first = True
    for lead, rest in s["bullets"]:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap)
        p.line_spacing = 1.18
        _run(p, "▪ ", size, bold=True, color=TEAL)
        _run(p, lead, size, bold=True, color=NAVY)
        _md(p, " — " + rest, size, DARK, bold_color=NAVY)


def _stats(slide, s):
    n = len(s["stats"])
    col_w = CW / n
    for i, (num, label) in enumerate(s["stats"]):
        x = ML + int(col_w) * i
        tf = _box(slide, x, Inches(2.55), col_w, Inches(1.3))
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, num, 34, bold=True, color=TEAL)
        tf = _box(slide, x, Inches(3.85), col_w, Inches(1.0))
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, label, 13.5, color=DARK)
    if s.get("note"):
        tf = _box(slide, ML, Inches(5.4), CW, Inches(0.6))
        p = tf.paragraphs[0]
        p.alignment = 2
        _md(p, s["note"], 12, GRAY, bold_color=DARK)


def _cell(cell, text, size, color, fill, bold=False, left=False):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.08)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
    _md(p, text, size, color, base_bold=bold)


def _table(slide, s):
    """举证表: 表头 navy/白字, 首列左对齐, emph 行浅青底加粗, note 为脚注。"""
    headers, rows = s["table"]
    emph = set(s.get("emph", []))
    widths = s.get("widths") or [1.0 / len(headers)] * len(headers)
    size = s.get("size", 13)
    row_h = s.get("row_h", 0.46)
    y = Inches(s.get("y", 1.95))
    gt = slide.shapes.add_table(len(rows) + 1, len(headers), ML, y, CW,
                                Inches(0.42 + row_h * len(rows))).table
    gt.first_row = False
    gt.horz_banding = False
    for j, frac in enumerate(widths):
        gt.columns[j].width = Emu(int(CW * frac))
    gt.rows[0].height = Inches(0.42)
    for i in range(1, len(rows) + 1):
        gt.rows[i].height = Inches(row_h)
    for j, h in enumerate(headers):
        _cell(gt.cell(0, j), h, size, WHITE, NAVY, bold=True)
    for i, row in enumerate(rows):
        hot = i in emph
        for j, v in enumerate(row):
            fill = RGBColor(0xE2, 0xF3, 0xF4) if hot else WHITE
            _cell(gt.cell(i + 1, j), str(v), size,
                  NAVY if (hot or j == 0) else DARK, fill,
                  bold=hot or j == 0, left=(j == 0))
    if s.get("note"):
        y_n = y + Inches(0.42 + row_h * len(rows)) + Inches(0.18)
        tf = _box(slide, ML, y_n, CW, Inches(1.0))
        p = tf.paragraphs[0]
        p.line_spacing = 1.25
        _md(p, s["note"], 11.5, GRAY, bold_color=DARK)


def _image(slide, s):
    y = Inches(1.72)
    if s.get("bullets"):
        _bullets(slide, s, y=y, size=14, gap=6)
        y = Inches(2.95)
    path = s["_path"]
    box_w, box_h = CW, Inches(6.72) - y
    size = png_size(path) or (1500, 900)
    scale = min(int(box_w) / size[0], int(box_h) / size[1])
    w, h = Emu(int(size[0] * scale)), Emu(int(size[1] * scale))
    pic = slide.shapes.add_picture(
        path, ML + Emu((int(box_w) - int(w)) // 2),
        y + Emu((int(box_h) - int(h)) // 2), w, h)
    pic.shadow.inherit = False
    if s.get("caption"):
        y_end = y + Emu((int(box_h) - int(h)) // 2) + h
        cy = y_end + Inches(0.06)
        if int(cy) > int(Inches(6.9)):
            cy = Inches(6.92)
        tf = _box(slide, ML, cy, CW, Inches(0.35))
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, s["caption"], 11, color=GRAY)


def _icon(slide, kind, cx, cy):
    """步骤图标: data=数据池+15 簇点 / d20,d28=模型方块 / lgbm=递增柱。cx,cy = 中心(int EMU)。"""
    if kind == "data":
        can = slide.shapes.add_shape(MSO_SHAPE.CAN, Emu(int(cx - Inches(0.60))),
                                     Emu(int(cy - Inches(0.21))), Inches(0.36), Inches(0.42))
        can.fill.solid()
        can.fill.fore_color.rgb = RGBColor(0xD5, 0xE9, 0xEB)
        can.line.color.rgb = TEAL
        can.line.width = Pt(1)
        can.shadow.inherit = False
        ar = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Emu(int(cx - Inches(0.17))),
                                    Emu(int(cy - Inches(0.06))), Inches(0.16), Inches(0.12))
        ar.fill.solid()
        ar.fill.fore_color.rgb = TEAL
        ar.line.fill.background()
        ar.shadow.inherit = False
        palette = [TEAL, NAVY, RGBColor(0x8A, 0x8A, 0x8A)]
        for r in range(3):
            for c in range(5):
                ov = slide.shapes.add_shape(
                    MSO_SHAPE.OVAL,
                    Emu(int(cx + Inches(0.06) + Inches(0.105) * c)),
                    Emu(int(cy - Inches(0.155) + Inches(0.105) * r)),
                    Inches(0.075), Inches(0.075))
                ov.fill.solid()
                ov.fill.fore_color.rgb = palette[(r * 5 + c) % 3]
                ov.line.fill.background()
                ov.shadow.inherit = False
    elif kind in ("d20", "d28"):
        big = kind == "d28"
        w, h = (Inches(0.58), Inches(0.5)) if big else (Inches(0.46), Inches(0.4))
        cb = slide.shapes.add_shape(MSO_SHAPE.CUBE, Emu(int(cx - w // 2)),
                                    Emu(int(cy - h // 2)), w, h)
        cb.fill.solid()
        cb.fill.fore_color.rgb = (RGBColor(0x6C, 0xD9, 0xDE) if big
                                  else RGBColor(0xB7, 0xDE, 0xE2))
        cb.line.color.rgb = NAVY if big else TEAL
        cb.line.width = Pt(1)
        cb.shadow.inherit = False
        tf = cb.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, kind, 10.5 if big else 9.5, bold=True, color=NAVY)
    elif kind == "lgbm":
        heights = [0.10, 0.17, 0.24, 0.32]
        base = int(cy + Inches(0.16))
        x = int(cx) - int(Inches(0.305))
        for i, hh in enumerate(heights):
            b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x),
                                       Emu(base - Inches(hh)), Inches(0.11), Inches(hh))
            b.fill.solid()
            b.fill.fore_color.rgb = TEAL if i == 3 else RGBColor(0x6E, 0x9E, 0xC4)
            b.line.fill.background()
            b.shadow.inherit = False
            x += int(Inches(0.11)) + int(Inches(0.07))


def _flow(slide, s):
    """算法流程图: 图标 + ①②③ 搜索层(浅青) + ④ 验证层(navy); ①→② 之间 α 配比芯片;
    ③→② 回环箭头。"""
    steps = s["steps"]
    n = len(steps)
    gap = Inches(0.58)
    bw = Emu((int(CW) - int(gap) * (n - 1)) // n)
    by, bh = Inches(2.02), Inches(2.7)
    step = int(bw) + int(gap)
    centers = []
    for i, (head, icon, body, key) in enumerate(steps):
        x = ML + Emu(step * i)
        last = i == n - 1
        sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, by, bw, bh)
        sp.adjustments[0] = 0.055
        sp.fill.solid()
        sp.fill.fore_color.rgb = NAVY if last else RGBColor(0xE8, 0xF5, 0xF6)
        sp.line.color.rgb = TEAL if last else RGBColor(0x9D, 0xC8, 0xCC)
        sp.line.width = Pt(1.5 if last else 0.75)
        sp.shadow.inherit = False
        centers.append(int(x) + int(bw) // 2)
        _icon(slide, icon, centers[-1], int(by) + int(Inches(0.40)))
        tf = _box(slide, x + Inches(0.14), by + Inches(0.72), bw - Inches(0.28), Inches(0.38))
        _run(tf.paragraphs[0], head, 15, bold=True, color=WHITE if last else NAVY)
        tf = _box(slide, x + Inches(0.14), by + Inches(1.12), bw - Inches(0.28), Inches(1.0))
        p = tf.paragraphs[0]
        p.line_spacing = 1.22
        _run(p, body, 11.5, color=WHITE if last else DARK)
        tf = _box(slide, x + Inches(0.14), by + bh - Inches(0.58), bw - Inches(0.28), Inches(0.52))
        p = tf.paragraphs[0]
        p.line_spacing = 1.18
        _run(p, key, 11, bold=True,
             color=RGBColor(0x9B, 0xE3, 0xE7) if last else TEAL)
        if not last:
            ar = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW, x + bw + Inches(0.07),
                by + bh // 2 - Inches(0.15), gap - Inches(0.14), Inches(0.3))
            ar.fill.solid()
            ar.fill.fore_color.rgb = TEAL
            ar.line.fill.background()
            ar.shadow.inherit = False
    # α 配比芯片 (①→② 之间, 箭头下方): 一个配比 = 15 个簇权重
    gcx = (centers[0] + centers[1]) // 2
    chy = by + bh // 2 + Inches(0.24)
    chip = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                  Emu(int(gcx - Inches(0.23))), chy,
                                  Inches(0.46), Inches(0.32))
    chip.adjustments[0] = 0.18
    chip.fill.solid()
    chip.fill.fore_color.rgb = WHITE
    chip.line.color.rgb = NAVY
    chip.line.width = Pt(1.25)
    chip.shadow.inherit = False
    p = chip.text_frame.paragraphs[0]
    p.alignment = 2
    _run(p, "α", 13, bold=True, color=NAVY)
    tf = _box(slide, Emu(int(gcx - Inches(0.29))), chy + Inches(0.36), Inches(0.58), Inches(0.24))
    p = tf.paragraphs[0]
    p.alignment = 2
    _run(p, "配比", 9, color=GRAY)
    # 迭代回环: ③ → ②
    lx, rx = centers[1], centers[2]
    lo = slide.shapes.add_shape(MSO_SHAPE.LEFT_ARROW, Emu(lx),
                                by + bh + Inches(0.18), Emu(rx - lx), Inches(0.3))
    lo.fill.solid()
    lo.fill.fore_color.rgb = RGBColor(0xE2, 0xF3, 0xF4)
    lo.line.color.rgb = TEAL
    lo.line.width = Pt(1)
    lo.shadow.inherit = False
    tf = _box(slide, Emu(lx), by + bh + Inches(0.54), Emu(rx - lx), Inches(0.32))
    p = tf.paragraphs[0]
    p.alignment = 2
    _run(p, s["loop_label"], 11.5, bold=True, color=TEAL)
    if s.get("note"):
        tf = _box(slide, ML, Inches(5.9), CW, Inches(0.9))
        p = tf.paragraphs[0]
        p.line_spacing = 1.25
        _md(p, s["note"], 11.5, GRAY, bold_color=DARK)


def build(run_dir, out):
    run_name = os.path.basename(os.path.normpath(run_dir))
    missing = []
    for s in SLIDES:
        if s.get("image"):
            p = resolve_chart(run_dir, s["image"])
            if p is None:
                missing.append(s["image"])
            s["_path"] = p
    if missing:
        print(f"✗ 缺图 {len(missing)} 张: {', '.join(missing)}")
        print("  (run 目录根层/state//detail/ 都没有; 先跑 report_charts.py"
              " + reinforcement_charts.py)")
        if os.environ.get("DECK_STRICT", "1") == "1":
            return 1
    prs = Presentation()
    prs.slide_width, prs.slide_height = PAGE_W, PAGE_H
    blank = prs.slide_layouts[6]
    total = len(SLIDES)
    for idx, s in enumerate(SLIDES, 1):
        slide = prs.slides.add_slide(blank)
        if s["kind"] == "title":
            _rect(slide, 0, 0, PAGE_W, PAGE_H, NAVY)
            tf = _box(slide, Inches(1.0), Inches(2.35), PAGE_W - Inches(2.0), Inches(1.2))
            p = tf.paragraphs[0]
            p.alignment = 2
            _run(p, s["title"], 40, bold=True, color=WHITE)
            _rect(slide, (PAGE_W - Inches(2.4)) // 2, Inches(3.75),
                  Inches(2.4), Inches(0.04), TEAL)
            tf = _box(slide, Inches(1.0), Inches(4.0), PAGE_W - Inches(2.0), Inches(0.8))
            p = tf.paragraphs[0]
            p.alignment = 2
            _run(p, s["subtitle"], 19, color=WHITE)
            if s.get("stats"):
                n = len(s["stats"])
                col_w = (PAGE_W - Inches(2.0)) / n
                for i, (num, label) in enumerate(s["stats"]):
                    x = Inches(1.0) + int(col_w) * i
                    tf = _box(slide, x, Inches(4.95), col_w, Inches(0.7))
                    p = tf.paragraphs[0]
                    p.alignment = 2
                    _run(p, num, 29, bold=True, color=RGBColor(0x6C, 0xD9, 0xDE))
                    tf = _box(slide, x, Inches(5.7), col_w, Inches(0.55))
                    p = tf.paragraphs[0]
                    p.alignment = 2
                    _run(p, label, 12.5, color=RGBColor(0xBB, 0xC8, 0xD4))
            tf = _box(slide, Inches(1.0), Inches(6.35), PAGE_W - Inches(2.0), Inches(0.5))
            p = tf.paragraphs[0]
            p.alignment = 2
            _run(p, f"{s['meta']} · {run_name}", 12, color=RGBColor(0xBB, 0xC8, 0xD4))
            continue
        _header(slide, s)
        if s["kind"] == "bullets":
            _bullets(slide, s)
        elif s["kind"] == "stats":
            _stats(slide, s)
        elif s["kind"] == "table":
            _table(slide, s)
        elif s["kind"] == "flow":
            _flow(slide, s)
        elif s["kind"] == "image":
            if s["_path"]:
                _image(slide, s)
            else:
                tf = _box(slide, ML, Inches(3.2), CW, Inches(1.0))
                p = tf.paragraphs[0]
                p.alignment = 2
                _run(p, f"[缺图: {s['image']}]", 20, bold=True,
                     color=RGBColor(0xC0, 0x39, 0x2B))
        _footer(slide, idx, total, run_name)
    prs.save(out)
    print(f"✓ {total} 页 → {out}")
    return 1 if missing else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="run 目录 (prod5 归档)")
    ap.add_argument("-o", "--output", default=None,
                    help="输出 pptx (默认 <run_dir>/<basename>_deck.pptx)")
    ap.add_argument("--check", action="store_true", help="只验证图表存在, 不生成")
    args = ap.parse_args()
    if not os.path.isdir(args.run_dir):
        sys.exit(f"✗ 目录不存在: {args.run_dir}")
    if args.check:
        ok = True
        for name in CHARTS_USED:
            p = resolve_chart(args.run_dir, name)
            print(f"  {'✓' if p else '✗'} {name}" + (f" → {p}" if p else ""))
            ok = ok and bool(p)
        sys.exit(0 if ok else 1)
    out = args.output or os.path.join(
        args.run_dir, f"{os.path.basename(os.path.normpath(args.run_dir))}_deck.pptx")
    sys.exit(build(args.run_dir, out))


if __name__ == "__main__":
    main()
