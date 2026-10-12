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
               "proxy_target_consistency.png", "cluster_alpha_vs_score.png"]

# ── 幻灯片文案 (v2: 每页 1 个核心论点 + 当页举证 — 表/图/数字;
#    数字已对 CP4 判定节与 experiment_prod5.md 判读记录核对) ──────────
SLIDES = [
    dict(kind="title",
         title="从 quadmix 到 CLIMB",
         subtitle="数据配比的系统搜索 — prod5 实验结果",
         meta="数据团队汇报 · 2026-10",
         stats=[("+0.043", "STEM 提升(z=+3.8)"),
                ("10.4×", "gsm8k vs base(0.029→0.301)"),
                ("88.5%", "赢家配方集中于 2 个数学簇")]),
    dict(kind="table", title="从手调到搜索",
         takeaway="quadmix 证明了配比有影响;系统搜索给出可复现的量化答案",
         table=(["配比策略", "STEM", "Δ vs base", "判定(vs 均匀)"],
                [["base(不继续训练)", "0.1752", "—", "锚点"],
                 ["均匀(15 簇等权)", "0.1657", "−0.0095", "参考臂"],
                 ["手调域配比 domainfix(math 0.60)", "0.1773", "+0.0021", "噪声内(z=+1.03)"],
                 ["系统搜索 climb(终选 cfg#101)", "0.2086", "+0.0334", "WINS**(z=+3.80)"]]),
         emph=[3], y=2.0, row_h=0.5,
         widths=[0.37, 0.13, 0.15, 0.35],
         note="同池 · 同 base(d28,~1.5B)· 每臂同预算 3B tokens — 继续训练本身不涨分,"
              "增益全部来自数据选择;四代搜索 vs 当轮基线 gap:"
              "−0.004 → +0.010 → +0.016~0.032 → +0.036~0.043"),
    dict(kind="table", title="结论先行",
         takeaway="三个搜索配方全部显著胜出 — 增益来自数据选择,不来自继续训练",
         table=(["臂", "STEM", "Δ vs 均匀", "z", "判定"],
                [["climb(终选 cfg#101)", "0.2086", "+0.0429", "+3.80", "WINS**"],
                 ["climb-cfg90", "0.2078", "+0.0421", "+3.67", "WINS**"],
                 ["climb-cfg100", "0.2013", "+0.0356", "+3.16", "WINS**"],
                 ["domainfix(手调域配比)", "0.1773", "+0.0116", "+1.03", "噪声内"],
                 ["natural(池占比)", "0.1723", "+0.0067", "+0.59", "噪声内"],
                 ["uniform(等权)", "0.1657", "(ref)", "—", "—"]]),
         emph=[0], y=1.9, row_h=0.46,
         widths=[0.32, 0.14, 0.18, 0.12, 0.24],
         note="符号检验:climb 对 uniform 6/6 benchmark 全胜(exact p=0.016);"
              "全部非搜索基线 ≈ base(−0.010~+0.002,噪声内)→ 3B 继续训练本身不产生 STEM 增益"),
    dict(kind="image", title="怎么做的",
         takeaway="两层评测:快而糙的代理搜索 + 慢而准的全量裁决",
         bullets=[
             ("搜索空间", "116M 文档 / ~92B tokens 的 STEM 池 → 15 个宏簇;一个配比 = 15 个簇权重 α"),
             ("两层", "代理 = d20 短训打分(每配比 640M tokens,SNR 加权目标);"
                    "全量 = d28 完整训练(每臂 3B tokens = 70% 配方 + 30% 通用,单遍)"),
             ("本轮规模", "115 个配比实测候选;6 训练臂 + base 锚点(共 18B tokens);"
                       "9,110 题逐题二值评测"),
         ],
         image="search_convergence.png",
         caption="115 个配比点的搜索轨迹(每轮 best 连线)"),
    dict(kind="image", title="主结果",
         takeaway="3 个 CLIMB 配方全部显著胜出三个基线",
         image="arms_main_results.png",
         caption="STEM 主指标 + 第二任务;每臂 3B tokens;误差棒 = 每臂实算 SE"),
    dict(kind="table", title="提升来自哪里",
         takeaway="增益集中在生成式数学推理 — 判别题几乎不动",
         table=(["benchmark(题量)", "climb", "uniform", "Δ", "倍数"],
                [["gsm8k_cot(1319,生成)", "0.3010", "0.1054", "+0.196", "2.9×"],
                 ["math_cot_500(500,生成)", "0.0580", "0.0160", "+0.042", "3.6×"],
                 ["mmlu_stem(3545,判别)", "0.2920", "0.2872", "+0.005", "1.02×"],
                 ["arc_easy(2376,判别)", "0.7294", "0.7269", "+0.003", "1.00×"],
                 ["arc_challenge(1172,判别)", "0.4360", "0.4334", "+0.003", "1.01×"],
                 ["gpqa_diamond(198,判别)", "0.2121", "0.2071", "+0.005", "1.02×"]]),
         emph=[0, 1], y=1.9, row_h=0.46,
         widths=[0.34, 0.16, 0.16, 0.17, 0.17],
         note="vs base(不训练):gsm8k 0.0288 → 0.3010(10.4×),math 0.0040 → 0.0580(14.5×)"
              "— 以上为每臂 3B tokens 的 mid-training 买到"),
    dict(kind="image", title="代理模型可信吗",
         takeaway="代理排序与全量实测一致 — 搜索没有在优化幻觉",
         bullets=[
             ("代理→实测一致", "搜索在代理分上选中的配方,d28 实测全部兑现:三臂 WINS**,终选 top-1"),
             ("独立复现", "冷启动(115 点全新采样)0.2086 vs 热启动(prod4)0.2142,"
                       "落在种子带 ±0.016~0.032 内 — 两次搜索抵达同一平台"),
         ],
         image="proxy_target_consistency.png",
         caption="climb 臂:代理分 vs d28 实测"),
    dict(kind="table", title="我们在 CLIMB 上的方法改进",
         takeaway="三处修复,每处都对应上一代的实测代价",
         table=(["修复", "上一代实测代价", "prod5 结果"],
                [["终选守卫(D19)", "prod4:外推终选 0.1972,输自家实测点 0.2142(−0.017)",
                  "守卫触发 → 终选 = 实测 top-1(0.2086)"],
                 ["宏簇平衡划分(D14)", "论文距离合并把 99% tokens 塌成巨簇(K=14 时留 99.0%)",
                  "恰好 15 簇,簇容量受约束 — 15 个真旋钮"],
                 ["SNR 加权目标(D8)", "prod1:搜索 0.1623 < 随机 0.1659(−0.004,噪声内)",
                  "搜索 +0.043 vs 均匀(z=+3.80)"]]),
         y=1.95, row_h=0.78, size=12, widths=[0.21, 0.43, 0.36],
         note="四代累计(vs 当轮基线):−0.004 → +0.010 → +0.016~0.032 → +0.043;"
              "论文机制不设防的地方,各加一道闸"),
    dict(kind="table", title="实验关键发现",
         takeaway="三条实测发现 — 每条都改变了流程",
         table=(["发现", "实测证据", "处置"],
                [["剪枝要单列下限(D13)",
                  "69/1000 簇过均值线,但 knowledge_value 1.8–2.0(格式分把知识贫瘠平均掉)",
                  "任一质量列 < 2.0 即剪(保留均值线 3.0)"],
                 ["生成评测适配 base(D17)",
                  "无 EOS:hit_cap 99.85% @256;gold 30% 的答案超 256(p95=643)",
                  "停止串 + math 预算 1024:18.8% → 25.8%"],
                 ["单遍必须真单遍(D16)",
                  "prod1/2 实测混料被重复消费 2.5–4 epoch",
                  "修复后构造性单遍(epoch ≈ 0.7)"]]),
         y=1.95, row_h=0.78, size=12, widths=[0.22, 0.44, 0.34],
         note="三条都来自实测代价,不是理论洁癖 — 修一条,数字才可信一分"),
    dict(kind="image", title="赢家是谁",
         takeaway="搜索把 80% 预算押在统计练习题簇 — gsm8k 的机制闭环",
         bullets=[
             ("赢家配方(cfg#101)", "C10『初等/应用统计练习题与解答体』α = 0.80 = 池占比 5.5% 的 14.6×;"
                                "C0 形式数学解题 0.08;其余 13 簇全压地板(0.009)"),
             ("收敛稳健", "top-3 配方全部 C10-heavy(α 0.55~0.80)— 搜索结论不是单点运气"),
         ],
         image="cluster_alpha_vs_score.png",
         caption="逐簇 α 与得分(赢家配方的簇权重分布)"),
    dict(kind="table", title="一个尖锐的发现:质量分 ≠ 训练效用",
         takeaway="赢家簇质量分最低、训练贡献最大",
         table=(["簇", "质量分(stem_rel / knowl.)", "配方角色", "d28 效果"],
                [["C10 统计练习题(赢家重仓)", "3.88 / 3.65 — 三簇最低", "α = 0.80(14.6× 池占比)",
                  "gsm8k 2.9× / math 3.6×(vs uniform)"],
                 ["C0 形式数学解题", "4.33 / 4.12", "α = 0.08", "与 C10 并列的机制解释"],
                 ["C6 综述长文", "4.45 / 4.07 — 最高", "cfg100 下注 α = 0.16",
                  "cfg100 math 减半(0.026 vs 0.058)"]]),
         emph=[0], y=1.95, row_h=0.62, size=12, widths=[0.26, 0.26, 0.23, 0.25],
         note="形状相近、域不同:C2(化学短练习)vs C10(统计短练习)= 0.1× vs 14.6× 池占比权重 — "
              "同为短练习,域对了才有用;质量分衡量『像不像好文本』,不衡量『训练有没有用』"),
    dict(kind="table", title="怎么保证没骗自己",
         takeaway="四道检查,全部量化、全部在案",
         table=(["检查", "数值", "判定"],
                [["评测管线锚点", "远端复评 base 0.1752 vs 本地基准 0.1738(|Δ| = 0.0014)",
                  "PASS(阈 0.002)"],
                 ["误差实算", "6/6 臂:SE = 逐题二项(9,110 题),非默认值", "全部结论带 z / p"],
                 ["独立复现", "冷/热两次搜索:0.2086 vs 0.2142(种子带 ±0.016~0.032)", "同平台"],
                 ["判定预注册", "V1–V5 判决规则实验前写死(含加冕规则)", "按规则执行,留档可查"]]),
         y=1.95, row_h=0.62, size=12.5, widths=[0.2, 0.52, 0.28],
         note="规则先于结果、误差实算、管线锚定、结论复现 — 四者齐备,结论才出手"),
    dict(kind="stats", title="这意味着什么",
         takeaway="配比是一个可搜索、可复现、可解释的旋钮",
         stats=[("+0.043", "STEM 提升(z=+3.8,6/6 benchmark 全胜)"),
                ("2 次", "独立搜索同平台(0.209 vs 0.214)"),
                ("≈3,900 NPU·h", "两轮全量验证 + 一次性全池嵌入")],
         note="赢家配方可直接复用;自然延伸:更大规模验证(20B,天级)+ 质量信号协作"
              "(P-3 已预注册)— 值得一起做"),
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
        _run(p, " — " + rest, size, color=DARK)


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
        _run(p, s["note"], 12, color=GRAY)


def _cell(cell, text, size, color, fill, bold=False, left=False):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = cell.margin_right = Inches(0.08)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    p = cell.text_frame.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT if left else PP_ALIGN.CENTER
    _run(p, text, size, bold=bold, color=color)


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
        _run(p, s["note"], 11.5, color=GRAY)


def _image(slide, s):
    y = Inches(1.72)
    if s.get("bullets"):
        _bullets(slide, s, y=y, size=14, gap=6)
        y = Inches(2.95)
    path = s["_path"]
    box_w, box_h = CW, Inches(6.82) - y
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
        print("  (run 目录根层/state//detail/ 都没有; 先跑 report_charts.py)")
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
