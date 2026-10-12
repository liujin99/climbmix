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

# ── 幻灯片文案 (S1 审稿版; 数字待 S3 与 CP4 判定节核对) ──────────────
SLIDES = [
    dict(kind="title",
         title="从 quadmix 到 CLIMB",
         subtitle="数据配比的系统搜索 — prod5 实验结果",
         meta="数据团队汇报 · 2026-10"),
    dict(kind="bullets", title="上次与这次", takeaway="同一个池,问法升级了",
         bullets=[
             ("quadmix(上次)", "手动设定几组配比做对比 — 验证了『配比有影响』"),
             ("这次(CLIMB)", "不再手动试:让搜索自动找配比,用全量训练实测裁决"),
             ("问题升级", "从『有没有差』到『最优配比怎么找、能不能复现』"),
         ]),
    dict(kind="stats", title="结论先行",
         takeaway="搜索找到的配比,显著优于自然配比与均匀配比",
         stats=[("+0.036~+0.043", "STEM 提升(3 个 CLIMB 配方 vs 均匀)"),
                ("z = +3.8 / +3.1", "vs 均匀 / vs 自然(实算 SE)"),
                ("2 次复现", "冷启动与热启动搜索到达同一平台")],
         note="每臂训练 = 2B tokens(d28,~1.5B 模型);误差棒 = 实算 SE(逐任务分数与题量),非默认值"),
    dict(kind="image", title="怎么做的",
         takeaway="两层评测:快而糙的代理搜索 + 慢而准的全量裁决",
         bullets=[
             ("流程", "池 → 聚 15 簇 → 代理搜 115 个配比 → 全量评测 5 臂(3 CLIMB + 均匀 + 自然)"),
             ("两层", "代理 = d20 短训练打分(每配比 640M tokens);全量 = d28 完整训练(每臂 2B tokens,~1.5B 模型)"),
             ("数据量", "每臂 2B tokens = 1.4B 搜索配方 + 0.6B 通用(70/30,单遍);全舰队 5 臂共 10B"),
         ],
         image="search_convergence.png",
         caption="115 个配比点的搜索轨迹(每轮 best 连线)"),
    dict(kind="image", title="主结果",
         takeaway="3 个 CLIMB 配方全部显著胜出两个基线",
         image="arms_main_results.png",
         caption="STEM 主指标 + 第二任务;每臂 2B tokens 训练;误差棒 = 每臂实算 SE"),
    dict(kind="stats", title="提升来自哪里",
         takeaway="收益集中在生成式数学推理 — 结构性提升,不是全面膨胀",
         stats=[("2.4–2.9×", "gsm8k 对 uniform 家族"),
                ("10–11×", "gsm8k 对 base 模型"),
                ("+0.04", "STEM 宏平均(vs 均匀)")],
         note="STEM = 6 个 benchmark 宏平均;gsm8k 为生成式数学推理 — 以上 = 每臂 2B tokens 的 mid-training 买到"),
    dict(kind="image", title="代理模型可信吗",
         takeaway="代理分数与全量实测一致 — 搜索没有在优化幻觉",
         bullets=[("复现", "冷启动(prod5)与热启动(prod4)两次独立搜索到达同一平台(~0.21)")],
         image="proxy_target_consistency.png",
         caption="climb 臂:代理分 vs d28 实测"),
    dict(kind="bullets", title="我们在 CLIMB 上的方法改进",
         takeaway="论文机制不设防的地方,我们各加了一道闸",
         bullets=[
             ("终选守卫", "论文让『外推赢家』直接上位;我们要求其预测优势显著超过噪声才放行,"
                        "否则回落最佳实测点。prod5 实战:守卫触发,其选点赢了全舰队"),
             ("宏簇构造重设计", "我们的池在嵌入空间是致密连续流形,论文的距离合并会把 99% tokens "
                              "塌成一个巨簇;改容量平衡划分后,搜索才真正有 15 个旋钮"),
             ("SNR 加权目标", "论文优化裸 accuracy — 难 benchmark 上那是抛硬币;"
                            "信噪比加权让代理模型学信号、不学噪声"),
         ]),
    dict(kind="bullets", title="实验关键发现",
         takeaway="三条用算力换来的发现 — 每条都改变了流程",
         bullets=[
             ("剪枝需要单列下限", "只看簇均值会漏检『格式干净但知识贫瘠』的簇;这些簇混进搜索空间后,"
                               "配比旋钮几乎转不动(prod1 实证,已修复)"),
             ("评测协议要适配 base 模型", "base 无 EOS,论文惯例的 256 生成预算让答案后的幻觉污染 "
                                      "math 分(实测 30% 正确答案需要更长);停止串 + 1024 预算后"
                                      "测到的才是真分数"),
             ("单遍退火必须是真单遍", "早期轮次混料实测被重复消费 2.5–4 epoch,预算口径失真;"
                                   "修复后构造性保证单遍"),
         ]),
    dict(kind="image", title="赢家是谁",
         takeaway="搜索自己找到了 gsm8k 的机制闭环",
         bullets=[
             ("赢家配方(cfg100)", "给『初等/应用统计练习题与解答体』簇最高权重"),
             ("簇语义", "统计练习题 = gsm8k 同机制,不是泛泛的『高质量文本』"),
         ],
         image="cluster_alpha_vs_score.png",
         caption="逐簇 α 与得分(赢家配方的簇权重分布)"),
    dict(kind="bullets", title="一个尖锐的发现:质量分 ≠ 训练效用",
         takeaway="现有质量分与下游训练效用不是一回事",
         bullets=[
             ("事实", "赢家簇的质量分在宏簇中最低 — 但它的下游 STEM 贡献最大"),
             ("解释", "质量分衡量『文本像不像好文本』,不衡量『训练有没有用』"),
             ("含义", "配比的裁判只能是下游实测 — 这正是系统搜索存在的理由;"
                    "也是质量信号值得一起改进的地方"),
         ]),
    dict(kind="bullets", title="怎么保证没骗自己",
         takeaway="判定先写死,误差实算,结论复现",
         bullets=[
             ("判定规则预注册", "判读标准在实验前写死,不是事后找说法"),
             ("误差棒实算", "SE 来自逐任务分数与题量;修正后旧结论 z 值仍然成立"),
             ("两次独立搜索复现", "冷启动与热启动同平台(~0.21),结果不依赖初始化"),
         ]),
    dict(kind="bullets", title="这意味着什么",
         takeaway="配比是一个可搜索、可复现、可解释的旋钮",
         bullets=[
             ("可搜索", "不靠直觉 — 由下游实测驱动迭代"),
             ("可复现", "两次独立搜索给出同一个答案"),
             ("可解释", "赢家配方有簇语义,不是黑盒"),
             ("自然延伸", "更大规模验证(20B,天级);质量信号协作 — 值得一起做"),
         ]),
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
