#!/usr/bin/env python3
"""生成图内中文文案的字体子集 (维护工具, 只在有文泉驿正黑的机器上跑)。

为什么: 算力服务器无 root、无系统 CJK 字体; 仓库内置一个只含图内实际
用到字符的文泉驿正黑子集 (原字体 GPL + 字体例外, 可再分发),
reinforcement_charts.setup_cjk_font() 优先使用内置字体, git pull 即得,
服务器不需要 apt / --font / ~/.fonts。

子集覆盖 = 三个脚本 (reinforcement_charts / report_charts / make_deck)
的全部源码字符 + ASCII + 常用标点/箭头/数学/希腊/带圈数字/CJK 标点/
全角区段。什么时候重跑: 上述脚本的图内文案 (matplotlib 渲染部分)
新增字符后 — 不重跑的话, 子集外字符会渲染成方框 (pptx 文本不受影响,
那由 PowerPoint 字体渲染)。改完重跑本脚本并提交新 ttf。

用法:
  python3 scripts/diagnostics/fonts/make_font_subset.py \
      --source /usr/share/fonts/truetype/wqy/wqy-zenhei.ttc
"""
import argparse
import os
import sys

from fontTools import subset
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "wqy-zenhei-deck.ttf")
FAMILY = "WQY Zen Hei Deck"   # 改名避免与系统 wqy 同名 (findfont 二义)

# 字符来源: reinforcement_charts (中文, matplotlib 渲染) 为必须覆盖;
# report_charts v10.1 起图内标签保持英文 (正经实验结果图), 仍收录进
# 必须集 (超集无害, 日后有中文字符进图也稳); make_deck 文案由
# PowerPoint 渲染, 只作富余量收录 (其中的 ▪ − 等 wqy 缺字不要求)
MUST_SOURCES = [os.path.join(HERE, "..", n) for n in
                ("reinforcement_charts.py", "report_charts.py")]
BONUS_SOURCES = [os.path.join(HERE, "..", "make_deck.py")]

# 只进 stdout 不进图的字符 (wqy 缺这两个符号; U+2212 由
# axes.unicode_minus=False 规避, ASCII 负号替代)
SKIP_CHARS = {"✓", "✗"}

# 富余区段 (码位闭区间): 常见符号/全角, 防文案小改就出方框
EXTRA_RANGES = [
    (0x20, 0x7E),     # ASCII 可打印
    (0xA0, 0xFF),     # 拉丁补充: × ÷ ° ± · …
    (0x391, 0x3C9),   # 希腊: σ Δ μ β
    (0x2010, 0x2027),  # 通用标点: – — ‘ ’ “ ” …
    (0x2030, 0x205E),  # ‰ ′ ″ ※ 等
    (0x2190, 0x21FF),  # 箭头: → ← ↑ ↓ ↔
    (0x2200, 0x22FF),  # 数学算符: ≈ ≠ ≤ ≥ ∑ √ ∞ ∂
    (0x2460, 0x24FF),  # 带圈数字 ①-⑳
    (0x3000, 0x303F),  # CJK 标点: 。 、 「 」
    (0xFF01, 0xFF5E),  # 全角形式
]

# matplotlib 渲染用不到的位图/高级排版表, 剔掉以缩小体积
DROP_TABLES = ["EBDT", "EBLC", "EBSC", "BASE", "JSTF", "DSIG"]


def collect_chars():
    must, bonus = set(), set()
    for p in MUST_SOURCES:
        with open(p, encoding="utf-8") as f:
            must.update(f.read())
    for p in BONUS_SOURCES:
        with open(p, encoding="utf-8") as f:
            bonus.update(f.read())
    for lo, hi in EXTRA_RANGES:
        bonus.update(chr(c) for c in range(lo, hi + 1))
    return must - SKIP_CHARS, bonus


def rename_family(font, family):
    """改家族名 (1/3/4/6/16/17), 与系统 wqy 区分 — findfont 结果确定。"""
    name = font["name"]
    for nid in (1, 3, 4, 6, 16, 17):
        name.removeNames(nameID=nid)
    for pid, eid, lid in ((3, 1, 0x409), (1, 0, 0)):
        name.setName(family, 1, pid, eid, lid)
        name.setName("Regular", 2, pid, eid, lid)
        name.setName(f"{family} Regular (subset)", 3, pid, eid, lid)
        name.setName(family, 4, pid, eid, lid)
        name.setName(family.replace(" ", ""), 6, pid, eid, lid)


def coverage(font):
    have = set()
    for t in font["cmap"].tables:
        if t.isUnicode():
            have.update(t.cmap.keys())
    return have


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default="/usr/share/fonts/truetype/wqy/"
                                        "wqy-zenhei.ttc")
    args = ap.parse_args()
    if not os.path.isfile(args.source):
        sys.exit(f"✗ 源字体不存在: {args.source} (本工具需要在有文泉驿正黑"
                 "的机器上跑; 其他 CJK 字体也可, 需覆盖基本汉字+符号)")

    must, bonus = collect_chars()
    chars = must | bonus
    print(f"字符集: 必须 {len(must)} + 富余 {len(bonus)} = {len(chars)} 个")

    opts = subset.Options()
    opts.font_number = 0            # TTC 第 0 个 = WenQuanYi Zen Hei
    opts.drop_tables = set(opts.drop_tables) | set(DROP_TABLES)
    opts.layout_features = ["*"]    # 保留 kern 等 (体积代价很小)
    font = subset.load_font(args.source, opts)
    sub = subset.Subsetter(options=opts)
    sub.populate(unicodes=[ord(c) for c in chars])
    sub.subset(font)
    rename_family(font, FAMILY)
    subset.save_font(font, OUT, opts)

    # 自验: 重载成品 — 图表脚本用字必须全覆盖 (富余缺字仅计数提示)
    check = TTFont(OUT)
    have = coverage(check)
    missing_must = sorted(c for c in must if ord(c) not in have)
    if missing_must:
        os.remove(OUT)
        sys.exit(f"✗ 图表脚本用字未覆盖, 子集不完整, 已删除: {missing_must!r}")
    n_glyphs = check["maxp"].numGlyphs
    n_bonus = sum(1 for c in bonus if ord(c) in have)
    print(f"✓ {OUT}")
    print(f"  家族名: {FAMILY} | 字形: {n_glyphs} | 富余覆盖: {n_bonus}/{len(bonus)}"
          f" | 大小: {os.path.getsize(OUT) / 1024:.0f} KB")
    print("  提交: git add scripts/diagnostics/fonts/ && git commit")


if __name__ == "__main__":
    main()
