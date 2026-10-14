#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_deck.py — prod5 → 数据团队汇报 PPTX 生成器 (2026-10-12)。

用法 (服务器, run 目录含 report_charts 的图):
    python3 scripts/diagnostics/make_deck.py result/prod5_20260929_201108
    python3 scripts/diagnostics/make_deck.py <run_dir> --check    # 只验图
    python3 scripts/diagnostics/make_deck.py <run_dir> -o out.pptx

依赖: python-pptx (pip install python-pptx; 若报 PEP 668 加
--break-system-packages 或用 venv)。图表按三代布局解析 (root → state/
→ detail/)。文案 = 本文件 SLIDES 常量 (审稿在对话里, 落地在这里)。

输出 (v10.2 实验结果/汇报分离): 汇报包 = run 目录下的 deck/ 子目录 —
pptx + 全部 7 张图都在里面 (5 张 deck 专属图由 reinforcement_charts.py
生成, 2 张实验结果图每次构建时从 run 目录复制进来, 包自含、可整目录
打包外发); run 目录根层只留实验结果。
16:9, 中文字体 Microsoft YaHei。
"""
import argparse
import glob
import os
import shutil
import struct
import sys

try:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
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

CHARTS_USED = ["search_convergence.png", "proxy_target_consistency.png",
               "recipe_distribution.png", "cluster_content_map.png",
               "reinforcement_d14_cluster_balance.png",
               "reinforcement_d8_dual_signal.png",
               "reinforcement_d8_scoring_snr.png",
               "reinforcement_d19_extrapolation.png",
               "finding_quality_value.png", "finding_pool_manifold.png"]

# deck 专属 (reinforcement 系图, 中文, 由 reinforcement_charts.py 写进
# 汇报包目录); 其余 2 张 = report_charts 实验结果图 (英文, 住 run 目录,
# 权威源在 run 目录 — 每次构建重新复制进汇报包, 不会用到旧副本)。
# v10.4: arms_main_results(主结果页删, 数字全在结论先行表+总表) 与
# cluster_alpha_vs_score(专业向小倍数图, 换成领导友好的配比全景) 不再
# 进汇报包, 仍是 run 目录的实验产物
DECK_OWNED = {"reinforcement_d14_cluster_balance.png",
              "reinforcement_d8_dual_signal.png",
              "reinforcement_d8_scoring_snr.png",
              "reinforcement_d19_extrapolation.png",
              "recipe_distribution.png",
              "cluster_content_map.png",
              "finding_quality_value.png", "finding_pool_manifold.png"}

# 流程页小插图 (deck_assets/, 无文字的意象图; 缺件回退矢量图标)
ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "deck_assets")
FLOW_ICON_IMG = {"data": "flow_cluster.png", "sample": "flow_sample.png",
                 "d20": "flow_proxy.png", "lgbm": "flow_learner.png",
                 "d28": "flow_final.png"}


def deck_dir_for(run_dir):
    """汇报包目录 = run 目录下的 deck/ 子目录 (v10.2 分离实验结果与汇报
    产物; 用户裁决: deck 跟着 run 走)。reinforcement_charts.py 有同名同
    实现副本 — 不互相 import, 免得把 numpy/matplotlib/pptx 依赖拉进
    对方的裸跑路径。"""
    return os.path.join(os.path.normpath(run_dir), "deck")


def resolve_chart(dirs, name):
    """按优先级在目录列表里找图 (各自含 根层/state//detail/ 三代布局)。"""
    for d in dirs:
        for sub in ("", "state/", "detail/"):
            p = os.path.join(d, sub, name)
            if os.path.isfile(p):
                return p
    return None

# ── 幻灯片文案 (v10.15 评审反馈: ① 发现③ 与改进① 前后呼应 — 改进①
#    结论条补『根因 = 池是连续流形, 详见关键发现③』, 发现③ 的『怎么防』
#    回指机制改进①; ② 发现③ 往『窄域池预警』角度立题(核心价值 = 告诉
#    大家专用领域上此算法前先查池结构), 问答行序改为陷阱优先; ③ 发现块
#    补机理图 — 新图 finding_quality_value.png(① 质量分 × 赢家权重排序
#    倒挂, 数字取档案三簇)与 finding_pool_manifold.png(③ 通用池成团 vs
#    窄域池流形+孤岛, 纯示意), ①③ 改 table_image; 发现② 已有图不动。
#    v10.14 评审反馈: ① 发现③重写成问答 — 单域池共性还是
#    本池特例 / 论文为何没遇到 / 别人会不会遇到 / 我们的答案, 价值自明;
#    ② p5 只描述评测体系(六项基准各测什么能力), 增益结论归 p6 —
#    『两类能力』实为题型, 能力粒度 = 逐基准能力点; ③ 改进②补机理图
#    reinforcement_d8_dual_signal.png(问题·做法·反证三联, 与 d8 同房样),
#    纯文字页太干; ④ 簇内容图标签去关键词, 只留簇名 + 中位长度(单行,
#    字号 8→9), 关键词/摘录住 §2b。改进②改 table_image, 页数仍 17。
#    v10.13 评审反馈: ①② p5/p6 主线从『题型』改『能力』 —
#    题型只是表现方式, 训模型关注的是能力: p5 = 评测体系(六项基准 × 两类
#    能力: 判别式知识/常识推理 vs 生成式多步推理)+ 总表; p6 = 能力归因
#    (最大提升 = 生成式推理, 两项数学基准); ③ 评分优化拆回两页 — 双信号
#    (解决: 代理小模型 + speedrun 下精度信号弱)与信噪比定权(解决: 基准
#    本身题量小、二项噪声大)动机不同, 各自走四段因果链; 改进块 三处→
#    四处(① 簇均衡 ② 双信号 ③ SNR 定权 ④ 终选守卫), 页数 16→17;
#    d8 机理图归 ③, ② 为纯链表页。
#    v10.12 评审反馈: ① 方法页分隔线与 body 文本重叠(实际
#    渲染行距 = 字体自然行高 × lnSpc, msyh 自然行高 ≈ 1.41 em, 4 行 body
#    实际底界比估算低 ~0.15in) — 分区整体下移 + lnSpc 1.22→1.05, 分隔线
#    2.22, 卡高 2.84, ③ 卡文案再剪; ② 总表页基准列加考点(测什么能力);
#    ③ p5/p6 主旨句定稿; ④ p6 结论条去口语(『换最强对手再对决』);
#    ⑤ 评分设计两页合并为一页, 与 ①③ 同构: 论文做法 → 实测证伪 →
#    我们的改进 → 改进验证 四段因果链 + order_arrow + d8 机理图;
#    ⑥ 发现② 改写为『赢家明确, 排序有噪声』(小模型搜大模型兑现是业界
#    常规, 不算发现); ⑦ 新增发现③ 池结构(连续流形 + 格式孤岛, D14);
#    ⑧ 『发现』→『关键发现』。页数不变(16): ② 合并 -1, 发现③ +1。
#    v10.11 评审反馈: ① 方法页五步卡 body 区与 key 行重叠
#    (③ 卡 body 折 5 行溢出 box, 压到卡底加粗 key 行) — 卡高 2.7→2.78,
#    body/key 分区加细分隔线, key 行垂直居中, ③⑤ 文案各剪一行;
#    ② 总表页结论条补基准构成总述(测了什么: 选择 4 + 写过程数学 2);
#    ③ 评分设计页并入机制改进优化为 ②(设计), 原 ② 改 ②(续);
#    ④ 质量分/代理有效性两页归『发现』节(发现①②); ⑤ 没骗自己页
#    增首行『度量衡校准』(D17 评测协议修复, A/B 18.8%→25.8%)。
#    v10.10 评审反馈: 簇内容图第 2 行的『样本开头』只有个
#    开头, 读者看不出所以然 — 移出图面, 第 2 行只留关键词; 标签区缩窄,
#    条形区放宽; 完整摘录仍在 §2b 实验档案。
#    v10.9 评审反馈: 簇内容页 11 簇第 1 行回退到『域纯度』
#    (C9 数学 97%…), 满页抽象的数学/化学/物理 — 15 簇全部命名(§2b n=25
#    抽样 + 对比 IDF 关键词, 2026-10-14); 更正 C10 旧名『统计练习题』
#    (n=1 误读, 关键词 divide·percentage·multiply + 样本"22% of 12" =
#    算术应用题, 与 gsm8k 机制闭环更贴), C6/C2 格式名 → 内容名;
#    p5/p6/p12 文案同步。页序不变 (16 页)。
#    v10.8 评审反馈: ① 方法页五步卡统一配色(原 ⑤ navy 只为
#    标记搜索层/验证层, 评审不需要), key 行只留模型规模、token 量拆进
#    body — 原两行 key 在渲染器里折成三行,『tokens』溢出卡片下缘;
#    ② 『机制修复』→『机制改进优化』; ③ 表列名『轮次』→『实验轮次』、
#    『第x轮』→『第x轮实验』, 单轮实验内的 3 轮搜索改称『迭代』(方法页/
#    搜索过程页/图内标签), 内外两层不再混淆; ④ 改进② takeaway 去口语,
#    表压缩(row_h 0.52)给图让位; ⑤ 改进③ 表加 ①→④ 顺序箭头(order_
#    arrow), 改进②③ 各加设计公式; ⑥ 新增评分设计页 — 精度×损失混合、
#    SNR 定权(我们的设计, paper_deviations D8); ⑦ 簇内容图标签第 2 行 =
#    关键词(cluster_peek 对比 IDF, 新特性, prod5 试点)+ 更长样本摘录。
#    页序 (17 页): 标题 / 结论先行 / 方法 / 搜索过程 / 总表 / 提升来源 /
#    机制改进优化①~④ / 配比全景 / 附:簇内容 / 关键发现①②③ /
#    没骗自己 / 意味着什么。
#    v10.7 评审反馈: ① 方法页小图按卡片宽度缩放(五步卡片不再溢出), 两级
#    模型规模改报有效参数 — 代理 d20 435M / 验证 d28 1.5B(总参数 0.9B /
#    ~2.4B, 比例 ≈ 1:3.4); ② 『三处加固』→『三处机制修复』, 方法页 note
#    框成五轮实验叙事(证据 = 第1~4轮, 验证 = 第5轮); ③ 加固①表末列改
#    『同轮对决:climb vs 簇等权臂』; ④ 全 deck 弃用 prodN 内部代号 →
#    『第N轮(实验)』(副标题/页脚/表格/图内标签), 归档名保留在页脚供溯源;
#    ⑤ 加固②改因果链表并讲清为什么单拎 gsm8k / math(全部净增益来自这两
#    项, 且题量最小最易被误判); ⑥ 加固③改四段式故事链(论文做法 → 实测
#    证伪 → 修复 → 验证); ⑦ 簇内容页去 bullets 放大图, 图内每簇标签附 1
#    篇样本摘录(§2b)。
#    v10.6 评审反馈: ① 结论页/总表页脚注精简; ② 方法页步骤重排 — 补
#    『配比采样』步、『代理模型训练与验证』、LightGBM 拟合对象写明、
#    『正式规模』改『中等规模验证』, 写明两级模型规模与训练量; ③ 修复
#    ①② takeaway 写明后果; ④ 修复③重构为因果链表; ⑤ 簇内容页标签差异
#    化 + 地板分隔线, 后移为配比全景的『附』页。
#    v10.5 评审反馈: ① 全 deck 术语专业化 — 代理模型 / LightGBM 拟合,
#    删口语表述; ② 两层成本设计(代理试错 vs 正式裁决)归入方法页, 搜索
#    过程页改讲『三轮迭代收敛』; ③ 代理模型有效性 = 附加验证, 移至靠后;
#    ④ 三处加固页改『表(数据)+ 图(机理)』; ⑤ 加固③与 prod4 对决表合并;
#    ⑥ 新增 15 语义簇内容分析页。
#    v10.4 评审反馈: 结论先行表以 base 为基准, 三个 climb 臂整组底色;
#    主结果页删; 流程页意象小图; 配比全景页。
#    v10.3: 表格得分百分点口径 xx.xx% + Δ(pt) + 相对提升(%); d19 极性
#    修复 (见 reinforcement_charts.py)。
#    **…** = 段内加粗; 数字已对 experiment_prod2_k15bal/prod4/prod5.md、
#    TODO.md prod3 记录、scoring_metric_design.md、paper_deviations.md
#    D8/D14/D19) ──
SLIDES = [
    dict(kind="title",
         title="训练数据怎么配?让算法自己搜",
         subtitle="CLIMB 数据配比搜索 · 第5轮实验结果",
         meta="数据团队汇报 · 2026-10",
         stats=[("+19.1%", "STEM 平均分相对提升(vs base;对 uniform 6 项全胜)"),
                ("10.4×", "数学应用题 gsm8k:2.9% → 30.1%"),
                ("88.5%", "赢家配方预算集中在 2 个数学簇")]),
    dict(kind="table", title="结论先行",
         takeaway="三个搜索配方全部显著优于 base(最高 +19.1%);三种传统配比均未超过 base — 增益来自数据配比,而非继续训练本身",
         table=(["配方", "STEM", "Δ vs base", "相对提升", "结论"],
                [["climb(搜索终选)", "**20.86%**", "+3.34pt", "**+19.1%**",
                  "显著胜出"],
                 ["climb-cfg90", "20.78%", "+3.26pt", "+18.6%", "显著胜出"],
                 ["climb-cfg100", "20.13%", "+2.61pt", "+14.9%", "显著胜出"],
                 ["domainfix(人工域配比)", "17.73%", "+0.21pt", "+1.2%",
                  "无显著差异"],
                 ["natural(池自然占比)", "17.23%", "−0.29pt", "−1.7%",
                  "无显著差异"],
                 ["uniform(15 簇等权)", "16.57%", "−0.95pt", "−5.4%",
                  "无显著差异"],
                 ["base(不继续训练)", "17.52%", "(基准)", "—", "参照基准"]]),
          emph=[0, 1, 2], y=1.9, row_h=0.46, size=12.5,
          widths=[0.28, 0.12, 0.15, 0.13, 0.32],
          note="Δ 与相对提升均 vs base;同池、同底座、同预算(每臂 3B tokens)— 增益全部来自配比选择。"
               "显著性(vs uniform,逐题二项 SE):三搜索臂 z = **+3.80 / +3.67 / +3.16**;"
               "climb **6 项全胜**(p=0.016)。cfg90 / cfg100 = 搜索 top-3 的另两个;"
               "domainfix / natural / uniform = 三种传统配比;得分 = Centered×100"),
    dict(kind="flow", title="方法 — CLIMB:把数据配比变成可搜索参数",
         takeaway="采样候选配比 → 代理模型训练与验证得到实测得分 → 预测器拟合响应面引导下一轮迭代 — 在 15 维配比空间逐步逼近最优",
         steps=[
             ("① 语义聚类", "data",
              "116M 篇 STEM 文档聚成 15 个语义簇;一个配比 = 15 维权重向量 α",
              "数据池 116M 篇 · 91.8B tokens"),
             ("② 配比采样", "sample",
              "第 1 次迭代随机采样覆盖全空间;之后由预测器引导,向预测高分区域集中",
              "3 次迭代 64/32/16 ≈ 115 候选"),
              ("③ 代理模型训练与验证", "d20",
               "按配比短训代理模型(640M tokens/候选),在 6 项基准上验证得实测得分",
               "代理模型 435M 有效参数"),
             ("④ 预测器拟合", "lgbm",
              "LightGBM 用已实测的『配比 α → 得分』样本拟合响应面,预测新配比的得分",
              "拟合样本:全部 115 个实测配比"),
              ("⑤ 中等规模验证", "d28",
               "top-3 配方与 3 种传统配比、base 同池同预算对决(每臂 3B tokens),裁定最终赢家",
               "验证模型 1.5B 有效参数"),
         ],
         loop_label="预测得分反馈,引导下一轮迭代采样",
         note="CLIMB = NVIDIA 论文方法(arXiv:2504.13161)。两级模型:代理 d20 = "
              "435M 有效参数(总参数 0.9B)/ 验证 d28 = 1.5B 有效参数(总参数 "
              "~2.4B),比例 ≈ 1:3.4,与论文的 1:3.7 相当(有效参数 = 参与"
              "矩阵乘法的参数,不含嵌入查找表)。两层成本:搜索层 115 候选 × "
              "640M ≈ 74B tokens;验证层 6 臂 × 3B = 18B。论文机制在四处"
              "不设防 — 我们用五轮连续实验逐一定位并改进优化(改进证据 = "
              "第1~4轮实验,改进验证 = 第5轮实验),见后文『四处机制改进优化』"),
    dict(kind="image", title="搜索过程:三次迭代收敛",
         takeaway="第 1 次迭代随机采样覆盖空间,第 2 次起学习器引导 — 最优逐次提升,第 3 次迭代产出全场最优",
         bullets=[
             ("迭代收敛", "候选分布逐次向高分区域收拢;**第 3 次迭代产出全场最优配方"
                        "(cfg#101,即 climb)**"),
             ("采样预算", "**3 次迭代 64/32/16 ≈ 115 个实测配比**覆盖 15 维空间;"
                        "top-3 配方全部落在 C10 主导的同一热区"),
         ],
          image="search_convergence.png",
          caption="115 个候选配比的得分轨迹(各迭代最佳连成折线)"),
    dict(kind="table", title="六项基准测什么能力 — 得分总表",
          takeaway="覆盖知识与推理两端:知识类 4 项选择题(科学常识 / 科学推理 / 学科知识 / 专家级科学),数学推理类 2 项(应用 / 竞赛,要写解题过程)",
          table=(["基准 · 能力(题量·题型)", "climb", "climb-cfg90", "climb-cfg100", "domainfix", "natural", "uniform", "base"],
                 [["arc_easy · 科学常识(2376·选择)", "72.94%", "72.47%", "72.14%", "72.73%", "72.10%", "72.69%", "**74.24%**"],
                   ["arc_challenge · 科学推理(1172·选择)", "43.60%", "44.03%", "44.71%", "42.83%", "43.34%", "43.34%", "**46.08%**"],
                   ["mmlu_stem · 学科知识(3545·选择)", "29.20%", "29.05%", "**30.16%**", "29.28%", "29.05%", "28.72%", "29.79%"],
                   ["gpqa_diamond · 专家级科学(198·选择)", "21.21%", "24.75%", "21.72%", "23.23%", "24.75%", "20.71%", "**26.26%**"],
                   ["gsm8k · 应用数学推理(1319·写过程)", "**30.10%**", "25.93%", "26.54%", "12.81%", "9.48%", "10.54%", "2.88%"],
                   ["math · 竞赛数学推理(500·写过程)", "**5.80%**", "5.00%", "2.60%", "2.80%", "1.60%", "1.60%", "0.40%"],
                   ["STEM 平均(centered)", "**20.86%**", "20.78%", "20.13%", "17.73%", "17.23%", "16.57%", "17.52%"]]),
            emph=[6], y=1.85, row_h=0.42, size=11,
            widths=[0.30] + [0.10] * 7,
            note="数值 = Centered×100(chance 校正准确率,CLIMB 论文同口径);"
                 "『STEM 平均』= 6 任务 Centered 的算术平均;加粗 = 行最优 — "
                 "选择题 4 行的行最优都是 base(不训练):3B 继续训练的共性回退,"
                 "与配比无关;逐题 ±SE 明细见 report"),
    dict(kind="table", title="提升来自哪里:能力归因 — 对最强传统方法",
          takeaway="提升全部落在数学推理能力 — 两项写过程基准对最强传统配比 2× 大赢;知识类四项全打平:增益在推理,不在知识",
          table=(["基准 · 能力(题量·题型)", "climb", "最强传统方法", "Δ", "结论"],
                 [["gsm8k · 应用数学推理(1319·写过程)", "30.10%", "12.81%(domainfix)", "+17.29pt", "2.3× 大赢"],
                   ["math · 竞赛数学推理(500·写过程)", "5.80%", "2.80%(domainfix)", "+3.00pt", "2.1× 大赢"],
                   ["mmlu_stem · 学科知识(3545·选择)", "29.20%", "29.28%(domainfix)", "−0.08pt", "打平"],
                   ["arc_easy · 科学常识(2376·选择)", "72.94%", "72.73%(domainfix)", "+0.21pt", "打平"],
                   ["arc_challenge · 科学推理(1172·选择)", "43.60%", "43.34%(uniform)", "+0.26pt", "打平"],
                   ["gpqa_diamond · 专家级科学(198·选择)", "21.21%", "24.75%(natural)", "−3.54pt", "打平(噪声内)"]]),
          emph=[0, 1], y=1.9, row_h=0.46,
          widths=[0.33, 0.12, 0.24, 0.12, 0.19],
          note="传统方法 = 非搜索配比:domainfix(人工域配比)/ natural(池自然占比)/ uniform(15 簇等权),"
               "逐基准取最强;选择题 Δ 全部 ≪ SE(逐题 SE 见 report §3);vs base:gsm8k **10.4×**、math **14.5×**"),
    dict(kind="table_image", title="四处机制改进优化 ①:簇大小强制均衡",
          takeaway="失衡的代价:99.05% token 塌缩进单一巨簇,15 维配比退化成 1 个旋钮 — 搜索无从调优,climb 得分反而低于等权臂(−0.36pt);均衡划分恢复 15 个旋钮后,+1.02 → +4.29pt(根因 = 池是连续流形,详见『关键发现③』)",
         table=(["实验轮次", "簇划分策略", "最大簇 token 占比", "有效旋钮",
                 "同轮实验对决:climb vs 簇等权臂"],
                [["第1轮实验(改进前)", "论文式距离合并", "**99.05%**", "1",
                  "**−0.36pt**(方向为负)"],
                 ["第2轮实验", "容量约束均衡(K=15)", "≤ 8.8%", "15",
                  "**+1.02pt**(符号首次转正)"],
                 ["第5轮实验(本轮)", "容量约束均衡(K=15)", "≤ 8.8%", "15",
                  "**+4.29pt**(z=+3.80,6/6 全胜)"]]),
          emph=[2], y=1.72, row_h=0.44, size=12,
          widths=[0.19, 0.25, 0.19, 0.12, 0.25],
          image="reinforcement_d14_cluster_balance.png"),
    dict(kind="table_image", title="四处机制改进优化 ②:评分双信号 — 精度为主,损失补位",
         order_arrow=True,
         takeaway="代理模型小、每任务只测 100 题 — 单靠 accuracy 排名不稳定;引入损失作第二信号,但永不接管(w ≥ 0.5)",
         table=(["环节", "发生了什么 / 我们做了什么"],
                 [["① 论文的做法", "优化目标 = 下游验证集 accuracy,单信号直评"
                   "(1B 模型全量题)"],
                  ["② 我们的场景问题", "搜索层 = 435M 代理 + 每任务 100 题 "
                   "speedrun:单题 = 1pt、SE ≈ ±5pt,相邻配比分辨不开(左图);"
                   "第1轮还把 NLL 缺失的 mmlu_stem 整项丢出目标"],
                  ["③ 我们的改进优化", "score = w·acc_z + (1−w)·nll_z — 精度为主、"
                   "损失补位、**永不接管**(w ≥ 0.5,中图);NLL 缺失按精度单独"
                   "计分"],
                  ["④ 改进验证", "第2~5轮全部在双信号目标下稳定运行;第3轮 w=0"
                   "(纯损失排名)『赢家』gsm8k **−56%** — 反向证明下限必要(右图)"]]),
          y=1.72, row_h=0.52, size=11.5,
          widths=[0.20, 0.80],
          image="reinforcement_d8_dual_signal.png"),
    dict(kind="table_image", title="四处机制改进优化 ③:权重按实测信噪比定",
         order_arrow=True,
         takeaway="题量小的基准,精度抽样噪声大 — 权重必须按实测噪声定,不能拍先验;第3轮先验误判让最强信号权重归零",
         table=(["环节", "发生了什么 / 我们做了什么"],
                 [["① 论文的做法", "多任务精度平均,无逐基准噪声定权 — 题量大小与"
                   "抽样噪声不进入目标函数"],
                  ["② 实测证伪(第3轮实验)", "噪声先验(0.25/K,按选择题口径换算)把 "
                   "gsm8k 噪声高估约 20× → **w = 0**,全场最强信号被判为噪声;"
                   "搜索目标静默翻转为纯损失排名 — 该轮『赢家』gsm8k **−56%**、"
                   "math **−78%**"],
                  ["③ 我们的改进优化", "σ²_noise = p(1−p)/(K_b·(1−c_b)²) **逐基准"
                   "实测**(p = 实测均值,K_b 题量,c_b 蒙对率);w = clip((1+f)/2, "
                   "0.5, 1),f = 1 − σ²_noise/σ²_between — 零人工参数,每次迭代"
                   "用全部实测点重估"],
                  ["④ 改进验证(第5轮实验)", "gsm8k 权重回放 **0.93**(6 项最高);"
                   "gsm8k **30.1%**(uniform 的 2.9×)/ math **5.8%**(base 的 "
                   "14.5×)— 题量最小的两项反成主要得分来源,终选 **6/6 全胜**"]]),
          y=1.72, row_h=0.56, size=11.5,
          widths=[0.20, 0.80],
          image="reinforcement_d8_scoring_snr.png"),
    dict(kind="table_image", title="四处机制改进优化 ④:终选守卫 — 终选必须是实测第一",
         order_arrow=True,
         takeaway="论文式终选只信预测、不信实测;我们的规则反过来:终选必须是实测第一 — 第5轮实验验证,终选即实测赢家,6/6 全胜",
         table=(["环节", "发生了什么 / 我们做了什么"],
                [["① 论文的做法", "终选 = 预测器外推的最优点(argmax 预测分,可为任意"
                  "未实测配比)— 只信预测曲线,该配比从未真正训练过"],
                 ["② 实测证伪(第4轮实验)", "外推点 **19.72%**,输给自家实测点 cfg72 "
                  "**21.42%**(**−1.70pt**)— 预测出来的增益是拟合噪声;"
                  "换种子外推同样垫底(19.90%)"],
                 ["③ 我们的改进优化", "守卫规则:终选 = 实测第一;外推点仅当预测分领先 "
                  "≥ m 才可上场,m = max(0.30, 留出残差 σ)— 领先不足即视为"
                  "拟合噪声"],
                 ["④ 改进验证(第5轮实验)", "守卫生效:终选 cfg#101 = 实测第一,"
                  "**20.86%** 全场最佳,**6/6 全胜 uniform** — 外推反例不可复现"]]),
         y=1.72, row_h=0.56, size=11.5,
         widths=[0.20, 0.80],
         image="reinforcement_d19_extrapolation.png"),
    dict(kind="image", title="最终配比全景:赢家 vs 传统方法 vs 池",
         takeaway="搜索给出的答案与任何人工先验都不同 — 80.4% 押一个池占比仅 5.5% 的簇",
         bullets=[
             ("赢家配方", "climb:**80.4% 给『算术应用题』(C10,池占比 5.5% 的 "
                       "**14.6×**)** + 8.1% 给『形式数学解题』— 两簇合计 88.5%"),
             ("结果稳定", "top-3 配方全部 C10 主导(0.55~0.80);cfg100 另押"
                       "『代数与数论长文』15.8% — 质量分最高,math 反而减半"),
             ("对照组", "uniform = 15 簇等权;池自然占比 ≡ natural 臂;"
                      "domainfix 是域标签配比(数学 60%),**不在语义簇空间,右图单列**"),
         ],
         image="recipe_distribution.png",
         caption="各策略 token 预算在 15 个语义簇上的分配(100% 堆叠,按赢家 α 降序,仅标注 ≥8% 的段);"
                 "数据:optimal_mixture_weights.json / topk_mixture_candidates.json / macro_info.json;"
                 "逐簇内容见下页"),
    dict(kind="image", title="附:15 个语义簇分别是什么",
         takeaway="决定权重的是域与文档功能,不是体量 — C10 池占比仅 5.5%,却拿到 80.4% 预算;其余 13 簇(含 3 个数学簇)全部压在地板",
         image="cluster_content_map.png",
          caption="读法:条长 = 池内体量,右列 = 赢家 α,分隔线以下全部压到 0.1× 地板;"
                  "15 簇全部经抽样细鉴命名(算术应用题 / 形式数学解题 / 代数与数论长文 / "
                  "SMILES 分子性质问答 / 海洋与生态生物学 / 凝聚态与量子光学 / …);"
                  "每簇标签 = 簇名 + 中位长度;逐簇关键词与样本摘录存实验档案 "
                  "cluster_semantics.md(§2b,每簇 50 篇抽样,对比 IDF,零 NPU,seed 固定可复现)"),
    dict(kind="table_image", title="关键发现 ①:文本质量分 ≠ 训练价值",
          takeaway="赢家重仓的簇质量分反而最低,排序完全倒挂(下图)— 质量分衡量『像不像好文本』,不衡量『喂了有没有用』;同样是短练习题:化学没用(0.1× 地板)、算术大赚(14.6×)— 押对领域,比押『高分文本』重要",
          table=(["簇", "质量分(相关性 / 知识性)", "在配方中的角色", "实测效果"],
                   [["算术应用题(赢家重仓)", "3.88 / 3.65 — 三簇最低", "权重 0.80(池占比的 14.6×)",
                     "gsm8k 2.9× / math 3.6×"],
                    ["形式数学解题", "4.33 / 4.12", "权重 0.08", "与赢家并列的机制解释"],
                    ["代数与数论长文(质量分最高)", "4.45 / 4.07", "climb-cfg100 押注(0.16)",
                     "cfg100 的 math 直接减半(2.60% vs 5.80%)"]]),
           emph=[0], y=1.72, row_h=0.55, size=12, widths=[0.24, 0.25, 0.24, 0.27],
           image="finding_quality_value.png"),
    dict(kind="image", title="关键发现 ②:赢家明确,排序有噪声",
          takeaway="top-3 配方全部显著胜出 — 『好配比』这层结论可靠;但配方间名次差落在种子噪声带内,排序不可较真;两次独立搜索也只保证同水平,不保证同一配方",
          bullets=[
              ("赢家明确", "代理分挑出的 3 个决赛配方,中等规模验证**全部显著胜出**"
                          "(z = +3.80 / +3.67 / +3.16)— 搜索锁定『好配比』可靠"),
              ("排序有噪声", "climb 20.86% / cfg90 20.78% / cfg100 20.13% — "
                          "名次差 0.08~0.73pt,远小于种子噪声带(±1.6~3.2pt):"
                          "**top-k 之间不必较真名次**"),
              ("独立搜索 = 同水平", "冷启动重搜(115 个配比全新采样):20.9% vs 热启动 "
                          "21.4%,同一噪声带 — **两次独立搜索同水平、不同配方**"),
          ],
          image="proxy_target_consistency.png",
          caption="climb 配方:代理分 vs 中等规模实测(分数兑现;名次带噪声)"),
    dict(kind="table_image", title="关键发现 ③:窄域池预警 — 用 CLIMB 前先查『簇塌缩』",
          takeaway="这是『领域内配比』的共性陷阱,不是本池特例:窄域池数据同质 → 嵌入成一条连续流形(99% tokens)→ 论文的距离合并必然链式塌缩 — 在专用领域上用此算法,先查池结构",
          table=(["问题", "我们的实证答案"],
                  [["陷阱是什么", "距离合并必然链式塌缩:K=14/21/24/32 实测最大簇 "
                    "**99.0 / 98.5 / 98.3 / 97.5%**;去掉地板直接塌到 K=3;912/912 "
                    "次合并全部『合法』— 塌缩是结构性的,调参救不了"],
                   ["为什么论文没遇到", "论文池 = 800B 通用网页数据 — 多域天然成团,"
                    "合并正好切在域边界上;窄域池没有域边界可切(下图)"],
                   ["谁会遇到", "任何『领域内配比』场景 — 法律 / 医学 / 代码等窄域池:"
                    "数据同质 → 流形致密 → 同样塌缩;通用大池不会"],
                   ["怎么防(我们的答案)", "配比前先查池结构(最大簇占比 / 合并链);"
                    "均衡划分构造 15 个旋钮(= 机制改进①);簇语义靠细鉴审计(§2b)"]]),
           y=1.72, row_h=0.55, size=12, widths=[0.18, 0.82],
           image="finding_pool_manifold.png"),
    dict(kind="table", title="怎么保证没骗自己",
          takeaway="度量衡先校准、规则先于结果、误差逐题实算、结论独立复现 — 五道检查全部量化、全部在案",
          table=(["检查", "数值", "判定"],
                  [["度量衡校准(生成式评测协议)", "停止串 + math 生成预算 256→1024 — 修复前 base 无 EOS、"
                     "99.9% 触顶,答后幻觉污染抽取(A/B:可抽取答案 18.8%→25.8%)",
                     "**PASS**(修复后全轮统一)"],
                   ["评测对账(换环境复评)", "同一底座复评:17.52% vs 基准 17.38%(差 0.14pt)",
                     "**PASS**(阈值 0.2pt)"],
                   ["误差逐题实算", "全部 6 臂 × 6 项基准,逐题二项分布实算误差,不用默认值",
                     "**所有结论带 z / p**"],
                   ["独立复现", "两次独立搜索:20.9% vs 21.4%(种子噪声带 ±1.6~3.2pt)",
                    "**同一水平**"],
                  ["规则先于结果", "判决标准(V1–V5,含加冕规则)实验前写死,留档可查",
                   "**按规则执行**"]]),
          y=1.95, row_h=0.62, size=12.5, widths=[0.24, 0.5, 0.26],
          note="四者齐备,结论才出手;原始数据与判决记录全部在案,可复查"),
    dict(kind="stats", title="这意味着什么",
         takeaway="配比是一个可搜索、可复现、可解释的旋钮 — 这次把它拧对了",
         stats=[("+19.1%", "STEM 平均分相对提升(vs base;对 uniform 6 项全胜)"),
                ("2 次", "独立搜索同水平复现(20.9% / 21.4%)"),
                ("≈3,900 NPU·h", "全部实验总开销(约 163 张卡 · 天)")],
         note="赢家配方可直接复用到下一版训练;自然延伸:更大规模验证 + 质量信号协作 — 值得一起做"),
]


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
    tf = _box(slide, ML, Inches(7.08), Inches(9.2), Inches(0.3))
    _run(tf.paragraphs[0], f"数据配比系统搜索 · 第5轮实验 · 归档 {run_name}",
         9, color=GRAY)
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
    if s.get("order_arrow") and len(rows) >= 2:
        # v10.8: 表左侧竖箭头 — 标注行序 ①→②→③④ (评审反馈: 环节链
        # 的先后更直观)。首行中心 → 末行中心, 线端三角箭头。
        ya0 = y + Inches(0.42 + row_h * 0.5)
        ya1 = y + Inches(0.42 + row_h * (len(rows) - 0.5))
        cx = ML - Inches(0.20)
        conn = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, cx, ya0, cx, ya1)
        conn.line.color.rgb = TEAL
        conn.line.width = Pt(2)
        ln = conn.line._get_or_add_ln()
        ln.append(ln.makeelement(
            qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
        conn.shadow.inherit = False


def _place_image(slide, path, y0, y1):
    """图片保持宽高比, 缩放进 [y0, y1] 高度带 (水平居中, 宽不过 CW)。"""
    size = png_size(path) or (1500, 900)
    scale = min(int(CW) / size[0], int(y1 - y0) / size[1])
    w, h = Emu(int(size[0] * scale)), Emu(int(size[1] * scale))
    pic = slide.shapes.add_picture(
        path, ML + Emu((int(CW) - int(w)) // 2),
        y0 + Emu((int(y1 - y0) - int(h)) // 2), w, h)
    pic.shadow.inherit = False
    return pic


def _image(slide, s):
    y = Inches(1.72)
    nb = len(s.get("bullets") or [])
    if nb:
        # 三点式 (问题/做法/结果) 行数多 → 字号小一档、图下移一档
        three = nb >= 3
        _bullets(slide, s, y=y, size=13 if three else 14,
                 gap=5 if three else 6)
        y = Inches(3.18) if three else Inches(2.95)
    pic = _place_image(slide, s["_path"], y, Inches(6.72))
    if s.get("caption"):
        cy = pic.top + pic.height + Inches(0.06)
        if int(cy) > int(Inches(6.9)):
            cy = Inches(6.92)
        tf = _box(slide, ML, cy, CW, Inches(0.35))
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, s["caption"], 11, color=GRAY)


def _table_image(slide, s):
    """改进优化页 (v10.5 引入): 举证表在上 (问题/结果数据), 机理图在下 — 表格承载
    全部数字 (含 **加粗** 的关键贡献), 图只讲机理; 不设 note/caption,
    空间让给图。"""
    _table(slide, s)
    _, rows = s["table"]
    y0 = (Inches(s.get("y", 1.72))
          + Inches(0.42 + s.get("row_h", 0.46) * len(rows))
          + Inches(0.14))
    if s["_path"]:
        _place_image(slide, s["_path"], y0, Inches(6.95))
    else:
        tf = _box(slide, ML, y0 + Inches(0.5), CW, Inches(0.8))
        p = tf.paragraphs[0]
        p.alignment = 2
        _run(p, f"[缺图: {s['image']}]", 20, bold=True,
             color=RGBColor(0xC0, 0x39, 0x2B))


def _icon(slide, kind, cx, cy):
    """步骤图标: data=数据池+15 簇点 / sample=候选配比堆叠条 / d20,d28=模型方块
    / lgbm=递增柱。cx,cy = 中心(int EMU)。"""
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
    elif kind == "sample":
        # 三条不同切分的堆叠条 = 三个候选配比 (采样多样化)
        cols = [TEAL, NAVY, RGBColor(0x8A, 0x8A, 0x8A),
                RGBColor(0x6C, 0xD9, 0xDE)]
        splits = [(0.55, 0.20, 0.15, 0.10), (0.25, 0.40, 0.20, 0.15),
                  (0.12, 0.18, 0.55, 0.15)]
        y = int(cy) - int(Inches(0.27))
        for sp in splits:
            x = int(cx) - int(Inches(0.42))
            for f, c in zip(sp, cols):
                w = int(Inches(0.84) * f)
                b = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Emu(x),
                                           Emu(y), Emu(w), Inches(0.11))
                b.fill.solid()
                b.fill.fore_color.rgb = c
                b.line.fill.background()
                b.shadow.inherit = False
                x += w
            y += int(Inches(0.19))
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
    """算法流程图: 图标 + ①~⑤ 同色浅青卡片(v10.8 统一 — 原 ⑤ 验证层
    用 navy 区分搜索层/验证层, 评审反馈不需要); ①→② 之间 α 配比芯片;
    ④→② 回环箭头 (预测器引导下一轮迭代采样)。v10.6 起五步(补『配比采样』)
    — 标题允许两行, 字号随列数降档; key 行只放短语(token 量在 body,
    长了会在部分渲染器折三行溢出卡片)。v10.11: 卡内三段分区(标题/body/
    key), body 与 key 之间加细分隔线, key 行垂直居中于下区。v10.12:
    渲染器实际行距 = lnSpc × 字体自然行高(msyh ≈ 1.41 em, 远大于字号),
    v10.11 的分区按『lnSpc × 字号』估算, 4 行 body 实际底界比分隔线低
    不到 0.1in → 重叠; 现按 1.41em 实距重排: lnSpc 1.22→1.05, body 区
    1.20~2.12, 分隔线 2.22, key 区 2.34~2.74, 卡高 2.84。"""
    steps = s["steps"]
    n = len(steps)
    gap = Inches(0.58)
    bw = Emu((int(CW) - int(gap) * (n - 1)) // n)
    by, bh = Inches(2.02), Inches(2.84)
    step = int(bw) + int(gap)
    five = n >= 5
    head_sz, body_sz, key_sz = ((13.5, 11, 10.5) if five
                                else (15, 11.5, 11))
    centers = []
    for i, (head, icon, body, key) in enumerate(steps):
        x = ML + Emu(step * i)
        last = i == n - 1
        sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, by, bw, bh)
        sp.adjustments[0] = 0.055
        sp.fill.solid()
        sp.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xF6)
        sp.line.color.rgb = RGBColor(0x9D, 0xC8, 0xCC)
        sp.line.width = Pt(0.75)
        sp.shadow.inherit = False
        centers.append(int(x) + int(bw) // 2)
        # 意象小图优先 (deck_assets/, 无文字); 缺件回退矢量图标。
        # v10.7: 宽度按卡片宽缩放 (五步卡片 1.96in, 原固定 2.1in 会横向
        # 溢出卡片边界), 等比 640:200
        icon_img = os.path.join(ASSET_DIR, FLOW_ICON_IMG.get(icon, ""))
        if os.path.isfile(icon_img):
            iw = min(Inches(2.1), int(bw) - Inches(0.26))
            ih = Emu(int(iw) * 200 // 640)
            slide.shapes.add_picture(
                icon_img, Emu(int(centers[-1]) - int(iw) // 2),
                Emu(int(by) + int(Inches(0.32)) - int(ih) // 2), iw, ih)
        else:
            _icon(slide, icon, centers[-1], int(by) + int(Inches(0.32)))
        tf = _box(slide, x + Inches(0.14),
                  by + (Inches(0.62) if five else Inches(0.72)),
                  bw - Inches(0.28),
                  Inches(0.58) if five else Inches(0.38))
        p = tf.paragraphs[0]
        if five:
            p.line_spacing = 0.95
        _run(p, head, head_sz, bold=True, color=NAVY)
        tf = _box(slide, x + Inches(0.14),
                  by + (Inches(1.20) if five else Inches(1.12)),
                  bw - Inches(0.28),
                  Inches(0.92) if five else Inches(1.0))
        p = tf.paragraphs[0]
        p.line_spacing = 1.05
        _run(p, body, body_sz, color=DARK)
        if five:
            # v10.11/12: body/key 分隔线 — key 行有了自己的明确区间
            # (位置按 1.41em 实际行距校核, 见 _flow docstring)
            _rect(slide, x + Inches(0.14), by + Inches(2.22),
                  bw - Inches(0.28), Inches(0.012),
                  RGBColor(0xC2, 0xDD, 0xE0))
        tf = _box(slide, x + Inches(0.14),
                  by + (Inches(2.34) if five else bh - Inches(0.58)),
                  bw - Inches(0.28),
                  Inches(0.40) if five else Inches(0.52))
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.line_spacing = 1.05
        _run(p, key, key_sz, bold=True, color=TEAL)
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
    # 迭代回环: ④ 预测器 → ② 采样 (跨 ③, 箭头在卡片下方)
    lx, rx = centers[1], centers[n - 2]
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
    ddir = deck_dir_for(run_dir)
    os.makedirs(ddir, exist_ok=True)
    missing = []
    n_img = 0
    for s in SLIDES:
        img = s.get("image")
        if not img:
            continue
        n_img += 1
        # deck 专属图先找汇报包 (旧布局曾写 run 根层, 兜底兼容并迁移);
        # 实验结果图只认 run 目录 (权威源, 防止用到汇报包里的旧副本)
        p = resolve_chart([ddir, run_dir] if img in DECK_OWNED else [run_dir],
                          img)
        s["_path"] = p
        if p is None:
            missing.append(img)
            continue
        dst = os.path.join(ddir, img)
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy2(p, dst)      # 复制进汇报包 → 包自含可整目录外发
            s["_path"] = dst
    # 清掉汇报包里已退出 CHARTS_USED 的旧图 (历史构建产物, 留着会让人
    # 误以为 pptx 用到; 只动本脚本管理的 *.png)
    for f in glob.glob(os.path.join(ddir, "*.png")):
        if os.path.basename(f) not in CHARTS_USED:
            os.remove(f)
    if missing:
        print(f"✗ 缺图 {len(missing)} 张: {', '.join(missing)}")
        print("  (实验结果图应在 run 目录、deck 专属图应在汇报包目录;"
              " 先跑 report_charts.py + reinforcement_charts.py)")
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
        elif s["kind"] == "table_image":
            _table_image(slide, s)
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
    print(f"  汇报包: {ddir}/ (pptx + {n_img} 张图, 与实验结果目录分离,"
          "可整目录打包外发)")
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
        ddir = deck_dir_for(args.run_dir)
        ok = True
        for name in CHARTS_USED:
            p = resolve_chart([ddir, args.run_dir], name)
            print(f"  {'✓' if p else '✗'} {name}" + (f" → {p}" if p else ""))
            ok = ok and bool(p)
        sys.exit(0 if ok else 1)
    out = args.output or os.path.join(
        deck_dir_for(args.run_dir),
        f"{os.path.basename(os.path.normpath(args.run_dir))}_deck.pptx")
    sys.exit(build(args.run_dir, out))


if __name__ == "__main__":
    main()
