#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tidy_result_dir.py — 【legacy 工具】2026-10-09 深整理一代的归档整理。

2026-10-10 起写入端统一 state/ 布局 (utils/paths.py), 新 run 的根层
出生即发布形态 — 不再需要本工具 (旧注"mark_completed 归档时自动追加深
整理"已随 state/ 迁移移除, 现无自动调用方)。保留它服务两类手工场景:
  - 2026-10 前的平铺归档 (prod1-4): --deep --apply 可整理成 9 项形态
    (该代形态的根层保留集与本工具一致)
  - 任意形态 (含 state/ 形态): NON_READER_MDS 扫收纳 — 搜索期管线
    (validation_report.md) 与判读工具 (cluster_peek.md) 落在根层的
    非读者 md 归位 detail/
  - 读取端 (cp4_report.resolve_run_file 三代链) 对新旧形态通吃,
    整理与否不影响任何工具

整理规则 (只对已归档 run, archive_meta.json 守卫):
    logs/ <- 根目录全部 *.log
    exps/ <- exp_NNNN/ 搜索工件目录
    detail/ <- 其余全部 (深度档 --deep; report.md 链接同步改写)

为什么安全:
  - 分析/报告脚本经双布局解析读状态文件 (cp4_report.resolve_run_file);
    exp_NNNN/ 与根日志只在【运行中/续跑】被 remote_executor 读写 — 因此
    本工具只允许在已归档 (archive_meta.json 存在, 根或 detail/) 的目录上
    执行; 未归档需 --force 自担风险。
  - fleet_monitor / preflight_launch 读 run 根的 search.log /
    dispatch_*.log, 但那是【运行中】的监控工具, 归档后不再使用。

幂等: 重复执行无副作用 (已挪过的不再匹配, 同名冲突直接拒绝, 链接改写
只匹配裸文件名 — 已带 detail/ 前缀的不会二次改写)。

注: 2026-10-09 起写入端已改 (日志进 logs/, 搜索工件进 exps/) — 新 run
出生即整洁 (自动深整理已移除, 见上)。

用法:
    python3 scripts/diagnostics/tidy_result_dir.py result/prod5_xxx            # dry-run
    python3 scripts/diagnostics/tidy_result_dir.py result/prod5_xxx --apply    # 基础档
    python3 scripts/diagnostics/tidy_result_dir.py result/prod5_xxx --deep --apply  # 深度档
"""
import argparse
import os
import re
import shutil
import sys

EXP_RE = re.compile(r"^exp_\d{4,}$")

    # 深度档的根层保留集 = 读者层 (2026-10-12 修正): 全部图表 (*.png) +
    # 全部报告 (*.md)。旧版固定集只保 report.md + 5 张决策图, 把
    # report_generator (domain_distribution / predictor_scatter) 与
    # recipe_report (recipe_winner_vs_baselines) 的根层图表和次级报告
    # MD 全扫进了 detail/ — 违反 "根层 = 发布形态" 裁决 (prod5 归档实锤,
    # 用 --restore-reader-layer 恢复)。
ROOT_KEEP_SUFFIXES = (".png", ".md", ".pptx")   # pptx = make_deck 发布物
# state/ = 2026-10-10 后的机器层目录, 整体保留 (扫进 detail/state/ 会让
# 三代解析 state/ 代整层失效 — fixture 实锤)
ROOT_KEEP_DIRS = {"state", "logs", "exps", "detail"}

# 后缀规则的显式例外 (2026-10-12 裁决: 根层 = 纯读者层, report.md +
# 决策图): 这三个 md 不是读者层 — report.md 不链接它们, 内容已内嵌或
# 被覆盖。深度档扫入 detail/, restore 不再拉回:
#   cluster_semantics.md  §2b 生成缓存 — 全文内嵌 report.md §2b, 对读者
#                          纯冗余; recipe_report 读三代/写恒 state/
#   cluster_peek.md       判读深潜全量版 (每簇 40 样本; §2b 为其紧凑版),
#                          手工 CLI 产物, 结论已回填 docs/report.md
#   validation_report.md  搜索期双臂中间视图 (climb vs uniform, 只落盘
#                          不上屏 — 2026-09-29 裁决, 终报 = report.md)
NON_READER_MDS = ("cluster_semantics.md", "cluster_peek.md",
                  "validation_report.md")


def _exists_anywhere(rd, name):
    # 三代全查 (state/ → 根层 → detail/): 入口守卫对 2026-10-10 后的
    # state/ 形态 run 同样放行 — NON_READER_MDS 扫收纳靠手工跑本工具,
    # 新形态 run 不能被挡在门外 (fixture 实锤: search_state.json 在
    # state/ 时旧版只查根层/detail 误拒)
    return any(os.path.exists(os.path.join(rd, sub, name))
               for sub in ("state", "", "detail"))


def plan_deep(rd, claimed):
    """深度档移动计划: 根层除保留集外的一切 (含目录与隐藏点文件) ->
    detail/。claimed = 基础档已认领的条目 — 不重复挪。

    隐藏点文件 (.done_* / .dispatch_*.lock / .fingerprint_* /
    .validation_fleet/ / .ipynb_checkpoints/ ...) 是运行期生命周期工件,
    归档 run 上惰性 — 一并收进 detail/; 复活 (stage_gate
    _restore_completed) 时自动归位根层。"""
    moves = []
    for e in sorted(os.listdir(rd)):
        if e in claimed:
            continue
        if e in ROOT_KEEP_DIRS and os.path.isdir(os.path.join(rd, e)):
            continue
        if (e.endswith(ROOT_KEEP_SUFFIXES) and e not in NON_READER_MDS
                and os.path.isfile(os.path.join(rd, e))):
            continue
        moves.append(e)
    return moves


def rewrite_report_links(rd, moved):
    """report.md 内嵌的 PNG/MD 相对链接改写为 detail/ 前缀 (只改真被挪走
    的文件名; 已带前缀的不匹配 — 幂等)。返回改写处数。"""
    rp = os.path.join(rd, "report.md")
    if not os.path.isfile(rp):
        return 0
    with open(rp, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    n = 0
    for name in moved:
        if not (name.endswith(".png") or name.endswith(".md")):
            continue
        old, new = f"]({name})", f"](detail/{name})"
        if old in txt:
            n += txt.count(old)
            txt = txt.replace(old, new)
    if n:
        with open(rp, "w", encoding="utf-8") as f:
            f.write(txt)
    return n


def plan_restore_reader(rd):
    """恢复计划: detail/ 内的读者层文件 (*.png / *.md) 搬回根层 —
    修正 2026-10-12 前深度档的误收纳 (见 ROOT_KEEP_SUFFIXES 注释)。
    非读者 md (NON_READER_MDS) 不拉回 — 它们留在 detail/ 是归位而非误收。"""
    dpath = os.path.join(rd, "detail")
    if not os.path.isdir(dpath):
        return []
    return [e for e in sorted(os.listdir(dpath))
            if e.endswith(ROOT_KEEP_SUFFIXES)
            and e not in NON_READER_MDS
            and os.path.isfile(os.path.join(dpath, e))]


def rewrite_report_links_back(rd, restored):
    """report.md 内被深度档改写成 detail/ 前缀的链接改回根层相对
    路径 (只针对本次真正搬回的文件名; 幂等)。返回改写处数。"""
    rp = os.path.join(rd, "report.md")
    if not os.path.isfile(rp):
        return 0
    with open(rp, encoding="utf-8", errors="replace") as f:
        txt = f.read()
    n = 0
    for name in restored:
        old, new = f"](detail/{name})", f"]({name})"
        if old in txt:
            n += txt.count(old)
            txt = txt.replace(old, new)
    if n:
        with open(rp, "w", encoding="utf-8") as f:
            f.write(txt)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(
        description="整理已收官的搜索结果目录: 基础档 *.log -> logs/, "
                    "exp_NNNN/ -> exps/; 深度档 --deep 再收 detail/, "
                    "根层只留读者层 (全部 *.png / *.md, 非读者 md 除外)")
    ap.add_argument("run_dir", help="结果目录 (如 result/prod5_20260929_201108)")
    ap.add_argument("--apply", action="store_true",
                    help="实际移动 (缺省只打印计划)")
    ap.add_argument("--deep", action="store_true",
                    help="深度档: 根层只留读者层 (*.png/*.md, "
                         "NON_READER_MDS 除外) + logs/ + exps/ + detail/, "
                         "其余进 detail/")
    ap.add_argument("--restore-reader-layer", action="store_true",
                    help="恢复模式: detail/ 内的 *.png/*.md 搬回根层 + "
                         "report.md 链接归位 (修正旧深度档误收纳)")
    ap.add_argument("--force", action="store_true",
                    help="目录无 archive_meta.json (未归档) 时仍执行")
    args = ap.parse_args()

    rd = args.run_dir
    if not os.path.isdir(rd):
        print(f"refuse: 不是目录: {rd}")
        return 2
    if not _exists_anywhere(rd, "search_state.json"):
        print(f"refuse: {rd} 下无 search_state.json — 不是搜索结果目录")
        return 2
    if not args.force and not _exists_anywhere(rd, "archive_meta.json"):
        print("refuse: 无 archive_meta.json (未收官归档) — 收官后再整理, "
              "确认无碍用 --force")
        return 2

    if args.restore_reader_layer:
        names = plan_restore_reader(rd)
        if not names:
            print("  detail/ 内无 *.png / *.md — 无可恢复项")
            return 0
        clash = [e for e in names if os.path.exists(os.path.join(rd, e))]
        if clash:
            print(f"  ⚠ 根层已存在同名项, 跳过: {', '.join(clash)}")
            names = [e for e in names if e not in clash]
        if not names:
            return 0
        print(f"  恢复读者层: {len(names)} 项 detail/ -> 根层:")
        for e in names:
            print(f"    {e}")
        if not args.apply:
            print("  (dry-run: 加 --apply 执行移动)")
            return 0
        for e in names:
            os.replace(os.path.join(rd, "detail", e), os.path.join(rd, e))
        n = rewrite_report_links_back(rd, names)
        print(f"  ✓ 搬回 {len(names)} 项; report.md 链接归位 {n} 处")
        return 0

    entries = sorted(os.listdir(rd))
    logs = [e for e in entries
            if e.endswith(".log") and os.path.isfile(os.path.join(rd, e))]
    exps = [e for e in entries
            if EXP_RE.match(e) and os.path.isdir(os.path.join(rd, e))]
    deep = plan_deep(rd, set(logs) | set(exps)) if args.deep else []

    n_before = len(entries)
    print(f"{rd}")
    if logs or exps:
        print(f"  基础档: 根条目 {n_before} -> "
              f"{n_before - len(logs) - len(exps) + (1 if logs else 0) + (1 if exps else 0)}"
              f" (logs/ {len(logs)} 项, exps/ {len(exps)} 项)")
    if deep:
        stay = [e for e in entries
                if e not in deep and e not in logs and e not in exps
                and not e.startswith(".")]
        print(f"  深度档: detail/ {len(deep)} 项; 根层保留 {len(stay)} 项:")
        print(f"    {', '.join(stay) if stay else '(无)'}")
    if not logs and not exps and not deep:
        print("  已整洁, 无可挪项")
        return 0

    plans = [("logs", logs), ("exps", exps), ("detail", deep)]

    # logs/ 与 exps/ 的同名冲突 = 拒绝 (绝不覆盖 — 它们不会被事后再生)。
    # detail/ 不在此列: 分析工具重跑会在根层再生成产物, 再整理时按
    # "根层新版归位" 覆盖 detail/ 旧档 (apply 段处理, 类型冲突仍跳过)。
    for sub, names in (("logs", logs), ("exps", exps)):
        if not names:
            continue
        dst = os.path.join(rd, sub)
        if os.path.isdir(dst):
            clash = [x for x in names if os.path.exists(os.path.join(dst, x))]
            if clash:
                print(f"refuse: {sub}/ 已存在同名项 {clash[:3]} — 不覆盖")
                return 2

    if not args.apply:
        print("  (dry-run: 加 --apply 执行移动)")
        return 0

    for sub, names in plans:
        if not names:
            continue
        dst = os.path.join(rd, sub)
        os.makedirs(dst, exist_ok=True)
        replaced = 0
        for x in names:
            src = os.path.join(rd, x)
            d = os.path.join(dst, x)
            if os.path.exists(d):
                if sub != "detail" or os.path.isdir(d) != os.path.isdir(src):
                    print(f"  refuse: {sub}/{x} 冲突 — 不覆盖, 留在根层")
                    continue
                if os.path.isdir(d):
                    shutil.rmtree(d)
                replaced += 1
            shutil.move(src, d)
        msg = f"  moved {len(names)} 项 -> {sub}/"
        if replaced:
            msg += f" (其中 {replaced} 项覆盖 detail/ 旧档 — 再生成归位)"
        print(msg)
    if deep:
        n = rewrite_report_links(rd, deep)
        if n:
            print(f"  report.md 链接改写 {n} 处 -> detail/")
    final = sorted(os.listdir(rd))
    print(f"  完成: 根条目 {n_before} -> {len(final)}"
          f"{'' if not args.deep else ' (' + ', '.join(final) + ')'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
