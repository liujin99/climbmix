#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tidy_result_dir.py — 【legacy 工具】2026-10-09 深整理一代的归档整理。

2026-10-10 起写入端统一 state/ 布局 (utils/paths.py), 新 run 的根层
出生即发布形态 — 不再需要本工具。保留它只为服务存量归档:
  - 2026-10 前的平铺归档 (prod1-4): --deep --apply 可整理成 9 项形态
    (该代形态的根层保留集与本工具一致)
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
出生即整洁; mark_completed 归档时自动追加深整理 (--deep --apply)。

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

    # 深度档的根层保留集: 主报告 + report_charts.py 五张决策图
    # (traineval 验证轮自 2026-10-10 起为独立平级 run, 不再住进实验目录)
ROOT_KEEP_FILES = {
    "report.md",
    "search_convergence.png",
    "arms_main_results.png",
    "proxy_target_consistency.png",
    "cluster_alpha_vs_score.png",
    "best_vs_worst_heatmap.png",
}
ROOT_KEEP_DIRS = {"logs", "exps", "detail"}


def _exists_anywhere(rd, name):
    return any(os.path.exists(os.path.join(rd, sub, name))
               for sub in ("", "detail"))


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
        if e in ROOT_KEEP_FILES and os.path.isfile(os.path.join(rd, e)):
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="整理已收官的搜索结果目录: 基础档 *.log -> logs/, "
                    "exp_NNNN/ -> exps/; 深度档 --deep 再收 detail/, "
                    "根层只留 report.md + 决策图")
    ap.add_argument("run_dir", help="结果目录 (如 result/prod5_20260929_201108)")
    ap.add_argument("--apply", action="store_true",
                    help="实际移动 (缺省只打印计划)")
    ap.add_argument("--deep", action="store_true",
                    help="深度档: 根层只留 report.md + 5 张决策图 + "
                         "logs/ + exps/ + detail/, 其余进 detail/")
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
