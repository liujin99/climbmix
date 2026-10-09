#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tidy_result_dir.py — 收官后的结果目录整理 (过程文件收进子目录)。

背景 (prod5_20260929_201108 实测): 结果目录根层 ~230 项, 其中
  - 115 个 exp_NNNN/ (d20 代理训练的 per-config 工件)
  - ~107 个 *.log (eval_node*/mid_train_node*/eval_*/dispatch_*/search.log)
  - ~45 个 结果+状态 (eval_*.csv, search_state.json, PNG/MD, ...)
人看的和脚本读的全混在一起。

整理规则 (只挪两类, 结果与状态全部留在根):
    logs/ <- 根目录全部 *.log
    exps/ <- exp_NNNN/ 搜索工件目录

为什么安全:
  - 分析/报告脚本 (cp4_report / report_charts / recipe_report /
    cluster_peek / final_report) 只读根目录的 search_state.json /
    eval_*.csv / topk_mixture_candidates.json / cluster_info*.json /
    launch_env.json / fleet_weights 等 — 不读根目录 .log, 不读 exp_NNNN/
    (final_report.py 的 exp_NNNN 只出现在注释里)。
  - exp_NNNN/ 与根日志只在【运行中/续跑】被 remote_executor 读写 — 因此
    本工具只允许在已归档 (archive_meta.json 存在) 的目录上执行; 未归档
    需 --force 自担风险。
  - fleet_monitor / preflight_launch 读 run 根的 search.log /
    dispatch_*.log, 但那是【运行中】的监控工具, 归档后不再使用。

幂等: 重复执行无副作用 (已挪过的不再匹配, 同名冲突直接拒绝)。

用法:
    python3 scripts/diagnostics/tidy_result_dir.py result/prod5_xxx           # dry-run
    python3 scripts/diagnostics/tidy_result_dir.py result/prod5_xxx --apply   # 执行
"""
import argparse
import os
import re
import shutil
import sys

EXP_RE = re.compile(r"^exp_\d{4,}$")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="整理已收官的搜索结果目录: *.log -> logs/, exp_NNNN/ -> exps/")
    ap.add_argument("run_dir", help="结果目录 (如 result/prod5_20260929_201108)")
    ap.add_argument("--apply", action="store_true",
                    help="实际移动 (缺省只打印计划)")
    ap.add_argument("--force", action="store_true",
                    help="目录无 archive_meta.json (未归档) 时仍执行")
    args = ap.parse_args()

    rd = args.run_dir
    if not os.path.isdir(rd):
        print(f"refuse: 不是目录: {rd}")
        return 2
    if not os.path.isfile(os.path.join(rd, "search_state.json")):
        print(f"refuse: {rd} 下无 search_state.json — 不是搜索结果目录")
        return 2
    if not args.force and not os.path.isfile(os.path.join(rd, "archive_meta.json")):
        print("refuse: 无 archive_meta.json (未收官归档) — 收官后再整理, "
              "确认无碍用 --force")
        return 2

    entries = sorted(os.listdir(rd))
    logs = [e for e in entries
            if e.endswith(".log") and os.path.isfile(os.path.join(rd, e))]
    exps = [e for e in entries
            if EXP_RE.match(e) and os.path.isdir(os.path.join(rd, e))]

    n_before = len(entries)
    n_after = (n_before - len(logs) - len(exps)
               + (1 if logs else 0) + (1 if exps else 0))
    print(f"{rd}")
    print(f"  根条目: {n_before} -> 约 {n_after}"
          f" (logs/ {len(logs)} 项, exps/ {len(exps)} 项)")
    if logs:
        print(f"  logs/: {len(logs)} 个日志, 例: {', '.join(logs[:3])}"
              f"{' ...' if len(logs) > 3 else ''}")
    if exps:
        print(f"  exps/: {len(exps)} 个搜索工件目录, 例: {', '.join(exps[:3])}"
              f"{' ...' if len(exps) > 3 else ''}")
    if not logs and not exps:
        print("  已整洁, 无可挪项")
        return 0

    # 目标子目录里的同名冲突 = 拒绝 (绝不覆盖)
    for sub, names in (("logs", logs), ("exps", exps)):
        dst = os.path.join(rd, sub)
        if os.path.isdir(dst):
            clash = [x for x in names if os.path.exists(os.path.join(dst, x))]
            if clash:
                print(f"refuse: {sub}/ 已存在同名项 {clash[:3]} — 不覆盖")
                return 2

    if not args.apply:
        print("  (dry-run: 加 --apply 执行移动)")
        return 0

    for sub, names in (("logs", logs), ("exps", exps)):
        if not names:
            continue
        dst = os.path.join(rd, sub)
        os.makedirs(dst, exist_ok=True)
        for x in names:
            shutil.move(os.path.join(rd, x), os.path.join(dst, x))
        print(f"  moved {len(names)} 项 -> {sub}/")
    print(f"  完成: 根条目 {n_before} -> {len(os.listdir(rd))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
