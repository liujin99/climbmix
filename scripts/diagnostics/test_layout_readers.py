#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_layout_readers.py — 布局不变量守卫 (2026-10-10 统一 state/ 布局)。

不变量: run 目录的状态文件 (机器层) 只准经三代解析访问 —
    state/ (现行) → 根级 (2026-10 前平铺归档) → detail/ (2026-10-09
    深整理一代) — 写入端一律 utils.paths.state_file()。

本测试静态扫描全部脚本/源码, 凡出现"状态文件名 + 直接路径拼接"且
不含任何解析标记 (state_file / _run_file / resolve_run_file / hrf /
sfile / _rf / stage1_pair_anywhere / "state" / "detail" / legacy) 的行
即为违规 — 新增直读会当场变红, 强制作者要么走解析器、要么有意识地
豁免 (在 EXEMPT 里登记并写明理由)。

背景: 2026-10-10 用户发现的 extend 流程断链 (run_extend_traineval 在
深整理归档上读不到 optimal_mixture_weights.json) 就是这个 bug 类的第
一次实锤 — 本测试在当时即能抓住。
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

# 机器层状态文件 (精确名 + glob 模式)
STATE_NAMES = [
    "search_state.json", "topk_mixture_candidates.json",
    "optimal_mixture_weights.json", "pipeline_summary.json",
    "cluster_info.json", "cluster_info_cache.json", "macro_info.json",
    "cluster_cache.npz", "macro_labels.npz", "balanced_profile.json",
    "prune_profile.json", "launch_env.json", "remote_config.json",
    "expected_arms.txt", "archive_meta.json", "sampled_dataset.parquet",
    "eval_base_remote.csv",
]
STATE_PATTERNS = [r"eval_[A-Za-z0-9_\-{}$]*\.csv",
                  r"target_arm_[A-Za-z0-9_\-{}`$]*\.json",
                  r"fleet_weights"]

# 解析标记: 行内含任一即视为合规访问
OK_MARKERS = [
    "state_file", "_run_file", "resolve_run_file", "hrf ", "hrf(",
    "sfile ", "sfile(", "_rf(", "_rf ", "stage1_pair_anywhere",
    '"state"', "'state'", "/state/", "state/", '"detail"', "'detail'",
    "detail/", "legacy", "STATE_DIR",
]

# 豁免 (file 相对路径, 正则行匹配) — 每条必须写理由
EXEMPT = [
    # 布局机制自身的定义处 (合法构造三代路径)
    (r"src/climbmix/utils/paths\.py", r".*"),
    (r"scripts/diagnostics/cp4_report\.py", r".*"),
    (r"scripts/diagnostics/tidy_result_dir\.py", r".*"),   # 旧归档整理工具
    # 升级/搬移/归档机制 (合法的目录操作, 非读取)
    (r"runs/lib/stage_gate\.sh", r".*"),
    (r"runs/run_extend_experiment\.sh", r".*"),   # hrf/_SS 三代链本体
    (r"runs/run_extend_traineval\.sh", r".*"),    # sfile 三代链本体
    (r"runs/run_extend_eval\.sh", r".*"),         # 三代判存本体
    (r"scripts/inject_history\.py", r".*"),       # 写 state/ + 新鲜度检查
    # exp_NNNN 工件目录内部 (per-config 产物, 非 run 根状态文件)
    (r"src/climbmix/pipeline/nanochat_cmds\.py", r".*"),
    (r"src/climbmix/remote/remote_executor\.py", r".*"),
    # 缓存目录相对路径 (跟随 --cluster-cache-dir, 布局无关)
    (r"src/climbmix/core/discovery\.py", r".*"),
    (r"src/climbmix/core/cluster_merge\.py", r".*"),
    # 池级临时文件, 非 run 目录
    (r"runs/preprocess_pool\.sh", r".*"),
    # 测试自建 fixture (各自定义自己的世界; 断言新写位置的测试已同步更新)
    (r"scripts/diagnostics/test_.*\.py", r".*"),
    # 文档/帮助/消息行 (非路径操作)
    (r".*", r'\s*#.*'),
    (r".*", r'.*echo .*'),
    (r".*", r'.*help=.*'),
    (r".*", r'.*print[(].*'),
]

SCAN_GLOBS = [
    os.path.join("scripts", "*.py"),
    os.path.join("scripts", "diagnostics", "*.py"),
    os.path.join("src", "climbmix", "**", "*.py"),
    os.path.join("runs", "*.sh"),
    os.path.join("runs", "lib", "*.sh"),
]
SKIP_FILES = {"test_layout_readers.py"}


def line_violates(line):
    if not re.search(r"join\(|/\$|\"\$|\bopen\(|isfile\(|-f ", line):
        return None
    for name in STATE_NAMES:
        if name in line:
            return name
    for pat in STATE_PATTERNS:
        if re.search(pat, line):
            return pat
    return None


def exempted(rel, line):
    for file_re, line_re in EXEMPT:
        if re.fullmatch(file_re, rel) and re.match(line_re, line):
            return True
    return False


def main():
    import glob
    violations = []
    for g in SCAN_GLOBS:
        for path in glob.glob(os.path.join(REPO, g), recursive=True):
            rel = os.path.relpath(path, REPO)
            if os.path.basename(rel) in SKIP_FILES or rel.endswith("_test.py"):
                continue
            if "/__pycache__/" in rel:
                continue
            with open(path, errors="replace") as f:
                for i, line in enumerate(f, 1):
                    line = line.rstrip("\n")
                    name = line_violates(line)
                    if name is None:
                        continue
                    if exempted(rel, line):
                        continue
                    if any(m in line for m in OK_MARKERS):
                        continue
                    violations.append(f"{rel}:{i}: [{name}] {line.strip()[:100]}")
    if violations:
        print(f"✗ 布局违规 {len(violations)} 处 (状态文件直连, 未走三代解析):")
        for v in violations:
            print(f"  {v}")
        print("\n修复: 读取走 resolve_run_file / _run_file / hrf / sfile, "
              "写入走 utils.paths.state_file();")
        print("或在此测试的 EXEMPT 登记豁免并写明理由。")
        return 1
    print("✓ 布局守卫通过 — 状态文件访问全部经三代解析/规范写入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
