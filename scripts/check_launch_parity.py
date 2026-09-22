#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════
#  check_launch_parity.py — 两轮发射配置对账（只读诊断）
#
#  用法:
#    python3 scripts/check_launch_parity.py result/prod4_current result/prod5_current
#
#  背景 (2026-09-22 用户裁决): 发射入口 = run_experiment.sh 本身, 大实验
#  以 EXP_NAME 区分, 不设每轮 launcher 包装。跨轮"同值续轮"的机器对照
#  价值保留为本工具: 引擎发射后 1 分钟内即写出 launch_env.json +
#  remote_config.json, diff 两个 run 目录, 人工过一眼 delta 清单 ——
#  预期变更 (本轮 EDIT 的键) 之外若出现意外偏离, 趁搜索还没烧几小时
#  就地停掉重来 (rm -rf result/<名>_current 后修正重发)。
#
#  只读 + stdlib-only; 不做判定 (哪些 delta 合法是人的裁决), 只列账。
# ═══════════════════════════════════════════════════════════════════
import json
import os
import sys


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src_dir, dst_dir = sys.argv[1], sys.argv[2]

    for fname in ("launch_env.json", "remote_config.json"):
        src = load(os.path.join(src_dir, fname))
        dst = load(os.path.join(dst_dir, fname))
        print(f"═══ {fname} ═══")
        if src is None or dst is None:
            missing = []
            if src is None:
                missing.append(f"{src_dir}/{fname}")
            if dst is None:
                missing.append(f"{dst_dir}/{fname}")
            print(f"  [缺文件] {', '.join(missing)}"
                  f"{'（目标轮还没发射?）' if dst is None else ''}")
            continue
        keys = sorted(set(src) | set(dst))
        same = changed = 0
        for k in keys:
            a, b = src.get(k, "<缺>"), dst.get(k, "<缺>")
            if a == b:
                same += 1
                continue
            changed += 1
            print(f"  [Δ] {k}: {src_dir.rsplit('/', 1)[-1]}={a!r} → "
                  f"{dst_dir.rsplit('/', 1)[-1]}={b!r}")
        print(f"  —— {same} 键同值, {changed} 键不同（上面清单 = 本轮全部"
              f"偏离, 逐项过目: 预期变更 vs 意外偏离）")
    print("\n[提示] 意外偏离的处理: 停引擎 → rm -rf 目标目录（缓存种子重拷）"
          " → 修正 env → 重发")


if __name__ == "__main__":
    main()
