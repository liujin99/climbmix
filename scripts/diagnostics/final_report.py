#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  final_report.py — 大报告终报步骤 (整个大实验的最后一步, QuaDMix 式)
#
#  用法:
#    python3 scripts/diagnostics/final_report.py result/prod5_current \
#        --base-expected 0.1746 \
#        --arms climb-cfg72,climb-cfg25,climb-cfg88,uniform,natural,domainfix
#
#  结构对应 (2026-09-22 用户裁决):
#    每配方结果目录   = 搜索层 exp_NNNN/ (meta.json = 权重+分数+逐任务+日志)
#    子报告           = report.md 的搜索节 + CP4 判定节 + 赢家配方节
#                       (随臂落地自动刷新, 过程可见)
#    大报告(本脚本)   = 实验全部完成后的最后一步: 臂盘点 → 判定节刷新
#                       (锚点正式判定) → 配方节刷新(链) → 终报印章节
#
#  输入: RUN_DIR; 可选 --base-expected (锚点 PASS/FAIL 校准值, 缺省
#  UNJUDGED 只报数); 可选 --arms (预期臂清单, 缺失 → 印章标 DRAFT 且
#  exit 1, 防止在臂未齐时误终报); --ref 默认 uniform。
#  幂等: 三节均 marker 替换, 重跑无害。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp4_report import discover_arms, parse_eval_csv

SEAL_BEGIN = "<!-- final_report:begin -->"
SEAL_END = "<!-- final_report:end -->"

# P-0 预注册的固定基线臂 (prod5 命名); 历史命名用 expected_arms.txt 覆盖
FIXED_BASELINES = ["uniform", "natural", "domainfix"]


def derive_expected_arms(run_dir):
    """从工件推导预期臂清单 (自动终报的核心 — "最后一臂落地"机器自判):
    topk_mixture_candidates.json 的 config_id → climb-cfg{id} ×k
    + 固定基线 uniform/natural/domainfix (P-0)
    + RUN_DIR/expected_arms.txt 覆盖/追加 (每行一个臂名, 历史命名或
    条件臂如 no-claim 精确控制用)。
    topk 缺失 → (None, 原因) — 不猜, 拒绝自动终报。"""
    topk_path = os.path.join(run_dir, "topk_mixture_candidates.json")
    if not os.path.isfile(topk_path):
        return None, "topk_mixture_candidates.json 缺失 — 无法推导预期臂"
    try:
        with open(topk_path) as f:
            topk = json.load(f)
        ids = [int(c["config_id"]) for c in topk.get("candidates") or []]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return None, f"topk 文件不可读 ({e})"
    if not ids:
        return None, "topk 候选为空 — 无法推导预期臂"
    expected = [f"climb-cfg{i}" for i in ids] + list(FIXED_BASELINES)
    override = os.path.join(run_dir, "expected_arms.txt")
    if os.path.isfile(override):
        with open(override) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        if names:
            return names, f"覆盖自 expected_arms.txt ({len(names)} 臂)"
    return expected, f"推导自 topk({len(ids)}) + 基线{FIXED_BASELINES}"


def main():
    ap = argparse.ArgumentParser(
        description="Final report: the closing step of the whole experiment")
    ap.add_argument("run_dir", nargs="?", default="result/prod5_current")
    ap.add_argument("--base-expected", type=float, default=None,
                    help="base 锚点校准值 (缺省 UNJUDGED 只报数)")
    ap.add_argument("--arms", default="",
                    help="预期臂清单 (逗号分隔; 缺失 → DRAFT + exit 1)")
    ap.add_argument("--ref", default="uniform", help="CP4 判定对照臂")
    ap.add_argument("--auto", action="store_true",
                    help="自动模式 (落臂钩子用): 预期臂清单从 topk+基线推导, "
                         "臂未齐只刷新判定/配方节不盖章; 齐了自动盖 FINAL 章")
    args = ap.parse_args()

    # ── 1) 刷新判定节 + 配方节 (cp4 内置配方链) ──
    here = os.path.dirname(os.path.abspath(__file__))
    cp4 = os.path.join(here, "cp4_report.py")
    cmd = [sys.executable, cp4, args.run_dir, "--ref", args.ref]
    if args.base_expected is not None:
        cmd += ["--base-expected", str(args.base_expected)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        tail = (r.stdout or r.stderr or "").strip().splitlines()
        print(f"[!] cp4_report 失败 — 无法终报: "
              f"{tail[-1] if tail else 'rc!=0'}")
        return 1
    for line in (r.stdout or "").strip().splitlines()[-2:]:
        if line.strip():
            print(line)

    # ── 2) 臂盘点 + 完整性 ──
    if args.auto:
        expected, src = derive_expected_arms(args.run_dir)
        if expected is None:
            print(f"[·] {src} — 判定/配方节已刷新, 不自动终报")
            return 0
        print(f"(预期臂清单 {src})")
    else:
        expected = [a.strip() for a in args.arms.split(",") if a.strip()]
    arms = discover_arms(args.run_dir)
    stems = {}
    for a in arms:
        d = parse_eval_csv(os.path.join(args.run_dir, f"eval_{a}.csv"))
        if d and d["stem"] is not None:
            stems[a] = d["stem"]
    ranked = sorted(stems, key=lambda a: -stems[a])
    base = parse_eval_csv(os.path.join(args.run_dir, "eval_base_remote.csv"))
    missing = [a for a in expected if a not in stems]
    complete = not missing

    if args.auto and missing:
        # 自动模式: 臂未齐 = 实验未完成, 判定/配方节已刷新即够,
        # 不盖 DRAFT 章 (过程态不落终报印章, 印章只属于完成时刻)
        print(f"[·] 臂未齐 ({len(expected) - len(missing)}/{len(expected)}), "
              f"缺: {', '.join(missing)} — 终报待最后一臂落地自动盖章")
        return 0

    # ── 3) 终报印章节 (report.md 末尾, marker 幂等) ──
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    L = ["## 终报 (Final Report)", "",
         f"- **生成:** {ts} — `final_report.py`（整个大实验的最后一步）",
         f"- 臂盘点: **{len(stems)} 臂落地**"
         + (f" / 预期 {len(expected)}" if expected else "")
         + (f"；缺: {', '.join(missing)}" if missing else "")]
    if ranked:
        top = " › ".join(f"{a} {stems[a]:.4f}" for a in ranked[:3])
        L.append(f"- 最终排名: {top}"
                 + (f" …（共 {len(ranked)} 臂）" if len(ranked) > 3 else ""))
    if base and base["stem"] is not None:
        judged = "（PASS/FAIL 见 CP4 判定节）" if args.base_expected is not None \
            else "（未判定 — 传 --base-expected）"
        L.append(f"- base 锚点: {base['stem']:.4f} {judged}")
    if ranked:
        uni = [a for a in ranked if a in ("uniform", "random3b", "random")]
        if uni:
            L.append(f"- 头条: 赢家 {ranked[0]} vs uniform 族最强 {uni[0]} "
                     f"Δ = {stems[ranked[0]] - stems[uni[0]]:+.4f}")
    L.append(f"- 状态: "
             + ("**FINAL — 预期臂全部落地**" if complete
                else "**DRAFT — 仍有预期臂未落地**"))
    L.append("")

    rp = os.path.join(args.run_dir, "report.md")
    existing = open(rp).read() if os.path.isfile(rp) else ""
    body = SEAL_BEGIN + "\n" + "\n".join(L).rstrip("\n") + "\n" + SEAL_END + "\n"
    if SEAL_BEGIN in existing and SEAL_END in existing:
        pre = existing.split(SEAL_BEGIN)[0].rstrip("\n")
        post = existing.split(SEAL_END, 1)[1].lstrip("\n")
        new = (pre + "\n\n" + body.rstrip("\n")
               + (("\n\n" + post) if post.strip() else "\n"))
    else:
        new = (existing.rstrip("\n") + "\n\n" if existing.strip() else "") + body
    tmp = rp + ".tmp"
    with open(tmp, "w") as f:
        f.write(new)
    os.replace(tmp, rp)

    print(f"\n[{'OK' if complete else 'DRAFT'}] 大报告完成 → {rp}")
    print(f"  结构: 搜索子报告 → CP4 判定 → 赢家配方 → 终报印章"
          f"（{len(stems)} 臂 + base）")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
