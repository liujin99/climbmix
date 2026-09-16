#!/usr/bin/env python3
# ═════════════════════════════════════════════════════════════════════
#  clean_derived_data.py — 臂本地派生数据清理 (战后清单 #8)
#
#  用法:
#    python3 scripts/clean_derived_data.py --run-dir result/prod4_current
#        # dry-run: 列出可清理的 {arm}_shards/{arm}_mixed + 可释放 GB
#    python3 scripts/clean_derived_data.py --run-dir result/prod4_current \
#        --arms climb,cfg11 --apply
#    python3 scripts/clean_derived_data.py --run-dir result/prod4_current --apply
#        # --arms 省略 = 自动发现所有已成功臂
#
#  背景 (prod4 2026-09-16): 臂产物上传 OBS 后本地不清理 (proxy 路径有
#  清理, 臂路径没有) — 每臂 ~31G (3B) / ~62G (6B) 常驻, 多臂并行吃穿
#  /work。OBS 上有内容键化的完整副本 (mixture_data_k*), 本地副本在臂
#  成功后即失去存在必要。
#
#  守卫 (防脚枪, --force 才可越过):
#    1. 臂未成功 (run_dir 无 .done_mid_train_{arm}) → 拒绝 — 数据是
#       失败重试的本钱 (dispatch 守卫 + 重新混料都依赖本地 .done 链)
#    2. 目录内无 .done → 拒绝 — 混料进行中或崩溃残留, 归 arm_engine
#       的 partial-wipe 逻辑管
#
#  删除内容: {arm}_shards/ 与 {arm}_mixed/ 整目录。eval CSV / 审计
#  台账 / cluster_cache 全部在 run_dir 平级, 不受影响; .done_mid_train
#  语义 (SUCCEEDED 臂重发 = no-op) 也不受影响。
#
#  只读+删除, stdlib-only; 默认 dry-run。
# ═════════════════════════════════════════════════════════════════════
import argparse
import os
import shutil
import sys


def _dir_gb(path: str) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2 ** 30


def discover_arms(run_dir: str) -> list:
    """run_dir 下所有 {arm}_mixed 目录 → 臂名 (有序去重)。"""
    arms = set()
    try:
        names = os.listdir(run_dir)
    except OSError:
        return []
    for n in names:
        if n.endswith("_mixed"):
            arm = n[:-len("_mixed")]
            if arm:
                arms.add(arm)
    return sorted(arms)


def plan_cleanup(run_dir: str, arms, force: bool) -> dict:
    """对每个臂判定 deletable / 拒绝原因。返回 {arm: {dirs, gb, reason}}."""
    out = {}
    for arm in arms:
        shards = os.path.join(run_dir, f"{arm}_shards")
        mixed = os.path.join(run_dir, f"{arm}_mixed")
        dirs = [d for d in (shards, mixed) if os.path.isdir(d)]
        if not dirs:
            out[arm] = dict(dirs=[], gb=0.0,
                            reason="no local dirs (already clean)")
            continue
        if not force and not os.path.isfile(
                os.path.join(run_dir, f".done_mid_train_{arm}")):
            out[arm] = dict(dirs=dirs, gb=sum(_dir_gb(d) for d in dirs),
                            reason="arm not SUCCEEDED "
                                   "(no .done_mid_train_{arm}) — data is "
                                   "retry capital; --force to override")
            continue
        incomplete = [d for d in dirs
                      if not os.path.isfile(os.path.join(d, ".done"))]
        if not force and incomplete:
            out[arm] = dict(dirs=dirs, gb=sum(_dir_gb(d) for d in dirs),
                            reason=f"no .done inside "
                                   f"{[os.path.basename(d) for d in incomplete]}"
                                   f" — mix in progress or crashed residue")
            continue
        out[arm] = dict(dirs=dirs, gb=sum(_dir_gb(d) for d in dirs),
                        reason=None)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="clean arm local derived data after OBS upload "
                    "(dry-run by default)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arms", default="",
                    help="逗号分隔臂名; 省略 = 自动发现所有 *_mixed")
    ap.add_argument("--apply", action="store_true",
                    help="真删 (默认 dry-run 只列清单)")
    ap.add_argument("--force", action="store_true",
                    help="越过守卫 (未成功臂 / 无 .done 目录也删)")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"✗ --run-dir 不存在: {args.run_dir}")

    arms = ([a.strip() for a in args.arms.split(",") if a.strip()]
            if args.arms else discover_arms(args.run_dir))
    if not arms:
        print("  (no arms with local *_mixed dirs — nothing to do)")
        return 0

    plan = plan_cleanup(args.run_dir, arms, args.force)
    total_deletable = 0.0
    for arm, info in plan.items():
        if info["reason"]:
            print(f"  [keep] {arm}: {info['reason']} "
                  f"({info['gb']:.1f} GB stays)")
        else:
            total_deletable += info["gb"]
            rel = [os.path.relpath(d, args.run_dir) for d in info["dirs"]]
            print(f"  [{'DEL' if args.apply else 'del'}] {arm}: "
                  f"{', '.join(rel)} — {info['gb']:.1f} GB")
    print(f"  total: {total_deletable:.1f} GB "
          f"{'deleted' if args.apply else 'reclaimable'} "
          f"({'dry-run' if not args.apply else 'applied'})")

    if args.apply:
        for arm, info in plan.items():
            if info["reason"]:
                continue
            for d in info["dirs"]:
                shutil.rmtree(d)
                print(f"  [rm] {os.path.relpath(d, args.run_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
