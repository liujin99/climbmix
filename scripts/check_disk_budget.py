#!/usr/bin/env python3
# ═════════════════════════════════════════════════════════════════════
#  check_disk_budget.py — 臂发射前的本地磁盘预算 preflight
#
#  用法 (由 runs/lib/arm_engine.sh 自动调用):
#    python3 scripts/check_disk_budget.py --run-dir result/prod4_current \
#        --arm climb --target-tokens 3B
#
#  背景 (prod4 2026-09-16 磁盘事故, 战后清单 #8): 3×6B 臂的选样+混料
#  全链本地落地 ~100G, 叠加旧归档把 /work 1.6T 打穿 100% → winner 臂
#  混到 95% 时 Errno 28 阵亡。本脚本在混料开始前估算本臂新增占用,
#  与 df 余量对比, 不够就 fail-loud (禁发), 而不是混到一半死。
#
#  模型 (prod4 3B 实测, STEM_RATIO=0.7): shards ≈ 4.3 GB/B tokens,
#  mixed ≈ 6.0 GB/B tokens; 各乘安全边际 1.15 → ~12 GB/B。
#  已有 .done 的目录 (复用, 不新增) 不计入。放行条件:
#    free >= 新增占用 × headroom (默认 2, 给同规模臂留一席之地)
#           + 绝对地板 (默认 50 GB)
#
#  逃生门: ARM_DISK_CHECK=0 (arm_engine) 或不调用本脚本。
#  只读不写; stdlib-only。
# ═════════════════════════════════════════════════════════════════════
import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from climbmix.utils.token_estimate import parse_token_count  # noqa: E402

SHARDS_GB_PER_B = 4.5   # prod4 实测 3B ≈ 13G (含边际前)
MIXED_GB_PER_B = 6.5    # prod4 实测 3B ≈ 18G (含边际前)


def estimate_arm_gb(target_tokens: int, *, shards_done: bool,
                    mixed_done: bool, shards_gb_per_b: float = SHARDS_GB_PER_B,
                    mixed_gb_per_b: float = MIXED_GB_PER_B,
                    margin: float = 1.15) -> float:
    """本臂将新增的本地占用 (GB); .done 已就位的产物不计入。"""
    budget_b = target_tokens / 1e9
    need = 0.0
    if not shards_done:
        need += shards_gb_per_b * budget_b * margin
    if not mixed_done:
        need += mixed_gb_per_b * budget_b * margin
    return need


def check_budget(run_dir: str, arm: str, target_tokens: int, *,
                 free_gb=None, headroom_x: float = 2.0,
                 min_free_gb: float = 50.0, **estimate_kw) -> dict:
    """返回 {ok, free_gb, need_gb, required_gb, shards_done, mixed_done}."""
    shards_done = os.path.isfile(
        os.path.join(run_dir, f"{arm}_shards", ".done"))
    mixed_done = os.path.isfile(
        os.path.join(run_dir, f"{arm}_mixed", ".done"))
    need = estimate_arm_gb(target_tokens, shards_done=shards_done,
                           mixed_done=mixed_done, **estimate_kw)
    if free_gb is None:
        free_gb = shutil.disk_usage(run_dir).free / 2 ** 30
    required = need * headroom_x + min_free_gb
    ok = need == 0.0 or free_gb >= required
    return dict(ok=ok, free_gb=free_gb, need_gb=need, required_gb=required,
                shards_done=shards_done, mixed_done=mixed_done)


def main():
    ap = argparse.ArgumentParser(
        description="arm local disk budget preflight (fail-loud)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--target-tokens", required=True,
                    help="token 预算 (同 TARGET_TOKENS, 支持 3B/6B 后缀)")
    ap.add_argument("--headroom-x", type=float, default=2.0,
                    help="新增占用倍数余量 (默认 2: 容得下再发一个同规模臂)")
    ap.add_argument("--min-free-gb", type=float, default=50.0,
                    help="绝对地板 GB (默认 50)")
    ap.add_argument("--margin", type=float, default=1.15,
                    help="估算安全边际 (默认 1.15)")
    args = ap.parse_args()

    target_tokens = parse_token_count(args.target_tokens)
    if not target_tokens:
        raise SystemExit(f"✗ --target-tokens 无法解析: {args.target_tokens}")
    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"✗ --run-dir 不存在: {args.run_dir}")

    r = check_budget(args.run_dir, args.arm, target_tokens,
                     headroom_x=args.headroom_x,
                     min_free_gb=args.min_free_gb, margin=args.margin)
    state = ("shards .done" if r["shards_done"] else "shards NEW") \
        + " + " + ("mixed .done" if r["mixed_done"] else "mixed NEW")
    print(f"  [disk] {args.arm} @{args.target_tokens}: {state}")
    print(f"  [disk] new local usage ~{r['need_gb']:.1f} GB; free "
          f"{r['free_gb']:.1f} GB; required {r['required_gb']:.1f} GB "
          f"(need x{args.headroom_x:g} + floor {args.min_free_gb:g})")
    if not r["ok"]:
        print(f"  [disk] ✗ NOT ENOUGH — a mid-mix Errno 28 killed the prod4 "
              f"winner arm at 95% (2026-09-16). Free space first, or "
              f"ARM_DISK_CHECK=0 to accept the risk.")
        return 1
    print(f"  [disk] ✓ budget OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
