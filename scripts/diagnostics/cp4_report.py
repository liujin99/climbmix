#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  cp4_report.py — CP4 最终对比报告 (climb vs random vs base 锚点)
#
#  用法:
#    python3 scripts/diagnostics/cp4_report.py [RUN_DIR]
#    python3 scripts/diagnostics/cp4_report.py result/prod1_20260906_204523
#    python3 scripts/diagnostics/cp4_report.py result/prod2_k15bal_current \
#        --base-expected 0.1738 --json cp4_verdict.json
#
#  输入 (RUN_DIR 下, 与 dispatch_target_arm.py 落地文件名一致):
#    eval_climb.csv / eval_random.csv  — 两臂 6 基准评测 CSV
#    eval_base_remote.csv              — 远端 d28 锚点 (可选)
#
#  CSV 行格式 (base_eval.py:537-550): label, raw_acc, centered_acc, nll
#    per-task 行:  arc_easy, 0.6123, 0.4831, 1.234
#    STEM 聚合行:  STEM, , 0.1738, 1.10   ← raw 列为空
#    centered = (acc − baseline)/(1 − baseline)  [0=随机, 1=满分]
#
#  报告内容:
#    1. 锚点校验: |远端 base stem − 本地期望| 分级 (PASS≤0.002 / WARN≤SE / FAIL)
#       — FAIL 说明远端评测管线有偏, 两臂对比不可信, 优先排查
#    2. 主指标: stem_metric (centered, 搜索目标) climb vs random
#       Δ ± √2·SE (SE=0.006 为 prod1 经验单次评测噪声), z + 单侧 p
#    3. 每基准表: raw acc 对比 + 二项 SE (BENCHMARK_SIZES) + 每基准 z
#    4. 符号检验: climb 赢几个基准, 精确二项 p (无分布假设)
#    5. 分级判定: WINS(significant≥95% / likely≥90%) / not significant / LOSES
#
#  只读, stdlib-only, 缺文件容错 (显示 pending 而非报错)。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import json
import math
import os
import sys

# 基准题量 (与 src/climbmix/core/types.py BENCHMARK_SIZES 一致;
# 独立硬编码副本 — 诊断脚本不依赖包导入, 服务器裸 python3 可跑)
BENCHMARK_SIZES = {
    "arc_easy": 2376,
    "arc_challenge": 1172,
    "mmlu_stem": 3545,
    "gpqa_diamond": 198,
    "gsm8k_cot": 1319,
    "math_cot_500": 500,
}
STEM_LABELS = list(BENCHMARK_SIZES)

# prod1 参照 (context footer)
PROD1 = {"climb": 0.1623, "random": 0.1659, "base": 0.1738}


def parse_eval_csv(path):
    """解析 base_eval CSV → {"stem", "stem_nll", "tasks": {name: {raw, centered, nll}}}.

    与 nanochat_cmds.parse_eval_results 同语义: 列 0=任务名, 1=raw acc
    (STEM 行为空), 2=centered, 3=nll; CORE 行跳过; 不可解析行跳过。
    """
    out = {"stem": None, "stem_nll": None, "tasks": {}}
    if not path or not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            name = parts[0]
            centered = _f(parts[2])
            nll = _f(parts[3]) if len(parts) >= 4 else None
            raw = _f(parts[1])
            if name in ("STEM", "CORE"):
                if name == "STEM" and centered is not None:
                    out["stem"] = centered
                    out["stem_nll"] = nll
                continue
            if centered is None:
                continue
            out["tasks"][name] = {"raw": raw, "centered": centered, "nll": nll}
    return out


def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def binom_se(p, n):
    """单基准二项 SE = sqrt(p(1-p)/N); p 越界/缺 N → None."""
    if p is None or n is None or n <= 0 or not (0.0 <= p <= 1.0):
        return None
    return math.sqrt(max(p * (1.0 - p), 0.0) / n)


def norm_sf(z):
    """标准正态单侧尾概率 P(Z >= z)."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def sign_test_p(wins, n):
    """精确二项符号检验单侧 p = P(X >= wins | X~Bin(n, 0.5))."""
    if n <= 0 or wins <= 0:
        return 1.0
    if wins > n:
        return 0.0
    tail = sum(math.comb(n, k) for k in range(wins, n + 1))
    return tail / (2 ** n)


def main():
    ap = argparse.ArgumentParser(description="CP4 final report: climb vs random")
    ap.add_argument("run_dir", nargs="?", default="result/prod2_k15bal_current")
    ap.add_argument("--arm-a", default="climb")
    ap.add_argument("--arm-b", default="random")
    ap.add_argument("--base-expected", type=float, default=0.1738,
                    help="本地 base (d28) stem_metric 期望值")
    ap.add_argument("--se", type=float, default=0.006,
                    help="单次评测 stem_metric 经验 SE (prod1 噪声地板)")
    ap.add_argument("--anchor-pass", type=float, default=0.002,
                    help="锚点 |Δ| PASS 阈值")
    ap.add_argument("--json", default="", help="可选: 机读 verdict 输出路径")
    args = ap.parse_args()

    a_path = os.path.join(args.run_dir, f"eval_{args.arm_a}.csv")
    b_path = os.path.join(args.run_dir, f"eval_{args.arm_b}.csv")
    base_path = os.path.join(args.run_dir, "eval_base_remote.csv")

    print("═" * 62)
    print(f"  CP4 report — {args.run_dir}  ({args.arm_a} vs {args.arm_b})")
    print("═" * 62)

    a = parse_eval_csv(a_path)
    b = parse_eval_csv(b_path)
    verdict = {"run_dir": args.run_dir, "anchor": None, "arms": None,
               "sign_test": None, "verdict": None}
    if a is None or b is None:
        for label, path, data in ((args.arm_a, a_path, a),
                                  (args.arm_b, b_path, b)):
            if data is None:
                print(f"  [·] {path} — not found (arm not finished yet?)")
        print("  → nothing to compare. CP4 fires after BOTH arms land.")
        if args.json:
            _dump_json(args.json, verdict)
        return 0

    # ── 1. 锚点校验 ─────────────────────────────────────────────────
    print("── 1. base anchor (remote eval pipeline check) ──")
    base = parse_eval_csv(base_path)
    if base is None:
        print(f"  [·] {base_path} — not found (skipped; optional arm)")
        print("      NOTE: without the anchor a systematic eval bias would "
              "shift BOTH arms equally — the Δ comparison stays valid,")
        print("      but absolute levels (e.g. vs prod1) are unverified.")
    else:
        d = (base["stem"] - args.base_expected) if base["stem"] is not None else None
        if d is None:
            print("  [!] base CSV has no STEM row — unreadable")
            verdict["anchor"] = "UNREADABLE"
        else:
            ad = abs(d)
            if ad <= args.anchor_pass:
                tag, sym = "PASS", "✓"
            elif ad <= args.se:
                tag, sym = "WARN", "·"
            else:
                tag, sym = "FAIL", "✗"
            print(f"  [{sym}] remote base stem = {base['stem']:.4f} vs local "
                  f"{args.base_expected:.4f}  |Δ|={ad:.4f} → {tag}")
            print(f"      (PASS ≤ {args.anchor_pass}, WARN ≤ SE={args.se}, "
                  f"FAIL above — FAIL means the remote eval pipeline is")
            print(f"       biased; investigate BEFORE trusting the arm "
                  f"comparison below)")
            verdict["anchor"] = {"stem": base["stem"], "delta": d, "tag": tag}

    # ── 2. 主指标: stem_metric ─────────────────────────────────────
    print(f"── 2. headline: stem_metric (centered, the search objective) ──")
    if a["stem"] is None or b["stem"] is None:
        print("  [!] STEM row missing in at least one arm CSV — "
              "falling back to recomputed mean over per-benchmark rows")
    a_stem = a["stem"] if a["stem"] is not None else _mean_centered(a)
    b_stem = b["stem"] if b["stem"] is not None else _mean_centered(b)
    if a_stem is None or b_stem is None:
        print("  [!] no usable stem score in both arms — abort")
        return 1
    delta = a_stem - b_stem
    se_delta = math.sqrt(2.0) * args.se
    z = delta / se_delta if se_delta > 0 else 0.0
    p = norm_sf(z)
    print(f"  {args.arm_a:<8} {a_stem:.4f}")
    print(f"  {args.arm_b:<8} {b_stem:.4f}")
    print(f"  Δ = {delta:+.4f}  (SE √2×{args.se:.3f} = {se_delta:.4f}, "
          f"z = {z:+.2f}, one-sided p = {p:.3f})")
    verdict["arms"] = {"a": a_stem, "b": b_stem, "delta": delta,
                       "z": z, "p_one_sided": p}

    # ── 3. 每基准表 (raw acc + 二项 SE) ────────────────────────────
    print("── 3. per-benchmark (raw accuracy, binomial SE) ──")
    common = [t for t in STEM_LABELS
              if t in a["tasks"] and t in b["tasks"]]
    missing = [t for t in STEM_LABELS if t not in common]
    hdr = (f"  {'benchmark':<15} {'N':>5}  {args.arm_a[:7]:>7} "
           f"{args.arm_b[:7]:>7}  {'Δraw':>8}  {'SE_Δ':>7}  {'z':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    wins = 0
    per_bench = []
    for t in common:
        ar = a["tasks"][t]["raw"]
        br = b["tasks"][t]["raw"]
        n = BENCHMARK_SIZES.get(t)
        se_a = binom_se(ar, n)
        se_b = binom_se(br, n)
        if ar is not None and br is not None and ar != br:
            wins += 1 if ar > br else 0
        row = {"benchmark": t, "n": n, "a_raw": ar, "b_raw": br}
        if None not in (ar, br, se_a, se_b):
            d = ar - br
            sd = math.sqrt(se_a ** 2 + se_b ** 2)
            zb = d / sd if sd > 0 else 0.0
            print(f"  {t:<15} {n:>5}  {ar:>7.4f} {br:>7.4f}  "
                  f"{d:>+8.4f}  {sd:>7.4f}  {zb:>+6.2f}")
            row.update({"delta": d, "se_delta": sd, "z": zb})
        else:
            print(f"  {t:<15} {str(n):>5}  {_fmt(ar):>7} {_fmt(br):>7}"
                  f"  {'—':>8}")
        per_bench.append(row)
    if missing:
        print(f"  (not scored on both arms: {', '.join(missing)})")

    # ── 4. 聚合 raw 均值 + 符号检验 ────────────────────────────────
    print("── 4. aggregate & sign test ──")
    pairs = [r for r in per_bench
             if r.get("delta") is not None and r.get("se_delta") is not None]
    if pairs:
        n_b = len(pairs)
        mean_d = sum(r["delta"] for r in pairs) / n_b
        se_mean = math.sqrt(sum(r["se_delta"] ** 2 for r in pairs)) / n_b
        z_mean = mean_d / se_mean if se_mean > 0 else 0.0
        # 注: centered=(acc−b)/(1−b), 4 选一基准 b=0.25 → SE 放大 ≤1.33×,
        # raw 空间的 z 是保守下界。
        print(f"  equal-weight raw mean Δ = {mean_d:+.4f} "
              f"(binomial SE {se_mean:.4f}, z {z_mean:+.2f}, "
              f"p {norm_sf(z_mean):.3f}) — conservative (centered "
              f"scales SE ≤1.33×)")
    if common:
        st = sign_test_p(wins, len(common))
        print(f"  sign test: {args.arm_a} wins {wins}/{len(common)} "
              f"benchmarks, exact one-sided p = {st:.3f}")
        verdict["sign_test"] = {"wins": wins, "n": len(common), "p": st}
    for arm_name, data in ((args.arm_a, a), (args.arm_b, b)):
        if data["stem_nll"] is not None:
            print(f"  {arm_name} stem NLL (secondary): "
                  f"{data['stem_nll']:.4f}")

    # ── 5. 判定 ────────────────────────────────────────────────────
    print("── 5. verdict ──")
    anchor_bad = verdict.get("anchor", {}).get("tag") if isinstance(
        verdict.get("anchor"), dict) else None
    if anchor_bad == "FAIL":
        v = "STOP — anchor FAIL: remote eval pipeline biased, comparison untrustworthy"
    elif delta > 0 and z >= 2.0:
        v = f"{args.arm_a.upper()} WINS — significant (z={z:+.2f}, ~95% one-sided)"
    elif delta > 0 and z >= 1.65:
        v = f"{args.arm_a.upper()} WINS — likely (z={z:+.2f}, ~90% one-sided)"
    elif delta < 0 and z <= -2.0:
        v = f"{args.arm_b.upper()} WINS — significant (z={z:+.2f}, ~95% one-sided)"
    elif delta < 0 and z <= -1.65:
        v = f"{args.arm_b.upper()} WINS — likely (z={z:+.2f}, ~90% one-sided)"
    else:
        v = (f"INCONCLUSIVE — Δ={delta:+.4f} within noise "
             f"(|z|={abs(z):.2f} < 1.65); treat as no significant difference")
    print(f"  ► {v}")
    verdict["verdict"] = v
    print(f"  reference: prod1 climb {PROD1['climb']:.4f} / random "
          f"{PROD1['random']:.4f} (Δ {PROD1['climb'] - PROD1['random']:+.4f}), "
          f"local base {PROD1['base']:.4f}")
    print("═" * 62)
    if args.json:
        verdict["per_benchmark"] = per_bench
        _dump_json(args.json, verdict)
    return 0


def _mean_centered(data):
    vals = [t["centered"] for t in data["tasks"].values()
            if t["centered"] is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v):
    return f"{v:.4f}" if v is not None else "—"


def _dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  (machine-readable verdict → {path})")


if __name__ == "__main__":
    sys.exit(main())
