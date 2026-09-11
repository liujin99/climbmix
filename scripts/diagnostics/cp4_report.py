#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  cp4_report.py — CP4 最终对比报告 (所有臂 vs base 锚点, 全景)
#
#  用法:
#    python3 scripts/diagnostics/cp4_report.py [RUN_DIR]
#    python3 scripts/diagnostics/cp4_report.py result/prod1_20260906_204523
#    python3 scripts/diagnostics/cp4_report.py result/prod2_k15bal_current \
#        --base-expected 0.1738 --json cp4_verdict.json
#
#  输入 (RUN_DIR 下, 与 dispatch_target_arm.py 落地文件名一致):
#    eval_<arm>.csv     — 每个臂一份 (climb / random / 自定义臂);
#                         全部自动发现, 有几个比几个
#    eval_base_remote.csv — 远端 d28 锚点 (可选)
#
#  CSV 行格式 (base_eval.py:537-550): label, raw_acc, centered_acc, nll
#    per-task 行:  arc_easy, 0.6123, 0.4831, 1.234
#    STEM 聚合行:  STEM, , 0.1738, 1.10   ← raw 列为空
#    centered = (acc − baseline)/(1 − baseline)  [0=随机, 1=满分]
#
#  报告内容 (对照臂 --ref, 默认 random = "搜索是否有价值"的基线):
#    1. 锚点校验: |远端 base stem − 本地期望| 分级 (PASS≤0.002 / WARN≤SE / FAIL)
#       — FAIL 说明远端评测管线有偏, 臂间对比不可信, 优先排查
#    2. 主指标: 所有臂的 stem_metric (centered, 搜索目标), 按分数排序;
#       各臂 vs ref: Δ ± √2·SE (SE=0.006 为 prod1 经验单次评测噪声), z + 单侧 p
#    3. 每基准表: 所有臂 raw acc ± 二项 SE (BENCHMARK_SIZES) 并排
#    4. 聚合 + 符号检验: 各臂 vs ref 的 raw 均值 Δ + z; 赢几个基准, 精确二项 p
#    5. 判定: 排名 + 各臂 vs ref 分级 (WINS significant≥95% / likely≥90% /
#       not significant / LOSES)
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

    STEM 行的 NLL 是 nan 时 (nanochat 的聚合对任何 per-task 缺失都产生
    nan — mmlu_stem 0-shot 无 gold span), 退化为有限 per-task NLL 的
    N 加权均值 (2026-09-10 CP4 两臂都打出 nan, 但 per-task 全部健康)。
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
    if out["stem_nll"] is None or not math.isfinite(out["stem_nll"]):
        pairs = [(t["nll"], BENCHMARK_SIZES.get(name, 0))
                 for name, t in out["tasks"].items()
                 if t["nll"] is not None and math.isfinite(t["nll"])]
        if pairs:
            tot_n = sum(n for _, n in pairs)
            if tot_n > 0:
                out["stem_nll"] = sum(v * n for v, n in pairs) / tot_n
            else:
                out["stem_nll"] = sum(v for v, _ in pairs) / len(pairs)
        else:
            out["stem_nll"] = None
    return out


def discover_arms(run_dir):
    """eval_<arm>.csv 全发现 (锚点 eval_base_remote.csv 除外) → 臂名列表."""
    arms = set()
    try:
        names = os.listdir(run_dir)
    except OSError:
        return []
    for n in names:
        if (n.startswith("eval_") and n.endswith(".csv")
                and n != "eval_base_remote.csv"):
            arms.add(n[len("eval_"):-len(".csv")])
    return sorted(arms)


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


def _tag(z):
    """Δ 的显著性分级标签 (z = Δ / SE_Δ, 单侧)."""
    if z >= 2.0:
        return "WINS**"
    if z >= 1.65:
        return "WINS*"
    if z <= -2.0:
        return "LOSES**"
    if z <= -1.65:
        return "LOSES*"
    return "·"


def main():
    ap = argparse.ArgumentParser(
        description="CP4 final report: all arms vs base anchor")
    ap.add_argument("run_dir", nargs="?", default="result/prod2_k15bal_current")
    ap.add_argument("--ref", default="random",
                    help="对照臂 (Δ/z/p 与判定的参照; 默认 random)")
    ap.add_argument("--base-expected", type=float, default=0.1738,
                    help="本地 base (d28) stem_metric 期望值")
    ap.add_argument("--se", type=float, default=0.006,
                    help="单次评测 stem_metric 经验 SE (prod1 噪声地板)")
    ap.add_argument("--anchor-pass", type=float, default=0.002,
                    help="锚点 |Δ| PASS 阈值")
    ap.add_argument("--json", default="", help="可选: 机读 verdict 输出路径")
    args = ap.parse_args()

    verdict = {"run_dir": args.run_dir, "anchor": None, "ref": None,
               "arms": None, "pairwise_vs_ref": None, "sign_test": None,
               "ranking": None, "verdict": None}

    arms = discover_arms(args.run_dir)
    if not arms:
        print("═" * 62)
        print(f"  CP4 report — {args.run_dir}")
        print("═" * 62)
        print(f"  [·] no eval_<arm>.csv under {args.run_dir}")
        print("  → nothing to compare. CP4 fires after arms land.")
        if args.json:
            _dump_json(args.json, verdict)
        return 0

    ref = args.ref if args.ref in arms else arms[0]
    if ref != args.ref:
        print(f"  (ref '{args.ref}' not among landed arms — "
              f"falling back to '{ref}')")

    # 解析 + stem (STEM 行缺失时回退 per-benchmark 均值)
    parsed, stems, no_stem = {}, {}, []
    for a in arms:
        d = parse_eval_csv(os.path.join(args.run_dir, f"eval_{a}.csv"))
        if d is None:
            print(f"  [·] eval_{a}.csv — not found (arm not finished yet?)")
            continue
        parsed[a] = d
        s = d["stem"] if d["stem"] is not None else _mean_centered(d)
        if s is None:
            no_stem.append(a)
        else:
            stems[a] = s
    if not stems:
        print("  [!] no usable stem score in any arm — abort")
        return 1
    for a in no_stem:
        print(f"  [!] {a}: no usable stem score — excluded from comparison")
    if ref not in stems:
        ref = sorted(stems)[0]
        print(f"  (ref has no usable stem — ref := {ref})")

    ranked = sorted(stems, key=lambda a: -stems[a])          # 分数降序
    others = [a for a in ranked if a != ref]                 # vs ref 的臂
    se_delta = math.sqrt(2.0) * args.se

    print("═" * 62)
    print(f"  CP4 report — {args.run_dir}  "
          f"({len(ranked)} arms; ref = {ref})")
    print("═" * 62)
    verdict["ref"] = ref
    verdict["arms"] = {a: {"stem": stems[a],
                           "stem_nll": parsed[a]["stem_nll"]}
                       for a in ranked}

    # ── 1. 锚点校验 ─────────────────────────────────────────────────
    print("── 1. base anchor (remote eval pipeline check) ──")
    base = parse_eval_csv(os.path.join(args.run_dir, "eval_base_remote.csv"))
    if base is None:
        print("  [·] eval_base_remote.csv — not found (skipped; optional)")
        print("      NOTE: without the anchor a systematic eval bias would "
              "shift ALL arms equally — the Δ comparisons stay valid,")
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

    # ── 2. 主指标: stem_metric (所有臂, vs ref) ─────────────────────
    print("── 2. headline: stem_metric (centered, the search objective) ──")
    print(f"  {'arm':<14} {'stem':>7}  {'Δvs ref':>8}  {'z':>6}  {'p':>5}  tag")
    pw = {}
    for a in ranked:
        if a == ref:
            print(f"  {a:<14} {stems[a]:>7.4f}  {'(ref)':>8}")
            continue
        delta = stems[a] - stems[ref]
        z = delta / se_delta if se_delta > 0 else 0.0
        p = norm_sf(z)
        print(f"  {a:<14} {stems[a]:>7.4f}  {delta:>+8.4f}  {z:>+6.2f}  "
              f"{p:>5.3f}  {_tag(z)}")
        pw[a] = {"delta": delta, "z": z, "p_one_sided": p, "tag": _tag(z)}
    if not others:
        print("  (only the ref arm has a usable score — nothing to compare yet)")
    print(f"  (SE_Δ = √2×{args.se:.3f} = {se_delta:.4f}; "
          f"tags: WINS** z≥2 (~95% one-sided), WINS* z≥1.65 (~90%), "
          f"· noise; LOSES* / LOSES** mirror)")
    verdict["pairwise_vs_ref"] = pw

    # ── 3. 每基准表 (所有臂 raw acc ± 二项 SE) ─────────────────────
    print("── 3. per-benchmark (raw accuracy ± binomial SE) ──")
    all_tasks = [t for t in STEM_LABELS
                 if any(t in d["tasks"] for d in parsed.values())]
    hdr = (f"  {'benchmark':<15} {'N':>5}  "
           + " ".join(f"{a[:11]:>17}" for a in ranked))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    per_bench = []
    for t in all_tasks:
        n = BENCHMARK_SIZES.get(t)
        cells, raws = [], {}
        for a in ranked:
            r = parsed[a]["tasks"].get(t, {}).get("raw") if a in parsed else None
            se = binom_se(r, n)
            raws[a] = r
            cells.append(f"{r:.4f}±{se:.4f}" if None not in (r, se) else "—")
        print(f"  {t:<15} {str(n):>5}  " + " ".join(f"{c:>17}" for c in cells))
        per_bench.append({"benchmark": t, "n": n, "raw": raws})
    absent = [t for t in STEM_LABELS if t not in all_tasks]
    if absent:
        print(f"  (not scored on any arm: {', '.join(absent)})")

    # ── 4. 聚合 raw 均值 + 符号检验 (各臂 vs ref) ──────────────────
    print(f"── 4. aggregate & sign test (vs ref = {ref}) ──")
    st = {}
    for a in others:
        pairs = []
        wins = n_cmp = 0
        for t in all_tasks:
            ra = parsed[a]["tasks"].get(t, {}).get("raw")
            rb = parsed[ref]["tasks"].get(t, {}).get("raw")
            if ra is None or rb is None:
                continue
            n = BENCHMARK_SIZES.get(t)
            sa, sb = binom_se(ra, n), binom_se(rb, n)
            n_cmp += 1
            wins += 1 if ra > rb else 0
            if None not in (sa, sb):
                pairs.append((ra - rb, math.sqrt(sa ** 2 + sb ** 2)))
        line = f"  {a}: "
        if pairs:
            n_p = len(pairs)
            mean_d = sum(d for d, _ in pairs) / n_p
            se_mean = math.sqrt(sum(s * s for _, s in pairs)) / n_p
            z_mean = mean_d / se_mean if se_mean > 0 else 0.0
            # 注: centered=(acc−b)/(1−b), 4 选一基准 b=0.25 → SE 放大 ≤1.33×,
            # raw 空间的 z 是保守下界。
            line += (f"raw mean Δ {mean_d:+.4f} (binomial SE {se_mean:.4f}, "
                     f"z {z_mean:+.2f}, p {norm_sf(z_mean):.3f}); ")
        line += f"wins {wins}/{n_cmp} benchmarks, exact one-sided p = {sign_test_p(wins, n_cmp):.3f}"
        print(line)
        st[a] = {"wins": wins, "n": n_cmp, "p": sign_test_p(wins, n_cmp)}
    verdict["sign_test"] = st
    nlls = [f"{a} {parsed[a]['stem_nll']:.4f}" for a in ranked
            if parsed[a]["stem_nll"] is not None]
    if nlls:
        print(f"  stem NLL (secondary): {' / '.join(nlls)}")

    # ── 5. 判定 ────────────────────────────────────────────────────
    print("── 5. verdict ──")
    lines = []
    anchor_tag = verdict.get("anchor", {}).get("tag") if isinstance(
        verdict.get("anchor"), dict) else None
    if anchor_tag == "FAIL":
        lines.append("STOP — anchor FAIL: remote eval pipeline biased, "
                     "comparison untrustworthy")
    if len(ranked) > 1:
        lines.append("ranking: "
                     + " › ".join(f"{a} {stems[a]:.4f}" for a in ranked))
    for a in others:
        d, z = pw[a]["delta"], pw[a]["z"]
        if z >= 2.0:
            v = f"{a.upper()} WINS vs {ref} — significant (z={z:+.2f}, ~95% one-sided)"
        elif z >= 1.65:
            v = f"{a.upper()} WINS vs {ref} — likely (z={z:+.2f}, ~90% one-sided)"
        elif z <= -2.0:
            v = f"{a.upper()} LOSES vs {ref} — significant (z={z:+.2f})"
        elif z <= -1.65:
            v = f"{a.upper()} LOSES vs {ref} — likely (z={z:+.2f})"
        else:
            v = (f"{a} vs {ref}: INCONCLUSIVE — Δ={d:+.4f} within noise "
                 f"(|z|={abs(z):.2f} < 1.65); treat as no significant difference")
        lines.append(v)
    if not lines:
        lines.append(f"only {ref} has landed — re-run after more arms finish")
    for v in lines:
        print(f"  ► {v}")
    verdict["ranking"] = ranked
    verdict["verdict"] = lines
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
