#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  report_charts.py — 结果叙事图表包 (用户读结果的六件套)
#
#  用法:
#    python3 scripts/diagnostics/report_charts.py result/prod5_20260929_201108
#    python3 scripts/diagnostics/report_charts.py result/prod5_current \
#        --task2 math_cot_500                       # 臂间图第二联换任务
#    python3 scripts/diagnostics/report_charts.py result/prod5_20260929_201108 \
#        --extra-run result/prod4_20260922_XXXX   # 图 C 并入 prod4 的 climb 臂
#
#  动机 (2026-10-08 推广裁决): 报告栈的表很全, 但三张最直观的结果图
#  缺位 — 搜索收敛(只有表)、臂间主结果(只有表)、代理→真实排名一致性
#  (只在 KEY_FINDINGS 手算)。另从 quadmix 借鉴两件: 逐簇小倍数图
#  (fig_optimizer_domain_vs_loss)、best-vs-worst 配方热力图
#  (fig_domain_heatmap), 以及指标状态判定 (report.py Reliability 节)。
#
#  输入 (RUN_DIR 下, 全部只读; 缺件降级跳过不报错):
#    search_state.json          搜索舰队: accumulated_configs/scores,
#                               realized_configs_per_iter (轮次重建),
#                               predictor_eval (val_preds/val_targets →
#                               Top-K Recall)
#    topk_mixture_candidates.json  图 C 的 climb 臂解析兜底: 现行
#                               _save_state 不落盘 selection 键, 落盘
#                               形态是本文件的 selection_mode + rank-1
#                               候选 (no-claim 族 → rank1 = 终选臂)
#    macro_info.json            簇标签/规模/质量分 (⑬r 新名;
#      或 cluster_info_cache.json   legacy 名仍读; 全缺 → C0..C{K-1})
#    eval_<arm>.csv             各臂 d28 评测 (cp4_report 同解析)
#
#  输出 (RUN_DIR 根, 与 report.md / 现有 PNG 同层):
#    search_convergence.png          A: 115 点按轮散布 + 每轮 best 连线
#                                    + Search Lift (vs 第 1 轮随机带, σ)
#    arms_main_results.png           B: 条形图 STEM + 第二任务 (--task2,
#                                    缺省 gsm8k_cot; 缺失时自动挑跨臂
#                                    区分度最大的任务), 误差棒 + vs 参照臂
#                                    (natural, 缺席回退 uniform) 的 z 值标注
#    proxy_target_consistency.png    C: climb 臂 proxy 实测分 vs d28 STEM
#                                    (逐臂名次标注 #p->#d + 评测噪声误差
#                                    棒 + 基线 d28 带 + proxy 名次轨迹线;
#                                    基线不在舰队 → 无 proxy 分, 如实缺席)
#    cluster_alpha_vs_score.png      D: 逐簇小倍数 (α vs proxy 分 + 线性
#                                    趋势线 + 逐簇 ρ 正负着色)
#    best_vs_worst_heatmap.png       E: Top-5 vs Bottom-5 逐簇 α 热力图
#                                    + 右联 Δ 条 (赢家比输家多/少放多少)
#    stdout 判定块 (F): pooled ρ 状态分级 (strong/good/moderate/weak)
#                                    + Top-K Recall + Search Lift
#                                    + D.10 论文语境 (94% @ 112 点)
#
#  原则: stdlib + numpy 必需, matplotlib 可选 (缺失只出文字摘要);
#  不依赖 climbmix 包 (服务器裸 python3 可跑, 与 cp4_report 同原则;
#  复用同目录 cp4_report 的 eval CSV 解析); 图内标签 ASCII (服务器无
#  CJK 字体, mfu_probe 同教训); 只读 RUN_DIR。
#
#  口径注意: proxy 分 = 搜索 utility (prod5 为 SNR 加权分, 越大越好;
#  --direction minimize 可切换); 臂分 = d28 STEM (centered)。图 C 的
#  一致性主张是 climb 族内排名传递 (KEY_FINDINGS #3), 不是逐点回归。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp4_report import BENCHMARK_SIZES, binom_se, discover_arms, parse_eval_csv

CLIMB_COLOR = "#DD8452"
BASELINE_COLOR = "#4C72B0"
ANCHOR_COLOR = "#9A9A9A"
ROUND_COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]


# ── 小工具 ─────────────────────────────────────────────────────────────

def load_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def parse_arm_name(a):
    """臂名 → {kind, config_id, rep}。kind ∈ cfg / climb_optimal /
    uniform / natural / domainfix / base / other。与 recipe_report 同语义,
    增加 base (无 mid-train 锚点臂)。"""
    rep = a.endswith("_rep")
    base = a[:-len("_rep")] if rep else a
    m = re.match(r"^(?:climb-)?cfg(\d+)$", base)
    if m:
        return {"kind": "cfg", "config_id": int(m.group(1)), "rep": rep}
    if base in ("climb", "noclaim", "no_claim", "argmin"):
        return {"kind": "climb_optimal", "rep": rep}
    if base in ("random", "random3b", "uniform"):
        return {"kind": "uniform", "rep": rep}
    if base == "natural":
        return {"kind": "natural", "rep": rep}
    if base == "domainfix":
        return {"kind": "domainfix", "rep": rep}
    if base.startswith("base"):
        return {"kind": "base", "rep": rep}
    return {"kind": "other", "rep": rep}


def _arm_color(a):
    k = parse_arm_name(a)
    if k["kind"] in ("cfg", "climb_optimal"):
        return CLIMB_COLOR
    if k["kind"] == "base":
        return ANCHOR_COLOR
    return BASELINE_COLOR


def _topk_idx(scores, k, maximize):
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    if not finite.any() or k <= 0:
        return np.array([], dtype=int)
    k = min(k, int(finite.sum()))
    if maximize:
        return np.argsort(np.where(finite, s, -np.inf))[-k:][::-1]
    return np.argsort(np.where(finite, s, np.inf))[:k]


def _status_tag(v, thresholds, below="weak"):
    """quadmix 式状态分级: thresholds = [(下限, 标签), ...] 降序;
    低于全部下限 → below (prod5 实测教训: 0/3 recall 曾被错标 moderate)。"""
    if v is None or not np.isfinite(v):
        return "n/a"
    for lo, tag in thresholds:
        if v >= lo:
            return tag
    return below


RHO_TAGS = [(0.70, "strong"), (0.50, "good"), (0.30, "moderate")]
LIFT_TAGS = [(1.00, "strong"), (0.50, "good"), (0.20, "moderate")]
RECALL_TAGS = [(0.70, "strong"), (0.50, "good"), (0.30, "moderate")]


# ── 工件读取 ───────────────────────────────────────────────────────────

def load_fleet(run_dir):
    """search_state → {scores, ids, W, iters, K, state} 或 None。"""
    state = load_json(os.path.join(run_dir, "search_state.json"))
    if not state:
        return None
    configs = state.get("accumulated_configs") or []
    scores = state.get("accumulated_scores") or []
    if len(configs) != len(scores) or not configs:
        return None
    W = np.array([c.get("weights") or [] for c in configs], dtype=np.float64)
    if W.ndim != 2 or W.shape[1] == 0 or W.shape[0] != len(configs):
        return None
    ids = np.array([int(c.get("config_id", i)) for i, c in enumerate(configs)])
    scores = np.array(scores, dtype=np.float64)
    K = W.shape[1]
    # 轮次重建: realized_configs_per_iter[k] = 第 k+1 轮的配置数
    # (与 iterative_bootstrapper._reconstruct_iteration_results 同逻辑)
    iters = np.ones(len(configs), dtype=int)
    per_iter = state.get("realized_configs_per_iter") or []
    offset = 0
    for k, n in enumerate(per_iter):
        n = int(n)
        if offset + n > len(configs):
            n = len(configs) - offset
        if n <= 0:
            break
        iters[offset:offset + n] = k + 1
        offset += n
    return {"scores": scores, "ids": ids, "W": W, "iters": iters,
            "K": K, "state": state}


def cluster_labels(run_dir, K):
    """macro_info.json (⑬r) / cluster_info_cache.json (legacy) → 标签列表。"""
    for name in ("macro_info.json", "cluster_info_cache.json"):
        ci = load_json(os.path.join(run_dir, name))
        if isinstance(ci, list) and ci:
            return [str(c.get("label") or f"C{c.get('cluster_id', i)}")
                    for i, c in enumerate(ci[:K])]
    return [f"C{i}" for i in range(K)]


def resolve_climb_config_id(run_dir, state):
    """D19 终选臂 (arm 'climb') 的 config_id。两级解析:
    ① search_state.selection (内存 extras 的形态 — 现行 _save_state
       不落盘该键, 留作未来兼容);
    ② topk_mixture_candidates.json (落盘形态): selection_mode 属
       best_measured 族 → rank-1 候选即终选 (no-claim 降级口径)。
    argmin 赢得槽位的轮次 (predictor_design_space_claimed) 该臂未经
    proxy 实测 → None (图 C 如实缺席, prod4 的 climb-winner 即此类)。"""
    sel = (state or {}).get("selection") or {}
    mode = str(sel.get("mode") or "")
    if mode:
        if "best_measured" in mode:
            bm = ((sel.get("claim") or {}).get("best_measured")) or {}
            cid = bm.get("config_id")
            return int(cid) if cid is not None else None
        return None
    topk = load_json(os.path.join(run_dir, "topk_mixture_candidates.json")) or {}
    if "best_measured" in str(topk.get("selection_mode") or ""):
        cands = topk.get("candidates") or []
        if cands and cands[0].get("config_id") is not None:
            return int(cands[0]["config_id"])
    return None


# ── F: 判定块 (stdout) ─────────────────────────────────────────────────

def verdict_block(fleet, maximize):
    state = fleet["state"]
    scores, iters = fleet["scores"], fleet["iters"]
    lines = []
    lines.append("=" * 66)
    lines.append("Predictor verdict (LightGBM) — 状态分级与论文语境")
    lines.append("=" * 66)

    # pooled held-out rho (val pairs across rounds)
    preds, targets = [], []
    for e in state.get("predictor_eval") or []:
        preds.extend(e.get("val_preds") or [])
        targets.extend(e.get("val_targets") or [])
    if len(preds) >= 2 and len(preds) == len(targets):
        rho = _spearman(preds, targets)
        tag = _status_tag(rho, RHO_TAGS)
        lines.append(f"Pooled held-out Spearman : rho = {rho:+.4f}  "
                     f"(n={len(preds)} pairs)  [{tag}]")
    else:
        rho = float("nan")
        lines.append("Pooled held-out Spearman : n/a (no val pairs in state)")

    # Top-K recall on held-out pairs
    for k in (3, 10):
        n = len(preds)
        if n < max(2 * k, 6):
            lines.append(f"Top-{k:<2d} recall (held-out)  : n/a "
                         f"(only {n} pairs)")
            continue
        p = np.asarray(preds, dtype=np.float64)
        t = np.asarray(targets, dtype=np.float64)
        if not maximize:
            p, t = -p, -t
        pk = set(_topk_idx(p, k, True).tolist())
        tk = set(_topk_idx(t, k, True).tolist())
        rec = len(pk & tk) / k
        lines.append(f"Top-{k:<2d} recall (held-out)  : "
                     f"{len(pk & tk)}/{k} = {rec:.2f}  "
                     f"[{_status_tag(rec, RECALL_TAGS)}]")

    # Search lift: top-k vs round-1 (random Dirichlet) band
    r1 = scores[iters == 1]
    r1 = r1[np.isfinite(r1)]
    for k in (3, 10):
        if len(r1) < 4 or fleet["W"].shape[0] < k:
            lines.append(f"Search lift (top-{k:<2d})     : n/a "
                         f"(round-1 band too small)")
            continue
        mu, sd = float(r1.mean()), float(r1.std(ddof=1))
        top = scores[_topk_idx(scores, k, maximize)]
        if sd > 0:
            lift = (float(top.mean()) - mu) / sd
            lines.append(f"Search lift top-{k:<2} : "
                         f"{float(top.mean()):+.4f} vs round-1 mean "
                         f"{mu:+.4f} = {lift:+.2f} sigma  "
                         f"[{_status_tag(lift, LIFT_TAGS)}]")
        else:
            lines.append(f"Search lift top-{k:<2} : n/a "
                         f"(round-1 sigma = 0)")

    lines.append("-" * 66)
    lines.append("语境: 论文 D.10 = 94% held-out Spearman @ 112 configs /")
    lines.append("350M proxy — 更小的 N 必然读数更低 (预算产物, 非缺陷)。")
    lines.append("分级: rho>=0.70 strong / >=0.50 good / >=0.30 moderate;")
    lines.append("       lift>=1.0sigma strong / >=0.5 good / >=0.2 moderate。")
    lines.append("=" * 66)
    return "\n".join(lines)


# ── A: 搜索收敛图 ──────────────────────────────────────────────────────

def chart_convergence(out_dir, fleet, maximize):
    scores, iters, ids = fleet["scores"], fleet["iters"], fleet["ids"]
    n = len(scores)
    rounds = sorted(set(iters.tolist()))
    better = "higher" if maximize else "lower"

    # 每轮 best (文本层始终输出)
    per_round = []
    for r in rounds:
        m = iters == r
        s = scores[m]
        s = s[np.isfinite(s)]
        if len(s):
            bi = _topk_idx(scores[m], 1, maximize)
            gi = int(np.where(m)[0][bi[0]]) if len(bi) else -1
            per_round.append((r, int(m.sum()), float(scores[gi]) if gi >= 0
                              else float("nan"), gi))
    r1 = scores[iters == 1]
    r1 = r1[np.isfinite(r1)]
    lifts = {}
    if len(r1) >= 4:
        mu, sd = float(r1.mean()), float(r1.std(ddof=1))
        if sd > 0:
            for k in (3, 10):
                if n >= k:
                    top = scores[_topk_idx(scores, k, maximize)]
                    lifts[k] = (float(top.mean()) - mu) / sd

    if not HAS_MPL:
        print("[A] matplotlib 不可用 — 文字版:")
        print(f"    {'round':>5} {'configs':>7} {'best':>9} {'mean':>9} {'std':>7}")
        for r, cnt, best, _ in per_round:
            m = scores[iters == r]
            m = m[np.isfinite(m)]
            print(f"    {r:>5} {cnt:>7} {best:>+9.4f} {m.mean():>+9.4f} "
                  f"{m.std(ddof=1) if len(m) > 1 else 0.0:>7.4f}")
        for k, v in lifts.items():
            print(f"    search lift (top-{k} vs round-1): {v:+.2f} sigma")
        return None

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(1, n + 1)
    for r in rounds:
        m = iters == r
        ax.scatter(x[m], scores[m], s=22, alpha=0.65,
                   color=ROUND_COLORS[(r - 1) % len(ROUND_COLORS)],
                   edgecolors="none", label=f"round {r}")
    # 轮次边界
    for r in rounds[1:]:
        b = int((iters < r).sum())
        ax.axvline(b + 0.5, color="gray", linestyle=":", linewidth=0.8)
    # 第 1 轮随机带 (mean ± std)
    if len(r1) >= 4:
        mu, sd = float(r1.mean()), float(r1.std(ddof=1))
        ax.axhspan(mu - sd, mu + sd, color="#4C72B0", alpha=0.10,
                   label="round-1 random baseline (mean+/-std)")
    # 每轮 best 连线
    bx = [gi + 1 for _, _, _, gi in per_round if gi >= 0]
    by = [b for _, _, b, _ in per_round if np.isfinite(b)]
    if len(bx) >= 2:
        ax.plot(bx, by, "o-", color="black", linewidth=1.4, markersize=6,
                label="best per round", zorder=5)
        for xi, yi in zip(bx, by):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)
    lift_txt = " | ".join(f"top-{k}: {v:+.2f} sigma" for k, v in lifts.items())
    if lift_txt:
        ax.text(0.02, 0.97, f"search lift vs round-1:  {lift_txt}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    ax.set_xlabel("config # (chronological)")
    ax.set_ylabel(f"proxy utility ({better} = better)")
    ax.set_title("Search convergence: per-config proxy score by round")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.text(0.5, 0.005,
             "proxy utility = SNR-weighted score of each mixture on the d20 "
             "proxy model (6 tasks, z-scale: round-1 mean ~ 0);\n"
             "blue band = round-1 random-sampling level - later rounds above "
             "it = the search is learning", ha="center", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = os.path.join(out_dir, "search_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── B: 臂间主结果图 ────────────────────────────────────────────────────

def chart_arms(out_dir, run_dir, se_stem, task2="gsm8k_cot"):
    arms = discover_arms(run_dir)
    rows = []
    for a in arms:
        ev = parse_eval_csv(os.path.join(run_dir, f"eval_{a}.csv"))
        if not ev or ev.get("stem") is None:
            continue
        rows.append({"arm": a, "stem": float(ev["stem"]),
                     "tasks": ev.get("tasks") or {}})
    if not rows:
        print("[B] 无可解析的 eval_<arm>.csv — 跳过臂间主结果图")
        return None
    rows.sort(key=lambda r: -r["stem"])

    # 第二联任务解析: 显式 --task2 缺席 → 自动挑跨臂区分度 (raw 极差)
    # 最大的任务; 完全无 per-task 行 → 只出 STEM 单联。
    t2, t2_auto = task2, False
    if not any(t2 in r["tasks"] for r in rows):
        common = set.intersection(*[set(r["tasks"]) for r in rows])
        best_name, best_spread = None, -1.0
        for nm in sorted(common):
            vals = [r["tasks"][nm].get("raw") for r in rows]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                spread = max(vals) - min(vals)
                if spread > best_spread:
                    best_name, best_spread = nm, spread
        if best_name is None:
            t2 = None
            print(f"[B] 指定任务 '{task2}' 缺失且无备选 per-task 行 — "
                  "只出 STEM 单联")
        else:
            t2, t2_auto = best_name, True
            print(f"[B] 指定任务 '{task2}' 缺失 — 自动改用区分度最大的 "
                  f"'{best_name}' (跨臂 raw 极差 {best_spread:.3f})")

    # 参照臂 = natural (实践者的无为默认: 池子按原占比顺序吃);
    # 缺席回退 uniform 族 (论文 App. C.1 对照)。2026-10-08 用户裁决:
    # 固定该链, 不设 --ref 参数。
    ref_row = next((r for r in rows if r["arm"] == "natural"), None)
    if ref_row is None:
        ref_row = next((r for r in rows
                        if parse_arm_name(r["arm"])["kind"] == "natural"),
                       None)
    if ref_row is None:
        ref_row = next((r for r in rows
                        if parse_arm_name(r["arm"])["kind"] == "uniform"),
                       None)
    if ref_row is None:
        print("[B] 找不到参照臂 (natural / uniform 族) — 显著性标注缺位, "
              "仍出图")

    def _z(v, v0, se):
        if v is None or v0 is None or not se:
            return None
        return (v - v0) / se

    n_t2 = BENCHMARK_SIZES.get(t2) if t2 else None
    if t2 and n_t2 is None:
        print(f"[B] 任务 '{t2}' 不在 BENCHMARK_SIZES — 第二联无二项误差棒/z")
    for r in rows:
        r["z_stem"] = (_z(r["stem"], ref_row["stem"], math.sqrt(2) * se_stem)
                       if ref_row else None)
        r["g2"] = ((r["tasks"].get(t2) or {}).get("raw")
                   if t2 else None)
        se_a = binom_se(r["g2"], n_t2) if r["g2"] is not None else None
        se_b = (binom_se(ref_row.get("g2"), n_t2)
                if ref_row and ref_row.get("g2") is not None else None)
        r["se_g"] = se_a
        r["z_g"] = (_z(r["g2"], ref_row.get("g2"),
                       math.sqrt((se_a or 0) ** 2 + (se_b or 0) ** 2))
                    if ref_row and se_a and se_b else None)

    if not HAS_MPL:
        print("[B] matplotlib 不可用 — 文字版 (STEM 降序):")
        for r in rows:
            z = f" z={r['z_stem']:+.2f}" if r["z_stem"] is not None else ""
            g2 = (f"{r['g2']:.4f}" if r["g2"] is not None else "n/a")
            print(f"    {r['arm']:<18} STEM {r['stem']:.4f}{z}  "
                  f"{t2 or '-'} {g2}")
        return None

    def _panel_labels(z_key):
        out = []
        for r in rows:
            z = r[z_key]
            tag = f"\n(z={z:+.1f})" if z is not None else ""
            mark = " (ref)" if ref_row and r["arm"] == ref_row["arm"] else ""
            out.append(r["arm"] + mark + tag)
        return out

    colors = [_arm_color(r["arm"]) for r in rows]
    hatches = ["//" if r["arm"].endswith("_rep") else "" for r in rows]

    if t2:
        fig, axes = plt.subplots(1, 2,
                                 figsize=(max(12, 1.5 * len(rows)), 5.5))
        panels = [
            (axes[0], "stem", None, "z_stem",
             "STEM (centered acc, 6 tasks)", "STEM centered"),
            (axes[1], "g2", "se_g", "z_g",
             f"{t2} (raw acc)" + ("  [auto-picked]" if t2_auto else ""),
             f"{t2} raw acc"),
        ]
    else:
        fig, axes = plt.subplots(1, 1,
                                 figsize=(max(8, 1.5 * len(rows)), 5.5))
        panels = [(axes, "stem", None, "z_stem",
                   "STEM (centered acc, 6 tasks)", "STEM centered")]
    for ax, key, se_key, z_key, ttl, ylab in panels:
        vals = [r[key] if r[key] is not None else float("nan")
                for r in rows]
        xs = np.arange(len(rows))
        bars = ax.bar(xs, vals, color=colors, width=0.62,
                      edgecolor="white", linewidth=0.5)
        for b, h in zip(bars, hatches):
            if h:
                b.set_hatch(h)
        if se_key:
            errs = [r[se_key] if r[se_key] is not None else 0.0 for r in rows]
        else:
            errs = [se_stem] * len(rows)
        ax.errorbar(xs, vals, yerr=errs, fmt="none",
                    ecolor="#333333", capsize=3, linewidth=1)
        for xi, v in zip(xs, vals):
            if v is not None and np.isfinite(v):
                ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(_panel_labels(z_key), fontsize=8,
                           rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    ref_name = ref_row["arm"] if ref_row else "n/a"
    fig.suptitle(f"Arm comparison at target scale (ref = {ref_name}; "
                 f"z = advantage in noise units, |z|>=2 significant)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "arms_main_results.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── C: 代理→真实一致性散点 ─────────────────────────────────────────────

def collect_pairs(run_dir, fleet, rows):
    """eval 臂 × 舰队 → (arm, proxy, d28) 对; 只收能落到实测 config_id 的
    climb 族臂 (cfg / climb_optimal 经 D19 best_measured 解析)。"""
    if fleet is None:
        return []
    ids_map = {int(cid): i for i, cid in enumerate(fleet["ids"].tolist())}
    climb_cid = resolve_climb_config_id(run_dir, fleet["state"])
    pairs = []
    for r in rows:
        k = parse_arm_name(r["arm"])
        cid = k.get("config_id")
        if k["kind"] == "climb_optimal":
            cid = climb_cid
            if cid is None:
                continue
        if k["kind"] != "cfg" and k["kind"] != "climb_optimal":
            continue
        if cid is None or int(cid) not in ids_map:
            continue
        i = ids_map[int(cid)]
        proxy = float(fleet["scores"][i])
        if not np.isfinite(proxy):
            continue
        pairs.append({"arm": r["arm"], "proxy": proxy, "d28": r["stem"]})
    return pairs


def chart_consistency(out_dir, pairs, baseline_stems=None, se=0.006):
    if len(pairs) < 2:
        print("[C] 可配对的 climb 臂不足 (需要 eval 臂 + 舰队 config_id) — "
              "跳过一致性散点")
        return None
    rho = _spearman([p["proxy"] for p in pairs], [p["d28"] for p in pairs])
    prank = {i: r + 1 for r, i in enumerate(
        sorted(range(len(pairs)), key=lambda i: -pairs[i]["proxy"]))}
    drank = {i: r + 1 for r, i in enumerate(
        sorted(range(len(pairs)), key=lambda i: -pairs[i]["d28"]))}
    txt = (f"proxy->target rank consistency: n={len(pairs)}, "
           f"Spearman rho = {rho:+.4f}" if len(pairs) >= 3
           else f"proxy->target rank consistency: n={len(pairs)} (rho "
                f"needs n>=3)")
    print(f"[C] {txt}")
    for i, p in enumerate(pairs):
        print(f"    {p['arm']:<18} proxy {p['proxy']:+.4f} (#{prank[i]})  "
              f"d28 {p['d28']:.4f} (#{drank[i]})")

    if not HAS_MPL:
        print("    注: 基线臂 (uniform/natural/domainfix) 不在搜索舰队, "
              "无实测 proxy 分 — 如实缺席")
        return None

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if baseline_stems:
        ax.axhspan(min(baseline_stems), max(baseline_stems),
                   color="#999999", alpha=0.20,
                   label="baseline arms d28 range "
                         "(uniform/natural/domainfix)")
    # 沿 proxy 名次连接各点: 线单调上行 = 排名传递; 折返 = 该处换位
    order = sorted(range(len(pairs)), key=lambda i: -pairs[i]["proxy"])
    ax.plot([pairs[i]["proxy"] for i in order],
            [pairs[i]["d28"] for i in order],
            "--", color="#888888", lw=1.0, zorder=3)
    ax.errorbar([p["proxy"] for p in pairs], [p["d28"] for p in pairs],
                yerr=[se] * len(pairs), fmt="none", ecolor="#333333",
                capsize=3, linewidth=1, zorder=4)
    ax.scatter([p["proxy"] for p in pairs], [p["d28"] for p in pairs],
               s=90, color=CLIMB_COLOR, edgecolors="white", linewidth=1.2,
               zorder=5)
    for i, p in enumerate(pairs):
        ax.annotate(f"{p['arm']}\n#{prank[i]} -> #{drank[i]}",
                    (p["proxy"], p["d28"]),
                    textcoords="offset points", xytext=(9, -4),
                    fontsize=8.5)
    ax.set_xlabel("d20 proxy utility (measured, search fleet)")
    ax.set_ylabel(f"d28 STEM (centered, +/- {se:.3f} eval noise)")
    ax.set_title(txt if len(pairs) >= 3
                 else "proxy->target rank consistency")
    ax.grid(alpha=0.3)
    if baseline_stems:
        ax.legend(fontsize=8, loc="lower right")
    fig.text(0.5, 0.005,
             "each point = one climb arm: its measured proxy score (x) vs "
             "its d28 STEM (y); '#p->#d' = its rank in each space.\n"
             "reading: top-1 holds; adjacent swaps inside the error bars "
             "are noise (trust regions, not points - KEY_FINDINGS #3); "
             "all climb arms sit above the gray baseline band.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    path = os.path.join(out_dir, "proxy_target_consistency.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── D: 逐簇小倍数图 ────────────────────────────────────────────────────

def chart_cluster_smallmult(out_dir, fleet, labels, maximize):
    W, scores = fleet["W"], fleet["scores"]
    K = fleet["K"]
    finite = np.isfinite(scores)
    W, scores = W[finite], scores[finite]
    if len(scores) < 3:
        print("[D] 舰队点数不足 — 跳过逐簇小倍数图")
        return None
    rhos = [_spearman(W[:, k], scores) for k in range(K)]
    if not HAS_MPL:
        print("[D] matplotlib 不可用 — 逐簇 rho (文字版):")
        for k in range(K):
            print(f"    {labels[k]:<6} rho = {rhos[k]:+.4f}")
        return None
    ncols = min(5, K)
    nrows = math.ceil(K / ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.0 * ncols, 2.8 * nrows),
                             squeeze=False)
    for k in range(K):
        ax = axes.flat[k]
        ax.scatter(W[:, k], scores, s=10, alpha=0.5,
                   color=ROUND_COLORS[k % len(ROUND_COLORS)],
                   edgecolors="none")
        mk = np.isfinite(W[:, k]) & np.isfinite(scores)
        if int(mk.sum()) >= 3 and float(np.ptp(W[mk, k])) > 0:
            fit = np.poly1d(np.polyfit(W[mk, k], scores[mk], 1))
            xs = np.linspace(float(W[mk, k].min()), float(W[mk, k].max()), 24)
            ax.plot(xs, fit(xs), "--", color="#333333", lw=1.0, alpha=0.75)
        col = ("#2A7F3E" if rhos[k] > 0.3 else
               "#C44E52" if rhos[k] < -0.3 else "#666666")
        ax.set_title(f"{labels[k]}  rho={rhos[k]:+.2f}", fontsize=9, color=col)
        ax.tick_params(labelsize=7)
    for k in range(K, nrows * ncols):
        axes.flat[k].axis("off")
    fig.suptitle("Per cluster: does putting MORE of it help? "
                 "(x = mixture weight, y = proxy score; dashed = linear fit)",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "rho = Spearman(weight, score) per cluster: green = the search "
             "rewards MORE of this cluster, red = less, gray = no clear "
             "direction", ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = os.path.join(out_dir, "cluster_alpha_vs_score.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── E: best-vs-worst 配方热力图 ─────────────────────────────────────────

def chart_heatmap(out_dir, fleet, labels, maximize, n_side=5):
    W, scores, ids = fleet["W"], fleet["scores"], fleet["ids"]
    finite = np.isfinite(scores)
    if int(finite.sum()) < 2 * n_side:
        print(f"[E] 舰队点数不足 ({int(finite.sum())} < {2 * n_side}) — "
              "跳过热力图")
        return None
    order = (_topk_idx(scores, len(scores), maximize)
             if maximize else
             np.argsort(np.where(finite, scores, np.inf)))
    order = np.asarray(order, dtype=int)
    order = order[np.isfinite(scores[order])]
    best_i = order[:n_side]
    worst_i = order[-n_side:][::-1]
    sel = np.concatenate([best_i, worst_i])
    M = W[sel]                       # (2*n_side, K)
    col_labels = ([f"b{r + 1}\ncfg{ids[i]}" for r, i in enumerate(best_i)] +
                  [f"w{r + 1}\ncfg{ids[i]}" for r, i in enumerate(worst_i)])

    if not HAS_MPL:
        print("[E] matplotlib 不可用 — 文字版 (best/worst 权重):")
        head = "      " + " ".join(f"{c.split(chr(10))[1]:>8}" for c in col_labels)
        print(head)
        for k in range(fleet["K"]):
            print(f"  {labels[k]:<4}" +
                  " ".join(f"{M[r, k]:>8.3f}" for r in range(M.shape[0])))
        return None

    K = fleet["K"]
    delta = W[best_i].mean(axis=0) - W[worst_i].mean(axis=0)
    fig = plt.figure(figsize=(max(8, 1.1 * M.shape[0]) + 3.5, 0.5 * K + 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[M.shape[0], 2.4], wspace=0.05)
    ax = fig.add_subplot(gs[0, 0])
    axd = fig.add_subplot(gs[0, 1], sharey=ax)
    im = ax.imshow(M.T, aspect="auto", cmap="viridis",
                   vmin=0, vmax=max(float(M.max()), 1e-9))
    ax.set_xticks(np.arange(M.shape[0]))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(np.arange(K))
    ax.set_yticklabels(labels, fontsize=8)
    for r in range(M.shape[0]):
        for k in range(K):
            v = M[r, k]
            ax.text(r, k, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 * float(M.max()) else "black")
    ax.axvline(n_side - 0.5, color="white", linewidth=2)
    ax.set_title("Winners vs losers: cluster mixture share")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mixture share (alpha)")
    axd.barh(np.arange(K), delta,
             color=["#2A7F3E" if v > 0 else "#C44E52" if v < 0 else "#999999"
                    for v in delta])
    axd.axvline(0, color="gray", linewidth=0.8)
    for k, v in enumerate(delta):
        axd.text(v, k, f" {v:+.2f}", va="center",
                 ha="left" if v >= 0 else "right", fontsize=7)
    lo, hi = float(delta.min()), float(delta.max())
    pad = 0.3 * max(hi - lo, 0.05)
    axd.set_xlim(lo - pad, hi + pad)
    axd.set_xlabel("winner - loser\n(mean share)")
    axd.tick_params(labelleft=False)
    axd.grid(axis="x", alpha=0.3)
    fig.text(0.5, 0.005,
             "columns b1..b5 = top-5 / w1..w5 = bottom-5 measured configs; "
             "right panel: how much MORE (green) or LESS (red) the winners "
             "put in each cluster", ha="center", fontsize=8, color="#555555")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    path = os.path.join(out_dir, "best_vs_worst_heatmap.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ── main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="结果叙事图表包: 收敛/臂间/一致性/逐簇/热力图 + 判定块")
    ap.add_argument("run_dir", nargs="?", default="result/prod5_current",
                    help="RUN_DIR (含 search_state.json / eval_*.csv)")
    ap.add_argument("--extra-run", default="",
                    help="另一 RUN_DIR: 图 C 并入其 climb 臂 (跨轮证据)")
    ap.add_argument("--direction", choices=["maximize", "minimize"],
                    default="maximize",
                    help="proxy 分方向 (prod SNR utility = maximize)")
    ap.add_argument("--se", type=float, default=0.006,
                    help="STEM 单次评测噪声 (cp4 同约定, 缺省 0.006)")
    ap.add_argument("--task2", default="gsm8k_cot",
                    help="臂间图第二联任务 (缺省 gsm8k_cot; 该任务缺失时"
                         "自动改用跨臂区分度最大的任务)")
    ap.add_argument("--out-dir", default="",
                    help="输出目录 (缺省 RUN_DIR)")
    args = ap.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        print(f"[!] RUN_DIR 不存在: {run_dir}")
        return 1
    out_dir = args.out_dir or run_dir
    maximize = args.direction == "maximize"
    t0 = time.time()
    made = []

    fleet = load_fleet(run_dir)
    if fleet is None:
        print("[!] search_state.json 缺失或不可解析 — A/C/D/E/F 降级, "
              "仅尝试 B (臂间图)")
        labels = []
    else:
        labels = cluster_labels(run_dir, fleet["K"])

    # F: 判定块 (只要舰队在就出)
    if fleet is not None:
        print(verdict_block(fleet, maximize))
        print()

    # A: 收敛
    if fleet is not None:
        p = chart_convergence(out_dir, fleet, maximize)
        if p:
            made.append(p)

    # B: 臂间主结果 (+ 供 C 用的 rows)
    rows = []
    for a in discover_arms(run_dir):
        ev = parse_eval_csv(os.path.join(run_dir, f"eval_{a}.csv"))
        if ev and ev.get("stem") is not None:
            rows.append({"arm": a, "stem": float(ev["stem"])})
    p = chart_arms(out_dir, run_dir, args.se, args.task2) \
        if rows else None
    if p:
        made.append(p)

    # C: 一致性 (本 run + 可选 extra run)
    pairs = collect_pairs(run_dir, fleet, rows) if fleet is not None else []
    if args.extra_run and os.path.isdir(args.extra_run):
        fleet2 = load_fleet(args.extra_run)
        if fleet2 is not None:
            rows2 = []
            for a in discover_arms(args.extra_run):
                ev = parse_eval_csv(
                    os.path.join(args.extra_run, f"eval_{a}.csv"))
                if ev and ev.get("stem") is not None:
                    rows2.append({"arm": a, "stem": float(ev["stem"])})
            extra = collect_pairs(args.extra_run, fleet2, rows2)
            for e in extra:
                e["arm"] = e["arm"] + "@" + os.path.basename(
                    args.extra_run.rstrip("/"))
            pairs.extend(extra)
    p = chart_consistency(out_dir, pairs,
                          [r["stem"] for r in rows
                           if parse_arm_name(r["arm"])["kind"]
                           in ("uniform", "natural", "domainfix")],
                          args.se)
    if p:
        made.append(p)

    # D / E
    if fleet is not None:
        p = chart_cluster_smallmult(out_dir, fleet, labels, maximize)
        if p:
            made.append(p)
        p = chart_heatmap(out_dir, fleet, labels, maximize)
        if p:
            made.append(p)

    print("-" * 66)
    if made:
        print(f"生成 {len(made)} 个图表 ({time.time() - t0:.1f}s):")
        for p in made:
            print(f"  {p}")
    else:
        print("未生成任何图表 (输入不足或 matplotlib 缺失 — 见上方降级说明)")
    if not HAS_MPL:
        print("注: matplotlib 不可用, 已输出文字版; 图表需安装 matplotlib")
    return 0


if __name__ == "__main__":
    sys.exit(main())
