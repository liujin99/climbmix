#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  recipe_report.py — 赢家配方解剖 (what did the winning arm actually
#  train on, and how does that recipe differ from the losers)
#
#  用法:
#    python3 scripts/diagnostics/recipe_report.py [RUN_DIR]
#    python3 scripts/diagnostics/recipe_report.py result/prod4_current \
#        --natural-weights /home/ma-user/work/tmp/natural_weights.json
#    python3 scripts/diagnostics/recipe_report.py result/prod4_current \
#        --arm climb-cfg72 --arm-weights domainfix=/path/w.json
#
#  输入 (RUN_DIR 下, 全部只读; 缺件降级不报错):
#    eval_<arm>.csv              各臂 d28 评测 (cp4_report 同格式);
#                                赢家 = stem 最高臂 (--arm 可指定)
#    topk_mixture_candidates.json top-k 搜索候选 (config_id + 标签键权重);
#                                臂名 climb-cfg<config_id> 由此解析配方
#    search_state.json           搜索历史 (全部已测配置+分数) → 舰队语境
#    cluster_info_cache.json     簇标签/规模/token/质量分
#    optimal_mixture_weights.json 设计空间 argmin 权重 (参照线, D19)
#    --natural-weights           natural 臂权重文件; 缺省按簇 token 占比重算
#    --arm-weights NAME=PATH     其他自定义臂 (如 domainfix) 的权重, 可多次
#
#  输出 (RUN_DIR/recipe_analysis/):
#    recipe_report.md       主报告 (图+表嵌入, 可整段抄进实验文档)
#    recipe_tables.md       纯表格
#    recipe_analysis.json   机读
#    figs/winner_vs_baselines.png  逐簇 α: 赢家 vs 其他 climb 臂 vs 基线
#    figs/winner_vs_fleet.png      赢家 α vs 全搜索舰队均值±散布
#    figs/alpha_vs_quality.png     赢家 α vs 簇质量分 (气泡 = 池 token 占比)
#
#  背景缺口 (experiment_prod4 §6 / experiment_prod5 预注册): 报告了 d28
#  臂间得分对比, 但没拆赢家配方 — 各簇配比是什么、簇是什么、与输家差在
#  哪。零 NPU; numpy 必需, matplotlib 可选 (缺失时只出表); 不依赖
#  climbmix 包 (服务器裸 python3 可跑, 与 cp4_report.py 同原则)。
#
#  口径注意: 臂分数 = d28 stem (centered, eval CSV); 舰队分数 = d20
#  proxy SNR 分 (search_state) — 两个尺度不可直接比大小, 只做各自语境。
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

CLIMB_ARM_RE = re.compile(r"^(?:climb-)?cfg(\d+)$")


# ── 工件读取 ───────────────────────────────────────────────────────────

def _f(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_json(path):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [!] {os.path.basename(path)} unreadable: {e}")
        return None


def parse_eval_csv(path):
    """eval CSV → {"stem": float|None, "stem_nll": float|None}.
    与 cp4_report.parse_eval_csv 同语义 (STEM 聚合行; centered 列)."""
    if not path or not os.path.isfile(path):
        return None
    stem = stem_nll = None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            if parts[0] == "STEM":
                stem = _f(parts[2])
                stem_nll = _f(parts[3]) if len(parts) >= 4 else None
                break
    return {"stem": stem, "stem_nll": stem_nll}


def discover_arms(run_dir):
    """eval_<arm>.csv 全发现 (锚点除外) → 臂名列表."""
    arms = []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return []
    for n in names:
        if n.startswith("eval_") and n.endswith(".csv") and n != "eval_base_remote.csv":
            arms.append(n[len("eval_"):-len(".csv")])
    return sorted(arms)


def weights_vec_from_payload(payload, labels):
    """权重载荷 (标签键 dict | list | 最优权重的 {"weights": {...}} 变体)
    → 按 labels 顺序的向量; 长度不符/缺标签 → None."""
    if payload is None:
        return None
    if isinstance(payload, dict):
        payload = payload.get("weights", payload)
    if isinstance(payload, dict):
        try:
            return np.array([float(payload[l]) for l in labels], dtype=np.float64)
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(payload, list):
        v = np.array(payload, dtype=np.float64)
        return v if len(v) == len(labels) else None
    return None


def l1(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).sum())


# ── 主流程 ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Winner recipe dissection: what data mix won, and why-ish")
    ap.add_argument("run_dir", nargs="?", default="result/prod4_current")
    ap.add_argument("--arm", default="",
                    help="指定赢家臂 (默认 = d28 stem 最高的臂)")
    ap.add_argument("--natural-weights", default="",
                    help="natural 臂权重 json (缺省按簇 token 占比重算)")
    ap.add_argument("--arm-weights", action="append", default=[],
                    metavar="NAME=PATH",
                    help="自定义臂权重文件, 可多次 (如 domainfix=w.json)")
    ap.add_argument("--se", type=float, default=0.006,
                    help="单次评测 stem SE (噪声带 |Δ|<=√2·SE)")
    ap.add_argument("--top-l1", type=int, default=6,
                    help="差异分解表每对显示的簇数")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = os.path.join(run_dir, "recipe_analysis")
    fig_dir = os.path.join(out_dir, "figs")
    notes = []          # 降级/口径注记

    # ── 簇信息 (地基; 缺它什么都做不了) ──
    ci_path = os.path.join(run_dir, "cluster_info_cache.json")
    cluster_info = load_json(ci_path)
    if not isinstance(cluster_info, list) or not cluster_info:
        print(f"[!] no usable cluster_info_cache.json under {run_dir} — abort")
        return 1
    labels = [c.get("label") or f"C{c.get('cluster_id', i)}"
              for i, c in enumerate(cluster_info)]
    K = len(labels)
    num_tokens = np.array([int(c.get("num_tokens") or 0) for c in cluster_info],
                          dtype=np.float64)
    quality = np.array([float(c.get("quality_score") or 0.0)
                        for c in cluster_info], dtype=np.float64)
    tok_share = num_tokens / num_tokens.sum() if num_tokens.sum() > 0 else num_tokens

    # ── 臂分数 (d28) ──
    arms = discover_arms(run_dir)
    if not arms:
        print(f"[!] no eval_<arm>.csv under {run_dir} — CP4 未落地, 无从判赢家")
        return 1
    scores = {}
    for a in arms:
        d = parse_eval_csv(os.path.join(run_dir, f"eval_{a}.csv"))
        if d and d["stem"] is not None:
            scores[a] = d
    if not scores:
        print("[!] 没有任何臂有可用 stem 分数 — abort")
        return 1
    ranked = sorted(scores, key=lambda a: -scores[a]["stem"])

    winner = args.arm if args.arm else ranked[0]
    if winner not in scores:
        print(f"[!] --arm {winner} 没有可用分数 (可用: {', '.join(ranked)})")
        return 1
    if args.arm and args.arm != ranked[0]:
        notes.append(f"赢家由 --arm 指定为 {winner} (按分数本是 {ranked[0]})")
    noise_band = math.sqrt(2.0) * args.se

    # ── 配方解析: 每个臂 → 权重向量 (能解则解, 解不了注明) ──
    topk = load_json(os.path.join(run_dir, "topk_mixture_candidates.json")) or {}
    topk_by_id = {}
    for c in topk.get("candidates") or []:
        cid = c.get("config_id")
        if cid is not None:
            topk_by_id[int(cid)] = c

    arm_weights = {}    # arm -> np.ndarray
    arm_src = {}        # arm -> 人类可读的配方来源
    for a in ranked:
        m = CLIMB_ARM_RE.match(a)
        if m:
            cid = int(m.group(1))
            c = topk_by_id.get(cid)
            v = weights_vec_from_payload((c or {}).get("weights"), labels)
            if v is None:
                arm_src[a] = f"cfg{cid}: topk 文件缺失或权重无法对齐标签"
            else:
                arm_weights[a] = v
                arm_src[a] = (f"topk 候选 cfg{cid} "
                              f"(rank {c.get('rank')}, d20 分 {c.get('score'):+.4f})")
        elif a in ("random", "random3b", "uniform"):
            arm_weights[a] = np.full(K, 1.0 / K)
            arm_src[a] = "uniform α=1/K (论文 App. C.1 Random)"
        elif a == "natural":
            v = weights_vec_from_payload(
                load_json(args.natural_weights), labels) \
                if args.natural_weights else None
            if v is None:
                v = tok_share.copy()
                arm_src[a] = "池 token 占比 (由 cluster_info_cache 重算)"
            else:
                arm_src[a] = f"natural 权重文件 ({args.natural_weights})"
            arm_weights[a] = v
        else:
            found = None
            for pair in args.arm_weights:
                if "=" in pair:
                    name, path = pair.split("=", 1)
                    if name == a:
                        found = weights_vec_from_payload(load_json(path), labels)
                        arm_src[a] = f"权重文件 ({path})"
                        break
            if found is not None:
                arm_weights[a] = found
            else:
                arm_src[a] = "未知配方 (可用 --arm-weights 提供)"
    for a in list(arm_weights):
        s = arm_weights[a].sum()
        if s > 0:
            arm_weights[a] = arm_weights[a] / s

    if winner not in arm_weights:
        print(f"[!] 赢家 {winner} 的配方无法解析 ({arm_src.get(winner)}) — abort")
        return 1
    W = arm_weights[winner]

    # 参照: 设计空间 argmin (D19: 搜索输出是一个区域)
    argmin_v = weights_vec_from_payload(
        load_json(os.path.join(run_dir, "optimal_mixture_weights.json")), labels)

    # ── 舰队语境 (d20 search_state) ──
    state = load_json(os.path.join(run_dir, "search_state.json")) or {}
    fleet = None
    fleet_iter_of = {}
    if state.get("accumulated_configs"):
        cfgs = state["accumulated_configs"]
        sc = state.get("accumulated_scores") or []
        Ws = []
        ok_ids = []
        ok_sc = []
        for i, c in enumerate(cfgs):
            v = c.get("weights")
            if isinstance(v, list) and len(v) == K:
                Ws.append(v)
                ok_ids.append(c.get("config_id"))
                ok_sc.append(sc[i] if i < len(sc) else None)
        if Ws:
            fleet = {
                "W": np.array(Ws, dtype=np.float64),
                "ids": ok_ids,
                "scores": np.array(
                    [np.nan if s is None else float(s) for s in ok_sc],
                    dtype=np.float64),
            }
            # 轮次溯源: accumulated 按批次追加, realized_configs_per_iter
            # 给出每轮追加数 → 位置 → 轮次 (派生口径)
            rc = state.get("realized_configs_per_iter") or []
            if rc and sum(rc) == len(Ws):
                pos = 0
                for it, n in enumerate(rc, start=1):
                    for _ in range(n):
                        fleet_iter_of[pos] = it
                        pos += 1
            else:
                notes.append("realized_configs_per_iter 与累计配置数不符 — 轮次溯源不可用")
        else:
            notes.append("search_state 中没有 K 维配置 — 舰队语境跳过")
    else:
        notes.append("search_state.json 缺失 — 舰队语境跳过")

    # 赢家 config_id 的舰队排名 + 轮次
    wm = CLIMB_ARM_RE.match(winner)
    fleet_ctx = None
    if fleet is not None and wm:
        cid = int(wm.group(1))
        idx = None
        for i, cidx in enumerate(fleet["ids"]):
            if cidx == cid:
                idx = i
                break
        if idx is not None:
            sc = fleet["scores"]
            fin = np.isfinite(sc)
            # 简洁可靠的排名: 有限分中严格大于该配置分的个数 + 1
            rank = int((sc[fin] > sc[idx]).sum()) + 1 if fin[idx] else None
            pct = 100.0 * rank / max(1, int(fin.sum())) if rank else None
            fleet_ctx = {
                "config_id": cid, "rank": rank, "n": int(fin.sum()),
                "percentile": pct,
                "d20_score": float(sc[idx]) if fin[idx] else None,
                "iteration": fleet_iter_of.get(idx),
            }

    # ── 输出 ──
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    figs = {}
    if HAS_MPL:
        figs = make_figs(fig_dir, labels, W, winner, arm_weights,
                         ranked, fleet, quality, tok_share)
    else:
        notes.append("matplotlib 不可用 — 只出表不出图")

    report_lines, table_lines, machine = build_report(
        run_dir, labels, K, num_tokens, tok_share, quality,
        ranked, scores, winner, noise_band, arm_weights, arm_src,
        W, argmin_v, fleet, fleet_ctx, fleet_iter_of, figs,
        notes, args)

    for name, text in (("recipe_report.md", report_lines),
                       ("recipe_tables.md", table_lines)):
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(text) + "\n")
        print(f"[Save] {path}")
    jpath = os.path.join(out_dir, "recipe_analysis.json")
    with open(jpath, "w") as f:
        json.dump(machine, f, indent=2, ensure_ascii=False, default=_json_default)
        f.write("\n")
    print(f"[Save] {jpath}")
    print(f"\n赢家: {winner} "
          f"(stem {scores[winner]['stem']:.4f}) — 报告: {out_dir}/recipe_report.md")
    return 0


# ── 图 ─────────────────────────────────────────────────────────────────

def make_figs(fig_dir, labels, W, winner, arm_weights, ranked, fleet,
              quality, tok_share):
    figs = {}
    order = np.argsort(-W)                      # 按赢家 α 降序

    # 1) 赢家 vs 其他 climb 臂 vs uniform/natural
    others = [a for a in ranked
              if a != winner and a in arm_weights
              and CLIMB_ARM_RE.match(a)][:2]
    baselines = [a for a in ranked
                 if a != winner and a in arm_weights
                 and not CLIMB_ARM_RE.match(a)][:2]
    show = [winner] + others + baselines
    if len(show) >= 2:
        n = len(show)
        width = 0.8 / n
        fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(labels)), 6))
        colors = ["#DD8452", "#4C72B0", "#8172B2", "#937860", "#DA8BC3"]
        for j, a in enumerate(show):
            ax.bar(np.arange(len(labels)) + (j - n / 2 + 0.5) * width,
                   arm_weights[a][order], width, label=a,
                   color=colors[j % len(colors)], alpha=0.9)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([labels[i] for i in order], rotation=45,
                           ha="right", fontsize=8)
        ax.set_ylabel("mixture weight α")
        ax.set_title(f"Per-cluster recipe: {winner} vs others")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, "winner_vs_baselines.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["vs_baselines"] = os.path.basename(p)

    # 2) 赢家 vs 舰队
    if fleet is not None and len(fleet["W"]):
        mean = fleet["W"].mean(axis=0)
        std = fleet["W"].std(axis=0)
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(10, 1.1 * len(labels)), 6))
        ax.bar(x - 0.2, W[order], 0.4, label=f"{winner}",
               color="#DD8452", alpha=0.9)
        ax.bar(x + 0.2, mean[order], 0.4, yerr=std[order],
               label=f"fleet mean ±sd (n={len(fleet['W'])})",
               color="#4C72B0", alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[i] for i in order], rotation=45,
                           ha="right", fontsize=8)
        ax.set_ylabel("mixture weight α")
        ax.set_title("Winner recipe vs search-fleet distribution (d20 space)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, "winner_vs_fleet.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["vs_fleet"] = os.path.basename(p)

    # 3) α vs 簇质量 (气泡 = 池 token 占比)
    if float(quality.std()) > 0:
        fig, ax = plt.subplots(figsize=(7, 6))
        size = 40 + 360 * tok_share / max(tok_share.max(), 1e-9)
        ax.scatter(quality, W, s=size, color="#4C72B0", alpha=0.75,
                   edgecolors="none")
        for i in range(len(labels)):
            if W[i] >= np.sort(W)[-min(10, len(labels))]:
                ax.annotate(labels[i], (quality[i], W[i]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("cluster quality score")
        ax.set_ylabel(f"winner α ({winner})")
        ax.set_title("Winner weights vs cluster quality (bubble = pool token share)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        p = os.path.join(fig_dir, "alpha_vs_quality.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["vs_quality"] = os.path.basename(p)
    return figs


# ── 报告组装 ───────────────────────────────────────────────────────────

def build_report(run_dir, labels, K, num_tokens, tok_share, quality,
                 ranked, scores, winner, noise_band, arm_weights, arm_src,
                 W, argmin_v, fleet, fleet_ctx, fleet_iter_of, figs,
                 notes, args):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    R = []            # recipe_report.md
    T = []            # recipe_tables.md
    M = {"run_dir": run_dir, "generated": ts, "winner": winner,
         "winner_stem": scores[winner]["stem"], "clusters": [], "notes": notes}

    R += [f"# 赢家配方解剖 — {os.path.basename(run_dir.rstrip('/'))}",
          "",
          f"**生成:** {ts}   **赢家臂:** `{winner}` "
          f"(d28 stem {scores[winner]['stem']:.4f})",
          "",
          "口径: 臂分数 = d28 STEM (centered acc, eval CSV); "
          "舰队分数 = d20 proxy SNR 分。两个尺度只做各自语境, 不互比。",
          ""]

    # 1. 臂间记分板
    R += ["## 1. 臂间记分板 (d28)", "",
          "| 臂 | stem | Δ vs 赢家 | 噪声带 | 配方来源 |",
          "|---|---|---|---|---|"]
    base = parse_eval_csv(os.path.join(run_dir, "eval_base_remote.csv"))
    if base and base["stem"] is not None:
        R.append(f"| _base (锚点)_ | _{base['stem']:.4f}_ | "
                 f"_{base['stem'] - scores[winner]['stem']:+.4f}_ | | "
                 "_基模型, 无配方_ |")
    for a in ranked:
        d = scores[a]["stem"] - scores[winner]["stem"]
        band = "≈平" if abs(d) <= noise_band else ("落后" if d < 0 else "领先")
        if a == winner:
            R.append(f"| **{a}** | **{scores[a]['stem']:.4f}** | (赢家) | | "
                     f"{arm_src.get(a, '')} |")
        else:
            R.append(f"| {a} | {scores[a]['stem']:.4f} | {d:+.4f} | {band} | "
                     f"{arm_src.get(a, '—')} |")
    R += [f"(噪声带 |Δ| ≤ √2×SE = {noise_band:.4f}, SE={args.se}; "
          "显著性判定请以 cp4_report.py 为准)", ""]

    # 2. 逐簇配方表
    R += ["## 2. 赢家配方逐簇明细", ""]
    cmp_arms = [a for a in ranked if a != winner and a in arm_weights][:4]
    R += ["| 簇 | 池 token% | 簇质量分 | " +
          " | ".join(f"{a} α" for a in cmp_arms + [winner]) +
          " | 赢家/池 倍率 |", "|---|---|---|" + "---|" * (len(cmp_arms) + 2)]
    order = np.argsort(-W)
    for i in order:
        row = [f"{labels[i]} | {100*tok_share[i]:.1f}% | {quality[i]:.2f} | "]
        row += [f"{arm_weights[a][i]:.4f} | " for a in cmp_arms]
        ratio = (W[i] / tok_share[i]) if tok_share[i] > 0 else float("inf")
        row += [f"**{W[i]:.4f}** | {ratio:.1f}× |"]
        R.append("| " + "".join(row).rstrip("| ") + " |")
        M["clusters"].append({
            "label": labels[i], "pool_token_share": float(tok_share[i]),
            "quality": float(quality[i]),
            "winner_alpha": float(W[i]),
            "pool_ratio": float(ratio) if math.isfinite(ratio) else None,
            "others": {a: float(arm_weights[a][i]) for a in cmp_arms},
        })
    if argmin_v is not None:
        R += ["", f"设计空间 argmin (optimal_mixture_weights.json, D19 参照): "
              f"与赢家 L1 = {l1(W, argmin_v):.3f}", ""]
    if figs.get("vs_baselines"):
        R += [f"![逐簇配方对比](figs/{figs['vs_baselines']})", ""]
    R += ["", "### 逐簇表 (纯表版见 recipe_tables.md)", ""]

    # 3. 舰队语境
    R += ["## 3. 赢家在搜索舰队中的位置 (d20)", ""]
    if fleet_ctx:
        it = fleet_ctx.get("iteration")
        it_s = f"第 {it} 轮" if it else "轮次不可考"
        R += [f"- 赢家配置 cfg{fleet_ctx['config_id']} 在 "
              f"{fleet_ctx['n']} 个已测配置中排 **第 {fleet_ctx['rank']}** "
              f"(前 {fleet_ctx['percentile']:.0f}%), 出自{it_s}, "
              f"d20 SNR 分 {fleet_ctx['d20_score']:+.4f}"]
        R += [f"- 舰队权重散布 (均值±sd) 见下图 — 赢家的非常规选择 "
              "(偏离舰队均值最远的簇) 是配方分析的重点", ""]
    elif fleet is not None:
        R += ["- 赢家臂不是 cfg 臂 (或 config_id 未在搜索历史中找到) — "
              "舰队排名不适用", ""]
    else:
        R += ["- search_state.json 不可用 — 舰队语境跳过", ""]
    if figs.get("vs_fleet"):
        R += [f"![赢家vs舰队](figs/{figs['vs_fleet']})", ""]
    if fleet is not None and fleet_ctx:
        dev = W - fleet["W"].mean(axis=0)
        top_dev = np.argsort(-np.abs(dev))[:5]
        R += ["赢家 vs 舰队均值偏离最大的簇:", "",
              "| 簇 | 赢家 α | 舰队均值 | 偏离 |", "|---|---|---|---|"]
        for i in top_dev:
            R.append(f"| {labels[i]} | {W[i]:.4f} | "
                     f"{fleet['W'].mean(axis=0)[i]:.4f} | {dev[i]:+.4f} |")
        R.append("")

    # 4. 机制视图
    R += ["## 4. 配方机制视图", ""]
    if figs.get("vs_quality"):
        R += [f"![α vs 簇质量](figs/{figs['vs_quality']})", ""]
    hi_q = np.argsort(-quality)[: K // 3]
    w_hi = W[hi_q].sum()
    s_hi = tok_share[hi_q].sum()
    R += [f"- 赢家把 **{100*w_hi:.0f}%** 的 token 预算给了质量分最高的 "
          f"{len(hi_q)} 个簇 (它们占池 {100*s_hi:.0f}%) — "
          f"{'超配' if w_hi > s_hi else '低配'} "
          f"({w_hi - s_hi:+.2f})", ""]

    # 5. 差异分解
    R += ["## 5. 差异分解: 赢家 vs 最近对手", ""]
    rivals = [a for a in ranked if a != winner and a in arm_weights]
    if rivals:
        rival = rivals[0]
        dvec = W - arm_weights[rival]
        contrib = np.abs(dvec)
        top = np.argsort(-contrib)[:args.top_l1]
        R += [f"最近对手: `{rival}` (stem {scores[rival]['stem']:.4f}, "
              f"L1 = {l1(W, arm_weights[rival]):.3f})", "",
              "| 簇 | 赢家 α | 对手 α | Δ | 贡献占比 |", "|---|---|---|---|---|"]
        tot = contrib.sum() or 1.0
        for i in top:
            R.append(f"| {labels[i]} | {W[i]:.4f} | "
                     f"{arm_weights[rival][i]:.4f} | {dvec[i]:+.4f} | "
                     f"{100*contrib[i]/tot:.0f}% |")
        R.append("")
        M["top_rival"] = {"arm": rival, "l1": l1(W, arm_weights[rival]),
                          "top_contributors": [
                              {"label": labels[i], "delta": float(dvec[i])}
                              for i in top]}
    else:
        R += ["(没有可解析配方的对手臂)", ""]

    # 注记 + 附录
    if notes:
        R += ["## 注记", ""] + [f"- {n}" for n in notes] + [""]
    R += ["## 附: 输入", "",
          f"- 臂分数: `{run_dir}/eval_<arm>.csv` (cp4_report 同解析)",
          f"- top-k 候选: `{run_dir}/topk_mixture_candidates.json`",
          f"- 搜索历史: `{run_dir}/search_state.json`",
          f"- 簇信息: `{run_dir}/cluster_info_cache.json`", ""]

    # 纯表版
    T += [f"# 赢家配方表 — {winner}", "",
          "| 簇 | 池token% | 质量 | 赢家α | " +
          " | ".join(f"{a}" for a in cmp_arms) + " | 赢家/池 |",
          "|---|---|---|---|" + "---|" * (len(cmp_arms) + 1)]
    for i in order:
        ratio = (W[i] / tok_share[i]) if tok_share[i] > 0 else float("inf")
        T.append("| " + " | ".join(
            [labels[i], f"{100*tok_share[i]:.1f}%", f"{quality[i]:.2f}",
             f"**{W[i]:.4f}**"] +
            [f"{arm_weights[a][i]:.4f}" for a in cmp_arms] +
            [f"{ratio:.1f}×"]) + " |")
    T += ["", "## 臂间记分板", "", "| 臂 | stem | Δ vs 赢家 |", "|---|---|---|"]
    for a in ranked:
        d = scores[a]["stem"] - scores[winner]["stem"]
        T.append(f"| {a} | {scores[a]['stem']:.4f} | {d:+.4f} |")

    M["figs"] = figs
    M["arm_scores"] = {a: scores[a]["stem"] for a in ranked}
    return R, T, M


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    sys.exit(main())
