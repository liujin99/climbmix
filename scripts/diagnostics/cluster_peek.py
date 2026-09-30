#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════
#  cluster_peek.py — 簇语义抽样: "C10 到底是什么" (零 NPU, 只读)
#
#  用法:
#    python3 scripts/diagnostics/cluster_peek.py \
#        --pool-dir /home/ma-user/work/100B_stem_parquet_filtered \
#        --cluster-cache result/prod4_current/cluster_cache.npz \
#        --clusters C10,C11,C5,C0,C12 --n 40 \
#        --out result/prod4_current/cluster_peek.md
#
#  输入:
#    --pool-dir        STEM 池 (metadata_cache.npz + shard parquet)
#    --cluster-cache   搜索的 cluster_cache.npz (final_labels, 簇空间)
#    --clusters        逗号分隔簇标签 (C10 / 10 均可) 或 all; 默认 all
#    --n               每簇抽样文档数 (默认 40)
#    --out             markdown 输出 (默认: cluster_cache 同目录 cluster_peek.md)
#
#  输出 (markdown): 每簇一节 — 规模/est-token/文档长度统计 + 域构成表
#  (簇×域, 文档数与 est-token 占比 — 顺带回答 domainfix 的域配额落在
#  哪些簇) + 质量列均值 + n 篇抽样文档首 300 字符。
#
#  集成 (2026-09-30): recipe_report 的赢家配方节 §2b "簇内容速写" 自动
#  内嵌本工具的数据面 (collect + render_brief; 池目录取自 launch_env 的
#  DATA_DIR, 紧凑版缓存 cluster_semantics.md 一次生成复用) — 报告自动带
#  簇语义, 手工 CLI 保留用于深挖 (多样本/指定簇)。
#
#  动机: 赢家配方解剖 (report.md) 的逐簇 α 表只有匿名 C0..C{K-1} —
#  "簇分别是什么" 从未被回答 (用户点名缺口)。零 NPU; 复用池级 metadata
#  缓存 (秒级), 文本读取走 ShardMetadataManager 并行 IO; 自举 src 路径
#  (prepare_random_baseline 同款, 无需 PYTHONPATH 前缀)。
# ═══════════════════════════════════════════════════════════════════════
import argparse
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from climbmix.data.metadata_manager import ShardMetadataManager


def _fmt_pct(x, tot):
    return f"{100.0 * x / tot:.1f}%" if tot > 0 else "—"


def _wanted_clusters(spec, K):
    """'all'/None → 全部; 否则逗号分隔字符串或 id 列表 (C10/10 均可)。"""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() == "all"):
        return list(range(K))
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    wanted = []
    for tok in items:
        tok = str(tok).strip()
        if not tok:
            continue
        m = re.match(r"^[Cc](\d+)$", tok)
        c = int(m.group(1)) if m else int(tok)
        if not (0 <= c < K):
            raise SystemExit(f"ERROR: cluster {tok} 不在 [0, {K})")
        wanted.append(c)
    return wanted


def collect(pool_dir, cluster_cache, clusters=None, n_per=25, chars=300,
            seed=7):
    """簇语义数据面 (CLI 完整版与 recipe_report §2b 紧凑版共用)。

    返回 (info, profiles):
      info = {num_docs, total_est_tok, K, pool_dir}
      profiles = 按 clusters 顺序, 每簇:
        {id, n_docs, est_tok, tok_pct, doc_pct, mean_chars, med_chars,
         p90_chars, quality: {列名: 均值} (原列序), domains: [(域名, docs,
         est_tok)] (域 id 序, 含 0 行), dom_docs_total, dom_tok_total,
         samples: [{gi, chars, snippet}]}
    空簇: 除 id/n_docs=0 外字段为空/0 (渲染层各自决定怎么显示)。"""
    print(f"[peek] loading pool metadata: {pool_dir}")
    mm = ShardMetadataManager(pool_dir)
    final = np.load(cluster_cache,
                    allow_pickle=False)["final_labels"].astype(np.int64)
    if len(final) != mm.num_docs:
        raise SystemExit(
            f"ERROR: cluster cache has {len(final):,} labels but pool has "
            f"{mm.num_docs:,} docs — pool/cache mismatch")
    K = int(final.max()) + 1
    dom = mm.cluster_labels.astype(np.int64)
    char = mm.doc_char_counts.astype(np.float64)
    est_tok = char / 4.0  # 硬编码 char/4 启发式, 同 selection 口径 (非旋钮)
    dom_names = list(getattr(mm._schema, "domain_names", None) or [])
    qcols = list(getattr(mm._schema, "quality_cols", None) or [])
    qual = np.asarray(mm.quality_scores, dtype=np.float64)
    if qual.ndim == 1:
        qual = qual[:, None]
    total_tok = float(est_tok.sum())
    wanted = _wanted_clusters(clusters, K)
    rng = np.random.default_rng(seed)

    profiles = []
    for c in wanted:
        idx = np.where(final == c)[0]
        p = {"id": int(c), "n_docs": int(len(idx)), "est_tok": 0.0,
             "tok_pct": 0.0, "doc_pct": 0.0, "mean_chars": 0.0,
             "med_chars": 0.0, "p90_chars": 0.0, "quality": {},
             "domains": [], "dom_docs_total": 0, "dom_tok_total": 0.0,
             "samples": []}
        profiles.append(p)
        if len(idx) == 0:
            continue
        chars_c = char[idx]
        toks_c = est_tok[idx]
        p["est_tok"] = float(toks_c.sum())
        p["tok_pct"] = (100.0 * p["est_tok"] / total_tok
                        if total_tok > 0 else 0.0)
        p["doc_pct"] = 100.0 * len(idx) / mm.num_docs
        p["mean_chars"] = float(chars_c.mean())
        p["med_chars"] = float(np.median(chars_c))
        p["p90_chars"] = float(np.percentile(chars_c, 90))
        qm = qual[idx].mean(axis=0)
        for i, name in enumerate(qcols[:qm.shape[0]]):
            p["quality"][name or f"q{i}"] = float(qm[i])

        # 域构成 (域 id 序, 全行含 0 — 完整版渲染保持历史格式)
        valid = dom[idx] >= 0
        if valid.any():
            dvals = np.unique(dom[idx][valid])
            n_dom = max(len(dom_names), int(dvals.max()) + 1)
            d_docs = np.bincount(dom[idx][valid], minlength=n_dom)
            d_toks = np.bincount(dom[idx][valid],
                                 weights=toks_c[valid], minlength=n_dom)
            p["dom_docs_total"] = int(d_docs.sum())
            p["dom_tok_total"] = float(d_toks.sum())
            for d in range(n_dom):
                name = dom_names[d] if d < len(dom_names) else f"D{d}"
                p["domains"].append((name, int(d_docs[d]), float(d_toks[d])))

        sample = rng.choice(idx, size=min(n_per, len(idx)), replace=False)
        sample.sort()
        texts = mm.read_texts(np.asarray(sample, dtype=np.int64), verbose=False)
        for gi, txt in zip(sample, texts):
            snippet = " ".join(str(txt).split())[:chars]
            p["samples"].append({"gi": int(gi), "chars": int(char[gi]),
                                 "snippet": snippet})
        print(f"[peek] C{c}: {len(idx):,} docs, {len(sample)} texts read")

    info = {"num_docs": int(mm.num_docs), "total_est_tok": total_tok,
            "K": K, "pool_dir": pool_dir}
    return info, profiles


def render_full(args, info, profiles):
    """CLI 完整版 markdown (与历史输出同构)。"""
    md = [f"# 簇语义抽样 — {os.path.basename(args.cluster_cache)}",
          "",
          f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"池: `{info['pool_dir']}` ({info['num_docs']:,} docs, "
          f"{info['total_est_tok'] / 1e9:.1f}B est-tok, char/4)   "
          f"K = {info['K']}   每簇抽样 {args.n} 篇 (seed {args.seed})",
          "",
          "口径: est-token = char/4 (与选样同启发式); 域构成交叉表同时回答 "
          "domainfix 的四域配额落在哪些簇。", ""]
    for p in profiles:
        head = f"## C{p['id']}"
        if p["n_docs"] == 0:
            md += [head, "", "(空簇 — 无文档)", ""]
            continue
        md += [head, "",
               f"- **{p['n_docs']:,} docs / {p['est_tok'] / 1e9:.2f}B est-tok** "
               f"(占池 {_fmt_pct(p['est_tok'], info['total_est_tok'])} token, "
               f"{_fmt_pct(p['n_docs'], info['num_docs'])} docs)",
               f"- 文档长度 (chars): 均值 {p['mean_chars']:.0f} / "
               f"中位 {p['med_chars']:.0f} / P90 {p['p90_chars']:.0f}"]
        if p["quality"]:
            md.append("- 质量列均值: " + ", ".join(
                f"{k} {v:.2f}" for k, v in p["quality"].items()))
        if p["domains"]:
            md += ["", "域构成 (本簇内):", "",
                   "| 域 | docs | doc% | est-tok% |", "|---|---|---|---|"]
            for name, dd, dt in p["domains"]:
                md.append(f"| {name} | {dd:,} | "
                          f"{_fmt_pct(dd, p['dom_docs_total'])} | "
                          f"{_fmt_pct(dt, p['dom_tok_total'])} |")
        md += ["", f"### 抽样 {len(p['samples'])} 篇 (首 {args.chars} 字符)", ""]
        for s in p["samples"]:
            md.append(f"- **[{s['gi']}]** ({s['chars']:,} ch) {s['snippet']}")
        md.append("")
    return "\n".join(md)


def render_brief(profiles):
    """紧凑版 (recipe_report §2b 内嵌): 每簇一行统计 + 一篇样本开头。
    报告内嵌的自助语义层 — 用户不再需要手工跑 CLI 才知道簇是什么。"""
    md = ["| 簇 | docs | 池 tok% | 域构成 (前2, doc%) | 中位 ch | 质量列 (前2) |",
          "|---|---|---|---|---|---|"]
    for p in profiles:
        if p["n_docs"] == 0:
            md.append(f"| C{p['id']} | 0 | — | — | — | — |")
            continue
        rows = sorted(((n, dd) for n, dd, _ in p["domains"] if dd > 0),
                      key=lambda kv: -kv[1])
        dom2 = (" / ".join(f"{n} {100.0 * dd / p['dom_docs_total']:.1f}%"
                           for n, dd in rows[:2]) if rows else "—")
        q2 = (", ".join(f"{k} {v:.2f}"
                        for k, v in list(p["quality"].items())[:2]) or "—")
        md.append(f"| C{p['id']} | {p['n_docs']:,} | {p['tok_pct']:.1f}% | "
                  f"{dom2} | {p['med_chars']:.0f} | {q2} |")
    md += ["", "每簇 1 篇抽样开头 (seed 固定):", ""]
    for p in profiles:
        if p["samples"]:
            s = p["samples"][0]
            md.append(f"- **C{p['id']}** [{s['gi']}] ({s['chars']:,} ch): "
                      f"{s['snippet']}")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(
        description="Cluster semantics peek: what IS C10")
    ap.add_argument("--pool-dir", required=True)
    ap.add_argument("--cluster-cache", required=True)
    ap.add_argument("--clusters", default="all",
                    help="逗号分隔 (C10/10 均可) 或 all (默认)")
    ap.add_argument("--n", type=int, default=40, help="每簇抽样文档数")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chars", type=int, default=300,
                    help="每篇抽样文档展示的字符数")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    info, profiles = collect(args.pool_dir, args.cluster_cache,
                             clusters=args.clusters, n_per=args.n,
                             chars=args.chars, seed=args.seed)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.cluster_cache)),
        "cluster_peek.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(render_full(args, info, profiles) + "\n")
    print(f"\n[peek] done in {time.time() - t0:.0f}s → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
