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
    print(f"[peek] loading pool metadata: {args.pool_dir}")
    mm = ShardMetadataManager(args.pool_dir)
    final = np.load(args.cluster_cache,
                    allow_pickle=False)["final_labels"].astype(np.int64)
    if len(final) != mm.num_docs:
        raise SystemExit(
            f"ERROR: cluster cache has {len(final):,} labels but pool has "
            f"{mm.num_docs:,} docs — pool/cache mismatch")
    K = int(final.max()) + 1
    dom = mm.cluster_labels.astype(np.int64)
    char = mm.doc_char_counts.astype(np.float64)
    est_tok = char / 4.0  # CHAR_TO_TOKEN_EST, 同 selection 口径
    dom_names = list(getattr(mm._schema, "domain_names", None) or [])
    qcols = list(getattr(mm._schema, "quality_cols", None) or [])
    qual = np.asarray(mm.quality_scores, dtype=np.float64)
    if qual.ndim == 1:
        qual = qual[:, None]

    if args.clusters.strip().lower() == "all":
        wanted = list(range(K))
    else:
        wanted = []
        for tok in args.clusters.split(","):
            tok = tok.strip()
            m = re.match(r"^[Cc](\d+)$", tok)
            c = int(m.group(1)) if m else int(tok)
            if not (0 <= c < K):
                raise SystemExit(f"ERROR: cluster {tok} 不在 [0, {K})")
            wanted.append(c)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.cluster_cache)),
        "cluster_peek.md")

    rng = np.random.default_rng(args.seed)
    md = [f"# 簇语义抽样 — {os.path.basename(args.cluster_cache)}",
          "",
          f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}   "
          f"池: `{args.pool_dir}` ({mm.num_docs:,} docs, "
          f"{est_tok.sum() / 1e9:.1f}B est-tok, char/4)   "
          f"K = {K}   每簇抽样 {args.n} 篇 (seed {args.seed})",
          "",
          "口径: est-token = char/4 (与选样同启发式); 域构成交叉表同时回答 "
          "domainfix 的四域配额落在哪些簇。", ""]

    for c in wanted:
        idx = np.where(final == c)[0]
        head = f"## C{c}"
        if len(idx) == 0:
            md += [head, "", "(空簇 — 无文档)", ""]
            continue
        chars_c = char[idx]
        toks_c = est_tok[idx]
        med = float(np.median(chars_c))
        p90 = float(np.percentile(chars_c, 90))

        md += [head, "",
               f"- **{len(idx):,} docs / {toks_c.sum() / 1e9:.2f}B est-tok** "
               f"(占池 {_fmt_pct(toks_c.sum(), est_tok.sum())} token, "
               f"{_fmt_pct(len(idx), mm.num_docs)} docs)",
               f"- 文档长度 (chars): 均值 {chars_c.mean():.0f} / 中位 {med:.0f} "
               f"/ P90 {p90:.0f}"]
        if qcols:
            qm = qual[idx].mean(axis=0)
            md.append("- 质量列均值: " + ", ".join(
                f"{name or f'q{i}'} {qm[i]:.2f}"
                for i, name in enumerate(qcols[:qm.shape[0]])))

        # 域构成交叉表
        valid = dom[idx] >= 0
        if valid.any():
            dvals = np.unique(dom[idx][valid])
            n_dom = max(len(dom_names), int(dvals.max()) + 1)
            d_docs = np.bincount(dom[idx][valid], minlength=n_dom)
            d_toks = np.bincount(dom[idx][valid],
                                 weights=toks_c[valid], minlength=n_dom)
            md += ["", "域构成 (本簇内):", "",
                   "| 域 | docs | doc% | est-tok% |", "|---|---|---|---|"]
            for d in range(n_dom):
                name = dom_names[d] if d < len(dom_names) else f"D{d}"
                md.append(f"| {name} | {d_docs[d]:,} | "
                          f"{_fmt_pct(d_docs[d], d_docs.sum())} | "
                          f"{_fmt_pct(d_toks[d], d_toks.sum())} |")

        sample = rng.choice(idx, size=min(args.n, len(idx)), replace=False)
        sample.sort()
        texts = mm.read_texts(np.asarray(sample, dtype=np.int64), verbose=False)
        md += ["", f"### 抽样 {len(sample)} 篇 (首 {args.chars} 字符)", ""]
        for gi, txt in zip(sample, texts):
            snippet = " ".join(str(txt).split())[:args.chars]
            md.append(f"- **[{gi}]** ({int(char[gi]):,} ch) {snippet}")
        md.append("")
        print(f"[peek] C{c}: {len(idx):,} docs, "
              f"{len(sample)} texts read")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n[peek] done in {time.time() - t0:.0f}s → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
