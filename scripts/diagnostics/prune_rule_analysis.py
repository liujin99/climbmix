"""Offline prune-rule analysis from prune_profile.json — no re-embedding, no re-clustering.

The production prune step uses per-cluster FLAT average quality (avg < t →
prune). The 2026-08-31 20-shard run showed the flat average lets through
escape clusters (e.g. C705: avg=3.03 passes t=3.0 while stem_relevance=1.70,
knowledge_value=1.15 — clean formatting washing out off-topic content).
This script answers, purely from the profile dump (seconds, any host):

  1. rule families side by side:
       mean t=X      current semantics (avg >= X keeps the cluster)
       floor f=X     every quality column >= X keeps the cluster
       mean + floor  survive BOTH (the refinement candidate)
     each row: clusters pruned / docs% / tokens% kept + docs-weighted and
     token-weighted remaining pool quality.
  2. escape list — clusters that PASS the current mean rule but FAIL a
     per-column floor: the C705-type leaks, ranked by docs.
  3. efficiency — quality gained per % of tokens removed, per rule point
     (the docs/tokens asymmetry: weak clusters are short docs, so token
     cost of quality is steeper than doc counts suggest).
  4. sanity — recomputes the stored threshold sweep from the matrix; a
     mismatch means schema drift, trust nothing below it.

Usage:
  python3 scripts/diagnostics/prune_rule_analysis.py /tmp/prune_subset_test/prune_profile.json
  python3 scripts/diagnostics/prune_rule_analysis.py p.json --floors 1.75,2.0,2.25 --means 3.0,3.5
  python3 scripts/diagnostics/prune_rule_analysis.py p.json --columns stem_relevance,knowledge_value
"""

import argparse
import json
import sys

import numpy as np

BOX = "─" * 74


def _fmt_vec(vec):
    return " ".join(f"{v:.2f}" for v in vec)


def _pct(x):
    return f"{100.0 * x:.1f}%"


def load_profile(path):
    try:
        with open(path) as f:
            profile = json.load(f)
    except OSError as e:
        sys.exit(f"cannot read profile: {e}")
    if not profile.get("quality_available") or "cluster_quality_matrix" not in profile:
        sys.exit("profile has no cluster_quality_matrix (quality unavailable or old "
                 "format) — re-run prune_report to regenerate")
    return profile


def validate_against_stored_sweep(avg, sizes, profile):
    """Recompute the stored threshold sweep from the matrix; guards schema drift."""
    n_docs = sizes.sum()
    bad = 0
    for row in profile.get("threshold_sweep", []):
        t = row["threshold"]
        kept = avg >= t
        dp = float(sizes[kept].sum()) / n_docs * 100
        # docs-share is the load-bearing check; a borderline cluster can flip
        # the exact count after the matrix's 4-decimal rounding
        if abs(dp - row["docs_kept_pct"]) > 0.11:
            bad += 1
    return bad


def rule_rows(avg, sizes, tokens, name, mask, n_docs, n_tokens):
    kept_docs = float(sizes[mask].sum())
    kept_tok = float(tokens[mask].sum()) if tokens is not None and n_tokens > 0 else None
    doc_q = float((avg * sizes)[mask].sum()) / kept_docs if kept_docs > 0 else float("nan")
    tok_q = (float((avg * tokens)[mask].sum()) / kept_tok
             if tokens is not None and n_tokens > 0 and kept_tok > 0 else None)
    return {
        "name": name,
        "pruned": int((~mask).sum()),
        "k": int(mask.size),
        "docs_kept": kept_docs / n_docs,
        "tokens_kept": (kept_tok / n_tokens) if kept_tok is not None else None,
        "doc_quality": doc_q,
        "token_quality": tok_q,
    }


def print_rule_table(rows, baseline):
    print(BOX)
    print("  Rule comparison  (kept = pass rule; quality = docs/token-weighted avg)")
    print(BOX)
    hdr = (f"    {'rule':<28} {'pruned':>9} {'docs':>7} {'tokens':>8}"
           f" {'doc-q':>7} {'tok-q':>7}")
    print(hdr)
    for r in rows:
        tok = f"{_pct(r['tokens_kept'])}" if r["tokens_kept"] is not None else "-"
        tok_q = f"{r['token_quality']:.3f}" if r["token_quality"] is not None else "-"
        cur = "  ← current" if r["name"] == baseline else ""
        print(f"    {r['name']:<28} {r['pruned']:>4}/{r['k']:<4} {_pct(r['docs_kept']):>7}"
              f" {tok:>8} {r['doc_quality']:>7.3f} {tok_q:>7}{cur}")
    print(BOX)


def print_efficiency(rows, baseline_row):
    """Δquality per % tokens removed for mean-based rules (drop-in replacements
    for the current rule); floor-only rows are reference only — they keep
    low-average clusters the mean rule kills, so Δq is apples-to-oranges."""
    print("  Efficiency vs current rule  (Δremaining quality / Δtokens removed)")
    base_q = baseline_row["doc_quality"]
    base_t = baseline_row["tokens_kept"] if baseline_row["tokens_kept"] is not None \
        else baseline_row["docs_kept"]
    for r in rows:
        if r["name"] == baseline_row["name"] or "mean" not in r["name"]:
            continue
        tok = r["tokens_kept"] if r["tokens_kept"] is not None else r["docs_kept"]
        removed_pct = (base_t - tok) * 100
        dq = r["doc_quality"] - base_q
        if removed_pct <= 0.05:
            eff = "-"
        else:
            eff = f"{dq / removed_pct:+.4f}/%"
        print(f"    {r['name']:<28} Δq={dq:+.4f}  Δtokens=-{removed_pct:.1f}%  "
              f"eff={eff}")
    print(BOX)


def print_escapes(matrix, avg, sizes, cols, ids, survive, fail_floor, f_esc, top_n):
    esc = np.nonzero(survive & fail_floor)[0]
    # argsort positions are WITHIN esc — reindex to original rows
    order = esc[np.argsort(-sizes[esc])[:top_n]]
    n_docs = sizes.sum()
    esc_docs = sizes[esc].sum()
    print(f"  Escape list  (pass mean rule, but some column < {f_esc:.2f})")
    print(f"    {len(esc)} clusters, {_pct(esc_docs / n_docs)} of docs — the flat "
          f"average's blind spot")
    if len(esc) == 0:
        print("    (none — mean rule is not leaking)")
    else:
        for i in order:
            vec = matrix[i]
            weak = int(np.argmin(vec))
            print(f"      C{ids[i]:<5} docs={int(sizes[i]):>7,} avg={avg[i]:.2f} "
                  f"[{_fmt_vec(vec)}]  ← {cols[weak]}={vec[weak]:.2f}")
    print(BOX)
    return len(esc), esc_docs / n_docs


def main():
    ap = argparse.ArgumentParser(description="Offline prune-rule analysis")
    ap.add_argument("profile", help="path to prune_profile.json")
    ap.add_argument("--floors", default="1.5,2.0,2.5",
                    help="per-column floor candidates (comma-separated)")
    ap.add_argument("--means", default="",
                    help="extra mean-threshold candidates beyond the profile one")
    ap.add_argument("--columns", default="",
                    help="columns the floor rule applies to (default: all quality columns)")
    ap.add_argument("--escape-floor", type=float, default=2.0,
                    help="floor for the escape list (default 2.0)")
    ap.add_argument("--escapes", type=int, default=15, help="escape list rows")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    cols = list(profile["quality_columns"])
    matrix = np.array(profile["cluster_quality_matrix"], dtype=np.float64)
    sizes = np.array(profile["cluster_sizes"], dtype=np.float64)
    ids = list(profile["cluster_ids"])
    tokens = (np.array(profile["cluster_tokens"], dtype=np.float64)
              if profile.get("cluster_tokens") else None)
    avg = matrix.mean(axis=1)
    t0 = float(profile["prune_threshold"])
    n_docs = sizes.sum()
    n_tokens = float(tokens.sum()) if tokens is not None else 0.0

    if args.columns:
        core = [c for c in args.columns.split(",") if c in cols]
        missing = [c for c in args.columns.split(",") if c not in cols]
        if missing:
            sys.exit(f"unknown columns {missing}; profile has {cols}")
    else:
        core = cols
    core_idx = [cols.index(c) for c in core]
    core_min = matrix[:, core_idx].min(axis=1)

    floors = [float(x) for x in args.floors.split(",") if x.strip()]
    means = [float(x) for x in args.means.split(",") if x.strip()]

    print(f"\n  profile: {args.profile}")
    print(f"  {int(matrix.shape[0])} clusters × {len(cols)} columns, "
          f"{int(n_docs):,} docs"
          + (f", {n_tokens:,.0f} tokens" if n_tokens > 0 else " (no token counts)")
          + f"; current rule: mean t={t0}")

    bad = validate_against_stored_sweep(avg, sizes, profile)
    if bad:
        sys.exit(f"schema drift: {bad} stored sweep rows disagree with the matrix — "
                 "profile too old for this script, regenerate it")
    print("  sanity: stored threshold sweep reproduced from matrix ✓")

    # ── Rule families ──
    rows = []
    survive_mean = avg >= t0
    rows.append(rule_rows(avg, sizes, tokens,
                          f"mean t={t0:.2f}", survive_mean, n_docs, n_tokens))
    for t in means:
        if abs(t - t0) < 1e-9:
            continue
        rows.append(rule_rows(avg, sizes, tokens,
                              f"mean t={t:.2f}", avg >= t, n_docs, n_tokens))
    for f in floors:
        rows.append(rule_rows(avg, sizes, tokens,
                              f"floor f={f:.2f} (any of {len(core)})",
                              core_min >= f, n_docs, n_tokens))
    for f in floors:
        rows.append(rule_rows(avg, sizes, tokens,
                              f"mean t={t0:.2f} + floor f={f:.2f}",
                              survive_mean & (core_min >= f), n_docs, n_tokens))

    baseline = f"mean t={t0:.2f}"
    print_rule_table(rows, baseline)
    print_efficiency(rows, rows[0])

    # ── Escape list ──
    n_esc, esc_share = print_escapes(
        matrix, avg, sizes, cols, ids, survive_mean,
        core_min < args.escape_floor, args.escape_floor, args.escapes)

    # ── Advice (same ✓/·/! convention as production diagnostics) ──
    print("  diagnosis:")
    hybrid = [r for r in rows if "+" in r["name"]]
    if n_esc == 0:
        print("    ✓ mean rule leaks nothing at this floor — no refinement needed")
    else:
        best = max(hybrid, key=lambda r: r["doc_quality"]) if hybrid else None
        print(f"· mean rule leaks {n_esc} clusters ({_pct(esc_share)} docs) with a "
              f"column < {args.escape_floor:.2f}")
        if best is not None:
            tok = _pct(best["tokens_kept"]) if best["tokens_kept"] is not None else "-"
            print(f"· tightest hybrid {best['name']}: prunes {best['pruned']} clusters, "
                  f"keeps {tok} tokens, doc-q {best['doc_quality']:.3f} "
                  f"(vs {rows[0]['doc_quality']:.3f})")
            print("  → adopting it = editing the floor as a second knob in "
                  "prune_clusters, profile stays the evidence")
    for r in rows:
        if r["name"].startswith("floor") and r["tokens_kept"] is not None \
                and r["tokens_kept"] < 0.80:
            print(f"! {r['name']} alone drops {_pct(1 - r['tokens_kept'])} of tokens — "
                  f"floor-only is too blunt here; use the hybrid")
            break
    print(BOX)


if __name__ == "__main__":
    main()
