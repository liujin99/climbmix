#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  prod2_watch.sh — prod2 生产 run 的只读仪表盘 (CP0-CP4 检查点)
#
#  用法:
#    bash scripts/diagnostics/prod2_watch.sh [result_dir]
#    watch -n 600 'bash scripts/diagnostics/prod2_watch.sh result/prod2_k15bal_current'
#
#  只读、无副作用 (printf + grep + python -c 只读 JSON)。每节独立容错:
#  缺文件显示 "(pending)" 而非报错 — 搜索期、臂期、完成后三个阶段都能挂。
#
#  检查点 (与 prod2 计划对齐, 详见 .opencode/plans/prod2-bplus-*.md):
#    CP0  聚类结构: balanced_profile K_final=K_ENHANCED, overflow=0,
#         max share ≤ (1+slack)/K (~7.7%)
#    CP1  搜索健康 (h≈6.5): 每任务 f>0 的 SNR 行, mmlu_stem acc-only 行,
#         exp 在跑且成功率正常
#    CP2  预测器 (h≈9.5): online rho ≥ 0.4, val R² > 0
#    CP3  选料 (h≈12.5): "Selection mode:" 行已打印 (NO-SIGNAL GUARD 未触发
#         时为 predictor_design_space)
#    CP4  交付 (h≈23.5): 两臂 eval CSV + (可选) 远端 base 锚点对比
#  熔断建议: 红灯项集中出现时 pkill -f run_climbmix.sh + 杀 dispatch,
#  重跑同命令恢复 (步骤级 .done / 迭代级 search_state / 实验级 meta.json)。
# ═══════════════════════════════════════════════════════════════════════
set -uo pipefail

DIR="${1:-result/${EXP_NAME:-main}_current}"
PY=python3

mark() { # mark <ok:0|1> <text>
    if [ "$1" = "0" ]; then printf "  [✓] %s\n" "$2"
    else printf "  [·] %s\n" "$2"; fi
}

echo "════════════════════════════════════════════════════════════"
echo "  prod2 watch — $DIR  ($(date '+%F %T'))"
echo "════════════════════════════════════════════════════════════"

# ── 1. 阶段标记 ─────────────────────────────────────────────────────────
echo "── Stage markers ──"
for m in .done_mid_train_random .done_eval_random .done_mid_train_climb \
         .done_eval_climb; do
    if [ -f "$DIR/$m" ]; then printf "  [✓] %s\n" "$m"
    else printf "  [·] %s (pending)\n" "$m"; fi
done
[ -f "$DIR/sampled_dataset.parquet" ] && mark 0 "search complete (sampled_dataset.parquet)" \
    || mark 1 "search running / not started"

# search_state.json 摘要 (机读)
if [ -f "$DIR/search_state.json" ]; then
    $PY - "$DIR/search_state.json" <<'PYEOF'
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  [!] search_state.json unreadable: {e}"); raise SystemExit
acc = s.get("accumulated_configs") or []
pend = s.get("pending") or {}
online = s.get("online_eval") or []
print(f"  search_state: iter {s.get('last_completed_iter')}, "
      f"{len(acc)} accumulated configs"
      + (f", pending iter {pend.get('iteration')} "
         f"({len(pend.get('configs') or [])} configs)" if pend else ""))
if online:
    rho = [o.get("spearman") for o in online if o.get("spearman") is not None]
    if rho:
        print(f"  online backtest rho history: "
              + " ".join(f"{r:.3f}" for r in rho))
# 原始分 spread vs SE≈0.006: 各 config 的 per-task acc 均值的极差
# (远小于 SE = 纯噪声, 搜索无信号 — prod1 教训)
try:
    pb = [d.get("acc") or {} for d in s.get("accumulated_per_benchmark") or []]
    means = []
    for d in pb:
        vals = [v for v in d.values() if isinstance(v, (int, float))]
        if vals: means.append(sum(vals) / len(vals))
    if len(means) >= 4:
        spread = max(means) - min(means)
        verdict = "OK (signal >> noise)" if spread > 0.012 else \
            "THIN (≤ 2×SE=0.006 — watch CP2)"
        print(f"  raw-score spread (max-min config mean acc): "
              f"{spread:.4f} — {verdict}")
    # 权重分散度: 所有 config 的最大单簇权重均值 (prod1 病灶: 集中单簇)
    if acc:
        mx = [max(c["weights"]) for c in acc if c.get("weights")]
        if mx:
            print(f"  weight dispersion: mean max-cluster weight "
                  f"{sum(mx)/len(mx):.3f} over {len(mx)} configs")
except Exception:
    pass
PYEOF
else
    mark 1 "search_state.json (search not started or crashed before iter 1)"
fi

# ── 2. CP0: 聚类结构 ────────────────────────────────────────────────────
echo "── CP0: balanced cluster structure ──"
if [ -f "$DIR/balanced_profile.json" ]; then
    $PY - "$DIR/balanced_profile.json" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
k, kf = p.get("K_target"), p.get("K_final")
ovf, ms = p.get("overflow_assignments"), p.get("max_share", 0.0)
cap = (1.0 + p.get("slack", 0.15)) / max(1, k or 1)
ok_k = (k == kf)
ok_o = (ovf == 0)
ok_s = (ms <= cap + 1e-9)
for ok, txt in ((ok_k, f"K_final={kf} == K_target={k}"),
                (ok_o, f"overflow_assignments={ovf} == 0"),
                (ok_s, f"max share {ms:.1%} <= cap {cap:.1%}")):
    print(f"  [{'✓' if ok else '✗'}] {txt}")
print(f"  fine->anchor cosine mean="
      f"{(p.get('fine_anchor_cosine') or {}).get('mean')}")
print("  CP0 " + ("GREEN" if (ok_k and ok_o and ok_s) else "RED — investigate before h6"))
PYEOF
else
    mark 1 "balanced_profile.json (clustering not finished — CP0 pending)"
fi

# ── 3. CP1: 搜索健康 ────────────────────────────────────────────────────
echo "── CP1: search health ──"
N_EXP=$(ls "$DIR"/exp_*/meta.json 2>/dev/null | wc -l)
N_ALL=$(ls -d "$DIR"/exp_* 2>/dev/null | wc -l)
echo "  exps: $N_EXP completed / $N_ALL started"
if [ -f "$DIR/search.log" ]; then
    echo "  last SNR lines (per-task f, w):"
    grep -E '^\s+\w+: w=' "$DIR/search.log" 2>/dev/null | tail -6 | sed 's/^/    /'
    echo "  mmlu_stem acc-only line:"
    grep -m1 "acc-only" "$DIR/search.log" 2>/dev/null | sed 's/^/    /' \
        || echo "    (not yet — appears after the first iteration's scoring)"
    echo "  adaptive lines (last 3):"
    grep -E "\[Adaptive\]|\] adaptive:" "$DIR/search.log" 2>/dev/null | tail -3 | sed 's/^/    /' \
        || true
    echo "  last iteration line:"
    grep -E "^\[Iter [0-9]+\] (Complete|adaptive:)" "$DIR/search.log" 2>/dev/null | tail -3 | sed 's/^/    /'
    echo "  predictor quality:"
    grep -E "Online backtest|Predictor val R" "$DIR/search.log" 2>/dev/null | tail -4 | sed 's/^/    /'
else
    mark 1 "search.log (search not started)"
fi

# ── 4. CP2: 预测器 (由上面 online rho 历史判断) ─────────────────────────
if [ -f "$DIR/search_state.json" ] && [ -f "$DIR/search.log" ]; then
    RHO=$(grep -oE "Online backtest: Spearman rho=[-0-9.]+" "$DIR/search.log" 2>/dev/null | tail -1 | grep -oE "[-0-9.]+$" || true)
    if [ -n "$RHO" ]; then
        ok=$(awk -v r="$RHO" 'BEGIN{print (r>=0.4)?0:1}')
        mark "$ok" "CP2: latest online rho=$RHO (target ≥ 0.4)"
    else
        mark 1 "CP2: online rho not measured yet (needs a guided iteration)"
    fi
fi

# ── 5. CP3: 选料 ────────────────────────────────────────────────────────
echo "── CP3: selection ──"
if [ -f "$DIR/search.log" ] && grep -q "Selection mode:" "$DIR/search.log" 2>/dev/null; then
    grep -A2 "Selection mode:" "$DIR/search.log" | tail -3 | sed 's/^/    /'
    mark 0 "CP3: selection done"
else
    mark 1 "CP3: Selection mode line not printed yet (search still running)"
fi

# ── 6. CP4: 两臂 + base 锚点 ────────────────────────────────────────────
echo "── CP4: target arms ──"
for a in random climb; do
    if [ -f "$DIR/target_arm_$a.json" ]; then
        $PY - "$DIR/target_arm_$a.json" <<PYEOF
import json
d = json.load(open("$DIR/target_arm_$a.json"))
print(f"  [$a] job {d.get('job_id')} {d.get('status')} "
      f"ok={d.get('ok')} elapsed={round((d.get('elapsed_seconds') or 0)/60)}m"
      + (" SALVAGED(train-only)" if d.get('salvaged_train_only') else ""))
PYEOF
    else
        mark 1 "target_arm_$a.json (arm not dispatched yet)"
    fi
    for f in "mid_train_$a.log" "eval_$a.log"; do
        if [ -f "$DIR/$f" ]; then
            printf "  %s tail: " "$f"
            tail -1 "$DIR/$f" 2>/dev/null | cut -c1-120
        fi
    done
done
if [ -f "$DIR/eval_base_remote.csv" ]; then
    mark 0 "remote base anchor: eval_base_remote.csv (compare STEM row vs local 0.1738)"
else
    mark 1 "remote base anchor (optional: dispatch_target_arm.py --arm base_eval_check)"
fi

# ── 7. 汇总 ─────────────────────────────────────────────────────────────
echo "── quick verdict ──"
if [ -f "$DIR/.done_eval_climb" ] && [ -f "$DIR/.done_eval_random" ]; then
    echo "  RUN COMPLETE — see $DIR/report.md"
elif ls "$DIR"/exp_*/meta.json >/dev/null 2>&1; then
    echo "  search in progress — CP1 checks above should all turn ✓ by h≈6.5"
elif [ -f "$DIR/sampled_dataset.parquet" ]; then
    echo "  arms in progress — CP4 checks above"
else
    echo "  early stage — clustering/search starting"
fi
echo "════════════════════════════════════════════════════════════"
