# ═══════════════════════════════════════════════════════════════════════
#  Auto CP4 tail report — SOURCED by runs/*.sh (not executed).
#
#  auto_cp4_report <run_dir>
#    臂训完后的自动全景对比报告 (cp4_report.py): 自动发现 run 内所有
#    eval_<arm>.csv, 有几个臂比几个 (对照臂 --ref 默认 random)。
#    一个臂 CSV 都没有 → 明确跳过不报错 (本臂 eval 失败由 dispatch
#    的退出码负责, 走不到这里)。
#    BASE_EXPECTED / SE 可 env 覆盖 (cp4_report.py 默认值)。
# ═══════════════════════════════════════════════════════════════════════
auto_cp4_report() {
    local run_dir="$1"
    local n_arms
    n_arms="$(find "$run_dir" -maxdepth 1 -name 'eval_*.csv' \
        ! -name 'eval_base_remote.csv' 2>/dev/null | wc -l)"
    if [ "$n_arms" -eq 0 ]; then
        echo "(跳过自动对比报告: ${run_dir} 下无 eval_<arm>.csv — 臂还没落地?)"
        return 0
    fi
    echo
    echo "═══ CP4 全景对比 (${n_arms} 臂) ═══"
    python3 "${CLIMBMIX_DIR:-$(pwd)}/scripts/diagnostics/cp4_report.py" "$run_dir" \
        --base-expected "${BASE_EXPECTED:-0.1738}" --se "${SE:-0.006}"
}
