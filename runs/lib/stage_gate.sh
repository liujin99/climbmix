# ═══════════════════════════════════════════════════════════════════════
#  Stage-scoped fingerprint gate — SOURCED by runs/*.sh (not executed).
#
#  Caller contract (before calling run_stage_gate):
#    CLIMBMIX_DIR, OUTPUT_DIR, EXP_NAME              — paths (set by runner)
#    FP_SEARCH_PARAMS / FP_TARGET_PARAMS             — arrays of "key=value"
#                                                      semantic params
#    MIGRATE_LEGACY_FINGERPRINT (optional env, =1)   — one-time adoption of a
#                                                      legacy single-fingerprint
#                                                      dir (see below)
#
#  Behavior:
#    - .fingerprint_search + .fingerprint_target are (re)written in $OUTPUT_DIR
#    - search mismatch  -> WHOLE dir archived to result/${EXP_NAME}_stale_<ts>
#      (everything downstream depends on the search: weights, sampled_dataset)
#    - target mismatch  -> ONLY target products archived to
#      result/${EXP_NAME}_target_stale_<ts>; search products
#      (search_state.json, exp_*/, sampled_dataset.parquet, ...) are kept and
#      Steps 4-8 rerun from sampled_dataset.parquet
#    - legacy dir holding only the old single .fingerprint:
#        * MIGRATE_LEGACY_FINGERPRINT=1 -> adopted UNVERIFIED (you assert that
#          nothing semantic changed besides the fingerprint mechanism itself;
#          the per-step .done markers then decide what actually reruns)
#        * otherwise -> archived with a hint about the migration variable
#
#  num_npu is deliberately NOT part of any fingerprint: it changes only the
#  parallel shape (local pool size, future remote executors), never
#  per-experiment semantics — each proxy experiment is ws=1 with seed-driven
#  data order; NUM_NPU additionally affects only shard row-group layout, not
#  content or scores. This lets a campaign shrink/grow its NPU pool (e.g. a
#  remote node disappearing) without resetting the whole run.
# ═══════════════════════════════════════════════════════════════════════

run_stage_gate() {
    local args_search=() args_target=() kv fp_search fp_target stale item
    for kv in "${FP_SEARCH_PARAMS[@]}"; do args_search+=(--param "$kv"); done
    for kv in "${FP_TARGET_PARAMS[@]}"; do args_target+=(--param "$kv"); done

    fp_search=$(python3 -m climbmix.utils.fingerprint --base-dir "$CLIMBMIX_DIR" \
        --stage search "${args_search[@]}")
    fp_target=$(python3 -m climbmix.utils.fingerprint --base-dir "$CLIMBMIX_DIR" \
        --stage target "${args_target[@]}")

    mkdir -p "$CLIMBMIX_DIR/result"

    if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
        if [ -f "$OUTPUT_DIR/.fingerprint_search" ] && [ -f "$OUTPUT_DIR/.fingerprint_target" ]; then
            if [ "$(cat "$OUTPUT_DIR/.fingerprint_search")" != "$fp_search" ]; then
                stale="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_$(date +%Y%m%d_%H%M%S)"
                echo "  SEARCH fingerprint changed (code or params) — archiving everything:"
                echo "    $OUTPUT_DIR -> $stale"
                mv "$OUTPUT_DIR" "$stale"
            elif [ "$(cat "$OUTPUT_DIR/.fingerprint_target")" != "$fp_target" ]; then
                stale="$CLIMBMIX_DIR/result/${EXP_NAME}_target_stale_$(date +%Y%m%d_%H%M%S)"
                echo "  TARGET fingerprint changed — archiving target products only"
                echo "  (search products kept: search_state.json, exp_*/, sampled_dataset.parquet):"
                echo "    -> $stale"
                mkdir -p "$stale"
                for item in climb_shards random_shards climb_mixed random_mixed \
                            mid_train_climb.log mid_train_random.log \
                            eval_climb.log eval_random.log \
                            .done_mid_train_climb .done_mid_train_random \
                            .done_eval_climb .done_eval_random; do
                    if [ -e "$OUTPUT_DIR/$item" ]; then mv "$OUTPUT_DIR/$item" "$stale/"; fi
                done
            else
                echo "  RESUME: $OUTPUT_DIR (search+target fingerprints match)"
            fi
        elif [ -f "$OUTPUT_DIR/.fingerprint" ]; then
            if [ "${MIGRATE_LEGACY_FINGERPRINT:-0}" = "1" ]; then
                echo "  ⚠ MIGRATE_LEGACY_FINGERPRINT=1: adopting legacy single-fingerprint dir"
                echo "    UNVERIFIED — you assert nothing semantic changed besides the"
                echo "    fingerprint mechanism itself. Per-step .done markers decide"
                echo "    what reruns."
            else
                stale="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_$(date +%Y%m%d_%H%M%S)"
                echo "  Legacy single-fingerprint dir found (pre stage-scoping)."
                echo "  To adopt it unverified instead:"
                echo "    MIGRATE_LEGACY_FINGERPRINT=1 bash runs/$(basename "$0") ..."
                echo "  Archiving:"
                echo "    $OUTPUT_DIR -> $stale"
                mv "$OUTPUT_DIR" "$stale"
            fi
        else
            stale="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_$(date +%Y%m%d_%H%M%S)"
            echo "  Non-empty output dir without fingerprints — archiving:"
            echo "    $OUTPUT_DIR -> $stale"
            mv "$OUTPUT_DIR" "$stale"
        fi
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "$fp_search" > "$OUTPUT_DIR/.fingerprint_search"
    echo "$fp_target" > "$OUTPUT_DIR/.fingerprint_target"
    rm -f "$OUTPUT_DIR/.fingerprint"
    echo "  Fingerprints: search=${fp_search} target=${fp_target}"
}
