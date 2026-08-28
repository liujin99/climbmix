# ═══════════════════════════════════════════════════════════════════════
#  Stage-scoped fingerprint gate + result-dir lifecycle — SOURCED by
#  runs/*.sh (not executed).
#
#  Caller contract (before calling run_stage_gate):
#    CLIMBMIX_DIR, OUTPUT_DIR, EXP_NAME              — paths (set by runner)
#    FP_SEARCH_PARAMS / FP_TARGET_PARAMS             — arrays of "key=value"
#                                                      semantic params
#    COMPLETION_MARKERS (optional array)             — .done files that count
#                                                      as "run finished"
#                                                      (default: .done_eval_climb)
#    MIGRATE_LEGACY_FINGERPRINT (optional env, =1)   — one-time adoption of a
#      legacy single-fingerprint dir (see below)
#
#  Result-dir lifecycle (default OUTPUT_DIR only, i.e.
#  result/${EXP_NAME}_current — custom OUTPUT_DIR keeps plain gate behaviour):
#
#    result/${EXP_NAME}_current/     ACTIVE: running / resumable / reusable
#    result/${EXP_NAME}_<ts>/        COMPLETED: terminal .done markers all
#                                    present; renamed by mark_completed at the
#                                    end of a green run, or at archive time
#    result/${EXP_NAME}_stale_<scope>_<ts>/
#                                    ABANDONED mid-run (scope = what changed):
#                                    search (whole dir, search fingerprint),
#                                    target (target products only), legacy
#                                    (old single-fingerprint format), orphan
#                                    (non-empty dir without fingerprints)
#
#  Every archive gets an archive_meta.json (reason, timestamps, old->new
#  fingerprints, git HEAD, was_complete, moved items) — the dir name alone no
#  longer has to carry the whole story.
#
#  Re-running a completed experiment is idempotent: if _current is missing,
#  the newest completed ${EXP_NAME}_<ts> whose SEARCH fingerprint matches is
#  restored as _current (all .done markers skip, zero NPU work; a target
#  fingerprint mismatch then re-runs only Steps 4-8).
#
#  Behavior:
#    - .fingerprint_search + .fingerprint_target are (re)written in $OUTPUT_DIR
#    - search mismatch  -> WHOLE dir archived (everything downstream depends
#      on the search: weights, sampled_dataset)
#    - target mismatch  -> ONLY target products archived; search products
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

_write_archive_meta() {
    # usage: _write_archive_meta <dir> key=value [key=value ...]
    # Writes archive_meta.json (atomic) into <dir>; fills git_head +
    # archived_at automatically. Values must not contain newlines.
    local dir="$1"; shift
    CLIMBMIX_DIR="$CLIMBMIX_DIR" python3 - "$dir" "$@" <<'PYEOF'
import json, os, subprocess, sys
from datetime import datetime

d = sys.argv[1]
kv = {}
for a in sys.argv[2:]:
    k, _, v = a.partition("=")
    kv[k] = v
try:
    head = subprocess.run(
        ["git", "-C", os.environ.get("CLIMBMIX_DIR", "."), "rev-parse",
         "--short", "HEAD"],
        capture_output=True, text=True, timeout=10).stdout.strip()
    kv.setdefault("git_head", head or "unknown")
except Exception:
    kv.setdefault("git_head", "unknown")
kv["archived_at"] = datetime.now().isoformat(timespec="seconds")
os.makedirs(d, exist_ok=True)
tmp = os.path.join(d, "archive_meta.json.tmp")
with open(tmp, "w") as f:
    json.dump(kv, f, indent=2, sort_keys=True)
os.replace(tmp, os.path.join(d, "archive_meta.json"))
PYEOF
}

_is_complete() {
    # All completion markers present inside $1? (COMPLETION_MARKERS array,
    # default .done_eval_climb — the common terminal marker of both shells)
    local dir="$1" m
    for m in "${COMPLETION_MARKERS[@]:-.done_eval_climb}"; do
        [ -n "$m" ] || continue
        [ -e "$dir/$m" ] || return 1
    done
    return 0
}

_archive_name() {
    # usage: _archive_name <scope> <was_complete:true|false>
    # -> completed runs are archived as ${EXP_NAME}_<ts>; abandoned ones as
    #    ${EXP_NAME}_stale_<scope>_<ts>. Collisions get a numeric suffix.
    local scope="$1" complete="$2" base n i=2
    if [ "$complete" = "true" ]; then
        base="$CLIMBMIX_DIR/result/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)"
    else
        base="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_${scope}_$(date +%Y%m%d_%H%M%S)"
    fi
    n="$base"
    while [ -e "$n" ]; do n="${base}_${i}"; i=$((i + 1)); done
    printf '%s' "$n"
}

_migrate_old_naming() {
    # One-time, idempotent migration to the current lifecycle naming:
    #   ${EXP_NAME}                 -> ${EXP_NAME}_current
    #   ${EXP_NAME}_target_stale_T  -> ${EXP_NAME}_stale_target_T
    #   ${EXP_NAME}_stale_T (old, timestamp-only) -> classified by content:
    #       both fingerprints -> _stale_search_T / legacy single -> _stale_legacy_T
    #       / none -> _stale_orphan_T
    # New-format names (scope segment instead of leading digits) don't match
    # the patterns and are left alone.
    local d base ts scope new
    local legacy_active="$CLIMBMIX_DIR/result/$EXP_NAME"

    if [ -d "$legacy_active" ] && [ ! -d "$OUTPUT_DIR" ]; then
        mv "$legacy_active" "$OUTPUT_DIR"
        echo "  (lifecycle migration: $EXP_NAME -> ${EXP_NAME}_current)"
    elif [ -d "$legacy_active" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "  ⚠ both $EXP_NAME and ${EXP_NAME}_current exist — leaving $EXP_NAME"
        echo "    in place; inspect/move it manually"
    fi

    for d in "$CLIMBMIX_DIR/result/${EXP_NAME}_target_stale_"*; do
        [ -d "$d" ] || continue
        base="${d##*/}"
        ts="${base#*${EXP_NAME}_target_stale_}"
        new="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_target_${ts}"
        if [ ! -e "$new" ]; then
            mv "$d" "$new"
            echo "  (lifecycle migration: $base -> ${EXP_NAME}_stale_target_${ts})"
        fi
    done

    for d in "$CLIMBMIX_DIR/result/${EXP_NAME}_stale_"*; do
        [ -d "$d" ] || continue
        base="${d##*/}"
        ts="${base#${EXP_NAME}_stale_}"
        case "$ts" in
            [0-9]*) ;;
            *) continue ;;
        esac
        if [ -f "$d/.fingerprint_search" ] && [ -f "$d/.fingerprint_target" ]; then
            scope="search"
        elif [ -f "$d/.fingerprint" ]; then
            scope="legacy"
        else
            scope="orphan"
        fi
        new="$CLIMBMIX_DIR/result/${EXP_NAME}_stale_${scope}_${ts}"
        if [ ! -e "$new" ]; then
            mv "$d" "$new"
            echo "  (lifecycle migration: $base -> ${EXP_NAME}_stale_${scope}_${ts})"
        fi
    done
}

_restore_completed() {
    # _current missing: reactivate the NEWEST completed ${EXP_NAME}_<ts> whose
    # search fingerprint matches (a completed run's products are exactly what
    # the fingerprints validated). Search-only match: a target mismatch then
    # re-runs only Steps 4-8 through the normal gate below.
    local d best=""
    for d in "$CLIMBMIX_DIR/result/${EXP_NAME}_"[0-9]*; do
        [ -d "$d" ] || continue
        [ -f "$d/.fingerprint_search" ] || continue
        # glob expansion is sorted: later (newer) matches overwrite $best
        if [ "$(cat "$d/.fingerprint_search")" = "$fp_search" ]; then
            best="$d"
        fi
    done
    if [ -n "$best" ]; then
        mv "$best" "$OUTPUT_DIR"
        echo "${best##*/}" > "$OUTPUT_DIR/.restored_from"
        echo "  RESTORE: completed run ${best##*/} matches the current search"
        echo "    fingerprint — reactivated as ${EXP_NAME}_current (idempotent:"
        echo "    .done markers skip; target fingerprint checked next)"
    fi
}

run_stage_gate() {
    local args_search=() args_target=() kv fp_search fp_target stale item was_complete
    for kv in "${FP_SEARCH_PARAMS[@]}"; do args_search+=(--param "$kv"); done
    for kv in "${FP_TARGET_PARAMS[@]}"; do args_target+=(--param "$kv"); done

    fp_search=$(python3 -m climbmix.utils.fingerprint --base-dir "$CLIMBMIX_DIR" \
        --stage search "${args_search[@]}")
    fp_target=$(python3 -m climbmix.utils.fingerprint --base-dir "$CLIMBMIX_DIR" \
        --stage target "${args_target[@]}")

    mkdir -p "$CLIMBMIX_DIR/result"

    # Lifecycle mechanics only apply to the default active dir; a custom
    # OUTPUT_DIR keeps plain gate behaviour (nothing to migrate/restore).
    if [ "$OUTPUT_DIR" = "$CLIMBMIX_DIR/result/${EXP_NAME}_current" ]; then
        _migrate_old_naming
        [ -d "$OUTPUT_DIR" ] || _restore_completed
    fi

    if [ -d "$OUTPUT_DIR" ] && [ -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
        if [ -f "$OUTPUT_DIR/.fingerprint_search" ] && [ -f "$OUTPUT_DIR/.fingerprint_target" ]; then
            if [ "$(cat "$OUTPUT_DIR/.fingerprint_search")" != "$fp_search" ]; then
                was_complete=false
                _is_complete "$OUTPUT_DIR" && was_complete=true
                stale="$(_archive_name search "$was_complete")"
                echo "  SEARCH fingerprint changed (code or params) — archiving everything:"
                echo "    $OUTPUT_DIR -> $stale"
                mv "$OUTPUT_DIR" "$stale"
                _write_archive_meta "$stale" \
                    "experiment=$EXP_NAME" \
                    "reason=search_fingerprint_changed" \
                    "was_complete=$was_complete" \
                    "old_fingerprint_search=$(cat "$stale/.fingerprint_search")" \
                    "new_fingerprint_search=$fp_search" \
                    "old_fingerprint_target=$(cat "$stale/.fingerprint_target")" \
                    "new_fingerprint_target=$fp_target"
            elif [ "$(cat "$OUTPUT_DIR/.fingerprint_target")" != "$fp_target" ]; then
                was_complete=false
                _is_complete "$OUTPUT_DIR" && was_complete=true
                stale="$(_archive_name target "$was_complete")"
                echo "  TARGET fingerprint changed — archiving target products only"
                echo "  (search products kept: search_state.json, exp_*/, sampled_dataset.parquet):"
                echo "    -> $stale"
                local moved=""
                for item in climb_shards random_shards climb_mixed random_mixed \
                            mid_train_climb.log mid_train_random.log \
                            eval_climb.log eval_random.log \
                            .done_mid_train_climb .done_mid_train_random \
                            .done_eval_climb .done_eval_random; do
                    if [ -e "$OUTPUT_DIR/$item" ]; then
                        mkdir -p "$stale"
                        mv "$OUTPUT_DIR/$item" "$stale/"
                        moved="$moved$item "
                    fi
                done
                if [ -n "$moved" ]; then
                    _write_archive_meta "$stale" \
                        "experiment=$EXP_NAME" \
                        "reason=target_fingerprint_changed" \
                        "was_complete=$was_complete" \
                        "old_fingerprint_target=$(cat "$OUTPUT_DIR/.fingerprint_target")" \
                        "new_fingerprint_target=$fp_target" \
                        "moved_items=${moved% }"
                fi
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
                was_complete=false
                _is_complete "$OUTPUT_DIR" && was_complete=true
                stale="$(_archive_name legacy "$was_complete")"
                echo "  Legacy single-fingerprint dir found (pre stage-scoping)."
                echo "  To adopt it unverified instead:"
                echo "    MIGRATE_LEGACY_FINGERPRINT=1 bash runs/$(basename "$0") ..."
                echo "  Archiving:"
                echo "    $OUTPUT_DIR -> $stale"
                mv "$OUTPUT_DIR" "$stale"
                _write_archive_meta "$stale" \
                    "experiment=$EXP_NAME" \
                    "reason=legacy_single_fingerprint" \
                    "was_complete=$was_complete" \
                    "old_fingerprint=$(cat "$stale/.fingerprint")"
            fi
        else
            was_complete=false
            _is_complete "$OUTPUT_DIR" && was_complete=true
            stale="$(_archive_name orphan "$was_complete")"
            echo "  Non-empty output dir without fingerprints — archiving:"
            echo "    $OUTPUT_DIR -> $stale"
            mv "$OUTPUT_DIR" "$stale"
            _write_archive_meta "$stale" \
                "experiment=$EXP_NAME" \
                "reason=no_fingerprint" \
                "was_complete=$was_complete"
        fi
    fi

    mkdir -p "$OUTPUT_DIR"
    echo "$fp_search" > "$OUTPUT_DIR/.fingerprint_search"
    echo "$fp_target" > "$OUTPUT_DIR/.fingerprint_target"
    rm -f "$OUTPUT_DIR/.fingerprint"
    echo "  Fingerprints: search=${fp_search} target=${fp_target}"
}

mark_completed() {
    # Called at the END of a green run: if every completion marker is present,
    # rename the active dir to its completed form (result/${EXP_NAME}_<ts>).
    # Keeps _current active otherwise (crash / partial run — resumable).
    # No-op for custom OUTPUT_DIR locations outside result/.
    local dir="$OUTPUT_DIR" new
    [ -d "$dir" ] || return 0
    case "$dir" in
        "$CLIMBMIX_DIR/result/"*) ;;
        *) return 0 ;;
    esac
    if ! _is_complete "$dir"; then
        echo "  (completion markers missing — keeping $(basename "$dir") active/resumable)"
        return 0
    fi
    new="$(_archive_name search true)"
    mv "$dir" "$new"
    _write_archive_meta "$new" \
        "experiment=$EXP_NAME" \
        "reason=completed" \
        "was_complete=true" \
        "fingerprint_search=$(cat "$new/.fingerprint_search" 2>/dev/null || echo unknown)" \
        "fingerprint_target=$(cat "$new/.fingerprint_target" 2>/dev/null || echo unknown)" \
        "restored_from=$(cat "$new/.restored_from" 2>/dev/null || echo '')"
    echo "  ✓ Run complete — archived as $new"
}
