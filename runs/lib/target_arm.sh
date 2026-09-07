# ═══════════════════════════════════════════════════════════════════════
#  target_arm.sh — d28 目标臂 (climb/random) 训练 + 评测的共享 shell 库
#
#  被 runs/run_climbmix.sh Step 6+7 (本地执行路径 / 远端失败兜底) 与
#  scripts/dispatch_target_arm.py (远端路径的 argv 事实来源) 共同使用:
#  本文件的 torchrun argv 与
#  src/climbmix/pipeline/nanochat_cmds.py 的 build_target_mid_train_cmd /
#  build_target_eval_cmd 必须 **逐 token 一致** — 由
#  scripts/diagnostics/test_prod2_runtime.py 强制保证 (PATH 桩 torchrun
#  双侧捕获比对)。
#
#  所需环境变量 (run_climbmix.sh 主管道已导出):
#    CLIMBMIX_DIR  NANOCHAT_DIR  NANOCHAT_BASE_DIR  OUTPUT_DIR
#    TARGET_BASE_CKPT  (= base_checkpoints/d28 的绝对路径)
#    NUM_NPU  TARGET_STEPS  TARGET_LR_SCALE  TARGET_WARMUP  TARGET_WARMDOWN
#    CORE_METRIC_EVERY  MID_DEVICE_BATCH_SIZE  MID_TRAIN_LOADER
#    EVAL_BENCHMARKS  EVAL_MAX_PER_TASK  EVAL_DEVICE_BATCH_SIZE
#    EVAL_CORE_BATCH_SIZE
#
#  作用域纪律 (同 npu_env.sh): 只在臂训练/评测的子 shell 里 source
#  npu_env.sh, 绝不让它进入并行搜索阶段 (2026-08-26 事故)。
#  幂等性: 调用方 (run_arm) 以 .done_mid_train_<name> / .done_eval_<name>
#  标记守门; 本函数只负责执行, 不写标记。
# ═══════════════════════════════════════════════════════════════════════

target_arm_train() {
    # usage: target_arm_train <data_dir> <tag> <name>
    local data_dir="$1" tag="$2" name="$3"
    local link_dir="$NANOCHAT_BASE_DIR/base_checkpoints/$tag"
    # Clean a stale/broken symlink from a previous crashed attempt BEFORE the
    # `[ -e ] || ln -s`: a broken link fails `[ -e ]` yet still blocks ln -s
    # (EEXIST), which kills the script under set -e.
    if [ -L "$link_dir" ] && [ ! -e "$link_dir" ]; then rm -f "$link_dir"; fi
    [ -e "$link_dir" ] || ln -s "$TARGET_BASE_CKPT" "$link_dir"
    # Clear partial checkpoints from a crashed attempt (nanochat may otherwise
    # try to auto-resume from inconsistent state; whole-run atomicity instead)
    rm -rf "$NANOCHAT_BASE_DIR/mid_checkpoints/$tag"
    # 单 8-rank torchrun (quadmix 验证过 env 块安全的唯一形态);
    # 并行搜索阶段绝不 source npu_env.sh (2026-08-26 事故).
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.mid_train -- \
        --num-iterations="$TARGET_STEPS" \
        --lr-scale="$TARGET_LR_SCALE" --warmup-ratio="$TARGET_WARMUP" --warmdown-ratio="$TARGET_WARMDOWN" \
        --core-metric-every="$CORE_METRIC_EVERY" \
        --device-batch-size="$MID_DEVICE_BATCH_SIZE" \
        --loader="$MID_TRAIN_LOADER" \
        --sample-every=-1 \
        --eval-every=-1 \
        --run="${name}_mid" --model-tag="$tag" \
        --data-dir="$data_dir" 2>&1 | tee "$OUTPUT_DIR/mid_train_${name}.log"
    )
    # NOT `[ -L ] && rm` as the last statement: when link_dir is absent or not
    # a symlink the function would return 1, and under set -e the script dies
    # AFTER successful training with .done unwritten → retrain on every resume.
    if [ -L "$link_dir" ]; then rm -f "$link_dir"; fi
}

target_arm_eval() {
    # usage: target_arm_eval <tag> <name>
    local tag="$1" name="$2"
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --max-per-task="$EVAL_MAX_PER_TASK" \
        --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
        --core-eval-batch-size="$EVAL_CORE_BATCH_SIZE" \
        --model-tag="$tag" --model-type=mid 2>&1 | tee "$OUTPUT_DIR/eval_${name}.log"
    )
    # base_eval writes a step-only CSV name (mid_model_{step}.csv) into the
    # shared base dir; both arms train the same step count, so the second
    # eval would overwrite the first. Archive the newest CSV per arm right
    # after its eval (Step 8 reads the LOGS; this preserves the raw
    # 4-column CSVs for the final analysis). Evals are sequential here —
    # the dispatch (remote) path lands its own eval_<arm>.csv copy.
    local newest
    newest=$(ls -t "$NANOCHAT_BASE_DIR"/base_eval/mid_model_*.csv 2>/dev/null | head -1)
    if [ -n "$newest" ]; then
        cp -f "$newest" "$OUTPUT_DIR/eval_${name}.csv"
        echo "  Archived $(basename "$newest") -> eval_${name}.csv"
    else
        echo "  WARNING: no mid_model_*.csv found after eval ${name}"
    fi
}
