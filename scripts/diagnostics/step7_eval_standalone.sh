#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
#  step7_eval_standalone.sh — 脱离 speedrun 全流程, 单独重跑 Step 7 eval
#
#  背景 (2026-08-28): speedrun Step 7 eval 在第一个任务 arc_easy 末尾的
#  dist.barrier() 处 OOM — HCCL 申请 401MiB allreduce 通信缓冲失败
#  (EL0004, torch allocator 记账之外)。根因: --eval=core 下真正决定
#  显存的是 --core-eval-batch-size (默认 16, 整块 pad-to-longest forward),
#  不是 --device-batch-size (BPB-only, core 分支不读)。修复 = core_bs 16→8。
#  本脚本不触碰指纹 / 不触发 stage gate, 直接对已训练好的 d28 mid ckpt
#  重跑 eval, 实测 core_bs=8 是否仍 OOM; OOM 则自动降档 8→4→2→1。
#
#  用法 (服务器 8x910B4 上; eval 全程约 1h, 建议 nohup/tmux):
#    bash scripts/diagnostics/step7_eval_standalone.sh
#    ADOPT=1 bash scripts/diagnostics/step7_eval_standalone.sh
#
#    不带 ADOPT   → 只测 eval (Phase 1)。通过/失败证据落在活跃目录的
#                   eval_test/ 下, 不影响流水线任何状态。
#    ADOPT=1      → eval 通过后把结果"接回"流水线 (Phase 2): 写入新
#                   target 指纹 + .done_eval_climb + eval_climb.log/csv。
#                   之后 bash runs/speedrun_climbmix.sh 指纹全匹配,
#                   Steps 1-7 全跳过 (mark_completed 收尾归档) — 不重训
#                   Step 6。若本次调用已有成功记录, ADOPT=1 只做 Phase 2。
#
#  活跃目录解析 (兼容 result-dir lifecycle 新旧命名):
#    用户显式 OUTPUT_DIR > speedrun shell 当前默认 (新代码 = result/
#    ${EXP_NAME}_current) > 旧命名活跃目录 result/${EXP_NAME} (新代码
#    首跑前仍存在时) > shell 默认。旧命名目录在下一次全流程
#    run_stage_gate 时被整体 mv 成 _current, 内容原样保留 — 因此 adopt
#    写进旧命名目录同样有效, 迁移后指纹/标记随目录一起走。
#
#  ADOPT=1 前置条件 (缺一即拒, 全部有明确报错):
#    1. Phase-1 eval 已成功 (eval_test/last_success 存在; 本次或上次运行)
#    2. 服务器 climbmix 工作区已含修复: runs/speedrun_climbmix.sh 里出现
#       EVAL_CORE_BATCH_SIZE= (commit+push+pull 或直接 scp 补丁均可 —
#       指纹哈希的是工作区文件内容, 不看 git 状态; 本脚本自身在
#       scripts/diagnostics/ 下, 不进指纹)
#    3. 活跃目录 .done_mid_train_climb 存在 (Step 6 已完成)
#    4. 新旧两个活跃目录不得同时存在 (与 gate 的迁移告警同口径)
#    5. EXP_NAME=speedrun (单臂; 生产 run_climbmix 双臂不适用)
#
#  环境变量 (默认全部 = speedrun Step 7 口径):
#    EVAL_CORE_BATCH_SIZE=8  起始 core batch (OOM 自动降档)
#    EVAL_DEVICE_BATCH_SIZE=16 / EVAL_MAX_PER_TASK=100 / EVAL_BENCHMARKS=stem
#    MODEL_TAG=d28_speedrun / NANOCHAT_DIR / NANOCHAT_BASE_DIR / NUM_NPU=8
#
#  退出码: 0=eval 通过 (含降档后通过)  2=core_bs=1 仍 OOM
#          3=非 OOM 失败              4=前置检查/ADOPT 条件失败
# ═══════════════════════════════════════════════════════════════════════
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIMBMIX_DIR="${CLIMBMIX_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

NANOCHAT_DIR="${NANOCHAT_DIR:-/home/ma-user/work/nanochat-npu}"
NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/home/ma-user/work/nanochat_model_dir}"
NUM_NPU="${NUM_NPU:-8}"
EXP_NAME="${EXP_NAME:-speedrun}"
TARGET_DEPTH="${TARGET_DEPTH:-28}"
MODEL_TAG="${MODEL_TAG:-d${TARGET_DEPTH}_${EXP_NAME}}"
EVAL_BENCHMARKS="${EVAL_BENCHMARKS:-stem}"
EVAL_MAX_PER_TASK="${EVAL_MAX_PER_TASK:-100}"
EVAL_DEVICE_BATCH_SIZE="${EVAL_DEVICE_BATCH_SIZE:-16}"
EVAL_CORE_BATCH_SIZE="${EVAL_CORE_BATCH_SIZE:-8}"
NANOCHAT_DTYPE="${NANOCHAT_DTYPE:-bfloat16}"
USER_OUTPUT_DIR="${OUTPUT_DIR:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── 活跃目录解析 (见文件头注释) ──
resolve_output_dir() {
    local candidate="" line
    line=$(grep -m1 '^OUTPUT_DIR=' "$CLIMBMIX_DIR/runs/speedrun_climbmix.sh" 2>/dev/null || true)
    if [ -n "$line" ]; then
        # 在隔离子 shell 里求值该赋值行 (OUTPUT_DIR 未传给子 shell → 取默认)
        candidate=$(bash -c 'CLIMBMIX_DIR=$1; EXP_NAME=$2; eval "$3"; printf %s "$OUTPUT_DIR"' \
            _ "$CLIMBMIX_DIR" "$EXP_NAME" "$line")
    fi
    candidate="${candidate:-$CLIMBMIX_DIR/result/$EXP_NAME}"
    if [ -n "$USER_OUTPUT_DIR" ]; then
        OUTPUT_DIR="$USER_OUTPUT_DIR"
    elif [ -d "$candidate" ]; then
        OUTPUT_DIR="$candidate"
    elif [ -d "$CLIMBMIX_DIR/result/$EXP_NAME" ] && [ "$candidate" != "$CLIMBMIX_DIR/result/$EXP_NAME" ]; then
        OUTPUT_DIR="$CLIMBMIX_DIR/result/$EXP_NAME"   # 新命名 + gate 迁移尚未发生
    else
        OUTPUT_DIR="$candidate"
    fi
}
resolve_output_dir
TEST_DIR="$OUTPUT_DIR/eval_test"
SUCCESS_FILE="$TEST_DIR/last_success"

# ── 与 speedrun 相同的 base NPU env (最小块; 不加 allocator 覆盖项) ──
export OMP_NUM_THREADS=1 WANDB_MODE=offline NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
mkdir -p "$NANOCHAT_BASE_DIR" 2>/dev/null || true
export ASCEND_HCCL_PATH=/usr/local/Ascend/ascend-toolkit/latest/hccl
export LD_LIBRARY_PATH=${ASCEND_HCCL_PATH}/lib64:${LD_LIBRARY_PATH:-}
export HCCL_CONNECT_TIMEOUT=1200 HCCL_WHITELIST_DISABLE=1
export NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME=eth0
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_NPU - 1)))
export RANK_SIZE=$NUM_NPU MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export HCCL_EXEC_TIMEOUT=1200
export PYTHONUNBUFFERED=1
export NANOCHAT_DTYPE="$NANOCHAT_DTYPE" PYTHONWARNINGS="ignore::UserWarning:torch_npu"

die() { echo "✗ $*" >&2; exit 4; }

# ── Pre-flight ──
[ -f "$CLIMBMIX_DIR/runs/lib/npu_env.sh" ] \
    || die "npu_env.sh not found under $CLIMBMIX_DIR (set CLIMBMIX_DIR=)"
[ -d "$NANOCHAT_DIR" ] || die "nanochat-npu not found at $NANOCHAT_DIR (set NANOCHAT_DIR=)"
[ -d "$NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG" ] \
    && ls "$NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG"/model_*.pt >/dev/null 2>&1 \
    || die "mid checkpoint missing: $NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG (Step 6 not complete?)"
python3 -c "import torch_npu; import torch; assert torch.npu.is_available(), 'NPU not available'" \
    || die "NPU not available — run on the 8x910B4 server"
mkdir -p "$TEST_DIR"

echo "════════════════════════════════════════════════════════════"
echo "  Standalone Step-7 eval: model=$MODEL_TAG  core_bs=$EVAL_CORE_BATCH_SIZE"
echo "  active dir: $OUTPUT_DIR"
echo "════════════════════════════════════════════════════════════"

run_eval_once() {   # $1=core_bs  $2=logfile
    local bs="$1" log="$2" rc=0
    echo "── eval attempt: --core-eval-batch-size=$bs ──"
    npu-smi info > "$log.npu_before" 2>&1 || true
    (
        # shellcheck source=/dev/null
        source "$CLIMBMIX_DIR/runs/lib/npu_env.sh"
        cd "$NANOCHAT_DIR" && torchrun --standalone --nproc_per_node="$NUM_NPU" -m scripts.base_eval -- \
        --eval=core --eval-benchmarks="$EVAL_BENCHMARKS" \
        --max-per-task="$EVAL_MAX_PER_TASK" \
        --device-batch-size="$EVAL_DEVICE_BATCH_SIZE" \
        --core-eval-batch-size="$bs" \
        --model-tag="$MODEL_TAG" --model-type=mid 2>&1 | tee "$log"
    ) || rc=$?   # 显式捕获: 尾部 npu-smi || true 不能掩盖 torchrun 退出码
    npu-smi info > "$log.npu_after" 2>&1 || true
    return "$rc"
}

oom_in_log() {   # $1=logfile — NPU 的 OOM 报错串不含 'out of memory', 按编号匹配
    grep -qE 'EL0004|Failed to allocate|out of memory|OutOfMemoryError|207001' "$1" 2>/dev/null
}

# ── Phase 1: standalone eval, OOM 自动降档 ──
phase1() {
    local bs="$EVAL_CORE_BATCH_SIZE" rc=0 log csv snap
    while :; do
        log="$TEST_DIR/attempt_bs${bs}_$(date +%Y%m%d_%H%M%S).log"
        if run_eval_once "$bs" "$log"; then
            grep -q "Results written to:" "$log" \
                || { echo "✗ exit 0 but no 'Results written to:' in log — treating as failure"; return 3; }
            csv=$(ls -t "$NANOCHAT_BASE_DIR"/base_eval/mid_model_*.csv 2>/dev/null | head -1 || true)
            [ -n "$csv" ] || { echo "✗ no mid_model_*.csv found after eval"; return 3; }
            # 立即快照 CSV (后续任何 eval 覆盖同名文件都不影响本结果);
            # %q 保证含空格的值 (DATE/路径) source 回来时安全
            snap="$TEST_DIR/success_$(basename "$csv")"
            cp -f "$csv" "$snap" || return 3
            printf 'LOG=%q\nCSV=%q\nBS=%q\nDATE=%q\n' "$log" "$snap" "$bs" "$(date)" > "$SUCCESS_FILE"
            echo ""
            echo "✓ EVAL PASSED at core-eval-batch-size=$bs"
            grep -E "STEM metric|STEM NLL|CORE metric" "$log" || true
            return 0
        fi
        rc=$?
        if oom_in_log "$log"; then
            echo "✗ OOM at core_bs=$bs (log: $log)"
            if [ "$bs" -gt 1 ]; then
                bs=$((bs / 2))
                echo "   retrying with core_bs=$bs ..."
                continue
            fi
            echo "✗ core_bs=1 仍 OOM — 需要 nanochat-npu 侧补丁 (barrier 前 empty_cache +"
            echo "  core_eval.py:382 的 OOM 匹配串加宽), 见 TODO.md Known Limitations"
            return 2
        fi
        echo "✗ eval FAILED (non-OOM, rc=$rc) — see $log"
        return 3
    done
}

# ── Phase 2: ADOPT — 结果接回流水线, 免重跑 Steps 4-7 ──
phase2() {
    echo ""
    echo "── ADOPT: 把 eval 结果接回 speedrun (不重跑 Steps 4-7) ──"
    local speed_sh="$CLIMBMIX_DIR/runs/speedrun_climbmix.sh"

    [ "$EXP_NAME" = "speedrun" ] || die "ADOPT is speedrun-only (EXP_NAME=$EXP_NAME; 生产双臂请走全流程)"
    grep -q '^EVAL_CORE_BATCH_SIZE=' "$speed_sh" \
        || die "服务器仓库未含修复 (speedrun_climbmix.sh 无 EVAL_CORE_BATCH_SIZE) — 先 commit+push+pull 或 scp 补丁"
    [ -f "$SUCCESS_FILE" ] || die "无成功 eval 记录 — 先跑 Phase 1"
    # shellcheck source=/dev/null
    . "$SUCCESS_FILE"
    [ -n "${LOG:-}" ] && [ -n "${CSV:-}" ] && [ -f "$LOG" ] && [ -f "$CSV" ] \
        || die "成功记录损坏 ($SUCCESS_FILE) — 删掉后重跑 Phase 1"

    # 从 (新) speedrun shell 提取变量赋值行 + 两个 FP 数组 — 求值环境与下次
    # 全流程 run_stage_gate 逐字一致 (含 ${VAR:-default} 的 env 覆盖语义,
    # grep 保持文件行序 → GENERAL_DATA_DIR 引用的 NANOCHAT_BASE_DIR 先定义)
    eval "$(grep -E '^(EXP_NAME|DATA_DIR|NANOCHAT_DIR|NANOCHAT_BASE_DIR|GENERAL_DATA_DIR|PROXY_DEPTH|PROXY_NUM_ITERATIONS|PROXY_TARGET_TOKENS|CONFIGS_PER_ITER|SEARCH_NUM_ITERATIONS|K_ENHANCED|K_CLUSTER_MAX|K_INIT|FILTER_METHOD|PRUNE_THRESHOLD|MERGE_DISTANCE|EMBEDDING_MODEL|EMBEDDING_SAMPLE_SIZE|PROXY_LR_SCALE|PROXY_WARMUP|PROXY_WARMDOWN|NPU_PER_EXP|TARGET_DEPTH|TARGET_STEPS|TARGET_TOKENS|TARGET_LR_SCALE|TARGET_WARMUP|TARGET_WARMDOWN|MID_DEVICE_BATCH_SIZE|MID_TRAIN_LOADER|EVAL_DEVICE_BATCH_SIZE|EVAL_CORE_BATCH_SIZE|CORE_METRIC_EVERY|STEM_RATIO|EVAL_BENCHMARKS|EVAL_MAX_PER_TASK|NANOCHAT_DTYPE|OUTPUT_DIR)=' "$speed_sh")"
    eval "$(awk '/^FP_SEARCH_PARAMS=\(/,/^\)$/' "$speed_sh")"
    eval "$(awk '/^FP_TARGET_PARAMS=\(/,/^\)$/' "$speed_sh")"
    [ "${#FP_TARGET_PARAMS[@]}" -gt 0 ] && [ "${#FP_SEARCH_PARAMS[@]}" -gt 0 ] \
        || die "FP 数组提取失败 (speedrun_climbmix.sh 格式变了?)"

    # 活跃目录二次解析 (eval 已把 OUTPUT_DIR 重置为 shell 默认值):
    # 新命名(_current)存在用之; 否则旧命名目录 (gate 迁移前) 同样有效 —
    # mv 保内容, gate 迁移后指纹/标记随目录一起走。新旧同存 = 歧义, 拒绝。
    local active="$OUTPUT_DIR" legacy="$CLIMBMIX_DIR/result/$EXP_NAME"
    if [ "$active" != "$legacy" ]; then
        if [ -d "$active" ] && [ -d "$legacy" ]; then
            die "新旧两个活跃目录同时存在: $legacy 与 $active — 先手动处理 (gate 迁移逻辑同样会对此告警)"
        fi
        [ -d "$active" ] || active="$legacy"
    fi
    [ -d "$active" ] || die "活跃目录不存在: $active (候选 $OUTPUT_DIR / $legacy)"
    [ -f "$active/.done_mid_train_climb" ] \
        || die "$active/.done_mid_train_climb 缺失 — Step 6 未完成, 无法 adopt"

    local args_t=() args_s=() kv fp_t fp_s old_s old_t
    for kv in "${FP_TARGET_PARAMS[@]}"; do args_t+=(--param "$kv"); done
    for kv in "${FP_SEARCH_PARAMS[@]}"; do args_s+=(--param "$kv"); done
    fp_t=$(PYTHONPATH="$CLIMBMIX_DIR/src" python3 -m climbmix.utils.fingerprint \
        --base-dir "$CLIMBMIX_DIR" --stage target "${args_t[@]}")
    fp_s=$(PYTHONPATH="$CLIMBMIX_DIR/src" python3 -m climbmix.utils.fingerprint \
        --base-dir "$CLIMBMIX_DIR" --stage search "${args_s[@]}")

    # 安全断言: search 指纹必须与现存一致, 否则下次全流程仍会整目录归档
    old_s=$(cat "$active/.fingerprint_search" 2>/dev/null || true)
    if [ -n "$old_s" ] && [ "$old_s" != "$fp_s" ]; then
        die "search 指纹不匹配 — adopt 后下次全流程仍会归档全部 (含搜索产物)。\
检查当前 shell 是否残留 EXP_*/EVAL_* 等覆盖变量, 在干净 shell 里重试 ADOPT"
    fi

    old_t=$(cat "$active/.fingerprint_target" 2>/dev/null || true)
    echo "$fp_t" > "$active/.fingerprint_target"
    [ "$old_t" = "$fp_t" ] && echo "  (target 指纹已是最新 — 无需变更)" \
        || echo "  target 指纹已更新 ($old_t → $fp_t)"

    cp -f "$LOG" "$active/eval_climb.log"
    cp -f "$CSV" "$active/eval_climb.csv"
    touch "$active/.done_eval_climb"

    echo "  活跃目录: $active"
    echo "  采用的 target 指纹参数 (供核对):"
    printf '    %s\n' "${FP_TARGET_PARAMS[@]}"
    echo ""
    echo "✓ adopted: .fingerprint_target / .done_eval_climb / eval_climb.log / eval_climb.csv"
    echo "  下一步: bash runs/speedrun_climbmix.sh → 指纹全匹配 → Steps 1-7 全跳过"
    echo "  (若活跃目录还是旧命名 result/$EXP_NAME, gate 会先整体迁移成 _current, 内容不变)"
}

# ── Dispatch ──
rc=0
if [ "${ADOPT:-0}" = "1" ] && [ -f "$SUCCESS_FILE" ]; then
    phase2                      # 已有成功记录: 只 adopt, 不重跑 eval
else
    phase1 || rc=$?
    if [ "$rc" -eq 0 ] && [ "${ADOPT:-0}" = "1" ]; then
        phase2
    fi
fi
exit "$rc"
