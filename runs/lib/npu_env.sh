# ═══════════════════════════════════════════════════════════════════════
#  npu_env.sh — d28 target 训练/评测专用的完整 NPU 环境块
#
#  来源: quadmix/nanochat_mid_compare/run_stem_experiment.sh (dev/dataset-schema
#  分支实证无 OOM 的配置, 同一 nanochat-npu repo + 同一 d28 ckpt, dbs=1)。
#  用途: d28 Step-6 OOM 修复 —— 该 env 块含 unified memory / 融合 pass /
#  内存复用等开关, quadmix 在"单 8-rank torchrun"形态下验证过安全。
#
#  作用域纪律 (重要):
#  - 只在 Step 6 (mid_train) / Step 7 (base_eval) 的子 shell 里 source,
#    绝不让它进入并行搜索阶段 (8 进程 × 1 卡形态在 2026-08-26 出过事故:
#    allocator 记账之外内存膨胀 → device 0 满 29.4G, aclnnMean 内核加载失败)。
#  - 故意不含: MASTER_ADDR/MASTER_PORT (各阶段自行管理, 并行搜索用动态端口),
#    ASCEND_VISIBLE_DEVICES / RANK_SIZE (由 runner 按 NUM_NPU 设置),
#    NANOCHAT_DTYPE (调用方已导出, 保持单一来源)。
#  - TARGET_ENV_BLOCK=0 可整体关闭 (退回最小环境, 用于故障隔离)。
# ═══════════════════════════════════════════════════════════════════════
if [ "${TARGET_ENV_BLOCK:-1}" = "1" ]; then
    export PYTORCH_ALLOC_CONF=expandable_segments:True
    export ASCEND_COMPILE_OPT_LEVEL=O3
    export TORCH_NPU_LAZY_COMPILE=1
    export PYTHONPRELOAD=torch_npu
    export TORCH_NPU_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256,memory_pool:True"
    export PYTORCH_NPU_ALLOC_MAX_SIZE=60G
    export ASCEND_ENABLE_CACHE=1
    export ASCEND_CACHE_POLICY=2
    export ASCEND_FUSION_ENABLE=1
    export ASCEND_GEMM_DTiling=1
    export TORCH_NPU_ENABLE_NUMA=1
    export ASCEND_MEMORY_COPY_MODE=1
    export ASCEND_HBM_ALLOC_TYPE=1
    export ASCEND_OPP_LEVEL=O3
    export ASCEND_FUSION_PASS_ENABLE=1
    export ASCEND_GEMM_BTiling=1
    export ASCEND_GEMM_ATiling=1
    export ASCEND_CONV_ALGO_SELECTION=1
    export ASCEND_ENABLE_TRANSFORMER_FUSION=1
    export ASCEND_MEMORY_REUSE_MODE=2
    export ASCEND_ENABLE_PREFETCH=1
    # HBM 溢出可换页到主机内存 —— quadmix dbs=1 稳定运行的关键开关之一
    export ASCEND_NPU_ENABLE_UNIFIED_MEMORY=1
    export ASCEND_OPTIMIZER_AGGRESSIVE_MODE=1
    export ASCEND_SYNCHRONIZATION_MODE=0
    export PYTORCH_NPU_ENABLE_LARGE_CONCAT=1
    export PYTORCH_NPU_ENABLE_TORCHscript=1
    export NPU_PERF_MODE=high_performance
    export ASCEND_DISABLE_MEM_SWAP=1
    export ASCEND_LAUNCH_BLOCKING=0
    export NPU_DISABLE_RECORD=1
else
    echo "  (TARGET_ENV_BLOCK=0 — 使用最小 NPU 环境)"
fi
