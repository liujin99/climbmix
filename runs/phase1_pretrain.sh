#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Phase-1 pretraining: produce checkpoint for proxy/target models
# ──────────────────────────────────────────────────────────────
# Uses nanochat's base_train.py to train a model from scratch,
# then saves the checkpoint for continual pre-training in
# Stage 1/2/3.
#
# Usage:
#   bash runs/phase1_d5.sh     # depth=5, ~59M, for Stage 1
#   bash runs/phase1_d10.sh    # depth=10, ~196M, for Stage 2
#   bash runs/phase1_d24.sh    # depth=24, ~1.38B, for Stage 3
# ──────────────────────────────────────────────────────────────

DEPTH="${1:-10}"
TARGET_RATIO="${2:-20}"   # tokens/params ratio (Chinchilla=20)

case "$DEPTH" in
    5)  PARAMS_APPROX="59M";  TOKENS_APPROX="1B";  BATCH_SIZE=4;  EST_TIME="~10 min" ;;
    10) PARAMS_APPROX="196M"; TOKENS_APPROX="4B";  BATCH_SIZE=8;  EST_TIME="~30-40 min" ;;
    24) PARAMS_APPROX="1.38B"; TOKENS_APPROX="13B"; BATCH_SIZE=8; EST_TIME="~13.8h" ;;
    *)  echo "Unsupported depth: $DEPTH (use 5, 10, or 24)"; exit 1 ;;
esac

NANOCHAT_DIR="${NANOCHAT_DIR:-/home/liujin99/nanochat-npu}"
CKPT_DIR="${CKPT_DIR:-/home/liujin99/climbmix/checkpoints/phase1_d${DEPTH}}"
DATA_DIR="${DATA_DIR:-/home/liujin99/data/essential-web-v1-preprocessed}"

echo "╔══ Phase-1 Pretraining ═══╗"
echo ""
echo "  Depth:      $DEPTH (~$PARAMS_APPROX params)"
echo "  Data ratio: $TARGET_RATIO (Chinchilla optimal)"
echo "  Tokens:     ~$TOKENS_APPROX"
echo "  Batch:      $BATCH_SIZE per device"
echo "  Output:     $CKPT_DIR"
echo "  Time:       $EST_TIME on 8×910B3"
echo ""
echo "╚══════════════════════════╝"

mkdir -p "$CKPT_DIR"

torchrun --standalone --nproc_per_node=8 \
    -m scripts.base_train \
    --depth "$DEPTH" \
    --target-param-data-ratio "$TARGET_RATIO" \
    --device-batch-size "$BATCH_SIZE" \
    --model-tag "phase1_d${DEPTH}" \
    --run "climbmix_phase1_d${DEPTH}" \
    "$@"
