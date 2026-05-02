#!/usr/bin/env bash

# 中文简介：批量评估一个 ABPolicy 训练目录下的多个 checkpoint，便于横向比较训练过程表现。

set -euo pipefail

RUN_DIR="${RUN_DIR:-/home/szk/szk/Evo-RL/outputs/abpolicy_train_20260502_003711}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_BATCHES="${MAX_BATCHES:-20}"
OUTPUT_JSON="${OUTPUT_JSON:-${RUN_DIR}/ab_checkpoint_compare_60_70_80k.json}"

python /home/szk/szk/Evo-RL/compare_ab_checkpoints.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/060000/pretrained_model" \
  --checkpoint-path "${RUN_DIR}/checkpoints/070000/pretrained_model" \
  --checkpoint-path "${RUN_DIR}/checkpoints/080000/pretrained_model" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --max-batches "${MAX_BATCHES}" \
  --output-json "${OUTPUT_JSON}"
