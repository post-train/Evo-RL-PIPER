#!/usr/bin/env bash

# 中文简介：对已经训练好的 ABPolicy checkpoint 做离线评估或离线推理分析。

set -euo pipefail

RUN_DIR="${RUN_DIR:-/home/szk/szk/Evo-RL/outputs/abpolicy_train_20260502_003711}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DEVICE="${DEVICE:-cuda}"
WINDOWS_PER_EPISODE="${WINDOWS_PER_EPISODE:-2}"
EPISODES="${EPISODES:-0 10 20 30 40}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/offline_ab_70k_80k}"

python /home/szk/szk/Evo-RL/run_ab_offline.py \
  --checkpoint-path "${RUN_DIR}/checkpoints/070000/pretrained_model" \
  --checkpoint-path "${RUN_DIR}/checkpoints/080000/pretrained_model" \
  --dataset-root "${DATASET_ROOT}" \
  --dataset-repo-id "${DATASET_REPO_ID}" \
  --device "${DEVICE}" \
  --windows-per-episode "${WINDOWS_PER_EPISODE}" \
  --episode-indices ${EPISODES} \
  --output-dir "${OUTPUT_DIR}"
