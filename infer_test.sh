#!/usr/bin/env bash

# 中文简介：基于给定 checkpoint 和数据集做一次离线推理测试，验证模型能否正常加载并输出动作。

set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/szk/szk/Evo-RL/outputs/a2a_train_20260426_121524/checkpoints/050000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_trimmed_10hz_320x240}"
DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_trimmed_10hz_320x240}"
TASK_TEXT="${TASK_TEXT:-Fold a towel}"
DEVICE="${DEVICE:-cuda}"
CONTROL_HZ="${CONTROL_HZ:-10}"
DURATION_S="${DURATION_S:-20}"
CAMERA_FPS="${CAMERA_FPS:-30}"

cmd=(
  python /home/szk/szk/Evo-RL/run_a2a_live_policy.py
  --checkpoint-path "${CHECKPOINT_PATH}"
  --dataset-root "${DATASET_ROOT}"
  --dataset-repo-id "${DATASET_REPO_ID}"
  --task "${TASK_TEXT}"
  --device "${DEVICE}"
  --control-hz "${CONTROL_HZ}"
  --camera-fps "${CAMERA_FPS}"
  --duration-s "${DURATION_S}"
  --run-start-pose-reset
)

"${cmd[@]}"
