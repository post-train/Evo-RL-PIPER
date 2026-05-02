#!/usr/bin/env bash

# 中文简介：执行同步版 LeRobot 推理流程，通常用于和异步/RTC 推理结果做对照。

set -euo pipefail

CHECKPOINT_PATH="${CHECKPOINT_PATH:-/home/szk/szk/Evo-RL/outputs/original_a2a_train_20260427_210451/checkpoints/050000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
TASK="${TASK:-Fold a towel}"
DEVICE="${DEVICE:-cuda}"
CONTROL_HZ="${CONTROL_HZ:-10}"
CAMERA_FPS="${CAMERA_FPS:-30}"
DURATION_S="${DURATION_S:-30}"
LEFT_CAMERA_PATH="${LEFT_CAMERA_PATH:-/dev/video2}"
RIGHT_CAMERA_PATH="${RIGHT_CAMERA_PATH:-/dev/video0}"
WIDE_CAMERA_PATH="${WIDE_CAMERA_PATH:-/dev/video4}"
LEFT_PORT="${LEFT_PORT:-can3}"
RIGHT_PORT="${RIGHT_PORT:-can2}"
USE_AMP="${USE_AMP:-0}"
RUN_START_POSE_RESET="${RUN_START_POSE_RESET:-1}"
DRY_RUN="${DRY_RUN:-0}"

cmd=(
    python /home/szk/szk/Evo-RL/run_a2a_sync_policy.py
    --checkpoint-path "${CHECKPOINT_PATH}"
    --dataset-root "${DATASET_ROOT}"
    --dataset-repo-id "${DATASET_REPO_ID}"
    --task "${TASK}"
    --device "${DEVICE}"
    --control-hz "${CONTROL_HZ}"
    --camera-fps "${CAMERA_FPS}"
    --duration-s "${DURATION_S}"
    --left-camera-path "${LEFT_CAMERA_PATH}"
    --right-camera-path "${RIGHT_CAMERA_PATH}"
    --wide-camera-path "${WIDE_CAMERA_PATH}"
    --left-port "${LEFT_PORT}"
    --right-port "${RIGHT_PORT}"
)

if [[ "${USE_AMP}" == "1" ]]; then
    cmd+=(--use-amp)
fi

if [[ "${RUN_START_POSE_RESET}" == "1" ]]; then
    cmd+=(--run-start-pose-reset)
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
