#!/usr/bin/env bash

# 中文简介：基于给定 policy 做一次同步推理录制，生成带模型执行结果的数据集输出目录。

set -euo pipefail

POLICY_PATH="${POLICY_PATH:-/home/szk/szk/Evo-RL/outputs/a2a_train_20260427_160416/checkpoints/050000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/outputs/a2a_inference_official}"
DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_a2a_inference_official_10hz_320x240}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-3000}"
DATASET_FPS="${DATASET_FPS:-10}"
CAMERA_FPS="${CAMERA_FPS:-30}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"
RUN_SAFE_RESET="${RUN_SAFE_RESET:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -d "${POLICY_PATH}" ]]; then
    echo "Policy path does not exist: ${POLICY_PATH}" >&2
    exit 1
fi

# if [[ "${RUN_SAFE_RESET}" == "1" && "${DRY_RUN}" != "1" ]]; then
#     python -m src.lerobot.utils.safe_dual_piper
# fi

cmd=(
    lerobot-record
    --robot.type=bi_piper_follower
    --robot.id=bi_piper_follower
    --robot.cameras="{\"left_left_arm\": {\"type\": \"opencv\", \"index_or_path\": \"/dev/video2\", \"width\": 320, \"height\": 240, \"fps\": ${CAMERA_FPS}, \"backend\": \"V4L2\", \"rotation\": \"ROTATE_180\", \"fourcc\": \"MJPG\"}, \"right_right_arm\": {\"type\": \"opencv\", \"index_or_path\": \"/dev/video0\", \"width\": 320, \"height\": 240, \"fps\": ${CAMERA_FPS}, \"backend\": \"V4L2\", \"rotation\": \"ROTATE_180\", \"fourcc\": \"MJPG\"}, \"wide_angle\": {\"type\": \"opencv\", \"index_or_path\": \"/dev/video4\", \"width\": 320, \"height\": 240, \"fps\": ${CAMERA_FPS}, \"backend\": \"V4L2\", \"rotation\": \"ROTATE_180\", \"fourcc\": \"MJPG\"}}"
    --robot.left_arm_config.port=can3
    --robot.right_arm_config.port=can2
    --robot.left_arm_config.require_calibration=false
    --robot.right_arm_config.require_calibration=false
    --policy.path="${POLICY_PATH}"
    --policy.device="${POLICY_DEVICE}"
    --policy.use_amp="${POLICY_USE_AMP}"
    --dataset.fps="${DATASET_FPS}"
    --dataset.episode_time_s="${EPISODE_TIME_S}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.num_episodes="${NUM_EPISODES}"
    --dataset.single_task="Fold a towel"
    --display_data="${DISPLAY_DATA}"
    --play_sounds="${PLAY_SOUNDS}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
