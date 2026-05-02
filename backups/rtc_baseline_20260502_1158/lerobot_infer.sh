#!/usr/bin/env bash

set -euo pipefail

POLICY_PATH="${POLICY_PATH:-/home/szk/szk/Evo-RL/outputs/flow_matching_train_20260430_170525/checkpoints/080000/pretrained_model}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/outputs/flow_matching_infer_80k}"

if [[ -d "${DATASET_ROOT}" ]]; then
    echo "Removing existing output directory: ${DATASET_ROOT}"
    rm -rf "${DATASET_ROOT}"
fi
DATASET_REPO_ID="${DATASET_REPO_ID:-local/eval_flow_matching_infer_80k}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-3000}"
DATASET_FPS="${DATASET_FPS:-30}"
CAMERA_FPS="${CAMERA_FPS:-30}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-false}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"
RUN_SAFE_RESET="${RUN_SAFE_RESET:-1}"
DRY_RUN="${DRY_RUN:-0}"
TASK_NAME="${TASK_NAME:-Fold a towel}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_VIDEO="${DATASET_VIDEO:-true}"
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_ASYNC_INFERENCE="${RTC_ASYNC_INFERENCE:-true}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-1}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-1.0}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-EXP}"
export LEROBOT_RTC_ASYNC_INFERENCE="${RTC_ASYNC_INFERENCE}"

if [[ ! -d "${POLICY_PATH}" ]]; then
    echo "Policy path does not exist: ${POLICY_PATH}" >&2
    exit 1
fi

# Keep the shell layer as a single lerobot-record invocation.
# The control-step loop stays inside lerobot's record_loop().
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
    --dataset.single_task="${TASK_NAME}"
    --dataset.push_to_hub="${PUSH_TO_HUB}"
    --dataset.video="${DATASET_VIDEO}"
    --display_data="${DISPLAY_DATA}"
    --play_sounds="${PLAY_SOUNDS}"
    --rtc.enabled="${RTC_ENABLED}"
    --rtc.async_inference="${RTC_ASYNC_INFERENCE}"
    --rtc.execution_horizon="${RTC_EXECUTION_HORIZON}"
    --rtc.max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}"
    --rtc.prefix_attention_schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
