#!/usr/bin/env bash

# 中文简介：在单机环境下启动 ABPolicy 的异步推理/执行实验流程，便于联调整套链路。

set -euo pipefail

cd /home/szk/szk/Evo-RL

POLICY_PATH="${POLICY_PATH:-/home/szk/szk/Evo-RL/outputs/abpolicy_train_20260502_003711/checkpoints/080000/pretrained_model}"
TASK_NAME="${TASK_NAME:-Fold a towel}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
FPS="${FPS:-30}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
LOG_EVERY="${LOG_EVERY:-30}"
DRY_RUN="${DRY_RUN:-0}"

ROBOT_ID="${ROBOT_ID:-bi_piper_follower}"
CAMERA_FPS="${CAMERA_FPS:-30}"
LEFT_CAN="${LEFT_CAN:-can3}"
RIGHT_CAN="${RIGHT_CAN:-can2}"

LEFT_REQUIRE_CALIBRATION="${LEFT_REQUIRE_CALIBRATION:-false}"
RIGHT_REQUIRE_CALIBRATION="${RIGHT_REQUIRE_CALIBRATION:-false}"
LEFT_ENABLE_ON_CONNECT="${LEFT_ENABLE_ON_CONNECT:-true}"
RIGHT_ENABLE_ON_CONNECT="${RIGHT_ENABLE_ON_CONNECT:-true}"

LEFT_CAMERA_PATH="${LEFT_CAMERA_PATH:-/dev/video2}"
RIGHT_CAMERA_PATH="${RIGHT_CAMERA_PATH:-/dev/video0}"
WIDE_CAMERA_PATH="${WIDE_CAMERA_PATH:-/dev/video4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-320}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-240}"
CAMERA_BACKEND="${CAMERA_BACKEND:-V4L2}"
CAMERA_ROTATION="${CAMERA_ROTATION:-ROTATE_180}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"

if [[ ! -d "${POLICY_PATH}" ]]; then
    echo "Policy path does not exist: ${POLICY_PATH}" >&2
    exit 1
fi

ROBOT_CONFIG_PATH="$(mktemp /tmp/ab_async_robot_config.XXXXXX.json)"
cleanup() {
    rm -f "${ROBOT_CONFIG_PATH}"
}
trap cleanup EXIT

cat > "${ROBOT_CONFIG_PATH}" <<EOF
{
  "type": "bi_piper_follower",
  "id": "${ROBOT_ID}",
  "left_arm_config": {
    "port": "${LEFT_CAN}",
    "require_calibration": ${LEFT_REQUIRE_CALIBRATION},
    "enable_on_connect": ${LEFT_ENABLE_ON_CONNECT}
  },
  "right_arm_config": {
    "port": "${RIGHT_CAN}",
    "require_calibration": ${RIGHT_REQUIRE_CALIBRATION},
    "enable_on_connect": ${RIGHT_ENABLE_ON_CONNECT}
  },
  "cameras": {
    "left_left_arm": {
      "type": "opencv",
      "index_or_path": "${LEFT_CAMERA_PATH}",
      "width": ${CAMERA_WIDTH},
      "height": ${CAMERA_HEIGHT},
      "fps": ${CAMERA_FPS},
      "backend": "${CAMERA_BACKEND}",
      "rotation": "${CAMERA_ROTATION}",
      "fourcc": "${CAMERA_FOURCC}"
    },
    "right_right_arm": {
      "type": "opencv",
      "index_or_path": "${RIGHT_CAMERA_PATH}",
      "width": ${CAMERA_WIDTH},
      "height": ${CAMERA_HEIGHT},
      "fps": ${CAMERA_FPS},
      "backend": "${CAMERA_BACKEND}",
      "rotation": "${CAMERA_ROTATION}",
      "fourcc": "${CAMERA_FOURCC}"
    },
    "wide_angle": {
      "type": "opencv",
      "index_or_path": "${WIDE_CAMERA_PATH}",
      "width": ${CAMERA_WIDTH},
      "height": ${CAMERA_HEIGHT},
      "fps": ${CAMERA_FPS},
      "backend": "${CAMERA_BACKEND}",
      "rotation": "${CAMERA_ROTATION}",
      "fourcc": "${CAMERA_FOURCC}"
    }
  }
}
EOF

cmd=(
    python
    src/lerobot/scripts/run_ab_async_single_machine.py
    --policy_path="${POLICY_PATH}"
    --task="${TASK_NAME}"
    --device="${POLICY_DEVICE}"
    --fps="${FPS}"
    --execution_horizon="${EXECUTION_HORIZON}"
    --max_steps="${MAX_STEPS}"
    --log_every="${LOG_EVERY}"
    --robot="${ROBOT_CONFIG_PATH}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"
