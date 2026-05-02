#!/usr/bin/env bash

# 中文简介：启动或管理 LeRobot 异步推理相关流程的包装脚本，用于单机/分离式异步推理实验。

set -euo pipefail

MODE="${1:-}"

if [[ -z "${MODE}" ]]; then
    echo "Usage: $0 <server|client>"
    exit 1
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-60}"
POLICY_TYPE="${POLICY_TYPE:-abpolicy}"
POLICY_PATH="${POLICY_PATH:-outputs/abpolicy_train/checkpoints/last/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
if [[ "${POLICY_TYPE}" == "abpolicy" || "${POLICY_TYPE}" == "cage" ]]; then
    AGGREGATE_FN="${AGGREGATE_FN:-latest_only}"
    CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-1.0}"
else
    AGGREGATE_FN="${AGGREGATE_FN:-weighted_average}"
    CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
fi
TASK_NAME="${TASK_NAME:-fold towel}"

if [[ "${MODE}" == "server" ]]; then
    python -m lerobot.async_inference.policy_server \
        --host="${HOST}" \
        --port="${PORT}" \
        --fps="${FPS}" \
        --inference_latency="$(python - <<'PY'
fps = float(__import__("os").environ["FPS"])
print(1.0 / fps)
PY
)"
elif [[ "${MODE}" == "client" ]]; then
    python -m lerobot.async_inference.robot_client \
        --robot.type=bi_piper_follower \
        --robot.id=bi_piper_follower \
        --robot.left_arm_config.cameras='{"left_arm": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 60, "backend": "V4L2", "rotation": "ROTATE_180"}}' \
        --robot.right_arm_config.cameras='{"right_arm": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 60, "backend": "V4L2", "rotation": "ROTATE_180"}}' \
        --robot.cameras='{"wide_angle": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 60, "backend": "V4L2", "rotation": "ROTATE_180"}}' \
        --robot.left_arm_config.port=can3 \
        --robot.right_arm_config.port=can2 \
        --robot.left_arm_config.require_calibration=false \
        --robot.right_arm_config.require_calibration=false \
        --server_address="${HOST}:${PORT}" \
        --policy_type="${POLICY_TYPE}" \
        --pretrained_name_or_path="${POLICY_PATH}" \
        --policy_device="${POLICY_DEVICE}" \
        --client_device="${CLIENT_DEVICE}" \
        --fps="${FPS}" \
        --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
        --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
        --aggregate_fn_name="${AGGREGATE_FN}" \
        --task="${TASK_NAME}"
else
    echo "Unknown mode: ${MODE}"
    echo "Usage: $0 <server|client>"
    exit 1
fi
