#!/usr/bin/env bash

# 中文简介：把 datasets3 的图像统一缩放到 320x240，并输出新的数据集目录供训练使用。

set -euo pipefail

INPUT_ROOT="${INPUT_ROOT:-/home/szk/szk/Evo-RL/datasets3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
REPO_ID="${REPO_ID:-datasets3_320x240}"
TARGET_FPS="${TARGET_FPS:-30}"
TARGET_WIDTH="${TARGET_WIDTH:-320}"
TARGET_HEIGHT="${TARGET_HEIGHT:-240}"
MAX_EPISODES="${MAX_EPISODES:-}"
OVERWRITE="${OVERWRITE:-0}"
FFMPEG_PRESET="${FFMPEG_PRESET:-veryfast}"
FFMPEG_CRF="${FFMPEG_CRF:-18}"

cmd=(
  python /home/szk/szk/Evo-RL/src/lerobot/datasets/downsample_lerobot_dataset.py
  --input-root "${INPUT_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --repo-id "${REPO_ID}"
  --target-fps "${TARGET_FPS}"
  --target-width "${TARGET_WIDTH}"
  --target-height "${TARGET_HEIGHT}"
  --ffmpeg-preset "${FFMPEG_PRESET}"
  --ffmpeg-crf "${FFMPEG_CRF}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
  cmd+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

"${cmd[@]}"
