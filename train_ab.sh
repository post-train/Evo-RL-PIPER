#!/usr/bin/env bash

# 中文简介：训练 ABPolicy 策略的主脚本，默认面向当前处理后的双臂 Piper 数据集。

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/szk/szk/Evo-RL/outputs}"
CACHE_ROOT="${CACHE_ROOT:-/home/szk/szk/Evo-RL/.cache/huggingface}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
STEPS="${STEPS:-80000}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
LOG_FREQ="${LOG_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"

N_OBS_STEPS="${N_OBS_STEPS:-8}"
ACTION_HISTORY_HORIZON="${ACTION_HISTORY_HORIZON:-8}"
ACTION_HORIZON="${ACTION_HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
NUM_CTRL_POINTS="${NUM_CTRL_POINTS:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"

IMAGE_ENCODER_NAME="${IMAGE_ENCODER_NAME:-facebook/dinov2-base}"
IMAGE_BACKBONE_USED_LAYERS="${IMAGE_BACKBONE_USED_LAYERS:-8}"
OBS_DIM="${OBS_DIM:-512}"
IMG_SIZE="${IMG_SIZE:-224}"
IMG_PATCH_SIZE="${IMG_PATCH_SIZE:-14}"
IMAGE_CROP_H="${IMAGE_CROP_H:-240}"
IMAGE_CROP_W="${IMAGE_CROP_W:-240}"

OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
OPTIMIZER_PERCEIVER_LR="${OPTIMIZER_PERCEIVER_LR:-1e-4}"
QPOS_NOISE_STD="${QPOS_NOISE_STD:-0.01}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${CACHE_ROOT}/hub}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/abpolicy_train_$(date +%Y%m%d_%H%M%S)}"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=abpolicy \
    --policy.push_to_hub=false \
    --policy.device="${POLICY_DEVICE}" \
    --policy.n_obs_steps="${N_OBS_STEPS}" \
    --policy.action_history_horizon="${ACTION_HISTORY_HORIZON}" \
    --policy.action_horizon="${ACTION_HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.bspline_num_ctrl_points="${NUM_CTRL_POINTS}" \
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}" \
    --policy.image_encoder_name="${IMAGE_ENCODER_NAME}" \
    --policy.image_backbone_used_layers="${IMAGE_BACKBONE_USED_LAYERS}" \
    --policy.obs_dim="${OBS_DIM}" \
    --policy.img_size="${IMG_SIZE}" \
    --policy.img_patch_size="${IMG_PATCH_SIZE}" \
    --policy.image_crop_shape="[${IMAGE_CROP_H}, ${IMAGE_CROP_W}]" \
    --policy.optimizer_lr="${OPTIMIZER_LR}" \
    --policy.optimizer_perceiver_lr="${OPTIMIZER_PERCEIVER_LR}" \
    --policy.qpos_noise_std="${QPOS_NOISE_STD}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=false \
    --num_workers="${NUM_WORKERS}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable="${WANDB_ENABLE}"
