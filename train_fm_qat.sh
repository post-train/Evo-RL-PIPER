#!/usr/bin/env bash

# 中文简介：训练 Flow Matching QAT(INT8) 策略的主脚本，沿用 train_fm.sh 的超参数，并启用独立的 QAT policy。

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/szk/szk/Evo-RL/outputs}"
CACHE_ROOT="${CACHE_ROOT:-/home/szk/szk/Evo-RL/.cache/huggingface}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
STEPS="${STEPS:-100000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
N_OBS_STEPS="${N_OBS_STEPS:-8}"
HORIZON="${HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-AUTO}"
USE_GROUP_NORM="${USE_GROUP_NORM:-AUTO}"
BACKBONE_LR_SCALE="${BACKBONE_LR_SCALE:-1.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
SOLVER_TYPE="${SOLVER_TYPE:-euler}"
USE_EMA="${USE_EMA:-true}"
EMA_POWER="${EMA_POWER:-0.75}"
TRT_EXPORT_ON_SAVE="${TRT_EXPORT_ON_SAVE:-true}"
TRT_REFERENCE_PRETRAINED_PATH="${TRT_REFERENCE_PRETRAINED_PATH:-}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${CACHE_ROOT}/hub}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/flow_matching_qat_train_$(date +%Y%m%d_%H%M%S)}"

if [[ "${PRETRAINED_BACKBONE_WEIGHTS}" == "AUTO" ]]; then
    PRETRAINED_BACKBONE_WEIGHTS="ResNet18_Weights.IMAGENET1K_V1"
fi
if [[ "${USE_GROUP_NORM}" == "AUTO" ]]; then
    USE_GROUP_NORM="false"
fi

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=flow_matching_qat \
    --policy.push_to_hub=false \
    --policy.device="${POLICY_DEVICE}" \
    --policy.vision_backbone="${VISION_BACKBONE}" \
    --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}" \
    --policy.use_group_norm="${USE_GROUP_NORM}" \
    --policy.n_obs_steps="${N_OBS_STEPS}" \
    --policy.horizon="${HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.backbone_lr_scale="${BACKBONE_LR_SCALE}" \
    --policy.num_inference_steps="${NUM_INFERENCE_STEPS}" \
    --policy.solver_type="${SOLVER_TYPE}" \
    --policy.use_ema="${USE_EMA}" \
    --policy.ema_power="${EMA_POWER}" \
    --policy.trt_export_on_save="${TRT_EXPORT_ON_SAVE}" \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.video_backend=pyav \
    --dataset.use_imagenet_stats=true \
    --num_workers="${NUM_WORKERS}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --eval_freq="${EVAL_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --output_dir="${OUTPUT_DIR}" \
    --wandb.enable="${WANDB_ENABLE}" \
    ${TRT_REFERENCE_PRETRAINED_PATH:+--policy.trt_reference_pretrained_path="${TRT_REFERENCE_PRETRAINED_PATH}"}
