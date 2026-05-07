#!/usr/bin/env bash

# 中文简介：训练 VITA 策略的主脚本，包含动作编码器、流匹配和 EMA 等相关超参数。

set -euo pipefail

DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/szk/szk/Evo-RL/outputs}"
CACHE_ROOT="${CACHE_ROOT:-/home/szk/szk/Evo-RL/.cache/huggingface}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
STEPS="${STEPS:-80000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-6}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
N_OBS_STEPS="${N_OBS_STEPS:-8}"
HORIZON="${HORIZON:-32}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
ACTION_QUEUE_REFRESH_STEPS="${ACTION_QUEUE_REFRESH_STEPS:-8}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-ResNet18_Weights.IMAGENET1K_V1}"
USE_FROZEN_BATCH_NORM="${USE_FROZEN_BATCH_NORM:-true}"
USE_VARIATIONAL="${USE_VARIATIONAL:-false}"
ACTION_ENCODER_TYPE="${ACTION_ENCODER_TYPE:-cnn}"
ACTION_DECODER_TYPE="${ACTION_DECODER_TYPE:-simple}"
FLOW_MATCHER_NAME="${FLOW_MATCHER_NAME:-exact}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
OPTIMIZER_LR_BACKBONE="${OPTIMIZER_LR_BACKBONE:-1e-5}"
OPTIMIZER_WEIGHT_DECAY="${OPTIMIZER_WEIGHT_DECAY:-1e-6}"
USE_EMA="${USE_EMA:-true}"
EMA_POWER="${EMA_POWER:-0.75}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${CACHE_ROOT}/hub}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/vita_train_$(date +%Y%m%d_%H%M%S)}"

python src/lerobot/scripts/lerobot_train.py \
    --policy.type=vita \
    --policy.push_to_hub=false \
    --policy.device="${POLICY_DEVICE}" \
    --policy.vision_backbone="${VISION_BACKBONE}" \
    --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}" \
    --policy.use_frozen_batch_norm="${USE_FROZEN_BATCH_NORM}" \
    --policy.n_obs_steps="${N_OBS_STEPS}" \
    --policy.horizon="${HORIZON}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.action_queue_refresh_steps="${ACTION_QUEUE_REFRESH_STEPS}" \
    --policy.latent_dim=512 \
    --policy.flow_matcher_name="${FLOW_MATCHER_NAME}" \
    --policy.flow_sigma=0.0 \
    --policy.num_sampling_steps=10 \
    --policy.flow_hidden_dim=512 \
    --policy.flow_num_layers=4 \
    --policy.flow_mlp_ratio=4.0 \
    --policy.flow_dropout=0.0 \
    --policy.decode_flow_latents=true \
    --policy.consistency_weight=1.0 \
    --policy.enc_contrastive_weight=0.0 \
    --policy.flow_contrastive_weight=0.0 \
    --policy.use_variational="${USE_VARIATIONAL}" \
    --policy.action_encoder_type="${ACTION_ENCODER_TYPE}" \
    --policy.action_decoder_type="${ACTION_DECODER_TYPE}" \
    --policy.action_kl_weight=0.0 \
    --policy.flow_action_recon_weight=0.5 \
    --policy.enc_action_recon_weight=0.5 \
    --policy.action_ae_enc_hidden_dim=512 \
    --policy.action_ae_dec_hidden_dim=512 \
    --policy.action_ae_num_layers=4 \
    --policy.action_ae_num_heads=8 \
    --policy.action_ae_mlp_ratio=4.0 \
    --policy.action_ae_dropout=0.0 \
    --policy.optimizer_lr="${OPTIMIZER_LR}" \
    --policy.optimizer_lr_backbone="${OPTIMIZER_LR_BACKBONE}" \
    --policy.optimizer_weight_decay="${OPTIMIZER_WEIGHT_DECAY}" \
    --policy.use_ema="${USE_EMA}" \
    --policy.ema_power="${EMA_POWER}" \
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
