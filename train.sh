#!/usr/bin/env bash

# 中文简介：训练 A2A 或 Original A2A 策略的主脚本，可切换数据集、步数和主要超参数。

set -euo pipefail

POLICY_TYPE="${POLICY_TYPE:-original_a2a}"
DATASET_REPO_ID="${DATASET_REPO_ID:-datasets3_320x240}"
DATASET_ROOT="${DATASET_ROOT:-/home/szk/szk/Evo-RL/datasets3_320x240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/szk/szk/Evo-RL/outputs}"
CACHE_ROOT="${CACHE_ROOT:-/home/szk/szk/Evo-RL/.cache/huggingface}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
STEPS="${STEPS:-80000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
N_OBS_STEPS="${N_OBS_STEPS:-32}"
HORIZON="${HORIZON:-64}"
N_ACTION_STEPS="${N_ACTION_STEPS:-32}"
VISION_BACKBONE="${VISION_BACKBONE:-resnet18}"
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-AUTO}"
USE_GROUP_NORM="${USE_GROUP_NORM:-AUTO}"
ACTION_PREDICTION_MODE="${ACTION_PREDICTION_MODE:-delta}"
NORMALIZE_DELTA_TARGETS="${NORMALIZE_DELTA_TARGETS:-true}"
DELTA_STATS_EPS="${DELTA_STATS_EPS:-1e-6}"

ACTION_QUEUE_REFRESH_STEPS="${ACTION_QUEUE_REFRESH_STEPS:-8}"
MOTION_WEIGHTED_SAMPLING="${MOTION_WEIGHTED_SAMPLING:-false}"
MOTION_SAMPLING_THRESHOLD="${MOTION_SAMPLING_THRESHOLD:-0.2}"
MOTION_SAMPLING_STATIC_WEIGHT="${MOTION_SAMPLING_STATIC_WEIGHT:-0.4}"
MOTION_SAMPLING_MAX_WEIGHT="${MOTION_SAMPLING_MAX_WEIGHT:-5.0}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${CACHE_ROOT}/datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${CACHE_ROOT}/hub}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${OUTPUT_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${POLICY_TYPE}_train_$(date +%Y%m%d_%H%M%S)}"

case "${POLICY_TYPE}" in
    original_a2a)
        if [[ "${PRETRAINED_BACKBONE_WEIGHTS}" == "AUTO" ]]; then
            PRETRAINED_BACKBONE_WEIGHTS="null"
        fi
        if [[ "${USE_GROUP_NORM}" == "AUTO" ]]; then
            USE_GROUP_NORM="true"
        fi
        ;;
    *)
        if [[ "${PRETRAINED_BACKBONE_WEIGHTS}" == "AUTO" ]]; then
            PRETRAINED_BACKBONE_WEIGHTS="ResNet18_Weights.IMAGENET1K_V1"
        fi
        if [[ "${USE_GROUP_NORM}" == "AUTO" ]]; then
            USE_GROUP_NORM="false"
        fi
        ;;
esac

cmd=(
    python src/lerobot/scripts/lerobot_train.py
    --policy.type="${POLICY_TYPE}"
    --policy.push_to_hub=false
    --policy.device="${POLICY_DEVICE}"
    --policy.vision_backbone="${VISION_BACKBONE}"
    --policy.pretrained_backbone_weights="${PRETRAINED_BACKBONE_WEIGHTS}"
    --policy.use_group_norm="${USE_GROUP_NORM}"
    --policy.n_obs_steps="${N_OBS_STEPS}"
    --policy.horizon="${HORIZON}"
    --policy.n_action_steps="${N_ACTION_STEPS}"
    --policy.action_queue_refresh_steps="${ACTION_QUEUE_REFRESH_STEPS}"
    --policy.latent_dim=512
    --policy.flow_hidden_dim=512
    --policy.flow_num_layers=4
    --policy.flow_mlp_ratio=4.0
    --policy.flow_dropout=0.0
    --policy.flow_sigma=0.0
    --policy.num_sampling_steps=6
    --policy.decode_flow_latents=true
    --policy.consistency_weight=1.0
    --policy.enc_contrastive_weight=0.0
    --policy.flow_contrastive_weight=0.0
    --policy.enc_recon_weight=0.5
    --policy.flow_recon_weight=0.5
    --policy.history_hidden_dim=512
    --policy.action_ae_enc_hidden_dim=512
    --policy.action_ae_dec_hidden_dim=512
    --policy.action_ae_num_layers=4
    --policy.use_separate_rgb_encoder_per_camera=true
    --policy.imagenet_norm=true
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.video_backend=pyav
    --dataset.use_imagenet_stats=false
    --dataset.motion_weighted_sampling="${MOTION_WEIGHTED_SAMPLING}"
    --dataset.motion_sampling_threshold="${MOTION_SAMPLING_THRESHOLD}"
    --dataset.motion_sampling_static_weight="${MOTION_SAMPLING_STATIC_WEIGHT}"
    --dataset.motion_sampling_max_weight="${MOTION_SAMPLING_MAX_WEIGHT}"
    --num_workers="${NUM_WORKERS}"
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --eval_freq="${EVAL_FREQ}"
    --log_freq="${LOG_FREQ}"
    --output_dir="${OUTPUT_DIR}"
    --wandb.enable="${WANDB_ENABLE}"
)

case "${POLICY_TYPE}" in
    a2a)
        cmd+=(
            --policy.action_prediction_mode="${ACTION_PREDICTION_MODE}"
            --policy.normalize_delta_targets="${NORMALIZE_DELTA_TARGETS}"
            --policy.delta_stats_eps="${DELTA_STATS_EPS}"
        )
        ;;
    original_a2a)
        ;;
    *)
        echo "Unsupported POLICY_TYPE=${POLICY_TYPE}. Expected one of: a2a, original_a2a" >&2
        exit 1
        ;;
esac

"${cmd[@]}"
