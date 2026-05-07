#!/bin/bash

# =============================================================================
# FM Rollout Sweep Analysis Script
# =============================================================================
# 用途: 离线分析 Flow Matching 模型在不同推理参数下的动作抖动情况
#
# 功能:
#   - 扫描不同的推理步数 (num_inference_steps)
#   - 扫描不同的求解器 (euler, dopri5)
#   - 扫描不同的裁剪设置 (clip_sample)
#   - 生成动作轨迹对比图，分析抖动程度
#
# 输出:
#   - 动作轨迹可视化图
#   - 不同参数组合的抖动指标对比
#
# 使用场景:
#   - 调试推理时的动作抖动问题
#   - 选择最优的推理参数组合
#   - 评估模型在不同设置下的表现
# =============================================================================

HF_HOME=/tmp/hf_rollout \
  HF_DATASETS_CACHE=/tmp/hf_rollout/datasets \
  HUGGINGFACE_HUB_CACHE=/tmp/hf_rollout/hub \
  MPLCONFIGDIR=/tmp/mpl_rollout \
  python src/lerobot/scripts/analyze_fm_rollout_sweep.py \
    --checkpoint outputs/flow_matching_train_20260430_170525/checkpoints/080000/pretrained_model \
    --device cpu \
    --episode-index 0 \
    --start-frame 0 \
    --length 96 \
    --steps-list 5,10,20 \
    --solver-list euler,dopri5 \
    --clip-list true,false