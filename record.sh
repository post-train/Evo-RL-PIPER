#!/bin/bash

# 中文简介：使用双臂 Piper 和双臂 leader 进行真人遥操作录制，生成训练数据集。

rerun reset
#python src/lerobot/utils/record_prepare.py

lerobot-record \
    --dataset.repo_id=zk-code/fold_a_towel \
    --dataset.root=/home/szk/szk/Evo-RL/datasets4 \
    --dataset.num_episodes=80 \
    --dataset.episode_time_s=90 \
    --dataset.push_to_hub=false \
    --dataset.fps=30 \
    --dataset.single_task="Fold a towel" \
    \
    --display_data=false \
    \
    --robot.type=bi_piper_follower \
    --robot.left_arm_config.cameras='{"left_arm": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30, "backend": "V4L2", "rotation": "ROTATE_180", "fourcc": "MJPG"}}' \
    --robot.right_arm_config.cameras='{"right_arm": {"type": "opencv", "index_or_path": "/dev/video0", "width": 640, "height": 480, "fps": 30, "backend": "V4L2", "rotation": "ROTATE_180", "fourcc": "MJPG"}}' \
    --robot.cameras='{"wide_angle": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30, "backend": "V4L2", "rotation": "ROTATE_180", "fourcc": "MJPG"}}' \
    --robot.id=bi_piper_follower \
    --robot.left_arm_config.port=can3 \
    --robot.right_arm_config.port=can2 \
    --robot.left_arm_config.require_calibration=false \
    --robot.right_arm_config.require_calibration=false \
    \
    --teleop.type=bi_piper_leader \
    --teleop.id=bi_piper_leader \
    --teleop.left_arm_config.port=can0 \
    --teleop.right_arm_config.port=can1 \
    --teleop.left_arm_config.require_calibration=false \
    --teleop.right_arm_config.require_calibration=false \
    --teleop.left_arm_config.manual_control=true \
    --teleop.right_arm_config.manual_control=true \
    --teleop.left_arm_config.gravity_comp_tx_ratio="[0.2,0.2,0.2,0.2,0.2,0.2]" \
    --teleop.right_arm_config.gravity_comp_tx_ratio="[0.2,0.2,0.2,0.2,0.2,0.2]" \
    --resume=false
