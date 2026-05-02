# 中文简介：将已录制数据集中的动作序列回放到真实双臂 Piper 机器人上，用于复现轨迹。

lerobot-replay \
    --robot.type=bi_piper_follower \
    --robot.id=bi_piper_follower \
    --robot.left_arm_config.port=can3 \
    --robot.right_arm_config.port=can2 \
    --robot.left_arm_config.require_calibration=false \
    --robot.right_arm_config.require_calibration=false \
    --dataset.repo_id=local/fold_a_towel \
    --dataset.root=/home/wtx/data/szk/Evo-RL/datasets \
    --dataset.episode=9
