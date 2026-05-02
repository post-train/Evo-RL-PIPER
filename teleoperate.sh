# 中文简介：直接用双臂 leader 遥操作双臂 Piper，不录数据，只做实时手动控制。

rerun reset

lerobot-teleoperate \
    --display_data=false \
    --robot.type=bi_piper_follower \
    --robot.id=bi_piper_follower \
    --robot.left_arm_config.port=can3 \
    --robot.right_arm_config.port=can2 \
    --robot.left_arm_config.require_calibration=false \
    --robot.right_arm_config.require_calibration=false \
    --teleop.type=bi_piper_leader \
    --teleop.id=bi_piper_leader \
    --teleop.left_arm_config.port=can0 \
    --teleop.right_arm_config.port=can1 \
    --teleop.left_arm_config.require_calibration=false \
    --teleop.right_arm_config.require_calibration=false \
    --teleop.left_arm_config.manual_control=true \
    --teleop.right_arm_config.manual_control=true \
    --teleop.left_arm_config.gravity_comp_tx_ratio="[0.2,0.2,0.2,0.2,0.2,0.2]" \
    --teleop.right_arm_config.gravity_comp_tx_ratio="[0.2,0.2,0.2,0.2,0.2,0.2]"
