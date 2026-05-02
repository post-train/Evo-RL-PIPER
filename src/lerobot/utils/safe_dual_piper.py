#!/usr/bin/env python3
# -*-coding:utf8-*-

import time
from lerobot.robots.piper_follower import PiperFollower
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig

# 目标位置 (单位：度)
TARGET_CAN2 = [-1.37, -2.13, 1.87, -3.34, 22.50, -1.96]
TARGET_CAN3 = [0.00, -2.14, 2.32, 0.00, 18.60, 0.00]

def deg_to_action(deg_list, prefix=""):
    """将角度转换为 action 字典格式"""
    action = {}
    for i, angle in enumerate(deg_list, 1):
        key = f"{prefix}joint_{i}.pos" if prefix else f"joint_{i}.pos"
        action[key] = angle
    return action

def main():
    # 创建配置
    config_can2 = PiperFollowerConfig(
        port="can2",
        require_calibration=False,
        enable_on_connect=True,
        speed_ratio=30,  # 速度限制 30%
    )
    config_can3 = PiperFollowerConfig(
        port="can3",
        require_calibration=False,
        enable_on_connect=True,
        speed_ratio=30,
    )
    
    # 创建机械臂实例
    print("创建机械臂实例...")
    arm_can2 = PiperFollower(config=config_can2)
    arm_can3 = PiperFollower(config=config_can3)
    
    # 连接机械臂
    print("连接 CAN2 机械臂...")
    arm_can2.connect(calibrate=False)
    print("连接 CAN3 机械臂...")
    arm_can3.connect(calibrate=False)
    
    print("\n机械臂已连接，等待使能...")
    time.sleep(2.0)
    
    # 读取当前位置
    obs_can2 = arm_can2.get_observation()
    obs_can3 = arm_can3.get_observation()
    
    print("\n当前位置:")
    print(f"CAN2: {[obs_can2.get(f'joint_{i}.pos', 0) for i in range(1, 7)]}")
    print(f"CAN3: {[obs_can3.get(f'joint_{i}.pos', 0) for i in range(1, 7)]}")
    
    # 转换目标位置为 action 格式
    target_action_can2 = deg_to_action(TARGET_CAN2)
    target_action_can3 = deg_to_action(TARGET_CAN3)
    
    print("\n目标位置:")
    print(f"CAN2: {TARGET_CAN2}")
    print(f"CAN3: {TARGET_CAN3}")
    
    # 软件插补平滑过渡
    steps = 200
    delay = 0.02
    
    print(f"\n开始移动 (步数={steps}, 间隔={delay}s)...")
    
    cur_pos_can2 = [obs_can2.get(f'joint_{i}.pos', 0) for i in range(1, 7)]
    cur_pos_can3 = [obs_can3.get(f'joint_{i}.pos', 0) for i in range(1, 7)]
    
    for i in range(1, steps + 1):
        ratio = i / steps
        
        # 计算插补位置
        step_pos_can2 = [cur + (tar - cur) * ratio for cur, tar in zip(cur_pos_can2, TARGET_CAN2)]
        step_pos_can3 = [cur + (tar - cur) * ratio for cur, tar in zip(cur_pos_can3, TARGET_CAN3)]
        
        # 发送动作
        arm_can2.send_action(deg_to_action(step_pos_can2))
        arm_can3.send_action(deg_to_action(step_pos_can3))
        
        time.sleep(delay)
    
    # 等待机械臂停止
    print("\n等待机械臂完全停止...")
    time.sleep(2.0)
    
    # 读取最终位置
    final_obs_can2 = arm_can2.get_observation()
    final_obs_can3 = arm_can3.get_observation()
    
    print("\n最终位置:")
    print(f"CAN2: {[final_obs_can2.get(f'joint_{i}.pos', 0) for i in range(1, 7)]}")
    print(f"CAN3: {[final_obs_can3.get(f'joint_{i}.pos', 0) for i in range(1, 7)]}")
    
    print("\n双臂已安全抵达目标位置！")
    
    # 断开连接
    print("\n正在断开连接...")
    arm_can2.disconnect()
    arm_can3.disconnect()
    print("完成！")

if __name__ == "__main__":
    main()

