#!/usr/bin/env python3

import time

from lerobot.robots.piper_follower import PiperFollower
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig

# Median first-frame joint pose across datasets3 episodes.
# Order:
# left_joint_1..6, left_gripper, right_joint_1..6, right_gripper
TARGET_LEFT = [-0.554, 4.500, -0.265, -5.666, 21.151, 11.765, 31.900]
TARGET_RIGHT = [0.552, 0.354, -0.337, 2.804, 22.196, -2.891, 30.600]


def to_action(target):
    return {
        "joint_1.pos": target[0],
        "joint_2.pos": target[1],
        "joint_3.pos": target[2],
        "joint_4.pos": target[3],
        "joint_5.pos": target[4],
        "joint_6.pos": target[5],
        "gripper.pos": target[6],
    }


def read_pose(arm):
    obs = arm.get_observation()
    return [
        obs.get("joint_1.pos", 0.0),
        obs.get("joint_2.pos", 0.0),
        obs.get("joint_3.pos", 0.0),
        obs.get("joint_4.pos", 0.0),
        obs.get("joint_5.pos", 0.0),
        obs.get("joint_6.pos", 0.0),
        obs.get("gripper.pos", 0.0),
    ]


def lerp_pose(start, end, ratio):
    return [s + (e - s) * ratio for s, e in zip(start, end, strict=True)]


def main():
    left = PiperFollower(
        PiperFollowerConfig(
            port="can3",
            require_calibration=False,
            enable_on_connect=True,
            speed_ratio=20,
        )
    )
    right = PiperFollower(
        PiperFollowerConfig(
            port="can2",
            require_calibration=False,
            enable_on_connect=True,
            speed_ratio=20,
        )
    )

    print("Connecting follower arms...")
    left.connect(calibrate=False)
    right.connect(calibrate=False)
    time.sleep(1.5)

    left_start = read_pose(left)
    right_start = read_pose(right)
    print("Current left :", [round(x, 3) for x in left_start])
    print("Current right:", [round(x, 3) for x in right_start])
    print("Target left  :", TARGET_LEFT)
    print("Target right :", TARGET_RIGHT)

    steps = 180
    delay_s = 0.02
    for i in range(1, steps + 1):
        ratio = i / steps
        left.send_action(to_action(lerp_pose(left_start, TARGET_LEFT, ratio)))
        right.send_action(to_action(lerp_pose(right_start, TARGET_RIGHT, ratio)))
        time.sleep(delay_s)

    time.sleep(1.0)
    print("Final left   :", [round(x, 3) for x in read_pose(left)])
    print("Final right  :", [round(x, 3) for x in read_pose(right)])

    left.disconnect()
    right.disconnect()
    print("Reset complete.")


if __name__ == "__main__":
    main()
