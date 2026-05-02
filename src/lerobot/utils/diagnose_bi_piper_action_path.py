#!/usr/bin/env python3

import argparse
import time

from lerobot.robots.bi_piper_follower import BiPiperFollower
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower import PiperFollowerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose BiPiperFollower action execution on selected joints.")
    parser.add_argument("--joint", type=str, default="left_joint_2.pos")
    parser.add_argument("--delta", type=float, default=-4.0)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--send-interval-s", type=float, default=0.05)
    parser.add_argument("--settle-s", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    robot = BiPiperFollower(
        BiPiperFollowerConfig(
            id="bi_piper_follower",
            left_arm_config=PiperFollowerConfig(
                port="can3",
                require_calibration=False,
                enable_on_connect=True,
                speed_ratio=20,
                disable_on_disconnect=False,
            ),
            right_arm_config=PiperFollowerConfig(
                port="can2",
                require_calibration=False,
                enable_on_connect=True,
                speed_ratio=20,
                disable_on_disconnect=False,
            ),
        )
    )

    print("Connecting BiPiperFollower...")
    robot.connect(calibrate=False)
    time.sleep(1.0)

    before = robot.get_observation()
    print("Before:")
    inspect_keys = [
        args.joint,
        "left_joint_1.pos",
        "left_joint_2.pos",
        "left_joint_4.pos",
        "right_joint_1.pos",
        "right_joint_4.pos",
    ]
    # preserve order while dropping duplicates
    inspect_keys = list(dict.fromkeys(inspect_keys))
    for key in inspect_keys:
        print(f"  {key}: {before[key]:.3f}")

    test_action = {
        "left_joint_1.pos": before["left_joint_1.pos"],
        "left_joint_2.pos": before["left_joint_2.pos"],
        "left_joint_3.pos": before["left_joint_3.pos"],
        "left_joint_4.pos": before["left_joint_4.pos"],
        "left_joint_5.pos": before["left_joint_5.pos"],
        "left_joint_6.pos": before["left_joint_6.pos"],
        "left_gripper.pos": before["left_gripper.pos"],
        "right_joint_1.pos": before["right_joint_1.pos"],
        "right_joint_2.pos": before["right_joint_2.pos"],
        "right_joint_3.pos": before["right_joint_3.pos"],
        "right_joint_4.pos": before["right_joint_4.pos"],
        "right_joint_5.pos": before["right_joint_5.pos"],
        "right_joint_6.pos": before["right_joint_6.pos"],
        "right_gripper.pos": before["right_gripper.pos"],
    }
    test_action[args.joint] = before[args.joint] + args.delta

    print(f"Sending test action for {args.joint}: target={test_action[args.joint]:.3f} delta={args.delta:.3f}")
    for _ in range(args.repeats):
        robot.send_action(test_action)
        time.sleep(args.send_interval_s)

    time.sleep(args.settle_s)
    after = robot.get_observation()
    print("After:")
    for key in inspect_keys:
        print(f"  {key}: {after[key]:.3f}")

    print("Delta:")
    for key in inspect_keys:
        print(f"  {key}: {after[key] - before[key]:.3f}")

    robot.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
