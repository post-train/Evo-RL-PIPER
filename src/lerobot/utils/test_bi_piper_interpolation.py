#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from lerobot.robots.bi_piper_follower import BiPiperFollower
from lerobot.robots.bi_piper_follower.config_bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower import PiperFollowerConfig


ALL_ACTION_KEYS = [
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_joint_6.pos",
    "left_gripper.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_joint_6.pos",
    "right_gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether the bi-piper hardware/sdK follows sparse commands smoothly. "
            "Alternates between two far-apart targets at 1Hz and records joint states."
        )
    )
    parser.add_argument("--joint", type=str, default="left_joint_2.pos")
    parser.add_argument("--delta", type=float, default=20.0)
    parser.add_argument("--switch-hz", type=float, default=1.0)
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--sample-hz", type=float, default=50.0)
    parser.add_argument("--warmup-s", type=float, default=1.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/outputs/bi_piper_interpolation_test.csv"),
    )
    return parser.parse_args()


def build_robot(args: argparse.Namespace) -> BiPiperFollower:
    return BiPiperFollower(
        BiPiperFollowerConfig(
            id="bi_piper_follower",
            left_arm_config=PiperFollowerConfig(
                port=args.left_port,
                require_calibration=False,
                enable_on_connect=True,
                speed_ratio=20,
                disable_on_disconnect=False,
            ),
            right_arm_config=PiperFollowerConfig(
                port=args.right_port,
                require_calibration=False,
                enable_on_connect=True,
                speed_ratio=20,
                disable_on_disconnect=False,
            ),
        )
    )


def make_base_action(obs: dict[str, float]) -> dict[str, float]:
    return {key: float(obs[key]) for key in ALL_ACTION_KEYS}


def summarize(records: list[dict[str, float]], joint: str) -> dict[str, float | str]:
    joint_values = [float(r[joint]) for r in records]
    time_values = [float(r["t"]) for r in records]
    target_values = [float(r["target"]) for r in records]
    if len(joint_values) < 3:
        return {"classification": "insufficient_samples"}

    velocities = []
    accelerations = []
    turning_points = 0
    prev_sign = 0

    for i in range(1, len(joint_values)):
        dt = max(1e-6, time_values[i] - time_values[i - 1])
        velocities.append((joint_values[i] - joint_values[i - 1]) / dt)

    for i in range(1, len(velocities)):
        accelerations.append(velocities[i] - velocities[i - 1])
        sign = 1 if velocities[i] > 1e-6 else -1 if velocities[i] < -1e-6 else 0
        if prev_sign != 0 and sign != 0 and sign != prev_sign:
            turning_points += 1
        if sign != 0:
            prev_sign = sign

    target_changes = sum(1 for i in range(1, len(target_values)) if target_values[i] != target_values[i - 1])
    moving_samples = sum(1 for v in velocities if abs(v) > 0.5)
    moving_ratio = moving_samples / max(1, len(velocities))
    peak_velocity = max(abs(v) for v in velocities) if velocities else 0.0
    median_abs_velocity = statistics.median(abs(v) for v in velocities) if velocities else 0.0
    peak_acceleration = max(abs(a) for a in accelerations) if accelerations else 0.0

    if moving_ratio > 0.6 and turning_points <= target_changes + 2:
        classification = "smooth_continuous_follow"
    elif moving_ratio < 0.15:
        classification = "mostly_stopped_or_step_hold"
    else:
        classification = "mixed_or_piecewise_follow"

    return {
        "classification": classification,
        "moving_ratio": moving_ratio,
        "turning_points": float(turning_points),
        "target_changes": float(target_changes),
        "peak_velocity": peak_velocity,
        "median_abs_velocity": median_abs_velocity,
        "peak_acceleration": peak_acceleration,
        "observed_min": min(joint_values),
        "observed_max": max(joint_values),
        "target_min": min(target_values),
        "target_max": max(target_values),
    }


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    robot = build_robot(args)
    print("Connecting BiPiperFollower...")
    robot.connect(calibrate=False)
    try:
        time.sleep(args.warmup_s)
        before = robot.get_observation()
        base_action = make_base_action(before)
        low_target = float(before[args.joint])
        high_target = low_target + args.delta
        period_s = 1.0 / args.switch_hz
        sample_period_s = 1.0 / args.sample_hz
        total_duration_s = args.cycles * period_s

        print("Before:")
        print(f"  joint={args.joint}")
        print(f"  current={low_target:.3f}")
        print(f"  target_a={low_target:.3f}")
        print(f"  target_b={high_target:.3f}")
        print(f"  switch_hz={args.switch_hz:.3f}")
        print(f"  sample_hz={args.sample_hz:.3f}")
        print(f"  cycles={args.cycles}")

        records: list[dict[str, float]] = []
        start_t = time.perf_counter()
        next_sample_t = start_t

        while True:
            now = time.perf_counter()
            elapsed = now - start_t
            if elapsed > total_duration_s:
                break

            phase_index = int(elapsed // period_s)
            target = low_target if (phase_index % 2 == 0) else high_target
            action = dict(base_action)
            action[args.joint] = target
            robot.send_action(action)

            if now >= next_sample_t:
                obs = robot.get_observation()
                records.append(
                    {
                        "t": elapsed,
                        "phase_index": phase_index,
                        "target": target,
                        args.joint: float(obs[args.joint]),
                    }
                )
                next_sample_t += sample_period_s

            time.sleep(0.001)

        time.sleep(args.settle_s)
        after = robot.get_observation()

        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["t", "phase_index", "target", args.joint])
            writer.writeheader()
            writer.writerows(records)

        summary = summarize(records, args.joint)
        print("After:")
        print(f"  current={float(after[args.joint]):.3f}")
        print(f"  delta={float(after[args.joint]) - low_target:.3f}")
        print("Summary:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"saved_csv: {args.output_csv}")
    finally:
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
