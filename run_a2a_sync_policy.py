#!/usr/bin/env python

from __future__ import annotations

import os
from pathlib import Path

_CACHE_ROOT = Path(__file__).resolve().parent / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_CACHE_ROOT / "hub"))

import argparse
import copy
import time
from contextlib import nullcontext

import numpy as np
import pyarrow.dataset as pa_ds
import torch
from PIL import Image

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig
from lerobot.utils.constants import OBS_STATE, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2A policy live with synchronous dual Piper execution.")
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", type=str, required=True)
    parser.add_argument("--task", type=str, default="Fold a towel")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--left-camera-path", type=str, default="/dev/video2")
    parser.add_argument("--right-camera-path", type=str, default="/dev/video0")
    parser.add_argument("--wide-camera-path", type=str, default="/dev/video4")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--run-start-pose-reset", action="store_true")
    parser.add_argument("--start-reset-steps", type=int, default=180)
    parser.add_argument("--start-reset-delay-s", type=float, default=0.02)
    parser.add_argument(
        "--save-first-frame-dir",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/outputs/live_policy_debug_sync"),
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def build_camera(path: str, width: int, height: int, fps: int) -> OpenCVCamera:
    return OpenCVCamera(
        OpenCVCameraConfig(
            index_or_path=Path(path),
            width=width,
            height=height,
            fps=fps,
            rotation=180,
            fourcc="MJPG",
            backend=200,
        )
    )


def build_arm(port: str) -> PiperFollower:
    return PiperFollower(
        PiperFollowerConfig(
            port=port,
            require_calibration=False,
            enable_on_connect=True,
            disable_on_disconnect=False,
        )
    )


def compute_dataset_start_pose(ds_meta: LeRobotDatasetMetadata) -> dict[str, float]:
    state_names = ds_meta.info["features"]["observation.state"]["names"]
    data_root = Path(ds_meta.root) / "data"
    table = pa_ds.dataset(sorted(data_root.glob("*/*.parquet")), format="parquet").to_table(
        columns=["frame_index", OBS_STATE]
    )
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    states = np.asarray(table[OBS_STATE].to_pylist(), dtype=np.float32)
    first_states = states[frame_index == 0]
    if len(first_states) == 0:
        raise ValueError("Failed to compute dataset start pose: no frame_index==0 rows found.")
    mean_state = first_states.mean(axis=0)
    return {name: float(mean_state[idx]) for idx, name in enumerate(state_names)}


def run_start_pose_reset(
    left_arm: PiperFollower,
    right_arm: PiperFollower,
    target: dict[str, float],
    steps: int,
    delay_s: float,
) -> None:
    left_obs = left_arm.get_observation()
    right_obs = right_arm.get_observation()

    start = {}
    for key in target:
        if key.startswith("left_"):
            start[key] = float(left_obs[key.removeprefix("left_")])
        elif key.startswith("right_"):
            start[key] = float(right_obs[key.removeprefix("right_")])

    for i in range(1, steps + 1):
        ratio = i / steps
        left_action = {}
        right_action = {}
        for key, value in target.items():
            interp = start[key] + (value - start[key]) * ratio
            if key.startswith("left_"):
                left_action[key.removeprefix("left_")] = interp
            elif key.startswith("right_"):
                right_action[key.removeprefix("right_")] = interp
        left_arm.send_action(left_action)
        right_arm.send_action(right_action)
        time.sleep(delay_s)
    time.sleep(1.0)


def collect_observation(
    left_arm: PiperFollower,
    right_arm: PiperFollower,
    left_camera: OpenCVCamera,
    right_camera: OpenCVCamera,
    wide_camera: OpenCVCamera,
) -> dict:
    left_obs = left_arm.get_observation()
    right_obs = right_arm.get_observation()
    obs = {f"left_{k}": v for k, v in left_obs.items()}
    obs.update({f"right_{k}": v for k, v in right_obs.items()})
    obs["left_left_arm"] = left_camera.async_read()
    obs["right_right_arm"] = right_camera.async_read()
    obs["wide_angle"] = wide_camera.async_read()
    return obs


def load_policy(args: argparse.Namespace):
    cfg = TrainPipelineConfig.from_pretrained(args.checkpoint_path)
    cfg = copy.deepcopy(cfg)
    cfg.policy.device = resolve_device(args.device)
    cfg.policy.use_amp = bool(args.use_amp)
    cfg.policy.pretrained_path = args.checkpoint_path
    cfg.dataset.root = args.dataset_root
    cfg.dataset.repo_id = args.dataset_repo_id

    ds_meta = LeRobotDatasetMetadata(cfg.dataset.repo_id, root=cfg.dataset.root)
    policy = make_policy(cfg.policy, ds_meta=ds_meta, rename_map=cfg.rename_map)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "normalizer_processor": {
                "stats": ds_meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
        postprocessor_overrides={
            "unnormalizer_processor": {
                "stats": ds_meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            }
        },
    )
    return cfg, ds_meta, policy, preprocessor, postprocessor


def compute_action_deltas(robot_action: dict[str, float], obs: dict[str, float]) -> dict[str, float]:
    return {name: float(target) - float(obs[name]) for name, target in robot_action.items() if name in obs}


def main() -> None:
    args = parse_args()
    init_logging()

    cfg, ds_meta, policy, preprocessor, postprocessor = load_policy(args)
    first_image_key = next(iter(cfg.policy.image_features))
    image_shape = cfg.policy.input_features[first_image_key].shape
    camera_height = int(image_shape[1])
    camera_width = int(image_shape[2])
    camera_fps = int(round(args.camera_fps if args.camera_fps is not None else ds_meta.fps))

    left_arm = build_arm(args.left_port)
    right_arm = build_arm(args.right_port)
    left_camera = build_camera(args.left_camera_path, camera_width, camera_height, camera_fps)
    right_camera = build_camera(args.right_camera_path, camera_width, camera_height, camera_fps)
    wide_camera = build_camera(args.wide_camera_path, camera_width, camera_height, camera_fps)

    left_arm.connect(calibrate=False)
    right_arm.connect(calibrate=False)
    left_camera.connect()
    right_camera.connect()
    wide_camera.connect()

    if args.run_start_pose_reset:
        print("Running start pose reset...")
        run_start_pose_reset(
            left_arm,
            right_arm,
            compute_dataset_start_pose(ds_meta),
            steps=args.start_reset_steps,
            delay_s=args.start_reset_delay_s,
        )

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    device = torch.device(cfg.policy.device)
    period_s = 1.0 / args.control_hz
    steps = max(1, int(args.duration_s * args.control_hz))

    print(
        f"device={cfg.policy.device} control_hz={args.control_hz} "
        f"camera_fps={camera_fps} steps={steps}"
    )
    print("Starting synchronous live policy loop...")

    try:
        for step in range(steps):
            t0 = time.perf_counter()
            obs = collect_observation(left_arm, right_arm, left_camera, right_camera, wide_camera)

            if step == 0:
                args.save_first_frame_dir.mkdir(parents=True, exist_ok=True)
                for key in ("left_left_arm", "right_right_arm", "wide_angle"):
                    Image.fromarray(obs[key]).save(args.save_first_frame_dir / f"{key}.png")
                print(f"saved_first_frames={args.save_first_frame_dir}")

            observation_frame = build_dataset_frame(ds_meta.features, obs, prefix=OBS_STR)
            with (
                torch.inference_mode(),
                torch.autocast(device_type=device.type) if device.type == "cuda" and cfg.policy.use_amp else nullcontext(),
            ):
                policy_action = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=cfg.policy.use_amp,
                    task=args.task,
                    robot_type="bi_piper_follower_sync",
                )

            robot_action = make_robot_action(policy_action, ds_meta.features)
            left_action = {k.removeprefix("left_"): v for k, v in robot_action.items() if k.startswith("left_")}
            right_action = {k.removeprefix("right_"): v for k, v in robot_action.items() if k.startswith("right_")}
            left_arm.send_action(left_action)
            right_arm.send_action(right_action)

            action_deltas = compute_action_deltas(robot_action, obs)
            if step % 10 == 0:
                max_delta_key = max(action_deltas, key=lambda k: abs(action_deltas[k]))
                print(
                    f"step={step} "
                    f"left_j1_obs={obs['left_joint_1.pos']:.3f} "
                    f"left_j1_tgt={robot_action['left_joint_1.pos']:.3f} "
                    f"left_j2_obs={obs['left_joint_2.pos']:.3f} "
                    f"left_j2_tgt={robot_action['left_joint_2.pos']:.3f} "
                    f"left_gripper_obs={obs['left_gripper.pos']:.3f} "
                    f"left_gripper_tgt={robot_action['left_gripper.pos']:.3f} "
                    f"right_j4_obs={obs['right_joint_4.pos']:.3f} "
                    f"right_j4_tgt={robot_action['right_joint_4.pos']:.3f} "
                    f"right_gripper_obs={obs['right_gripper.pos']:.3f} "
                    f"right_gripper_tgt={robot_action['right_gripper.pos']:.3f} "
                    f"max_delta_joint={max_delta_key} "
                    f"max_delta={action_deltas[max_delta_key]:.3f}"
                )

            dt = time.perf_counter() - t0
            if dt > period_s and step % 10 == 0:
                print(f"warning step={step} loop_dt={dt:.3f}s budget={period_s:.3f}s")
            time.sleep(max(period_s - dt, 0.0))
    finally:
        left_arm.disconnect()
        right_arm.disconnect()
        if left_camera.is_connected:
            left_camera.disconnect()
        if right_camera.is_connected:
            right_camera.disconnect()
        if wide_camera.is_connected:
            wide_camera.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
