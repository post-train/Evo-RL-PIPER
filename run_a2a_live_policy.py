#!/usr/bin/env python

from __future__ import annotations

import argparse
import copy
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.robots.bi_piper_follower import BiPiperFollower
from lerobot.robots import make_robot_from_config
from lerobot.robots.bi_piper_follower import BiPiperFollowerConfig
from lerobot.robots.piper_follower import PiperFollowerConfig
from lerobot.utils.constants import OBS_STATE, OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import init_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2A policy live on bi_piper_follower without lerobot-record.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/outputs/a2a_train_20260424_221424/checkpoints/last/pretrained_model"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("/home/szk/szk/Evo-RL/datasets3"))
    parser.add_argument("--dataset-repo-id", type=str, default="datasets3")
    parser.add_argument("--task", type=str, default="Fold a towel")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--policy-hz", type=float, default=None)
    parser.add_argument("--camera-fps", type=float, default=None)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--left-camera-path", type=str, default="/dev/video2")
    parser.add_argument("--right-camera-path", type=str, default="/dev/video0")
    parser.add_argument("--wide-camera-path", type=str, default="/dev/video4")
    parser.add_argument("--left-port", type=str, default="can3")
    parser.add_argument("--right-port", type=str, default="can2")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--run-start-pose-reset", action="store_true")
    parser.add_argument("--raw-policy-action", action="store_true")
    parser.add_argument("--sync-bimanual-grippers", action="store_true")
    parser.add_argument("--gripper-lead-steps", type=int, default=0)
    parser.add_argument("--action-gain", type=float, default=1.0)
    parser.add_argument("--min-joint-delta-deg", type=float, default=0.0)
    parser.add_argument("--min-gripper-delta", type=float, default=0.0)
    parser.add_argument("--max-joint-step-deg", type=float, default=2.0)
    parser.add_argument("--max-gripper-step", type=float, default=4.0)
    parser.add_argument("--freeze-gripper-steps", type=int, default=0)
    parser.add_argument("--left-joint-2-min-pos", type=float, default=-2.0)
    parser.add_argument("--left-joint-2-max-pos", type=float, default=127.5)
    parser.add_argument(
        "--save-first-frame-dir",
        type=Path,
        default=Path("/home/szk/szk/Evo-RL/outputs/live_policy_debug"),
    )
    return parser.parse_args()


def ensure_local_hf_cache(repo_root: Path) -> None:
    cache_root = repo_root / ".cache" / "huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "hub"))


def resolve_device(device: str) -> str:
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def build_robot(args: argparse.Namespace, camera_width: int, camera_height: int, camera_fps: int):
    left_camera_path = Path(args.left_camera_path)
    right_camera_path = Path(args.right_camera_path)
    wide_camera_path = Path(args.wide_camera_path)

    robot_cfg = BiPiperFollowerConfig(
        id="bi_piper_follower",
        left_arm_config=PiperFollowerConfig(
            port=args.left_port,
            require_calibration=False,
            enable_on_connect=True,
            disable_on_disconnect=False,
        ),
        right_arm_config=PiperFollowerConfig(
            port=args.right_port,
            require_calibration=False,
            enable_on_connect=True,
            disable_on_disconnect=False,
        ),
        cameras={
            "left_left_arm": OpenCVCameraConfig(
                index_or_path=left_camera_path,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                rotation=180,
                fourcc="MJPG",
                backend=200,
            ),
            "right_right_arm": OpenCVCameraConfig(
                index_or_path=right_camera_path,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                rotation=180,
                fourcc="MJPG",
                backend=200,
            ),
            "wide_angle": OpenCVCameraConfig(
                index_or_path=wide_camera_path,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
                rotation=180,
                fourcc="MJPG",
                backend=200,
            ),
        },
    )
    return make_robot_from_config(robot_cfg)


def compute_dataset_start_pose(ds_meta: LeRobotDatasetMetadata) -> dict[str, float]:
    info = ds_meta.info
    state_names = info["features"]["observation.state"]["names"]
    data_root = Path(ds_meta.root) / "data"
    import pyarrow.dataset as pa_ds

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


def run_start_pose_reset(robot: BiPiperFollower, target: dict[str, float]) -> None:
    obs = robot.get_observation()
    start = {k: float(obs[k]) for k in target}
    steps = 180
    for i in range(1, steps + 1):
        ratio = i / steps
        action = {k: start[k] + (target[k] - start[k]) * ratio for k in target}
        robot.send_action(action)
        time.sleep(0.02)
    time.sleep(1.0)


def get_fresh_observation(robot: BiPiperFollower) -> dict:
    return robot.get_observation()


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
    return {name: float(target) - float(obs[name]) for name, target in robot_action.items()}


def main() -> None:
    args = parse_args()
    ensure_local_hf_cache(Path(__file__).resolve().parent)
    init_logging()

    cfg, ds_meta, policy, preprocessor, postprocessor = load_policy(args)
    first_image_key = next(iter(cfg.policy.image_features))
    image_shape = cfg.policy.input_features[first_image_key].shape
    camera_height = int(image_shape[1])
    camera_width = int(image_shape[2])
    camera_fps = int(round(args.camera_fps if args.camera_fps is not None else ds_meta.fps))
    robot = build_robot(args, camera_width=camera_width, camera_height=camera_height, camera_fps=camera_fps)
    robot.connect(calibrate=False)
    if args.run_start_pose_reset:
        print("Running start pose reset...")
        run_start_pose_reset(robot, compute_dataset_start_pose(ds_meta))
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
    print("Starting live policy loop...")

    try:
        for step in range(steps):
            t0 = time.perf_counter()
            obs = get_fresh_observation(robot)

            if step == 0:
                args.save_first_frame_dir.mkdir(parents=True, exist_ok=True)
                for key in ("left_left_arm", "right_right_arm", "wide_angle"):
                    if key in obs:
                        image = Image.fromarray(obs[key])
                        image.save(args.save_first_frame_dir / f"{key}.png")
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
                    robot_type=robot.robot_type,
                )

            robot_action = make_robot_action(policy_action, ds_meta.features)
            if args.sync_bimanual_grippers:
                mean_gripper = 0.5 * (robot_action["left_gripper.pos"] + robot_action["right_gripper.pos"])
                robot_action["left_gripper.pos"] = mean_gripper
                robot_action["right_gripper.pos"] = mean_gripper

            sent_action = robot.send_action(robot_action)
            action_deltas = compute_action_deltas(sent_action, obs)

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
        robot.disconnect()
        print("Stopped.")


if __name__ == "__main__":
    main()
