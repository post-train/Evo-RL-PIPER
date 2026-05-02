#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time
from functools import cached_property

import numpy as np

from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.piper_follower import PiperFollower, PiperFollowerConfig, PiperXFollower, PiperXFollowerConfig
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import PIPER_JOINT_ACTION_KEYS

from ..robot import Robot
from .config_bi_piper_follower import BiPiperFollowerConfig, BiPiperXFollowerConfig

logger = logging.getLogger(__name__)


class _LatestActionDispatcher:
    """Continuously send only the newest arm action on a dedicated thread."""

    def __init__(self, send_fn, name: str):
        self._send_fn = send_fn
        self._name = name
        self._condition = threading.Condition()
        self._latest_action: RobotAction | None = None
        self._stop = False
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest_action is None and not self._stop:
                    self._condition.wait()
                if self._stop:
                    return
                action = self._latest_action
                self._latest_action = None

            try:
                if action is not None:
                    self._send_fn(action)
            except BaseException as exc:  # noqa: BLE001
                self._error = exc
                logger.exception("%s failed while sending action.", self._name)
                return

    def submit(self, action: RobotAction) -> None:
        if self._error is not None:
            raise RuntimeError(f"{self._name} failed.") from self._error
        with self._condition:
            self._latest_action = dict(action)
            self._condition.notify()
        if self._error is not None:
            raise RuntimeError(f"{self._name} failed.") from self._error

    def close(self) -> None:
        with self._condition:
            self._stop = True
            self._latest_action = None
            self._condition.notify_all()
        self._thread.join(timeout=2.0)
        if self._error is not None:
            raise RuntimeError(f"{self._name} failed.") from self._error


class BiPiperFollower(Robot):
    """Bimanual PiPER/PiPER-X follower arms."""

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"
    _side_field_names = (
        "port",
        "judge_flag",
        "can_auto_init",
        "log_level",
        "startup_sleep_s",
        "speed_ratio",
        "high_follow",
        "mode_refresh_interval_s",
        "enable_on_connect",
        "enable_timeout_s",
        "calibration_scale",
        "require_calibration",
        "sync_gripper",
        "gripper_effort_default",
        "gripper_status_code",
        "cameras",
        "disable_on_disconnect",
    )

    def _build_arm_config(self, arm_config_cls, side_cfg, side: str):
        kwargs = {name: getattr(side_cfg, name) for name in self._side_field_names}
        kwargs["id"] = f"{self.config.id}_{side}" if self.config.id else None
        kwargs["calibration_dir"] = self.config.calibration_dir
        return arm_config_cls(**kwargs)

    def __init__(self, config: BiPiperFollowerConfig | BiPiperXFollowerConfig):
        super().__init__(config)
        self.config = config
        self._observation_lock = threading.Lock()
        self._observation_stop_event: threading.Event | None = None
        self._observation_ready_event = threading.Event()
        self._observation_thread: threading.Thread | None = None
        self._latest_observation: RobotObservation | None = None
        self._left_action_dispatcher: _LatestActionDispatcher | None = None
        self._right_action_dispatcher: _LatestActionDispatcher | None = None

        if config.type == "bi_piperx_follower":
            arm_config_cls = PiperXFollowerConfig
            arm_cls = PiperXFollower
        else:
            arm_config_cls = PiperFollowerConfig
            arm_cls = PiperFollower

        left_arm_config = self._build_arm_config(arm_config_cls, config.left_arm_config, "left")
        right_arm_config = self._build_arm_config(arm_config_cls, config.right_arm_config, "right")

        self.left_arm = arm_cls(left_arm_config)
        self.right_arm = arm_cls(right_arm_config)

        # Only for compatibility with other parts of the codebase that expect `robot.cameras`.
        self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras}

    @property
    def _motors_ft(self) -> dict[str, type]:
        left_arm_motors_ft = self.left_arm._motors_ft
        right_arm_motors_ft = self.right_arm._motors_ft
        return {
            **{f"left_{k}": v for k, v in left_arm_motors_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_motors_ft.items()},
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # 如果有缓存且没有广角相机，直接返回
        if hasattr(self, '_cameras_ft_cache') and self._cameras_ft_cache is not None:
            return self._cameras_ft_cache
        
        left_arm_cameras_ft = self.left_arm._cameras_ft
        right_arm_cameras_ft = self.right_arm._cameras_ft
        result = {
            **{f"left_{k}": v for k, v in left_arm_cameras_ft.items()},
            **{f"right_{k}": v for k, v in right_arm_cameras_ft.items()},
        }
        
        # 添加广角相机 - 从配置中获取特征
        if hasattr(self.config, 'cameras') and self.config.cameras:
            for name, cam_config in self.config.cameras.items():
                # 相机特征格式：(height, width, channels)
                result[name] = (cam_config.height, cam_config.width, 3)
        
        self._cameras_ft_cache = result
        return result

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def _is_teleop_send_only_mode(self) -> bool:
        return bool(getattr(self.left_arm, "_teleop_send_only_mode", False)) and bool(
            getattr(self.right_arm, "_teleop_send_only_mode", False)
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        if self._left_action_dispatcher is None:
            self._left_action_dispatcher = _LatestActionDispatcher(
                self.left_arm.send_action,
                f"{self.name}_left_send",
            )
        if self._right_action_dispatcher is None:
            self._right_action_dispatcher = _LatestActionDispatcher(
                self.right_arm.send_action,
                f"{self.name}_right_send",
            )

        if hasattr(self.config, 'cameras') and self.config.cameras:
            from lerobot.cameras import make_cameras_from_configs
            self.wide_angle_cameras = make_cameras_from_configs(self.config.cameras)
            for name, camera in self.wide_angle_cameras.items():
                camera.connect()
                logger.info(f"[BiPiperFollower] ground camera: {name} connected")

            self.cameras = {**self.left_arm.cameras, **self.right_arm.cameras, **self.wide_angle_cameras}

            self._cameras_ft_cache = None  
        else:
            self.wide_angle_cameras = {}

        if not self._is_teleop_send_only_mode():
            self._start_observation_thread()

    def set_teleop_send_only_mode(self, enabled: bool) -> None:
        self.left_arm.set_teleop_send_only_mode(enabled)
        self.right_arm.set_teleop_send_only_mode(enabled)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def setup_motors(self) -> None:
        self.left_arm.setup_motors()
        self.right_arm.setup_motors()

    def _collect_observation(self) -> RobotObservation:
        obs_dict: RobotObservation = {}
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": value for key, value in left_obs.items()})
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": value for key, value in right_obs.items()})

        if hasattr(self, "wide_angle_cameras") and self.wide_angle_cameras:
            for name, camera in self.wide_angle_cameras.items():
                try:
                    obs_dict[name] = camera.async_read()
                except Exception as e:
                    logger.warning(f"Failed to capture image from {name}: {e}")
                    obs_dict[name] = np.zeros((camera.config.height, camera.config.width, 3), dtype=np.uint8)

        return obs_dict

    def _observation_loop(self) -> None:
        stop_event = self._observation_stop_event
        if stop_event is None:
            return

        while not stop_event.is_set():
            try:
                obs_dict = self._collect_observation()
                with self._observation_lock:
                    self._latest_observation = obs_dict
                self._observation_ready_event.set()
            except Exception:
                logger.exception("BiPiperFollower observation refresh failed.")
                time.sleep(0.005)

    def _start_observation_thread(self) -> None:
        self._stop_observation_thread()
        self._latest_observation = None
        self._observation_ready_event.clear()
        self._observation_stop_event = threading.Event()
        self._observation_thread = threading.Thread(
            target=self._observation_loop,
            name=f"{self.name}_observation_loop",
            daemon=True,
        )
        self._observation_thread.start()

    def _stop_observation_thread(self) -> None:
        if self._observation_stop_event is not None:
            self._observation_stop_event.set()
        if self._observation_thread is not None and self._observation_thread.is_alive():
            self._observation_thread.join(timeout=2.0)
        self._observation_thread = None
        self._observation_stop_event = None
        self._observation_ready_event.clear()

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        if self._is_teleop_send_only_mode():
            raise RuntimeError(
                f"{self} was connected in teleop send-only mode, so bimanual follower observations are unavailable."
            )
        if not self._observation_ready_event.wait(timeout=1.0):
            logger.warning("BiPiperFollower observation snapshot not ready; falling back to synchronous read.")
            return self._collect_observation()

        with self._observation_lock:
            if self._latest_observation is None:
                return self._collect_observation()
            return dict(self._latest_observation)

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        left_action: RobotAction = {}
        right_action: RobotAction = {}
        for key, value in action.items():
            if key.startswith("left_"):
                left_action[key.removeprefix("left_")] = value
            elif key.startswith("right_"):
                right_action[key.removeprefix("right_")] = value

        if self._left_action_dispatcher is None or self._right_action_dispatcher is None:
            raise RuntimeError("BiPiperFollower action dispatchers are not initialized. Call connect() first.")

        self._left_action_dispatcher.submit(left_action)
        self._right_action_dispatcher.submit(right_action)

        prefixed_sent_action_left = {
            f"left_{key}": value for key, value in self._preview_arm_sent_action(self.left_arm, left_action).items()
        }
        prefixed_sent_action_right = {
            f"right_{key}": value
            for key, value in self._preview_arm_sent_action(self.right_arm, right_action).items()
        }
        return {**prefixed_sent_action_left, **prefixed_sent_action_right}

    def _preview_arm_sent_action(self, arm: PiperFollower, action: RobotAction) -> RobotAction:
        sent_action: RobotAction = {}

        has_all_joints = all(key in action for key in PIPER_JOINT_ACTION_KEYS)
        if has_all_joints:
            if arm._use_uncalibrated_passthrough():
                joint_targets = [action[key] for key in PIPER_JOINT_ACTION_KEYS]
            else:
                joint_targets = [arm._offset_to_target(key, action[key]) for key in PIPER_JOINT_ACTION_KEYS]
            sent_action.update(
                {key: value for key, value in zip(PIPER_JOINT_ACTION_KEYS, joint_targets, strict=True)}
            )

        if arm.config.sync_gripper and "gripper.pos" in action:
            if arm._use_uncalibrated_passthrough():
                gripper_target = action["gripper.pos"]
            else:
                gripper_target = arm._offset_to_target("gripper.pos", action["gripper.pos"])
            sent_action["gripper.pos"] = gripper_target

        return sent_action

    @check_if_not_connected
    def disconnect(self):
        self._stop_observation_thread()
        try:
            if self._left_action_dispatcher is not None:
                self._left_action_dispatcher.close()
                self._left_action_dispatcher = None
            if self._right_action_dispatcher is not None:
                self._right_action_dispatcher.close()
                self._right_action_dispatcher = None
            self.left_arm.disconnect()
        finally:
            try:
                self.right_arm.disconnect()
            finally:
                self._left_action_dispatcher = None
                self._right_action_dispatcher = None


class BiPiperXFollower(BiPiperFollower):
    config_class = BiPiperXFollowerConfig
    name = "bi_piperx_follower"
