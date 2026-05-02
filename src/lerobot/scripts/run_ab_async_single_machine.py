#!/usr/bin/env python

import contextlib
import logging
import math
import os
import queue
import threading
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor import PolicyAction
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_piper_follower,
    make_robot_from_config,
    piper_follower,
)
from lerobot.robots.robot import Robot
from lerobot.scripts.recording_hil import PolicyAction as _UnusedPolicyActionAlias  # noqa: F401
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device, init_logging

from lerobot.async_inference.helpers import make_lerobot_observation, resize_robot_observation_image


REPO_ROOT = Path(__file__).resolve().parents[3]
HF_CACHE_ROOT = REPO_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_ROOT / "hub"))
os.environ.setdefault("DATASETS_CACHE", str(HF_CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logger = logging.getLogger(__name__)


@dataclass
class ABAsyncSingleMachineConfig:
    robot: RobotConfig
    policy_path: str
    task: str = ""
    device: str = "cuda"
    fps: int = 30
    execution_horizon: int = 8
    max_steps: int = 0
    log_every: int = 30
    warmup_timeout_s: float = 10.0


@dataclass
class InferenceRequest:
    obs_timestep: int
    history_snapshot: list[dict[str, torch.Tensor]]


@dataclass
class InferenceResult:
    obs_timestep: int
    ctrl_points: torch.Tensor | None
    full_actions: torch.Tensor | None
    inference_latency_s: float


@dataclass
class PendingAction:
    timestep: int
    action_dict: dict[str, float]
    action_tensor: torch.Tensor


class ABAsyncSingleMachineRunner:
    def __init__(self, cfg: ABAsyncSingleMachineConfig):
        self.cfg = cfg
        self.device = get_safe_torch_device(cfg.device)
        self.policy_path = Path(cfg.policy_path).expanduser().resolve()
        self.stop_event = threading.Event()
        self.request_queue: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
        self.result_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=1)
        self.worker_thread: threading.Thread | None = None

        self.policy, self.preprocessor, self.postprocessor = self._load_policy_stack()
        self.robot = make_robot_from_config(cfg.robot)
        self.action_names = list(self.robot.action_features.keys())
        self.action_dim = len(self.action_names)
        self.action_features = {ACTION: {"names": self.action_names}}
        self.lerobot_features = self._build_lerobot_features(self.robot)

        self.obs_history: deque[tuple[int, dict[str, torch.Tensor]]] = deque(maxlen=self.policy.config.n_obs_steps)
        self.executed_actions: deque[torch.Tensor] = deque(
            maxlen=self.policy.config.horizon + self.policy.config.n_action_steps + 64
        )
        self.pending_actions: deque[PendingAction] = deque()
        self.inflight_request: InferenceRequest | None = None
        self.last_sent_action_dict: dict[str, float] | None = None
        self.last_sent_action_tensor: torch.Tensor | None = None

    def _load_policy_stack(self):
        policy_cfg = PreTrainedConfig.from_pretrained(self.policy_path)
        policy_cfg = deepcopy(policy_cfg)
        policy_cfg.pretrained_path = self.policy_path
        policy_cfg.device = self.device.type if self.device.index is None else str(self.device)
        policy = make_policy(policy_cfg)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=policy_cfg.pretrained_path,
        )
        policy.eval()
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
        return policy, preprocessor, postprocessor

    def _build_lerobot_features(self, robot: Robot) -> dict[str, dict]:
        from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features

        return map_robot_keys_to_lerobot_features(robot)

    def _raw_observation_to_policy_input(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        lerobot_obs = make_lerobot_observation(raw_observation, self.lerobot_features)
        policy_obs: dict[str, Any] = {
            OBS_STATE: torch.as_tensor(lerobot_obs[OBS_STATE], dtype=torch.float32),
            "task": self.cfg.task,
            "robot_type": self.robot.robot_type,
        }

        for image_key, feature in self.policy.config.image_features.items():
            if image_key not in lerobot_obs:
                raise KeyError(f"Missing image key `{image_key}` in robot observation.")
            image = torch.as_tensor(lerobot_obs[image_key], dtype=torch.float32)
            policy_obs[image_key] = resize_robot_observation_image(image, feature.shape)

        return policy_obs

    def _prepare_single_step_observation(self, raw_observation: dict[str, Any]) -> dict[str, torch.Tensor]:
        policy_obs = self._raw_observation_to_policy_input(raw_observation)
        processed = self.preprocessor(policy_obs)
        prepared = self.policy._prepare_model_batch(processed, for_training=False)
        return {
            OBS_STATE: prepared[OBS_STATE].detach().clone(),
            OBS_IMAGES: prepared[OBS_IMAGES].detach().clone(),
        }

    def _build_history_snapshot(self) -> list[dict[str, torch.Tensor]]:
        return [
            {
                OBS_STATE: item[OBS_STATE].detach().clone(),
                OBS_IMAGES: item[OBS_IMAGES].detach().clone(),
            }
            for _, item in self.obs_history
        ]

    def _build_queue_batch(self, history_snapshot: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            OBS_STATE: torch.stack([item[OBS_STATE] for item in history_snapshot], dim=1),
            OBS_IMAGES: torch.stack([item[OBS_IMAGES] for item in history_snapshot], dim=1),
        }

    def _vector_from_action_dict(self, action: dict[str, float]) -> torch.Tensor:
        return torch.tensor([float(action[name]) for name in self.action_names], dtype=torch.float32)

    def _hold_action_from_observation(self, raw_observation: dict[str, Any]) -> tuple[dict[str, float], torch.Tensor]:
        action = {name: float(raw_observation[name]) for name in self.action_names}
        return action, self._vector_from_action_dict(action)

    def _enqueue_latest_result(self, result: InferenceResult) -> None:
        while True:
            try:
                self.result_queue.put_nowait(result)
                return
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

    def _start_worker(self) -> None:
        if self.worker_thread is not None:
            return

        def _worker() -> None:
            while not self.stop_event.is_set():
                try:
                    request = self.request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if request is None:
                    return

                queue_batch = self._build_queue_batch(request.history_snapshot)
                start_t = time.perf_counter()
                with torch.inference_mode():
                    with (
                        torch.autocast(device_type=self.device.type)
                        if self.device.type == "cuda" and self.policy.config.use_amp
                        else contextlib.nullcontext()
                    ):
                        latent = self.policy.model.generate_ctrl_points(queue_batch)
                        if self.policy.config.use_bspline:
                            ctrl_points = self.policy._denormalize_action_like(latent).squeeze(0).detach().cpu()
                            full_actions = None
                        else:
                            ctrl_points = None
                            full_actions = self.policy._denormalize_action_like(latent).squeeze(0).detach().cpu()

                self._enqueue_latest_result(
                    InferenceResult(
                        obs_timestep=request.obs_timestep,
                        ctrl_points=ctrl_points,
                        full_actions=full_actions,
                        inference_latency_s=time.perf_counter() - start_t,
                    )
                )

        self.worker_thread = threading.Thread(target=_worker, name="abpolicy-async-worker", daemon=True)
        self.worker_thread.start()

    def _stop_worker(self) -> None:
        self.stop_event.set()
        try:
            self.request_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None

    def _maybe_submit_request(self, current_step: int) -> None:
        if len(self.obs_history) < self.policy.config.n_obs_steps:
            return
        if self.inflight_request is not None:
            return
        if len(self.pending_actions) > self.cfg.execution_horizon:
            return

        latest_obs_step = self.obs_history[-1][0]
        request = InferenceRequest(
            obs_timestep=latest_obs_step,
            history_snapshot=self._build_history_snapshot(),
        )
        self.request_queue.put_nowait(request)
        self.inflight_request = request
        logger.info(
            "submitted async inference step=%d obs_step=%d queue_len=%d",
            current_step,
            latest_obs_step,
            len(self.pending_actions),
        )

    def _get_prefix_tensor(self, needed: int, fallback_action: torch.Tensor) -> torch.Tensor:
        prefix = list(self.executed_actions)[-needed:]
        if len(prefix) < needed:
            pad = [fallback_action.detach().clone() for _ in range(needed - len(prefix))]
            prefix = pad + prefix
        return torch.stack(prefix, dim=0)

    def _postprocess_chunk(self, chunk: torch.Tensor) -> list[dict[str, float]]:
        actions: list[dict[str, float]] = []
        for idx in range(chunk.shape[0]):
            processed_action: PolicyAction = self.postprocessor(chunk[idx].unsqueeze(0))
            actions.append(make_robot_action(processed_action, self.action_features))
        return actions

    def _rebuild_chunk_from_result(
        self,
        result: InferenceResult,
        current_step: int,
        fallback_action: torch.Tensor,
    ) -> torch.Tensor | None:
        delay_steps = max(0, current_step - result.obs_timestep)

        if self.policy.config.use_bspline:
            if result.ctrl_points is None:
                return None

            n_prefix = self.policy.config.action_history_horizon + delay_steps
            prefix = self._get_prefix_tensor(n_prefix, fallback_action).unsqueeze(0).to(self.device)
            ctrl_points = result.ctrl_points.unsqueeze(0).to(self.device)

            with torch.inference_mode():
                refit_ctrl = self.policy.model.projector.refit_prefix_w(
                    prefix,
                    ctrl_points,
                    n_prefix=n_prefix,
                    n_free=self.policy.config.refit_n_free,
                    last_pt_weight=self.policy.config.refit_last_pt_weight,
                )
                full_actions = self.policy.model.projector.rebuild_batch(refit_ctrl).squeeze(0).detach().cpu()

            start = self.policy.config.action_history_horizon + delay_steps
            end = min(start + self.policy.config.n_action_steps, full_actions.shape[0])
            if start >= end:
                return None
            return full_actions[start:end]

        if result.full_actions is None:
            return None
        start = delay_steps
        end = min(start + self.policy.config.n_action_steps, result.full_actions.shape[0])
        if start >= end:
            return None
        return result.full_actions[start:end]

    def _drain_results(self, current_step: int, fallback_action: torch.Tensor) -> None:
        latest_result: InferenceResult | None = None
        while True:
            try:
                latest_result = self.result_queue.get_nowait()
            except queue.Empty:
                break

        if latest_result is None:
            return

        self.inflight_request = None
        action_chunk = self._rebuild_chunk_from_result(latest_result, current_step, fallback_action)
        if action_chunk is None or action_chunk.shape[0] == 0:
            logger.warning(
                "dropping stale async result obs_step=%d current_step=%d latency=%.2fms",
                latest_result.obs_timestep,
                current_step,
                latest_result.inference_latency_s * 1e3,
            )
            return

        action_dicts = self._postprocess_chunk(action_chunk)
        self.pending_actions = deque(
            PendingAction(
                timestep=current_step + idx,
                action_dict=action_dicts[idx],
                action_tensor=action_chunk[idx].detach().cpu(),
            )
            for idx in range(action_chunk.shape[0])
        )
        logger.info(
            "accepted async result obs_step=%d current_step=%d latency=%.2fms delay_steps=%d queued=%d",
            latest_result.obs_timestep,
            current_step,
            latest_result.inference_latency_s * 1e3,
            max(0, current_step - latest_result.obs_timestep),
            len(self.pending_actions),
        )

    def run(self) -> None:
        init_logging()
        self.robot.connect()
        self._start_worker()

        logger.info(
            "ABPolicy single-machine async inference started. policy=%s fps=%d action_steps=%d exec_h=%d",
            self.policy_path,
            self.cfg.fps,
            self.policy.config.n_action_steps,
            self.cfg.execution_horizon,
        )

        start_t = time.perf_counter()
        step = 0

        try:
            while True:
                if self.cfg.max_steps > 0 and step >= self.cfg.max_steps:
                    break

                loop_start_t = time.perf_counter()
                raw_observation = self.robot.get_observation()
                hold_action_dict, hold_action_tensor = self._hold_action_from_observation(raw_observation)
                prepared = self._prepare_single_step_observation(raw_observation)
                self.obs_history.append((step, prepared))

                if len(self.executed_actions) == 0:
                    for _ in range(self.policy.config.action_history_horizon):
                        self.executed_actions.append(hold_action_tensor.detach().clone())

                self._drain_results(step, hold_action_tensor)
                self._maybe_submit_request(step)

                while self.pending_actions and self.pending_actions[0].timestep < step:
                    self.pending_actions.popleft()

                if self.pending_actions and self.pending_actions[0].timestep == step:
                    pending = self.pending_actions.popleft()
                    action_to_send = pending.action_dict
                    action_tensor = pending.action_tensor
                else:
                    action_to_send = self.last_sent_action_dict or hold_action_dict
                    action_tensor = self.last_sent_action_tensor or hold_action_tensor

                sent_action = self.robot.send_action(action_to_send)
                sent_action_tensor = self._vector_from_action_dict(sent_action if sent_action else action_to_send)
                self.executed_actions.append(sent_action_tensor)
                self.last_sent_action_dict = action_to_send
                self.last_sent_action_tensor = action_tensor.detach().cpu()

                if step % max(self.cfg.log_every, 1) == 0:
                    loop_dt_ms = (time.perf_counter() - loop_start_t) * 1e3
                    elapsed_s = time.perf_counter() - start_t
                    logger.info(
                        "step=%d elapsed=%.2fs loop=%.2fms hist=%d pending=%d inflight=%s",
                        step,
                        elapsed_s,
                        loop_dt_ms,
                        len(self.obs_history),
                        len(self.pending_actions),
                        self.inflight_request is not None,
                    )

                precise_sleep(max(1.0 / self.cfg.fps - (time.perf_counter() - loop_start_t), 0.0))
                step += 1
        finally:
            self._stop_worker()
            self.robot.disconnect()


@parser.wrap()
def main(cfg: ABAsyncSingleMachineConfig) -> None:
    runner = ABAsyncSingleMachineRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
