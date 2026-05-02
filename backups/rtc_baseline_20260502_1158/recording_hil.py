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

"""Human-in-loop recording helpers used by `lerobot_record.py`."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import contextlib
from copy import deepcopy
from dataclasses import dataclass, field
import math
import logging
import queue
import threading
from typing import Any

import numpy as np
import torch

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTracker
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues, prepare_observation_for_inference
from lerobot.processor import PolicyAction, PolicyProcessorPipeline, RobotAction
from lerobot.rl.acp_tags import build_acp_tagged_task
from lerobot.robots import Robot
from lerobot.teleoperators import Teleoperator
from lerobot.utils.control_utils import predict_action
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


logger = logging.getLogger(__name__)


@dataclass
class ACPInferenceConfig:
    enable: bool = False
    use_cfg: bool = False
    cfg_beta: float = 1.0


@dataclass
class RTCPolicyRuntime:
    """Runtime state for RTC-enabled chunked policy inference."""

    cfg: RTCConfig
    fps: int
    action_queue: ActionQueue
    latency_tracker: LatencyTracker
    chunk_counter: int = 0
    last_inference_delay: int = 0
    last_real_delay: int = 0
    last_inference_latency_s: float = 0.0
    last_queue_size_before: int = 0
    last_queue_size_after: int = 0
    last_prev_actions_len: int = 0
    last_action_index_before_inference: int = 0
    last_chunk_steps: int = 0
    last_returned_action: torch.Tensor | None = None
    inference_lock: threading.Lock = field(default_factory=threading.Lock)
    async_request_queue: queue.Queue | None = None
    async_result_queue: queue.Queue | None = None
    async_stop_event: threading.Event | None = None
    async_thread: threading.Thread | None = None
    pending_request_id: int | None = None
    request_counter: int = 0
    generation: int = 0
    observation_history: dict[str, deque] = field(default_factory=dict)
    history_warmup_logged: bool = False

    @classmethod
    def create(cls, cfg: RTCConfig, fps: int) -> "RTCPolicyRuntime":
        runtime = cls(
            cfg=deepcopy(cfg),
            fps=fps,
            action_queue=ActionQueue(cfg),
            latency_tracker=LatencyTracker(),
        )
        if runtime.cfg.async_inference:
            runtime.async_request_queue = queue.Queue(maxsize=1)
            runtime.async_result_queue = queue.Queue()
            runtime.async_stop_event = threading.Event()
        return runtime

    def reset(self) -> None:
        self.action_queue = ActionQueue(self.cfg)
        self.latency_tracker.reset()
        self.chunk_counter = 0
        self.last_inference_delay = 0
        self.last_real_delay = 0
        self.last_inference_latency_s = 0.0
        self.last_queue_size_before = 0
        self.last_queue_size_after = 0
        self.last_prev_actions_len = 0
        self.last_action_index_before_inference = 0
        self.last_chunk_steps = 0
        self.last_returned_action = None
        self.pending_request_id = None
        self.request_counter = 0
        self.generation += 1
        self.history_warmup_logged = False
        for queue_value in self.observation_history.values():
            queue_value.clear()
        if self.async_request_queue is not None:
            while True:
                try:
                    self.async_request_queue.get_nowait()
                except queue.Empty:
                    break
        if self.async_result_queue is not None:
            while True:
                try:
                    self.async_result_queue.get_nowait()
                except queue.Empty:
                    break

    def shutdown(self) -> None:
        if self.async_stop_event is not None:
            self.async_stop_event.set()
        if self.async_request_queue is not None:
            try:
                self.async_request_queue.put_nowait(None)
            except queue.Full:
                pass
        if self.async_thread is not None:
            self.async_thread.join(timeout=2.0)
            self.async_thread = None


@dataclass
class RTCInferenceRequest:
    request_id: int
    generation: int
    history_snapshot: dict[str, tuple[torch.Tensor, ...]]
    inference_delay: int
    prev_actions: torch.Tensor | None
    prev_actions_len: int
    action_index_before_inference: int
    queue_size_before: int
    execution_horizon: int
    is_bootstrap_chunk: bool
    use_rtc_guidance: bool


@dataclass
class RTCInferenceResult:
    request_id: int
    generation: int
    original_actions: torch.Tensor
    processed_actions: torch.Tensor
    inference_delay: int
    real_delay: int
    inference_latency_s: float
    prev_actions_len: int
    action_index_before_inference: int
    queue_size_before: int
    execution_horizon: int
    is_bootstrap_chunk: bool
    use_rtc_guidance: bool


POLICY_RUNTIME_STATE_KEYS = ("_action_queue", "_queues", "_prev_mean")


INTERVENTION_STATE_POLICY = 0.0
INTERVENTION_STATE_ACTIVE = 1.0
INTERVENTION_STATE_RELEASE = 2.0


def _get_torch_rng_state(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return cpu_state, cuda_state


def _set_torch_rng_state(
    device: torch.device, cpu_state: torch.Tensor, cuda_state: torch.Tensor | None
) -> None:
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def _clone_runtime_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, deque):
        return deque((_clone_runtime_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, dict):
        return {key: _clone_runtime_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_runtime_value(item) for item in value)
    return deepcopy(value)


def _capture_policy_runtime_state(policy: PreTrainedPolicy) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key in POLICY_RUNTIME_STATE_KEYS:
        if hasattr(policy, key):
            state[key] = _clone_runtime_value(getattr(policy, key))
    return state


def _restore_policy_runtime_state(policy: PreTrainedPolicy, state: dict[str, Any]) -> None:
    for key, value in state.items():
        setattr(policy, key, _clone_runtime_value(value))


def _predict_policy_action_with_runtime_state(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    runtime_state: dict[str, Any],
) -> PolicyAction:
    _restore_policy_runtime_state(policy, runtime_state)
    action = predict_action(
        observation=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
    )
    runtime_state.clear()
    runtime_state.update(_capture_policy_runtime_state(policy))
    return action


def _predict_policy_action_with_acp_inference(
    *,
    observation_frame: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    device: torch.device,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    use_amp: bool,
    task: str | None,
    robot_type: str | None,
    acp_inference: ACPInferenceConfig,
    cond_runtime_state: dict[str, Any] | None = None,
    uncond_runtime_state: dict[str, Any] | None = None,
) -> PolicyAction:
    if not acp_inference.enable:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=task,
            robot_type=robot_type,
        )

    conditional_task = build_acp_tagged_task(task, is_positive=True)
    if not acp_inference.use_cfg:
        return predict_action(
            observation=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=use_amp,
            task=conditional_task,
            robot_type=robot_type,
        )

    if cond_runtime_state is None or uncond_runtime_state is None:
        raise ValueError("CFG inference requires cond/uncond runtime states.")

    cpu_state, cuda_state = _get_torch_rng_state(device)
    action_cond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=conditional_task,
        robot_type=robot_type,
        runtime_state=cond_runtime_state,
    )
    _set_torch_rng_state(device, cpu_state, cuda_state)
    action_uncond = _predict_policy_action_with_runtime_state(
        observation_frame=observation_frame,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task=task,
        robot_type=robot_type,
        runtime_state=uncond_runtime_state,
    )
    return action_uncond + acp_inference.cfg_beta * (action_cond - action_uncond)


def enable_policy_rtc(policy: PreTrainedPolicy, rtc_config: RTCConfig) -> None:
    """Enable RTC on a loaded policy instance."""

    if not hasattr(policy.config, "rtc_config"):
        raise ValueError(f"Policy '{policy.name}' does not expose rtc_config and cannot run RTC inference.")

    policy.config.rtc_config = deepcopy(rtc_config)

    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
        return

    if hasattr(policy, "rtc_processor"):
        policy.rtc_processor = RTCProcessor(policy.config.rtc_config)
        return

    raise ValueError(f"Policy '{policy.name}' does not provide an RTC processor hook.")


def _prepare_policy_observation(
    observation: dict[str, np.ndarray],
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    device: torch.device,
    task: str | None,
    robot_type: str | None,
) -> dict[str, Any]:
    observation_for_policy = {
        key: value.copy() if isinstance(value, np.ndarray) else value for key, value in observation.items()
    }
    prepared_observation = prepare_observation_for_inference(observation_for_policy, device, task, robot_type)
    prepared_observation = preprocessor(prepared_observation)
    if ACTION in prepared_observation:
        prepared_observation.pop(ACTION)
    if getattr(policy.config, "image_features", None):
        prepared_observation = dict(prepared_observation)
        prepared_observation[OBS_IMAGES] = torch.stack(
            [prepared_observation[key] for key in policy.config.image_features],
            dim=1,
        )
    return prepared_observation


def initialize_rtc_observation_history(runtime: RTCPolicyRuntime, policy: PreTrainedPolicy) -> None:
    observation_history = {
        OBS_STATE: deque(maxlen=policy.config.n_obs_steps),
    }
    if getattr(policy.config, "image_features", None):
        observation_history[OBS_IMAGES] = deque(maxlen=policy.config.n_obs_steps)
    if getattr(policy.config, "env_state_feature", None):
        observation_history[OBS_ENV_STATE] = deque(maxlen=policy.config.n_obs_steps)
    runtime.observation_history = observation_history


def update_rtc_observation_history(
    runtime: RTCPolicyRuntime,
    *,
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    task: str | None,
    robot_type: str | None,
) -> None:
    if not runtime.observation_history:
        initialize_rtc_observation_history(runtime, policy)

    device = get_safe_torch_device(policy.config.device)
    prepared_observation = _prepare_policy_observation(
        observation,
        policy=policy,
        preprocessor=preprocessor,
        device=device,
        task=task,
        robot_type=robot_type,
    )
    for key, history_queue in runtime.observation_history.items():
        if key not in prepared_observation:
            continue
        history_queue.append(prepared_observation[key].detach().clone())


def _build_rtc_history_snapshot(runtime: RTCPolicyRuntime) -> dict[str, tuple[torch.Tensor, ...]]:
    history_snapshot: dict[str, tuple[torch.Tensor, ...]] = {}
    for key, history_queue in runtime.observation_history.items():
        if not history_queue:
            continue
        frames = list(history_queue)
        while len(frames) < history_queue.maxlen:
            frames.insert(0, frames[0].clone())
        history_snapshot[key] = tuple(frame.detach().clone() for frame in frames)
    return history_snapshot


def _get_rtc_history_lengths(runtime: RTCPolicyRuntime) -> dict[str, int]:
    return {key: len(history_queue) for key, history_queue in runtime.observation_history.items()}


def _rtc_history_ready(runtime: RTCPolicyRuntime) -> bool:
    return bool(runtime.observation_history) and all(
        len(history_queue) == history_queue.maxlen for history_queue in runtime.observation_history.values()
    )


def _compute_async_prefetch_threshold(
    runtime: RTCPolicyRuntime,
    policy: PreTrainedPolicy,
    *,
    inference_delay: int,
) -> int:
    chunk_steps = policy.config.n_action_steps
    if chunk_steps <= 1:
        return runtime.cfg.execution_horizon
    dynamic_threshold = max(runtime.cfg.execution_horizon, inference_delay)
    return min(chunk_steps - 1, dynamic_threshold)


def _should_use_async_rtc_guidance(
    runtime: RTCPolicyRuntime,
    policy: PreTrainedPolicy,
    *,
    prev_actions_len: int,
) -> bool:
    # Flow matching is currently more stable on the real robot when async inference
    # is used purely for prefetching, without RTC prefix guidance.
    if getattr(policy, "name", "") == "flow_matching":
        return False
    return prev_actions_len >= 2 and runtime.chunk_counter >= 2


def _get_rtc_history_hold_action(runtime: RTCPolicyRuntime, policy: PreTrainedPolicy) -> torch.Tensor | None:
    state_history = runtime.observation_history.get(OBS_STATE)
    if not state_history:
        return None
    last_state = state_history[-1]
    action_feature = getattr(policy.config, "action_feature", None)
    if action_feature is None or last_state.shape[-1] != action_feature.shape[0]:
        return None
    return last_state.squeeze(0).detach().clone()


def _restore_policy_history_snapshot(
    policy: PreTrainedPolicy,
    history_snapshot: dict[str, tuple[torch.Tensor, ...]],
) -> dict[str, torch.Tensor]:
    if not hasattr(policy, "_queues"):
        raise RuntimeError("RTC history snapshot restore requires a policy queue state.")

    restored_queues: dict[str, deque] = {}
    for key, queue_value in policy._queues.items():
        frames = history_snapshot.get(key)
        if frames is None:
            restored_queues[key] = deque(maxlen=queue_value.maxlen)
            continue
        restored_queues[key] = deque((frame.detach().clone() for frame in frames), maxlen=queue_value.maxlen)
    policy._queues = restored_queues
    return {key: frames[-1] for key, frames in history_snapshot.items() if frames}


def start_async_rtc_worker(
    runtime: RTCPolicyRuntime,
    *,
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
) -> None:
    if not runtime.cfg.async_inference:
        return
    if runtime.async_request_queue is None or runtime.async_result_queue is None or runtime.async_stop_event is None:
        raise RuntimeError("Async RTC runtime is missing worker queues.")
    if runtime.async_thread is not None:
        return

    def _worker() -> None:
        device = get_safe_torch_device(policy.config.device)
        while not runtime.async_stop_event.is_set():
            try:
                request = runtime.async_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                return

            import time as _time

            wall_start_t = _time.perf_counter()
            with (
                torch.autocast(device_type=device.type)
                if device.type == "cuda" and policy.config.use_amp
                else contextlib.nullcontext()
            ):
                with runtime.inference_lock:
                    prepared_observation = _restore_policy_history_snapshot(policy, request.history_snapshot)
                    action_chunk = policy.predict_action_chunk(
                        prepared_observation,
                        inference_delay=request.inference_delay if request.use_rtc_guidance else 0,
                        prev_chunk_left_over=request.prev_actions if request.use_rtc_guidance else None,
                        execution_horizon=request.execution_horizon,
                    )
                    original_actions = action_chunk.squeeze(0).detach().clone()

                    processed_actions = []
                    for idx in range(action_chunk.shape[1]):
                        processed_action = postprocessor(action_chunk[:, idx, :])
                        processed_actions.append(processed_action.detach())
                    processed_actions = torch.stack(processed_actions, dim=1).squeeze(0)

            inference_latency = _time.perf_counter() - wall_start_t
            real_delay = math.ceil(inference_latency / (1.0 / runtime.fps))
            runtime.async_result_queue.put(
                RTCInferenceResult(
                    request_id=request.request_id,
                    generation=request.generation,
                    original_actions=original_actions,
                    processed_actions=processed_actions,
                    inference_delay=request.inference_delay,
                    real_delay=real_delay,
                    inference_latency_s=inference_latency,
                    prev_actions_len=request.prev_actions_len,
                    action_index_before_inference=request.action_index_before_inference,
                    queue_size_before=request.queue_size_before,
                    execution_horizon=request.execution_horizon,
                    is_bootstrap_chunk=request.is_bootstrap_chunk,
                    use_rtc_guidance=request.use_rtc_guidance,
                )
            )

    runtime.async_thread = threading.Thread(target=_worker, name="rtc-policy-worker", daemon=True)
    runtime.async_thread.start()


def _merge_rtc_inference_result(runtime: RTCPolicyRuntime, result: RTCInferenceResult) -> None:
    runtime.latency_tracker.add(result.inference_latency_s)
    if result.is_bootstrap_chunk:
        runtime.action_queue.original_queue = result.original_actions.clone()
        runtime.action_queue.queue = result.processed_actions.clone()
        runtime.action_queue.last_index = 0
        if result.real_delay >= len(result.processed_actions):
            logger.warning(
                "RTC bootstrap fallback kept the full first chunk because real_delay=%d >= chunk_steps=%d. "
                "This indicates first inference latency exceeds the action chunk horizon.",
                result.real_delay,
                len(result.processed_actions),
            )
    else:
        effective_real_delay = result.real_delay
        min_actions_to_keep = 3
        if not result.use_rtc_guidance:
            min_actions_to_keep = max(
                min_actions_to_keep,
                result.inference_delay,
                max(result.real_delay - 1, 0),
            )
            min_actions_to_keep = min(len(result.processed_actions) - 1, min_actions_to_keep)
        max_safe_delay = max(len(result.processed_actions) - min_actions_to_keep, 0)
        if effective_real_delay > max_safe_delay:
            logger.warning(
                "RTC delay clipping applied: real_delay=%d chunk_steps=%d clipped_to=%d to keep at least %d actions available.",
                result.real_delay,
                len(result.processed_actions),
                max_safe_delay,
                min_actions_to_keep,
            )
            effective_real_delay = max_safe_delay
        runtime.action_queue.merge(
            original_actions=result.original_actions,
            processed_actions=result.processed_actions,
            real_delay=effective_real_delay,
            action_index_before_inference=result.action_index_before_inference,
        )

    runtime.chunk_counter += 1
    runtime.last_inference_delay = result.inference_delay
    runtime.last_real_delay = result.real_delay
    runtime.last_inference_latency_s = result.inference_latency_s
    runtime.last_queue_size_before = result.queue_size_before
    runtime.last_queue_size_after = runtime.action_queue.qsize()
    runtime.last_prev_actions_len = result.prev_actions_len
    runtime.last_action_index_before_inference = result.action_index_before_inference
    runtime.last_chunk_steps = len(result.original_actions)
    runtime.pending_request_id = None
    logger.info(
        "RTC chunk=%d q_before=%d q_after=%d action_idx=%d prev_left=%d inf_delay=%d real_delay=%d latency=%.2fms max_latency=%.2fms exec_h=%d chunk_steps=%d guidance=%s",
        runtime.chunk_counter,
        runtime.last_queue_size_before,
        runtime.last_queue_size_after,
        runtime.last_action_index_before_inference,
        runtime.last_prev_actions_len,
        runtime.last_inference_delay,
        runtime.last_real_delay,
        runtime.last_inference_latency_s * 1e3,
        (runtime.latency_tracker.max() or 0.0) * 1e3,
        runtime.cfg.execution_horizon,
        runtime.last_chunk_steps,
        result.use_rtc_guidance,
    )


def _drain_async_rtc_results(runtime: RTCPolicyRuntime) -> None:
    if runtime.async_result_queue is None:
        return
    while True:
        try:
            result = runtime.async_result_queue.get_nowait()
        except queue.Empty:
            return
        if result.generation != runtime.generation:
            logger.info(
                "Ignoring stale RTC async result request=%d generation=%d current_generation=%d",
                result.request_id,
                result.generation,
                runtime.generation,
            )
            continue
        _merge_rtc_inference_result(runtime, result)


def predict_action_with_rtc(
    *,
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    task: str | None,
    robot_type: str | None,
    runtime: RTCPolicyRuntime,
) -> PolicyAction:
    """Predict one action while maintaining RTC queue state across control steps."""

    if runtime.cfg.async_inference:
        return predict_action_with_rtc_async(
            observation=observation,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            task=task,
            robot_type=robot_type,
            runtime=runtime,
        )

    if runtime.action_queue.qsize() <= runtime.cfg.execution_horizon:
        action_index_before_inference = runtime.action_queue.get_action_index()
        prev_actions = runtime.action_queue.get_left_over()
        is_bootstrap_chunk = runtime.action_queue.empty() and prev_actions is None
        prev_actions_len = 0 if prev_actions is None else len(prev_actions)
        queue_size_before = runtime.action_queue.qsize()
        inference_latency = runtime.last_inference_latency_s or runtime.latency_tracker.max() or 0.0
        time_per_step = 1.0 / runtime.fps
        inference_delay = math.ceil(inference_latency / time_per_step)

        device = get_safe_torch_device(policy.config.device)
        import time as _time

        wall_start_t = _time.perf_counter()
        with (
            torch.autocast(device_type=device.type)
            if device.type == "cuda" and policy.config.use_amp
            else contextlib.nullcontext()
        ):
            prepared_observation = _prepare_policy_observation(
                observation,
                policy=policy,
                preprocessor=preprocessor,
                device=device,
                task=task,
                robot_type=robot_type,
            )
            if hasattr(policy, "_queues"):
                policy._queues = populate_queues(policy._queues, prepared_observation)
            action_chunk = policy.predict_action_chunk(
                prepared_observation,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_actions,
                execution_horizon=runtime.cfg.execution_horizon,
            )
            original_actions = action_chunk.squeeze(0).detach().clone()

            processed_actions = []
            for idx in range(action_chunk.shape[1]):
                processed_action = postprocessor(action_chunk[:, idx, :])
                processed_actions.append(processed_action.detach())
            processed_actions = torch.stack(processed_actions, dim=1).squeeze(0)

        inference_latency = _time.perf_counter() - wall_start_t
        _merge_rtc_inference_result(
            runtime,
            RTCInferenceResult(
                request_id=runtime.request_counter,
                generation=runtime.generation,
                original_actions=original_actions,
                processed_actions=processed_actions,
                inference_delay=inference_delay,
                real_delay=math.ceil(inference_latency / time_per_step),
                inference_latency_s=inference_latency,
                prev_actions_len=prev_actions_len,
                action_index_before_inference=action_index_before_inference,
                queue_size_before=queue_size_before,
                execution_horizon=runtime.cfg.execution_horizon,
                is_bootstrap_chunk=is_bootstrap_chunk,
            ),
        )

    action = runtime.action_queue.get()
    if action is None:
        raise RuntimeError("RTC action queue is empty after chunk prediction.")
    runtime.last_returned_action = action.clone()
    return action


def predict_action_with_rtc_async(
    *,
    observation: dict[str, np.ndarray],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    task: str | None,
    robot_type: str | None,
    runtime: RTCPolicyRuntime,
) -> PolicyAction:
    if runtime.async_request_queue is None or runtime.async_result_queue is None:
        raise RuntimeError("Async RTC runtime is not initialized.")

    _drain_async_rtc_results(runtime)

    if runtime.chunk_counter == 0 and not _rtc_history_ready(runtime):
        hold_action = _get_rtc_history_hold_action(runtime, policy)
        if hold_action is not None:
            runtime.last_returned_action = hold_action.clone()
            if not runtime.history_warmup_logged:
                logger.info(
                    "RTC async warmup holding current state until observation history is full: hist=%s",
                    _get_rtc_history_lengths(runtime),
                )
                runtime.history_warmup_logged = True
            return hold_action
        raise RuntimeError("RTC async warmup has no valid hold action while observation history is filling.")

    inference_latency = runtime.last_inference_latency_s or runtime.latency_tracker.max() or 0.0
    time_per_step = 1.0 / runtime.fps
    inference_delay = math.ceil(inference_latency / time_per_step)
    prefetch_threshold = _compute_async_prefetch_threshold(
        runtime,
        policy,
        inference_delay=inference_delay,
    )

    if runtime.action_queue.qsize() <= prefetch_threshold and runtime.pending_request_id is None:
        action_index_before_inference = runtime.action_queue.get_action_index()
        prev_actions = runtime.action_queue.get_left_over()
        prev_actions_len = 0 if prev_actions is None else len(prev_actions)
        queue_size_before = runtime.action_queue.qsize()
        use_rtc_guidance = _should_use_async_rtc_guidance(
            runtime,
            policy,
            prev_actions_len=prev_actions_len,
        )

        runtime.request_counter += 1
        request = RTCInferenceRequest(
            request_id=runtime.request_counter,
            generation=runtime.generation,
            history_snapshot=_build_rtc_history_snapshot(runtime),
            inference_delay=inference_delay,
            prev_actions=None if prev_actions is None else prev_actions.detach().clone(),
            prev_actions_len=prev_actions_len,
            action_index_before_inference=action_index_before_inference,
            queue_size_before=queue_size_before,
            execution_horizon=runtime.cfg.execution_horizon,
            is_bootstrap_chunk=runtime.action_queue.empty() and prev_actions is None,
            use_rtc_guidance=use_rtc_guidance,
        )
        runtime.async_request_queue.put_nowait(request)
        runtime.pending_request_id = request.request_id
        logger.info(
            "RTC async submit request=%d q_before=%d action_idx=%d prev_left=%d inf_delay=%d exec_h=%d prefetch=%d guidance=%s hist=%s",
            request.request_id,
            queue_size_before,
            action_index_before_inference,
            prev_actions_len,
            inference_delay,
            runtime.cfg.execution_horizon,
            prefetch_threshold,
            use_rtc_guidance,
            _get_rtc_history_lengths(runtime),
        )

    action = runtime.action_queue.get()
    if action is not None:
        runtime.last_returned_action = action.clone()
        return action

    if runtime.chunk_counter == 0 and runtime.pending_request_id is not None:
        try:
            result = runtime.async_result_queue.get(timeout=5.0)
        except queue.Empty as exc:
            raise RuntimeError("Timed out waiting for the first RTC async action chunk.") from exc
        if result.generation == runtime.generation:
            _merge_rtc_inference_result(runtime, result)
            action = runtime.action_queue.get()
            if action is not None:
                runtime.last_returned_action = action.clone()
                return action

    if runtime.last_returned_action is not None:
        logger.warning(
            "RTC async queue underrun; reusing the last action while waiting for request=%s",
            runtime.pending_request_id,
        )
        return runtime.last_returned_action.clone()

    raise RuntimeError("RTC async action queue is empty and no fallback action is available.")


class PolicySyncDualArmExecutor:
    """Broadcast one policy-derived robot action to follower + teleop arm."""

    def __init__(self, robot: Robot, teleop: Teleoperator, parallel_dispatch: bool = True):
        self.robot = robot
        self.teleop = teleop
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            self.teleop.send_feedback(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        teleop_future = self._pool.submit(self.teleop.send_feedback, action)
        sent_action = robot_future.result()
        teleop_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
