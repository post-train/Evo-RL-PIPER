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

"""Core recording loop used by `lerobot_record.py`."""

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_ACTIVE,
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    PolicySyncDualArmExecutor,
    RTCPolicyRuntime,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
    initialize_rtc_observation_history,
    predict_action_with_rtc,
    start_async_rtc_worker,
    update_rtc_observation_history,
)
from lerobot.teleoperators import Teleoperator, koch_leader, omx_leader, so_leader
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.recording_annotations import resolve_collector_policy_id
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from lerobot.utils.visualization_utils import log_rerun_data

T = TypeVar("T")
TIMING_LOG_EVERY_STEPS = 120


class _AsyncEpisodeFrameWriter:
    """Write already-paired control-step frames off the control path."""

    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self._queue: queue.SimpleQueue[dict[str, Any] | None] = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._run, name="record-frame-writer", daemon=True)
        self._error: BaseException | None = None
        self._pending_count = 0
        self._pending_lock = threading.Lock()
        self._pending_empty = threading.Event()
        self._pending_empty.set()
        self._thread.start()

    def _run(self) -> None:
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    return
                try:
                    self.dataset.add_frame(frame)
                finally:
                    with self._pending_lock:
                        self._pending_count -= 1
                        if self._pending_count == 0:
                            self._pending_empty.set()
        except BaseException as exc:  # noqa: BLE001
            self._error = exc

    def submit(self, frame: dict[str, Any]) -> None:
        if self._error is not None:
            raise RuntimeError("Async episode frame writer failed.") from self._error
        with self._pending_lock:
            self._pending_count += 1
            self._pending_empty.clear()
        self._queue.put(frame)
        if self._error is not None:
            raise RuntimeError("Async episode frame writer failed.") from self._error

    def close(self) -> None:
        if self._error is not None:
            self._thread.join()
            raise RuntimeError("Async episode frame writer failed.") from self._error
        self._pending_empty.wait()
        self._queue.put(None)
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("Async episode frame writer failed.") from self._error


""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     .-----( ACTION LOGIC )------------------.
     V                                       V
     [ From Teleoperator ]                   [ From Policy ]
     |                                       |
     |  [teleop.get_action] -> raw_action    |   [predict_action]
     |          |                            |          |
     |          V                            |          V
     | [teleop_action_processor]             |          |
     |          |                            |          |
     '---> processed_teleop_action           '---> processed_policy_action
     |                                       |
     '-------------------------.-------------'
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
    policy_sync_executor: PolicySyncDualArmExecutor | None = None,
    intervention_state_machine_enabled: bool = True,
    collector_policy_id_policy: str = "policy",
    collector_policy_id_human: str = "human",
    acp_inference: ACPInferenceConfig | None = None,
    rtc: RTCConfig | None = None,
    communication_retry_timeout_s: float = 2.0,
    communication_retry_interval_s: float = 0.1,
):
    if acp_inference is None:
        acp_inference = ACPInferenceConfig()
    if rtc is None:
        rtc = RTCConfig()

    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    if dataset is None and policy is not None:
        raise ValueError("Policy-driven recording requires a dataset for feature mapping.")
    if rtc.enabled and policy is None:
        raise ValueError("RTC inference requires a policy.")
    if rtc.enabled and (preprocessor is None or postprocessor is None):
        raise ValueError("RTC inference requires preprocessor and postprocessor.")
    if rtc.enabled and acp_inference.enable:
        raise ValueError("RTC inference is not supported together with ACP inference in lerobot-record.")

    frame_writer = _AsyncEpisodeFrameWriter(dataset) if dataset is not None else None

    action_feature_names = dataset.features[ACTION]["names"] if dataset is not None else None
    if action_feature_names is None:
        if hasattr(robot.action_features, "keys"):
            action_feature_names = list(robot.action_features.keys())
        else:
            action_feature_names = list(robot.action_features)
    zero_policy_action = dict.fromkeys(action_feature_names, 0.0)
    has_teleop = isinstance(teleop, (Teleoperator, list))
    intervention_enabled = intervention_state_machine_enabled and policy is not None and has_teleop
    intervention_state = INTERVENTION_STATE_POLICY
    last_teleop_action: RobotAction | None = None
    teleop_fallback_warned = False

    teleop_arm_for_mode_switch: Any | None = None
    if isinstance(teleop, Teleoperator):
        teleop_arm_for_mode_switch = teleop
    elif isinstance(teleop, list):
        teleop_arm_for_mode_switch = teleop_arm

    def set_teleop_manual_control(enabled: bool) -> None:
        if teleop_arm_for_mode_switch is None:
            return
        if not hasattr(teleop_arm_for_mode_switch, "set_manual_control"):
            return
        try:
            teleop_arm_for_mode_switch.set_manual_control(enabled)
        except Exception:
            logging.exception("Failed to switch teleop manual-control mode to %s", enabled)

    if policy is None:
        # During reset/teleop-only loops keep leader backdrivable for manual dragging.
        set_teleop_manual_control(True)

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
    rtc_runtime = RTCPolicyRuntime.create(rtc, fps) if policy is not None and rtc.enabled else None
    if rtc_runtime is not None:
        initialize_rtc_observation_history(rtc_runtime, policy)
        if rtc_runtime.cfg.async_inference:
            start_async_rtc_worker(
                rtc_runtime,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
        logging.info(
            "RTC enabled fps=%d exec_h=%d max_guidance_weight=%.3f schedule=%s async=%s",
            fps,
            rtc.execution_horizon,
            rtc.max_guidance_weight,
            rtc.prefix_attention_schedule,
            rtc.async_inference,
        )
        if rtc.enabled and not rtc.async_inference:
            raise RuntimeError(
                "RTC inference is running in synchronous mode. "
                "This deployment path requires rtc.async_inference=true to avoid blocking the control loop."
            )

    cond_policy_runtime_state: dict[str, Any] | None = None
    uncond_policy_runtime_state: dict[str, Any] | None = None
    if policy is not None and acp_inference.enable and acp_inference.use_cfg:
        cond_policy_runtime_state = _capture_policy_runtime_state(policy)
        uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

    if intervention_enabled:
        # Start in S0: policy drives both arms, teleop arm should accept feedback commands.
        set_teleop_manual_control(False)

    def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
        timeout_s = max(communication_retry_timeout_s, 0.0)
        interval_s = max(communication_retry_interval_s, 0.0)
        deadline_t = time.perf_counter() + timeout_s
        attempts = 0
        first_error: ConnectionError | None = None

        while True:
            attempts += 1
            try:
                result = fn()
                if attempts > 1:
                    elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                    logging.warning(
                        "%s recovered after %d retries in %.2fs.",
                        action_name,
                        attempts - 1,
                        elapsed_s,
                    )
                return result
            except ConnectionError as error:
                if first_error is None:
                    first_error = error
                    logging.warning(
                        "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                        action_name,
                        timeout_s,
                        error,
                    )

                if timeout_s <= 0.0:
                    raise

                remaining_s = deadline_t - time.perf_counter()
                if remaining_s <= 0.0:
                    raise

                sleep_s = interval_s if interval_s > 0.0 else remaining_s
                time.sleep(min(sleep_s, remaining_s))

    try:
        timestamp = 0
        start_episode_t = time.perf_counter()
        step_idx = 0
        while timestamp < control_time_s:
            start_loop_t = time.perf_counter()

            if events.get("stop_recording") or events["exit_early"]:
                events["exit_early"] = False
                break

            if events.get("toggle_intervention", False):
                events["toggle_intervention"] = False
                if intervention_enabled:
                    if intervention_state == INTERVENTION_STATE_POLICY:
                        intervention_state = INTERVENTION_STATE_ACTIVE
                        set_teleop_manual_control(True)
                        logging.info("Intervention enabled (S1): teleop actions now override policy execution.")
                    else:
                        intervention_state = INTERVENTION_STATE_RELEASE
                        set_teleop_manual_control(False)
                        if policy is not None and preprocessor is not None and postprocessor is not None:
                            if rtc_runtime is not None:
                                rtc_runtime.reset()
                                with rtc_runtime.inference_lock:
                                    policy.reset()
                                    preprocessor.reset()
                                    postprocessor.reset()
                                initialize_rtc_observation_history(rtc_runtime, policy)
                            else:
                                policy.reset()
                                preprocessor.reset()
                                postprocessor.reset()
                            if acp_inference.enable and acp_inference.use_cfg:
                                cond_policy_runtime_state = _capture_policy_runtime_state(policy)
                                uncond_policy_runtime_state = _capture_policy_runtime_state(policy)
                        if policy is not None and preprocessor is not None and postprocessor is not None:
                            logging.info("Policy cache reset on release: next policy action is recomputed.")
                        logging.info("Intervention release requested (S2): returning control to policy.")
                else:
                    logging.info("Intervention toggle ignored because policy+teleop are not both active.")

            # Get robot observation
            obs = robot.get_observation()
            observation_dt_s = time.perf_counter() - start_loop_t

            # Applies a pipeline to the raw robot observation, default is IdentityProcessor
            obs_processed = robot_observation_processor(obs)
            observation_processing_dt_s = time.perf_counter() - start_loop_t - observation_dt_s

            if dataset is not None:
                observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

            # Get action from policy and/or teleop
            act_processed_policy: RobotAction | None = None
            act_processed_teleop: RobotAction | None = None
            action_start_t = time.perf_counter()
            if (
                policy is not None
                and preprocessor is not None
                and postprocessor is not None
                and not (intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE)
            ):
                if rtc_runtime is not None:
                    update_rtc_observation_history(
                        rtc_runtime,
                        observation=observation_frame,
                        policy=policy,
                        preprocessor=preprocessor,
                        task=single_task,
                        robot_type=robot.robot_type,
                    )
                    policy_action = predict_action_with_rtc(
                        observation=observation_frame,
                        policy=policy,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        task=single_task,
                        robot_type=robot.robot_type,
                        runtime=rtc_runtime,
                    )
                else:
                    policy_action = _predict_policy_action_with_acp_inference(
                        observation_frame=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                        acp_inference=acp_inference,
                        cond_runtime_state=cond_policy_runtime_state,
                        uncond_runtime_state=uncond_policy_runtime_state,
                    )
                act_processed_policy = make_robot_action(policy_action, dataset.features)

            if isinstance(teleop, Teleoperator):
                act = run_with_connection_retry("teleop.get_action", teleop.get_action)

                # Applies a pipeline to the raw teleop action, default is IdentityProcessor
                act_processed_teleop = teleop_action_processor((act, obs))

            elif isinstance(teleop, list):
                arm_action = run_with_connection_retry("teleop_arm.get_action", teleop_arm.get_action)
                arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
                keyboard_action = teleop_keyboard.get_action()
                base_action = robot._from_keyboard_to_base_action(keyboard_action)
                act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
                act_processed_teleop = teleop_action_processor((act, obs))
            action_dt_s = time.perf_counter() - action_start_t

            if act_processed_policy is None and act_processed_teleop is None:
                logging.info(
                    "No policy or teleoperator provided, skipping action generation."
                    "This is likely to happen when resetting the environment without a teleop device."
                    "The robot won't be at its rest position at the start of the next episode."
                )
                continue

            if act_processed_teleop is not None:
                last_teleop_action = act_processed_teleop
                teleop_fallback_warned = False

            policy_action_for_storage = (
                act_processed_policy if act_processed_policy is not None else zero_policy_action
            )

            is_intervention = 0.0
            if intervention_enabled and intervention_state == INTERVENTION_STATE_ACTIVE:
                is_intervention = 1.0
                if act_processed_teleop is not None:
                    action_values = act_processed_teleop
                elif last_teleop_action is not None:
                    action_values = last_teleop_action
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but no fresh teleop action is available; reusing last teleop action."
                        )
                        teleop_fallback_warned = True
                elif act_processed_policy is not None:
                    action_values = act_processed_policy
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but teleop action is unavailable; falling back to policy action."
                        )
                        teleop_fallback_warned = True
                else:
                    action_values = zero_policy_action
                    if not teleop_fallback_warned:
                        logging.warning(
                            "Intervention is active but no teleop/policy action is available; sending zero action."
                        )
                        teleop_fallback_warned = True
            else:
                action_values = act_processed_policy if act_processed_policy is not None else act_processed_teleop

            # Applies a pipeline to the action, default is IdentityProcessor
            action_processing_start_t = time.perf_counter()
            robot_action_to_send = robot_action_processor((action_values, obs))
            action_processing_dt_s = time.perf_counter() - action_processing_start_t

            # Send action to robot
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset. action = postprocessor.process(action)
            # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
            selected_from_policy = act_processed_policy is not None and action_values is act_processed_policy
            send_action_start_t = time.perf_counter()
            if policy_sync_executor is not None and selected_from_policy:
                _sent_action = run_with_connection_retry(
                    "policy_sync_executor.send_action",
                    lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                        robot_action_to_send
                    ),
                )
            else:
                _sent_action = run_with_connection_retry(
                    "robot.send_action",
                    lambda robot_action_to_send=robot_action_to_send: robot.send_action(robot_action_to_send),
                )
            send_action_dt_s = time.perf_counter() - send_action_start_t

            # Write to dataset
            dataset_dt_s = 0.0
            if dataset is not None:
                dataset_start_t = time.perf_counter()
                action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
                policy_action_frame = build_dataset_frame(
                    dataset.features, policy_action_for_storage, prefix="complementary_info.policy_action"
                )
                frame = {**observation_frame, **action_frame, **policy_action_frame, "task": single_task}

                if "complementary_info.is_intervention" in dataset.features:
                    frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
                if "complementary_info.state" in dataset.features:
                    frame["complementary_info.state"] = np.array([intervention_state], dtype=np.float32)
                if "complementary_info.collector_policy_id" in dataset.features:
                    frame["complementary_info.collector_policy_id"] = resolve_collector_policy_id(
                        intervention_enabled=intervention_enabled,
                        is_intervention=bool(is_intervention),
                        selected_from_policy=selected_from_policy,
                        policy_id=collector_policy_id_policy,
                        human_id=collector_policy_id_human,
                    )
                if frame_writer is None:
                    dataset.add_frame(frame)
                else:
                    frame_writer.submit(frame)
                dataset_dt_s = time.perf_counter() - dataset_start_t

            display_dt_s = 0.0
            if display_data:
                display_start_t = time.perf_counter()
                log_rerun_data(
                    observation=obs_processed, action=action_values, compress_images=display_compressed_images
                )
                display_dt_s = time.perf_counter() - display_start_t

            if intervention_state == INTERVENTION_STATE_RELEASE:
                intervention_state = INTERVENTION_STATE_POLICY

            dt_s = time.perf_counter() - start_loop_t
            budget_s = 1 / fps
            sleep_s = max(budget_s - dt_s, 0.0)
            precise_sleep(sleep_s)

            if rtc_runtime is not None and step_idx % TIMING_LOG_EVERY_STEPS == 0:
                logging.info(
                    "RTC runtime step=%d q_now=%d action_idx=%d chunks=%d pending=%s last_q_before=%d last_q_after=%d last_prev_left=%d last_inf_delay=%d last_real_delay=%d last_latency=%.2fms max_latency=%.2fms last_chunk_steps=%d",
                    step_idx,
                    rtc_runtime.action_queue.qsize(),
                    rtc_runtime.action_queue.get_action_index(),
                    rtc_runtime.chunk_counter,
                    rtc_runtime.pending_request_id,
                    rtc_runtime.last_queue_size_before,
                    rtc_runtime.last_queue_size_after,
                    rtc_runtime.last_prev_actions_len,
                    rtc_runtime.last_inference_delay,
                    rtc_runtime.last_real_delay,
                    rtc_runtime.last_inference_latency_s * 1e3,
                    (rtc_runtime.latency_tracker.max() or 0.0) * 1e3,
                    rtc_runtime.last_chunk_steps,
                )
            loop_total_dt_s = time.perf_counter() - start_loop_t

            if step_idx % TIMING_LOG_EVERY_STEPS == 0:
                logging.info(
                    "Record loop timing step=%d obs=%.2fms obs_proc=%.2fms act=%.2fms act_proc=%.2fms send=%.2fms dataset=%.2fms display=%.2fms loop=%.2fms sleep=%.2fms budget=%.2fms",
                    step_idx,
                    observation_dt_s * 1e3,
                    observation_processing_dt_s * 1e3,
                    action_dt_s * 1e3,
                    action_processing_dt_s * 1e3,
                    send_action_dt_s * 1e3,
                    dataset_dt_s * 1e3,
                    display_dt_s * 1e3,
                    loop_total_dt_s * 1e3,
                    sleep_s * 1e3,
                    budget_s * 1e3,
                )
            if dt_s > budget_s:
                logging.warning(
                    "Record loop overrun step=%d over=%.2fms obs=%.2fms obs_proc=%.2fms act=%.2fms act_proc=%.2fms send=%.2fms dataset=%.2fms display=%.2fms budget=%.2fms",
                    step_idx,
                    (dt_s - budget_s) * 1e3,
                    observation_dt_s * 1e3,
                    observation_processing_dt_s * 1e3,
                    action_dt_s * 1e3,
                    action_processing_dt_s * 1e3,
                    send_action_dt_s * 1e3,
                    dataset_dt_s * 1e3,
                    display_dt_s * 1e3,
                    budget_s * 1e3,
                )

            timestamp = time.perf_counter() - start_episode_t
            step_idx += 1
    finally:
        if rtc_runtime is not None:
            rtc_runtime.shutdown()
        if frame_writer is not None:
            frame_writer.close()
