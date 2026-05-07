#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Action queue management for Real-Time Chunking (RTC).

This module provides ActionQueue, a thread-safe queue for managing action chunks
in real-time control scenarios. It supports both RTC-enabled and non-RTC modes,
handling action merging and leftover tracking.
"""

import logging
from threading import Lock

import torch
from torch import Tensor

from lerobot.policies.rtc.configuration_rtc import RTCConfig

logger = logging.getLogger(__name__)


class ActionQueue:
    """Thread-safe queue for managing action chunks in real-time control.

    This queue handles two types of action sequences:
    - Original actions: Used for RTC to compute leftovers from previous chunks
    - Processed actions: Post-processed actions ready for robot execution

    The queue operates in two modes:
    1. RTC-enabled: Replaces the entire queue with new actions, accounting for inference delay
    2. RTC-disabled: Appends new actions to the queue, maintaining continuity

    Args:
        cfg (RTCConfig): Configuration for Real-Time Chunking behavior.

    Attributes:
        queue (Tensor | None): Processed actions for robot rollout (time_steps, action_dim).
        original_queue (Tensor | None): Original actions for RTC computation (time_steps, action_dim).
        last_index (int): Current consumption index in the queue.
    """

    def __init__(self, cfg: RTCConfig):
        """Initialize the action queue.

        Args:
            cfg: RTC configuration controlling queue behavior.
        """
        self.queue = None  # Processed actions for robot rollout
        self.original_queue = None  # Original actions for RTC
        self.lock = Lock()
        self.last_index = 0
        self.cfg = cfg

    def get(self) -> Tensor | None:
        """Get the next action from the queue.

        Returns:
            Tensor | None: The next action (action_dim,) or None if queue is empty.
                          Returns a clone to prevent external modifications.
        """
        with self.lock:
            if self.queue is None or self.last_index >= len(self.queue):
                return None

            action = self.queue[self.last_index]
            self.last_index += 1
            return action.clone()

    def qsize(self) -> int:
        """Get the number of remaining actions in the queue.

        Returns:
            int: Number of unconsumed actions.
        """
        if self.queue is None:
            return 0
        length = len(self.queue)
        return length - self.last_index

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            bool: True if no actions remain, False otherwise.
        """
        if self.queue is None:
            return True

        length = len(self.queue)
        return length - self.last_index <= 0

    def get_action_index(self) -> int:
        """Get the current action consumption index.

        Returns:
            int: Index of the next action to be consumed.
        """
        return self.last_index

    def get_left_over(self) -> Tensor | None:
        """Get leftover original actions for RTC prev_chunk_left_over.

        These are the unconsumed actions from the current chunk, which will be
        used by RTC to compute corrections for the next chunk.

        Returns:
            Tensor | None: Remaining original actions (remaining_steps, action_dim),
                          or None if no original queue exists.
        """
        with self.lock:
            if self.original_queue is None:
                return None
            return self.original_queue[self.last_index :]

    def merge(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        action_index_before_inference: int | None = 0,
        preserve_prefix_steps: int = 0,
        transition_blend_steps: int = 0,
    ):
        """Merge new actions into the queue.

        This method operates differently based on RTC mode:
        - RTC enabled: Replaces the queue, accounting for inference delay
        - RTC disabled: Appends to the queue, maintaining continuity

        Args:
            original_actions: Unprocessed actions from policy (time_steps, action_dim).
            processed_actions: Post-processed actions for robot (time_steps, action_dim).
            real_delay: Number of time steps of inference delay.
            action_index_before_inference: Index before inference started, for validation.
        """
        with self.lock:
            effective_delay = self._check_delays(real_delay, action_index_before_inference)

            if self.cfg.enabled:
                self._replace_actions_queue(
                    original_actions,
                    processed_actions,
                    effective_delay,
                    preserve_prefix_steps=preserve_prefix_steps,
                    transition_blend_steps=transition_blend_steps,
                )
                return

            self._append_actions_queue(original_actions, processed_actions)

    def _replace_actions_queue(
        self,
        original_actions: Tensor,
        processed_actions: Tensor,
        real_delay: int,
        preserve_prefix_steps: int = 0,
        transition_blend_steps: int = 0,
    ):
        """Replace the queue with new actions (RTC mode).

        Discards the first `real_delay` actions since they correspond to the time
        spent during inference, when the robot was executing previous actions.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
            real_delay: Number of time steps to skip due to inference delay.
        """
        new_original_queue = original_actions[real_delay:].clone()
        new_queue = processed_actions[real_delay:].clone()

        if (
            preserve_prefix_steps > 0 or transition_blend_steps > 0
        ) and self.queue is not None and self.last_index < len(self.queue):
            old_original_queue = self.original_queue[self.last_index :].to(
                device=new_original_queue.device,
                dtype=new_original_queue.dtype,
            ).clone()
            old_queue = self.queue[self.last_index :].to(
                device=new_queue.device,
                dtype=new_queue.dtype,
            ).clone()

            preserve_prefix_steps = min(preserve_prefix_steps, len(old_queue), len(new_queue))
            transition_blend_steps = min(
                transition_blend_steps,
                max(len(old_queue) - preserve_prefix_steps, 0),
                max(len(new_queue) - preserve_prefix_steps, 0),
            )

            if preserve_prefix_steps > 0 or transition_blend_steps > 0:
                original_parts = []
                processed_parts = []

                if preserve_prefix_steps > 0:
                    original_parts.append(old_original_queue[:preserve_prefix_steps])
                    processed_parts.append(old_queue[:preserve_prefix_steps])

                if transition_blend_steps > 0:
                    processed_blend_weights = torch.linspace(
                        1.0 / (transition_blend_steps + 1),
                        transition_blend_steps / (transition_blend_steps + 1),
                        steps=transition_blend_steps,
                        device=new_queue.device,
                        dtype=new_queue.dtype,
                    ).unsqueeze(-1)
                    original_blend_weights = torch.linspace(
                        1.0 / (transition_blend_steps + 1),
                        transition_blend_steps / (transition_blend_steps + 1),
                        steps=transition_blend_steps,
                        device=new_original_queue.device,
                        dtype=new_original_queue.dtype,
                    ).unsqueeze(-1)

                    old_original_blend = old_original_queue[
                        preserve_prefix_steps : preserve_prefix_steps + transition_blend_steps
                    ]
                    new_original_blend = new_original_queue[
                        preserve_prefix_steps : preserve_prefix_steps + transition_blend_steps
                    ]
                    original_parts.append(
                        (1.0 - original_blend_weights) * old_original_blend
                        + original_blend_weights * new_original_blend
                    )

                    old_processed_blend = old_queue[
                        preserve_prefix_steps : preserve_prefix_steps + transition_blend_steps
                    ]
                    new_processed_blend = new_queue[
                        preserve_prefix_steps : preserve_prefix_steps + transition_blend_steps
                    ]
                    processed_parts.append(
                        (1.0 - processed_blend_weights) * old_processed_blend
                        + processed_blend_weights * new_processed_blend
                    )

                original_parts.append(new_original_queue[preserve_prefix_steps + transition_blend_steps :])
                processed_parts.append(new_queue[preserve_prefix_steps + transition_blend_steps :])

                self.original_queue = torch.cat([part for part in original_parts if len(part) > 0], dim=0)
                self.queue = torch.cat([part for part in processed_parts if len(part) > 0], dim=0)

                seam_index = preserve_prefix_steps + transition_blend_steps
                if (
                    seam_index > 0
                    and seam_index < len(self.queue)
                    and len(old_queue) >= seam_index
                ):
                    # Keep the new chunk's long-term trend, but damp the first few
                    # post-seam steps so the handoff does not introduce a visible jerk.
                    seam_anchor = self.queue[seam_index - 1]
                    seam_target = self.queue[seam_index]
                    seam_offset = seam_anchor - seam_target
                    seam_smoothing_steps = min(4, len(self.queue) - seam_index)
                    if seam_smoothing_steps > 0:
                        seam_decay = torch.linspace(
                            1.0,
                            0.0,
                            steps=seam_smoothing_steps + 1,
                            device=self.queue.device,
                            dtype=self.queue.dtype,
                        )[1:].unsqueeze(-1)
                        self.queue[seam_index : seam_index + seam_smoothing_steps] = (
                            self.queue[seam_index : seam_index + seam_smoothing_steps]
                            + seam_decay * seam_offset.unsqueeze(0)
                        )
            else:
                self.original_queue = new_original_queue
                self.queue = new_queue
        else:
            self.original_queue = new_original_queue
            self.queue = new_queue

        logger.debug(f"original_actions shape: {self.original_queue.shape}")
        logger.debug(f"processed_actions shape: {self.queue.shape}")
        logger.debug(f"real_delay: {real_delay}")

        self.last_index = 0

    def _append_actions_queue(self, original_actions: Tensor, processed_actions: Tensor):
        """Append new actions to the queue (non-RTC mode).

        Removes already-consumed actions and appends new ones, maintaining
        queue continuity without replacement.

        Args:
            original_actions: Unprocessed actions from policy.
            processed_actions: Post-processed actions for robot.
        """
        if self.queue is None:
            self.original_queue = original_actions.clone()
            self.queue = processed_actions.clone()
            return

        existing_original_queue = self.original_queue.to(
            device=original_actions.device,
            dtype=original_actions.dtype,
        )
        self.original_queue = torch.cat([existing_original_queue, original_actions.clone()])
        self.original_queue = self.original_queue[self.last_index :]

        existing_queue = self.queue.to(
            device=processed_actions.device,
            dtype=processed_actions.dtype,
        )
        self.queue = torch.cat([existing_queue, processed_actions.clone()])
        self.queue = self.queue[self.last_index :]

        self.last_index = 0

    def _check_delays(self, real_delay: int, action_index_before_inference: int | None = None) -> int:
        """Validate that computed delays match expectations.

        Compares the delay computed from inference latency with the actual
        number of actions consumed during inference.

        Args:
            real_delay: Delay computed from inference latency.
            action_index_before_inference: Action index when inference started.
        """
        if action_index_before_inference is None:
            return max(real_delay, 0)

        indexes_diff = max(self.last_index - action_index_before_inference, 0)
        if indexes_diff != real_delay:
            # Let's check that action index difference (real delay calculated based on action queue)
            # is the same as delay calculated based on inference latency
            logger.warning(
                f"[ACTION_QUEUE] Indexes diff is not equal to real delay. "
                f"Indexes diff: {indexes_diff}, real delay: {real_delay}"
            )
        return indexes_diff
