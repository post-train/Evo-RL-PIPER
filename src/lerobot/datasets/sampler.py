#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
from collections.abc import Iterator

import numpy as np
import torch


class EpisodeAwareSampler:
    def __init__(
        self,
        dataset_from_indices: list[int],
        dataset_to_indices: list[int],
        episode_indices_to_use: list | None = None,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        shuffle: bool = False,
    ):
        """Sampler that optionally incorporates episode boundary information.

        Args:
            dataset_from_indices: List of indices containing the start of each episode in the dataset.
            dataset_to_indices: List of indices containing the end of each episode in the dataset.
            episode_indices_to_use: List of episode indices to use. If None, all episodes are used.
                                    Assumes that episodes are indexed from 0 to N-1.
            drop_n_first_frames: Number of frames to drop from the start of each episode.
            drop_n_last_frames: Number of frames to drop from the end of each episode.
            shuffle: Whether to shuffle the indices.
        """
        indices = []
        for episode_idx, (start_index, end_index) in enumerate(
            zip(dataset_from_indices, dataset_to_indices, strict=True)
        ):
            if episode_indices_to_use is None or episode_idx in episode_indices_to_use:
                indices.extend(range(start_index + drop_n_first_frames, end_index - drop_n_last_frames))

        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            for i in torch.randperm(len(self.indices)):
                yield self.indices[i]
        else:
            for i in self.indices:
                yield i

    def __len__(self) -> int:
        return len(self.indices)


def compute_motion_sampling_weights(
    dataset,
    *,
    threshold: float,
    static_weight: float,
    max_weight: float,
    drop_n_first_frames: int = 0,
    drop_n_last_frames: int = 0,
) -> torch.DoubleTensor:
    """Compute per-frame sampling weights that favor windows with real motion.

    The score is based on the maximum of adjacent action/state L2 deltas inside each episode.
    Frames with score below `threshold` are down-weighted to `static_weight`, while moving frames
    receive weights proportional to `score / threshold`, capped by `max_weight`.
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0 for motion-weighted sampling")
    if static_weight <= 0:
        raise ValueError("static_weight must be > 0 for motion-weighted sampling")
    if max_weight < static_weight:
        raise ValueError("max_weight must be >= static_weight")

    hf_dataset = dataset.hf_dataset.with_format(None)
    num_frames = len(hf_dataset)
    weights = np.full(num_frames, static_weight, dtype=np.float64)

    episode_ids = np.asarray(hf_dataset["episode_index"])
    actions = np.asarray(hf_dataset["action"], dtype=np.float32)
    states = np.asarray(hf_dataset["observation.state"], dtype=np.float32)

    unique_episodes = np.unique(episode_ids)
    for episode_id in unique_episodes:
        frame_ids = np.flatnonzero(episode_ids == episode_id)
        if len(frame_ids) == 0:
            continue

        ep_actions = actions[frame_ids]
        ep_states = states[frame_ids]

        action_delta = np.zeros(len(frame_ids), dtype=np.float32)
        state_delta = np.zeros(len(frame_ids), dtype=np.float32)
        if len(frame_ids) > 1:
            action_delta[1:] = np.linalg.norm(ep_actions[1:] - ep_actions[:-1], axis=1)
            state_delta[1:] = np.linalg.norm(ep_states[1:] - ep_states[:-1], axis=1)

        motion_score = np.maximum(action_delta, state_delta)
        ep_weights = np.clip(motion_score / threshold, static_weight, max_weight).astype(np.float64)

        valid_start = min(drop_n_first_frames, len(frame_ids))
        valid_end = max(valid_start, len(frame_ids) - drop_n_last_frames)

        if valid_start > 0:
            ep_weights[:valid_start] = 0.0
        if valid_end < len(frame_ids):
            ep_weights[valid_end:] = 0.0

        weights[frame_ids] = ep_weights

    if dataset.episodes is not None:
        allowed = set(dataset.episodes)
        for idx, episode_id in enumerate(episode_ids.tolist()):
            if episode_id not in allowed:
                weights[idx] = 0.0

    if not np.any(weights > 0):
        raise ValueError("Motion-weighted sampling produced no valid frames.")

    return torch.as_tensor(weights, dtype=torch.double)
