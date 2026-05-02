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

import importlib.util

import torch

from lerobot.datasets import video_utils


def test_get_safe_default_codec_falls_back_when_torchcodec_runtime_unavailable(monkeypatch):
    video_utils.is_torchcodec_available.cache_clear()

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "torchcodec" else None)

    real_import = __import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("torchcodec"):
            raise RuntimeError("libtorchcodec failed to load")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", failing_import)

    assert video_utils.get_safe_default_codec() == "pyav"


def test_decode_video_frames_falls_back_to_pyav_when_torchcodec_decode_fails(monkeypatch):
    expected = torch.ones(1, 3, 4, 4)

    def fail_torchcodec(*args, **kwargs):
        raise RuntimeError("libtorchcodec failed to load")

    def fake_pyav(video_path, timestamps, tolerance_s):
        return expected

    monkeypatch.setattr(video_utils, "decode_video_frames_torchcodec", fail_torchcodec)
    monkeypatch.setattr(video_utils, "decode_video_frames_pyav", fake_pyav)

    actual = video_utils.decode_video_frames(
        video_path="dummy.mp4",
        timestamps=[0.0],
        tolerance_s=0.1,
        backend="torchcodec",
    )

    assert torch.equal(actual, expected)
