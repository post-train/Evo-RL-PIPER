#!/usr/bin/env python

from typing import Any

import torch

from lerobot.policies.fm.processor_fm import make_flow_matching_pre_post_processors
from lerobot.policies.fm_qat.configuration_fm_qat import FlowMatchingQATConfig


def make_flow_matching_qat_pre_post_processors(
    config: FlowMatchingQATConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    return make_flow_matching_pre_post_processors(config=config, dataset_stats=dataset_stats)
