#!/usr/bin/env python

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.fm.configuration_fm import FlowMatchingConfig


@PreTrainedConfig.register_subclass("flow_matching_qat")
@dataclass
class FlowMatchingQATConfig(FlowMatchingConfig):
    compile_velocity_net: bool = False
    qat_enabled: bool = True
    qat_quantize_backbone: bool = False
    qat_activation_observer_decay: float = 0.99
    qat_activation_eps: float = 1e-8
    trt_export_on_save: bool = True
    trt_input_batch_size: int = 1
    trt_dynamic_batch_max: int = 4
    trt_opset_version: int = 17
    trt_workspace_size_mb: int = 2048
    trt_onnx_filename: str = "fm_qat_velocity_net.onnx"
    trt_engine_filename: str = "fm_qat_velocity_net.plan"
    trt_report_filename: str = "fm_qat_tensorrt_report.json"
    trt_reference_pretrained_path: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 <= self.qat_activation_observer_decay < 1.0:
            raise ValueError("`qat_activation_observer_decay` must be in [0, 1).")
        if self.qat_activation_eps <= 0:
            raise ValueError("`qat_activation_eps` must be > 0.")
        if self.trt_input_batch_size <= 0:
            raise ValueError("`trt_input_batch_size` must be > 0.")
        if self.trt_dynamic_batch_max < self.trt_input_batch_size:
            raise ValueError("`trt_dynamic_batch_max` must be >= `trt_input_batch_size`.")
        if self.trt_opset_version <= 0:
            raise ValueError("`trt_opset_version` must be > 0.")
        if self.trt_workspace_size_mb <= 0:
            raise ValueError("`trt_workspace_size_mb` must be > 0.")
