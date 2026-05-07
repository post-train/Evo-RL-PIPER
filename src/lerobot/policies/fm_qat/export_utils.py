#!/usr/bin/env python

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from lerobot.policies.fm.modeling_fm import FlowMatchingPolicy
from lerobot.policies.fm_qat.configuration_fm_qat import FlowMatchingQATConfig

logger = logging.getLogger(__name__)


class VelocityNetExportWrapper(nn.Module):
    def __init__(self, flow_matching_model: nn.Module):
        super().__init__()
        self.flow_matching_model = flow_matching_model

    def forward(self, x_t: Tensor, time: Tensor, global_cond: Tensor) -> Tensor:
        return self.flow_matching_model.velocity_net(x_t, time, global_cond=global_cond)


def export_policy_to_tensorrt(
    policy: nn.Module,
    config: FlowMatchingQATConfig,
    save_directory: Path,
) -> None:
    if not config.trt_export_on_save:
        return

    if not torch.cuda.is_available():
        logger.warning("Skipping TensorRT export because CUDA is unavailable.")
        return

    report: dict[str, Any] = {
        "policy_type": config.type,
        "status": "skipped",
        "onnx_path": str(save_directory / config.trt_onnx_filename),
        "engine_path": str(save_directory / config.trt_engine_filename),
        "reference_pretrained_path": config.trt_reference_pretrained_path,
    }

    was_training = policy.training
    try:
        policy.eval()
        report.update(_export_policy_to_tensorrt_impl(policy, config, save_directory))
        report["status"] = "ok"
    except Exception as exc:
        logger.exception("TensorRT export for flow_matching_qat failed: %s", exc)
        report["status"] = "failed"
        report["error"] = str(exc)
    finally:
        if was_training:
            policy.train()

    report_path = save_directory / config.trt_report_filename
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))


def _export_policy_to_tensorrt_impl(
    policy: nn.Module,
    config: FlowMatchingQATConfig,
    save_directory: Path,
) -> dict[str, Any]:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT Python package is not installed.") from exc

    flow_matching_model = getattr(policy, "flow_matching")
    device = next(flow_matching_model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError(f"TensorRT export requires the policy on CUDA, got device={device}.")

    wrapper = VelocityNetExportWrapper(flow_matching_model).eval()
    example_inputs = _build_example_inputs(config=config, device=device)
    onnx_path = save_directory / config.trt_onnx_filename
    engine_path = save_directory / config.trt_engine_filename

    torch.onnx.export(
        wrapper,
        example_inputs,
        onnx_path,
        input_names=["x_t", "time", "global_cond"],
        output_names=["velocity"],
        opset_version=config.trt_opset_version,
        dynamic_axes={
            "x_t": {0: "batch"},
            "time": {0: "batch"},
            "global_cond": {0: "batch"},
            "velocity": {0: "batch"},
        },
    )

    _build_trt_engine(
        onnx_path=onnx_path,
        engine_path=engine_path,
        config=config,
        trt=trt,
        sample_inputs=example_inputs,
    )

    ref_output = wrapper(*example_inputs).detach().float().cpu()
    trt_output = _run_tensorrt_engine(engine_path=engine_path, sample_inputs=example_inputs, trt=trt)

    report: dict[str, Any] = {
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "input_shapes": {
            "x_t": list(example_inputs[0].shape),
            "time": list(example_inputs[1].shape),
            "global_cond": list(example_inputs[2].shape),
        },
        "qat_pytorch_vs_tensorrt": _compute_error_metrics(ref_output, trt_output),
    }

    if config.trt_reference_pretrained_path:
        report["reference_fm_vs_tensorrt"] = _compare_reference_policy_to_tensorrt(
            reference_path=config.trt_reference_pretrained_path,
            config=config,
            example_inputs=example_inputs,
            trt_output=trt_output,
        )

    return report


def _build_example_inputs(config: FlowMatchingQATConfig, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    batch_size = config.trt_input_batch_size
    action_dim = config.action_feature.shape[0]
    cond_dim = _get_global_condition_dim(config)

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    x_t = torch.randn(batch_size, config.horizon, action_dim, device=device, generator=generator)
    time = torch.linspace(1.0, 0.0, batch_size, device=device)
    global_cond = torch.randn(batch_size, cond_dim, device=device, generator=generator)
    return x_t, time, global_cond


def _get_global_condition_dim(config: FlowMatchingQATConfig) -> int:
    cond_dim = config.robot_state_feature.shape[0]
    if config.image_features:
        num_images = len(config.image_features)
        cond_dim += config.spatial_softmax_num_keypoints * 2 * num_images
    if config.env_state_feature:
        cond_dim += config.env_state_feature.shape[0]
    return cond_dim * config.n_obs_steps


def _build_trt_engine(
    onnx_path: Path,
    engine_path: Path,
    config: FlowMatchingQATConfig,
    trt,
    sample_inputs: tuple[Tensor, Tensor, Tensor],
) -> None:
    logger_trt = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger_trt)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger_trt)

    if not parser.parse(onnx_path.read_bytes()):
        errors = [parser.get_error(i).desc() for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))

    config_builder = builder.create_builder_config()
    workspace_bytes = config.trt_workspace_size_mb * 1024 * 1024
    if hasattr(config_builder, "set_memory_pool_limit"):
        config_builder.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config_builder.max_workspace_size = workspace_bytes

    if builder.platform_has_fast_fp16:
        config_builder.set_flag(trt.BuilderFlag.FP16)
    if hasattr(trt.BuilderFlag, "INT8"):
        config_builder.set_flag(trt.BuilderFlag.INT8)

    profile = builder.create_optimization_profile()
    sample_x, sample_time, sample_cond = sample_inputs
    batch = config.trt_input_batch_size
    max_batch = config.trt_dynamic_batch_max
    horizon = sample_x.shape[1]
    action_dim = sample_x.shape[2]
    cond_dim = sample_cond.shape[1]

    profile.set_shape("x_t", (1, horizon, action_dim), (batch, horizon, action_dim), (max_batch, horizon, action_dim))
    profile.set_shape("time", (1,), (batch,), (max_batch,))
    profile.set_shape("global_cond", (1, cond_dim), (batch, cond_dim), (max_batch, cond_dim))
    config_builder.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config_builder)
    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build returned None.")
    engine_path.write_bytes(bytes(serialized_engine))


def _run_tensorrt_engine(engine_path: Path, sample_inputs: tuple[Tensor, Tensor, Tensor], trt) -> Tensor:
    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
    except ImportError as exc:
        raise RuntimeError("pycuda is required to run TensorRT inference for precision comparison.") from exc

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine.")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")

    host_inputs = [tensor.detach().float().cpu().numpy() for tensor in sample_inputs]
    num_bindings = engine.num_bindings
    device_buffers: list[Any] = [None] * num_bindings
    host_outputs: list[Any] = []
    bindings: list[int] = [0] * num_bindings
    stream = cuda.Stream()

    for binding_idx in range(num_bindings):
        if engine.binding_is_input(binding_idx):
            host_array = host_inputs.pop(0)
            context.set_binding_shape(binding_idx, host_array.shape)
            device_buffers[binding_idx] = cuda.mem_alloc(host_array.nbytes)
            bindings[binding_idx] = int(device_buffers[binding_idx])
            cuda.memcpy_htod_async(device_buffers[binding_idx], host_array, stream)
        else:
            shape = tuple(context.get_binding_shape(binding_idx))
            dtype = trt.nptype(engine.get_binding_dtype(binding_idx))
            host_array = torch.empty(shape, dtype=torch.float32).cpu().numpy().astype(dtype, copy=False)
            device_buffers[binding_idx] = cuda.mem_alloc(host_array.nbytes)
            bindings[binding_idx] = int(device_buffers[binding_idx])
            host_outputs.append((binding_idx, host_array))

    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    outputs = []
    for binding_idx, host_array in host_outputs:
        cuda.memcpy_dtoh_async(host_array, device_buffers[binding_idx], stream)
        outputs.append(host_array)
    stream.synchronize()

    if len(outputs) != 1:
        raise RuntimeError(f"Expected one TensorRT output, got {len(outputs)}.")
    return torch.from_numpy(outputs[0]).float()


def _compute_error_metrics(reference: Tensor, candidate: Tensor) -> dict[str, float]:
    diff = candidate - reference
    ref_norm = torch.norm(reference).item()
    return {
        "mae": float(diff.abs().mean().item()),
        "rmse": float(torch.sqrt((diff * diff).mean()).item()),
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float(torch.norm(diff).item() / max(ref_norm, 1e-12)),
    }


def _compare_reference_policy_to_tensorrt(
    reference_path: str,
    config: FlowMatchingQATConfig,
    example_inputs: tuple[Tensor, Tensor, Tensor],
    trt_output: Tensor,
) -> dict[str, float]:
    reference_policy = FlowMatchingPolicy.from_pretrained(reference_path, local_files_only=True)
    reference_policy = reference_policy.to(example_inputs[0].device)
    reference_policy.eval()

    wrapper = VelocityNetExportWrapper(reference_policy.flow_matching).eval()
    with torch.no_grad():
        ref_output = wrapper(*example_inputs).detach().float().cpu()
    del config
    return _compute_error_metrics(ref_output, trt_output)
