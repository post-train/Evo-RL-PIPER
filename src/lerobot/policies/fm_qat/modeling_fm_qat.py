#!/usr/bin/env python

import logging
import math
from collections import deque
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
import torchvision
from diffusers.training_utils import EMAModel
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.policies.fm.modeling_fm import ActionSelectKwargs, SpatialSoftmax, _replace_submodules
from lerobot.policies.fm_qat.configuration_fm_qat import FlowMatchingQATConfig
from lerobot.policies.fm_qat.export_utils import export_policy_to_tensorrt
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

logger = logging.getLogger(__name__)


def _symmetric_scale_from_amax(amax: Tensor, eps: float) -> Tensor:
    return torch.clamp(amax / 127.0, min=eps)


def _fake_quantize_tensor(x: Tensor, scale: Tensor) -> Tensor:
    scale = scale.to(device=x.device, dtype=x.dtype)
    q = torch.clamp(torch.round(x / scale), -127, 127)
    dequant = q * scale
    return x + (dequant - x).detach()


class QATActivationObserver(nn.Module):
    def __init__(self, decay: float, eps: float):
        super().__init__()
        self.decay = decay
        self.eps = eps
        self.register_buffer("amax", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    def forward(self, x: Tensor) -> Tensor:
        current_amax = x.detach().abs().amax()
        if self.training:
            if bool(self.initialized):
                self.amax.mul_(self.decay).add_(current_amax * (1.0 - self.decay))
            else:
                self.amax.copy_(current_amax)
                self.initialized.fill_(True)
        else:
            if not bool(self.initialized):
                self.amax.copy_(current_amax)
                self.initialized.fill_(True)

        scale = _symmetric_scale_from_amax(self.amax, self.eps)
        return _fake_quantize_tensor(x, scale)


class QATModuleMixin:
    def _make_activation_observer(self, config: FlowMatchingQATConfig) -> QATActivationObserver:
        return QATActivationObserver(
            decay=config.qat_activation_observer_decay,
            eps=config.qat_activation_eps,
        )

    def _fake_quant_weight(self, weight: Tensor, eps: float) -> Tensor:
        reduce_dims = tuple(range(1, weight.dim()))
        amax = weight.detach().abs().amax(dim=reduce_dims, keepdim=True)
        scale = _symmetric_scale_from_amax(amax, eps)
        return _fake_quantize_tensor(weight, scale)


class QATLinear(nn.Linear, QATModuleMixin):
    def __init__(self, in_features: int, out_features: int, config: FlowMatchingQATConfig, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        self.qat_enabled = config.qat_enabled
        self.qat_activation = self._make_activation_observer(config)
        self.qat_eps = config.qat_activation_eps

    def forward(self, x: Tensor) -> Tensor:
        if not self.qat_enabled:
            return F.linear(x, self.weight, self.bias)
        x = self.qat_activation(x)
        weight = self._fake_quant_weight(self.weight, self.qat_eps)
        return F.linear(x, weight, self.bias)


class QATConv1d(nn.Conv1d, QATModuleMixin):
    def __init__(self, *args, config: FlowMatchingQATConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_enabled = config.qat_enabled
        self.qat_activation = self._make_activation_observer(config)
        self.qat_eps = config.qat_activation_eps

    def forward(self, x: Tensor) -> Tensor:
        if not self.qat_enabled:
            return self._conv_forward(x, self.weight, self.bias)
        x = self.qat_activation(x)
        weight = self._fake_quant_weight(self.weight, self.qat_eps)
        return self._conv_forward(x, weight, self.bias)


class QATConv2d(nn.Conv2d, QATModuleMixin):
    def __init__(self, *args, config: FlowMatchingQATConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_enabled = config.qat_enabled
        self.qat_activation = self._make_activation_observer(config)
        self.qat_eps = config.qat_activation_eps

    def forward(self, x: Tensor) -> Tensor:
        if not self.qat_enabled:
            return self._conv_forward(x, self.weight, self.bias)
        x = self.qat_activation(x)
        weight = self._fake_quant_weight(self.weight, self.qat_eps)
        return self._conv_forward(x, weight, self.bias)


class QATConvTranspose1d(nn.ConvTranspose1d, QATModuleMixin):
    def __init__(self, *args, config: FlowMatchingQATConfig, **kwargs):
        super().__init__(*args, **kwargs)
        self.qat_enabled = config.qat_enabled
        self.qat_activation = self._make_activation_observer(config)
        self.qat_eps = config.qat_activation_eps

    def forward(self, x: Tensor, output_size=None) -> Tensor:
        if not self.qat_enabled:
            return super().forward(x, output_size=output_size)
        x = self.qat_activation(x)
        weight = self._fake_quant_weight(self.weight, self.qat_eps)
        output_padding = self._output_padding(
            x,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,
            1,
            self.dilation,
        )
        return F.conv_transpose1d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class FlowMatchingQATPolicy(PreTrainedPolicy):
    config_class = FlowMatchingQATConfig
    name = "flow_matching_qat"

    def __init__(self, config: FlowMatchingQATConfig, **kwargs):
        super().__init__(config)
        del kwargs
        config.validate_features()
        self.config = config
        self._queues = None

        self.rtc_processor: RTCProcessor | None = None
        if config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(config.rtc_config)

        self.flow_matching = FlowMatchingQATModel(config)

        self.ema = None
        self._ema_device_set = False
        if config.use_ema:
            self.ema = EMAModel(
                self.flow_matching.parameters(),
                power=config.ema_power,
            )

        self.reset()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        policy = super().from_pretrained(*args, **kwargs)
        policy.maybe_enable_compiled_velocity_net()
        return policy

    def maybe_enable_compiled_velocity_net(self) -> None:
        if not self.config.compile_velocity_net:
            return
        if getattr(self.flow_matching, "_velocity_net_compiled", False):
            return

        if not hasattr(torch, "compile"):
            logger.warning("torch.compile is unavailable in this PyTorch build; skipping velocity_net compilation.")
            return

        try:
            self.flow_matching.velocity_net = torch.compile(
                self.flow_matching.velocity_net,
                mode=self.config.compile_mode,
            )
            self.flow_matching._velocity_net_compiled = True
            logger.info(
                "Enabled torch.compile for FM QAT velocity_net with mode=%s",
                self.config.compile_mode,
            )
            self.flow_matching.warmup_compiled_velocity_net()
        except Exception as exc:
            logger.warning("Failed to compile FM QAT velocity_net; continuing without compile. Error: %s", exc)
            self.flow_matching._velocity_net_compiled = False

    def get_optim_params(self) -> dict:
        if self.config.backbone_lr_scale != 1.0 and self.config.image_features:
            backbone_params = []
            other_params = []
            for name, param in self.flow_matching.named_parameters():
                if "backbone" in name:
                    backbone_params.append(param)
                else:
                    other_params.append(param)
            backbone_lr = self.config.optimizer_lr * self.config.backbone_lr_scale
            return [
                {"params": other_params},
                {"params": backbone_params, "lr": backbone_lr},
            ]
        return self.flow_matching.parameters()

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.flow_matching.generate_actions(
            batch,
            noise=kwargs.get("noise"),
            rtc_processor=self.rtc_processor if self._rtc_enabled() else None,
            inference_delay=kwargs.get("inference_delay"),
            prev_chunk_left_over=kwargs.get("prev_chunk_left_over"),
            execution_horizon=kwargs.get("execution_horizon"),
        )

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=1)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, **kwargs)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        action = self._queues[ACTION].popleft()
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, None]:
        if self.config.image_features:
            batch = dict(batch)
            first_key = next(iter(self.config.image_features))
            first_img = batch[first_key]
            if first_img.dim() == 4:
                images = [batch[key].unsqueeze(1) for key in self.config.image_features]
            else:
                images = [batch[key] for key in self.config.image_features]
            batch[OBS_IMAGES] = torch.stack(images, dim=2)
        loss = self.flow_matching.compute_loss(batch, reduction=reduction)
        return loss, None

    def update(self):
        if self.ema is not None:
            if not self._ema_device_set:
                device = next(self.flow_matching.parameters()).device
                self.ema.to(device)
                self._ema_device_set = True
            self.ema.step(self.flow_matching.parameters())

    def use_ema_weights(self):
        if self.ema is not None:
            self.ema.store(self.flow_matching.parameters())
            self.ema.copy_to(self.flow_matching.parameters())

    def restore_training_weights(self):
        if self.ema is not None:
            self.ema.restore(self.flow_matching.parameters())

    def _save_pretrained(self, save_directory: Path) -> None:
        super()._save_pretrained(save_directory)
        export_policy_to_tensorrt(self, self.config, save_directory)


class FlowMatchingQATModel(nn.Module):
    def __init__(self, config: FlowMatchingQATConfig):
        super().__init__()
        self.config = config
        self._velocity_net_compiled = False

        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [FMRgbEncoderQAT(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = FMRgbEncoderQAT(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        self.velocity_net = FMConditionalUnet1dQAT(
            config,
            global_cond_dim=global_cond_dim * config.n_obs_steps,
        )

    @torch.no_grad()
    def warmup_compiled_velocity_net(self) -> None:
        if not self._velocity_net_compiled:
            return
        warmup_num_chunks = self.config.compile_warmup_num_chunks
        if warmup_num_chunks <= 0:
            return

        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action_dim = self.config.action_feature.shape[0]
        obs_state_dim = self.config.robot_state_feature.shape[0]
        cond_dim = obs_state_dim
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                cond_dim += self.rgb_encoder[0].feature_dim * num_images
            else:
                cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            cond_dim += self.config.env_state_feature.shape[0]
        cond_dim *= self.config.n_obs_steps

        was_training = self.velocity_net.training
        self.velocity_net.eval()
        try:
            for _ in range(warmup_num_chunks):
                x_t = torch.randn(
                    (1, self.config.horizon, action_dim),
                    dtype=dtype,
                    device=device,
                )
                time = torch.full((1,), 1.0, dtype=dtype, device=device)
                global_cond = torch.zeros((1, cond_dim), dtype=dtype, device=device)
                _ = self.velocity_net(x_t, time, global_cond=global_cond)
        finally:
            if was_training:
                self.velocity_net.train()

    def generate_actions(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        actions = self._sample_flow(
            batch_size,
            global_cond=global_cond,
            noise=noise,
            rtc_processor=rtc_processor,
            inference_delay=inference_delay,
            prev_chunk_left_over=prev_chunk_left_over,
            execution_horizon=execution_horizon,
        )

        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        if self.config.clip_sample:
            actions = actions.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

        return actions

    def _sample_flow(
        self,
        batch_size: int,
        global_cond: Tensor | None = None,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action_dim = self.config.action_feature.shape[0]

        x_t = (
            noise
            if noise is not None
            else torch.randn(
                size=(batch_size, self.config.horizon, action_dim),
                dtype=dtype,
                device=device,
            )
        )

        use_rtc = rtc_processor is not None
        use_ode = self.config.solver_type == "dopri5" and not use_rtc

        if use_ode:
            x_0 = self._sample_ode(x_t, global_cond)
        else:
            x_0 = self._sample_euler(
                x_t,
                global_cond,
                num_steps=num_steps,
                rtc_processor=rtc_processor,
                inference_delay=inference_delay,
                prev_chunk_left_over=prev_chunk_left_over,
                execution_horizon=execution_horizon,
            )

        return x_0

    def _sample_euler(
        self,
        x_t: Tensor,
        global_cond: Tensor | None,
        num_steps: int | None = None,
        rtc_processor: RTCProcessor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        dt = -1.0 / num_steps

        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.full((x_t.shape[0],), time, dtype=x_t.dtype, device=x_t.device)

            def denoise_step_partial(input_x_t, current_time=time_tensor):
                return self.velocity_net(input_x_t, current_time, global_cond=global_cond)

            if rtc_processor is not None:
                v_t = rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial(x_t)

            x_t = x_t + dt * v_t

            if self.config.clip_sample:
                x_t = x_t.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

            if rtc_processor is not None and rtc_processor.is_debug_enabled():
                rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def _sample_ode(self, x_t: Tensor, global_cond: Tensor | None) -> Tensor:
        from torchdiffeq import odeint

        device = x_t.device

        class ODEFunc(nn.Module):
            def __init__(self, velocity_net, global_cond):
                super().__init__()
                self.velocity_net = velocity_net
                self.global_cond = global_cond

            def forward(self, t, x):
                time_tensor = t.expand(x.shape[0])
                return self.velocity_net(x, time_tensor, global_cond=self.global_cond)

        ode_func = ODEFunc(self.velocity_net, global_cond)
        t_span = torch.tensor([1.0, 0.0], device=device)

        solution = odeint(
            ode_func,
            x_t,
            t_span,
            method="dopri5",
            atol=self.config.ode_atol,
            rtol=self.config.ode_rtol,
        )
        return solution[-1]

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> Tensor:
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        trajectory = batch[ACTION]
        batch_size = trajectory.shape[0]

        noise = torch.randn_like(trajectory)
        epsilon = 1e-5
        t = torch.rand(batch_size, device=trajectory.device) * (1.0 - epsilon) + epsilon

        t_expanded = t[:, None, None]
        x_t = t_expanded * noise + (1.0 - t_expanded) * trajectory
        u_t = noise - trajectory

        v_pred = self.velocity_net(x_t, t, global_cond=global_cond)

        loss = F.mse_loss(v_pred, u_t, reduction="none")

        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        per_sample_loss = loss.mean(dim=(1, 2))

        if self.config.rollout_consistency_weight > 0:
            rollout_num_steps = self.config.rollout_consistency_num_steps or self.config.num_inference_steps
            rollout_pred = self._sample_flow(
                batch_size=batch_size,
                global_cond=global_cond,
                noise=noise.clone(),
                num_steps=rollout_num_steps,
            )
            rollout_loss = F.mse_loss(rollout_pred, trajectory, reduction="none")
            if self.config.do_mask_loss_for_padding:
                rollout_loss = rollout_loss * in_episode_bound.unsqueeze(-1)
            per_sample_loss = per_sample_loss + self.config.rollout_consistency_weight * rollout_loss.mean(dim=(1, 2))

        if reduction == "mean":
            return per_sample_loss.mean()
        if reduction == "none":
            return per_sample_loss
        raise ValueError(f"Unsupported reduction: {reduction}")

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)


class FMRgbEncoderQAT(nn.Module):
    def __init__(self, config: FlowMatchingQATConfig):
        super().__init__()
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            self.maybe_random_crop = (
                torchvision.transforms.RandomCrop(config.crop_shape) if config.crop_is_random else self.center_crop
            )
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the weights!")
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        if config.qat_quantize_backbone:
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.Conv2d),
                func=lambda x: _make_qat_conv2d_from_conv(x, config),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = QATLinear(config.spatial_softmax_num_keypoints * 2, self.feature_dim, config=config)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _make_qat_conv2d_from_conv(conv: nn.Conv2d, config: FlowMatchingQATConfig) -> QATConv2d:
    qat_conv = QATConv2d(
        conv.in_channels,
        conv.out_channels,
        conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        config=config,
    )
    qat_conv.weight.data.copy_(conv.weight.data)
    if conv.bias is not None:
        qat_conv.bias.data.copy_(conv.bias.data)
    return qat_conv


class FMSinusoidalPosEmbQAT(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0) * 1000.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class FMConv1dBlockQAT(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups, config: FlowMatchingQATConfig):
        super().__init__()
        self.block = nn.Sequential(
            QATConv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2, config=config),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class FMConditionalResidualBlock1dQAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        config: FlowMatchingQATConfig,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()
        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = FMConv1dBlockQAT(in_channels, out_channels, kernel_size, n_groups=n_groups, config=config)

        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), QATLinear(cond_dim, cond_channels, config=config))

        self.conv2 = FMConv1dBlockQAT(out_channels, out_channels, kernel_size, n_groups=n_groups, config=config)

        self.residual_conv = (
            QATConv1d(in_channels, out_channels, 1, config=config) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)

        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


class FMConditionalUnet1dQAT(nn.Module):
    def __init__(self, config: FlowMatchingQATConfig, global_cond_dim: int):
        super().__init__()
        self.config = config

        self.time_encoder = nn.Sequential(
            FMSinusoidalPosEmbQAT(config.time_embed_dim),
            QATLinear(config.time_embed_dim, config.time_embed_dim * 4, config=config),
            nn.Mish(),
            QATLinear(config.time_embed_dim * 4, config.time_embed_dim, config=config),
        )

        cond_dim = config.time_embed_dim + global_cond_dim

        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "config": config,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FMConditionalResidualBlock1dQAT(dim_in, dim_out, **common_res_block_kwargs),
                        FMConditionalResidualBlock1dQAT(dim_out, dim_out, **common_res_block_kwargs),
                        QATConv1d(dim_out, dim_out, 3, stride=2, padding=1, config=config)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                FMConditionalResidualBlock1dQAT(
                    config.down_dims[-1],
                    config.down_dims[-1],
                    **common_res_block_kwargs,
                ),
                FMConditionalResidualBlock1dQAT(
                    config.down_dims[-1],
                    config.down_dims[-1],
                    **common_res_block_kwargs,
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FMConditionalResidualBlock1dQAT(dim_in * 2, dim_out, **common_res_block_kwargs),
                        FMConditionalResidualBlock1dQAT(dim_out, dim_out, **common_res_block_kwargs),
                        QATConvTranspose1d(dim_out, dim_out, 4, stride=2, padding=1, config=config)
                        if not is_last
                        else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            FMConv1dBlockQAT(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size, n_groups=config.n_groups, config=config),
            QATConv1d(config.down_dims[0], config.action_feature.shape[0], 1, config=config),
        )

    def forward(self, x: Tensor, time: Tensor, global_cond: Tensor | None = None) -> Tensor:
        x = einops.rearrange(x, "b t d -> b d t")

        time_embed = self.time_encoder(time)

        global_feature = torch.cat([time_embed, global_cond], dim=-1) if global_cond is not None else time_embed

        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, "b d t -> b t d")
        return x
