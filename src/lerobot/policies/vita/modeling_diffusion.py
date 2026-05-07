#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from pathlib import Path

import einops
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d
from diffusers.training_utils import EMAModel

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.policies.utils import populate_queues
from lerobot.policies.vita.configuration_diffusion import VitaConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE

try:
    from torchcfm.conditional_flow_matching import (
        ConditionalFlowMatcher as TorchCFMConditionalFlowMatcher,
        ExactOptimalTransportConditionalFlowMatcher as TorchCFMExactMatcher,
        SchrodingerBridgeConditionalFlowMatcher as TorchCFMSchrodingerMatcher,
        TargetConditionalFlowMatcher as TorchCFMTargetMatcher,
    )
except ModuleNotFoundError:
    TorchCFMConditionalFlowMatcher = None
    TorchCFMTargetMatcher = None
    TorchCFMSchrodingerMatcher = None
    TorchCFMExactMatcher = None


class VitaPolicy(PreTrainedPolicy):
    config_class = VitaConfig
    name = "vita"

    def __init__(self, config: VitaConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        del kwargs
        config.validate_features()
        self.config = config
        self.obs_horizon = config.obs_horizon
        self.action_horizon = config.action_horizon
        self.pred_horizon = config.pred_horizon
        self._queues = None
        self._action_queue = None
        self.model = VitaModel(config, dataset_stats=dataset_stats)
        self.ema = EMAModel(parameters=self.model.parameters(), power=config.ema_power) if config.use_ema else None
        self.rtc_processor: RTCProcessor | None = None
        if config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(config.rtc_config)
        self.reset()

    def get_optim_params(self) -> list[dict]:
        backbone_params = list(self.model.observer.backbone.parameters())
        backbone_param_ids = {id(param) for param in backbone_params}
        other_params = [param for param in self.model.parameters() if id(param) not in backbone_param_ids]
        return [
            {"params": other_params},
            {"params": backbone_params, "lr": self.config.optimizer_lr_backbone},
        ]

    def _save_pretrained(self, save_directory: Path) -> None:
        if self.ema is None:
            super()._save_pretrained(save_directory)
            return
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        try:
            super()._save_pretrained(save_directory)
        finally:
            self.ema.restore(self.model.parameters())

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
        }
        self._action_queue = deque([], maxlen=self.action_horizon)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        inference_delay: int | None = None,
        prev_chunk_left_over: Tensor | None = None,
        execution_horizon: int | None = None,
    ) -> Tensor:
        del noise
        del execution_horizon
        batch = self._prepare_observation_batch(batch)
        self._maybe_populate_history(batch)
        stacked = self._stack_history_batch()
        with self._ema_scope():
            pred_actions = self.model.generate_actions(stacked)
        pred_actions = pred_actions[:, : self.action_horizon]
        if prev_chunk_left_over is not None and inference_delay and inference_delay > 0:
            pred_actions = self._apply_prefix_guidance(
                pred_actions,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
            )
        pred_actions = self._apply_inference_safety_limits(pred_actions, stacked)
        return pred_actions

    def _queue_observation(self, batch: dict[str, Tensor]) -> None:
        self._queues = populate_queues(self._queues, batch)

    def _prepare_observation_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)
        if ACTION in batch:
            batch.pop(ACTION)
        if OBS_IMAGES not in batch and self.config.image_features:
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return batch

    def _maybe_populate_history(self, batch: dict[str, Tensor]) -> None:
        if self._queues is None:
            self.reset()
        state_history = self._queues[OBS_STATE]
        state_value = batch.get(OBS_STATE)
        if state_value is None:
            return
        expected_len = state_history.maxlen
        already_stacked = state_value.ndim >= 3 and state_value.shape[1] == expected_len
        if already_stacked:
            restored_queues: dict[str, deque] = {}
            for key, queue_value in self._queues.items():
                if key not in batch:
                    restored_queues[key] = deque(maxlen=queue_value.maxlen)
                    continue
                restored_queues[key] = deque(
                    (frame.detach().clone() for frame in batch[key].unbind(dim=1)),
                    maxlen=queue_value.maxlen,
                )
            self._queues = restored_queues
            return
        self._queue_observation(batch)

    def _stack_history_batch(self) -> dict[str, Tensor]:
        if self._queues is None:
            raise RuntimeError("VITA observation history is not initialized.")
        return {key: torch.stack(list(queue_value), dim=1) for key, queue_value in self._queues.items() if len(queue_value) > 0}

    def _apply_prefix_guidance(
        self,
        pred_actions: Tensor,
        *,
        prev_chunk_left_over: Tensor,
        inference_delay: int,
    ) -> Tensor:
        if prev_chunk_left_over.ndim == 2:
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)
        prefix_steps = min(inference_delay, pred_actions.shape[1], prev_chunk_left_over.shape[1])
        if prefix_steps <= 0:
            return pred_actions

        guided_actions = pred_actions.clone()
        guided_actions[:, :prefix_steps, :] = prev_chunk_left_over[:, :prefix_steps, :].to(
            device=guided_actions.device,
            dtype=guided_actions.dtype,
        )

        blend_steps = min(2, guided_actions.shape[1] - prefix_steps, prev_chunk_left_over.shape[1] - prefix_steps)
        if blend_steps > 0:
            prev_blend = prev_chunk_left_over[:, prefix_steps : prefix_steps + blend_steps, :].to(
                device=guided_actions.device,
                dtype=guided_actions.dtype,
            )
            weights = torch.linspace(
                1.0 / (blend_steps + 1),
                blend_steps / (blend_steps + 1),
                steps=blend_steps,
                device=guided_actions.device,
                dtype=guided_actions.dtype,
            ).view(1, blend_steps, 1)
            guided_actions[:, prefix_steps : prefix_steps + blend_steps, :] = (
                (1.0 - weights) * prev_blend
                + weights * guided_actions[:, prefix_steps : prefix_steps + blend_steps, :]
            )
        return guided_actions

    def _apply_inference_safety_limits(
        self,
        pred_actions: Tensor,
        stacked_observation: dict[str, Tensor],
    ) -> Tensor:
        if not self.config.infer_safe_delta_enabled:
            return pred_actions
        state_history = stacked_observation.get(OBS_STATE)
        if state_history is None or state_history.shape[-1] != pred_actions.shape[-1]:
            return pred_actions

        max_delta = float(self.config.infer_safe_delta_max_norm)
        if max_delta <= 0:
            return pred_actions

        clamped_actions = pred_actions.clone()
        reference = state_history[:, -1, :].to(device=pred_actions.device, dtype=pred_actions.dtype)
        for step in range(clamped_actions.shape[1]):
            delta = clamped_actions[:, step, :] - reference
            delta = delta.clamp(min=-max_delta, max=max_delta)
            clamped_actions[:, step, :] = reference + delta
            reference = clamped_actions[:, step, :]
        return clamped_actions

    def _ensure_ema_device(self) -> None:
        if self.ema is None:
            return
        model_device = next(self.model.parameters()).device
        if not hasattr(self.ema, "shadow_params"):
            return
        for idx, shadow_param in enumerate(self.ema.shadow_params):
            if shadow_param.device != model_device:
                self.ema.shadow_params[idx] = shadow_param.to(device=model_device)
        if hasattr(self.ema, "temp_stored_params") and self.ema.temp_stored_params is not None:
            self.ema.temp_stored_params = [
                param.to(device=model_device) for param in self.ema.temp_stored_params
            ]

    @contextmanager
    def _ema_scope(self):
        if self.ema is None:
            yield
            return
        self._ensure_ema_device()
        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())
        try:
            yield
        finally:
            self.ema.restore(self.model.parameters())

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        batch = self._prepare_observation_batch(batch)
        self._queue_observation(batch)
        refresh_after_steps = self.config.action_queue_refresh_steps
        if self.config.infer_no_rtc_replan_every_step and not self._rtc_enabled():
            refresh_after_steps = max(1, self.config.infer_no_rtc_refresh_steps)
        refresh_threshold = max(self.action_horizon - refresh_after_steps, 0)
        if len(self._action_queue) == 0 or len(self._action_queue) <= refresh_threshold:
            previous_actions = None
            if not self._rtc_enabled() and len(self._action_queue) > 0:
                previous_actions = torch.stack(list(self._action_queue), dim=1)
            pred_actions = self.predict_action_chunk(batch)
            pred_actions = pred_actions[:, : self.action_horizon]
            if previous_actions is not None:
                pred_actions = self._blend_with_existing_plan(pred_actions, previous_actions)
            self._action_queue.clear()
            self._action_queue.extend(pred_actions.transpose(0, 1))
        action = self._action_queue.popleft()
        return action

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _blend_with_existing_plan(
        self,
        pred_actions: Tensor,
        previous_actions: Tensor,
    ) -> Tensor:
        blend_steps = min(
            self.config.infer_no_rtc_blend_steps,
            pred_actions.shape[1],
            previous_actions.shape[1],
        )
        if blend_steps <= 0:
            return pred_actions

        blended_actions = pred_actions.clone()
        prev = previous_actions[:, :blend_steps, :].to(device=pred_actions.device, dtype=pred_actions.dtype)
        weights = torch.linspace(
            1.0 / (blend_steps + 1),
            blend_steps / (blend_steps + 1),
            steps=blend_steps,
            device=pred_actions.device,
            dtype=pred_actions.dtype,
        ).view(1, blend_steps, 1)
        blended_actions[:, :blend_steps, :] = (1.0 - weights) * prev + weights * blended_actions[:, :blend_steps, :]
        return blended_actions

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        batch = dict(batch)
        batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.model.compute_loss(batch, reduction=reduction)

    def update(self):
        if self.ema is not None:
            self._ensure_ema_device()
            self.ema.step(self.model.parameters())


class VitaModel(nn.Module):
    def __init__(self, config: VitaConfig, dataset_stats: dict | None = None):
        super().__init__()
        self.config = config
        self.obs_horizon = config.obs_horizon
        self.action_horizon = config.action_horizon
        self.pred_horizon = config.pred_horizon
        self.num_sampling_steps = config.num_sampling_steps
        self.observer = ResNetObserver(config)
        self.obs_dim = self.observer.output_dim
        self.obs_encoder = nn.Linear(self.obs_dim, config.latent_dim)
        self.FM = TorchFlowMatcherFactory.make(config.flow_matcher_name, config.flow_sigma, config.num_sampling_steps)
        self.freeze_action_encoder = config.freeze_action_encoder
        self.freeze_action_decoder = config.freeze_action_decoder
        self.flow_action_recon_weight = config.flow_action_recon_weight
        self.enc_action_recon_weight = config.enc_action_recon_weight
        self.action_kl_weight = config.action_kl_weight
        self.use_action_vae = config.use_variational
        self.latent_dim = config.latent_dim
        self.enc_contrastive_weight = config.enc_contrastive_weight
        self.flow_contrastive_weight = config.flow_contrastive_weight

        if config.action_recon_loss_type == "l1":
            self.recon_loss_fn = F.l1_loss
        else:
            self.recon_loss_fn = F.mse_loss

        action_dim = config.action_feature.shape[0]
        self.action_encoder, self.action_decoder = build_action_autoencoder(config, action_dim)
        self.flow_net = SimpleFlowNet(
            input_dim=config.latent_dim,
            hidden_dim=config.flow_hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.flow_num_layers,
            mlp_ratio=config.flow_mlp_ratio,
            dropout=config.flow_dropout,
            time_embed_dim=config.flow_time_embed_dim,
        )
        del dataset_stats

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        obs_features = self.observer(batch)
        obs_latents = self.obs_encoder(obs_features)
        gt_actions = batch[ACTION][:, : self.pred_horizon, :]
        batch_size = gt_actions.shape[0]
        metrics: dict[str, float] = {}

        with torch.no_grad() if self.freeze_action_encoder else torch.enable_grad():
            if self.use_action_vae:
                action_posterior, action_latents = self.action_encoder(gt_actions, deterministic=not self.training)
            else:
                action_posterior = None
                action_latents = self.action_encoder(gt_actions)

        flow_loss, _ = self.FM.compute_loss(
            self.flow_net,
            target=action_latents,
            start=obs_latents,
        )
        per_sample_loss = flow_loss
        metrics["flow_loss"] = float(flow_loss.mean().detach().item())

        if self.enc_contrastive_weight > 0:
            contrastive_loss = compute_contrastive_loss(obs_latents.view(batch_size, -1), action_latents.view(batch_size, -1))
            per_sample_loss = per_sample_loss + self.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = float(contrastive_loss.detach().item())

        if not self.freeze_action_encoder and self.use_action_vae and self.action_kl_weight > 0 and action_posterior is not None:
            action_kl_loss = action_posterior.kl()
            per_sample_loss = per_sample_loss + self.action_kl_weight * action_kl_loss
            metrics["action_kl_loss"] = float(action_kl_loss.mean().detach().item())

        if self.config.decode_flow_latents and not self.freeze_action_encoder and not self.freeze_action_decoder:
            action_latents_pred = self.FM.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=obs_latents,
                num_steps=self.num_sampling_steps,
            )

            if self.config.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents, reduction="none").mean(dim=-1)
                per_sample_loss = per_sample_loss + self.config.consistency_weight * consistency_loss
                metrics["consistency_loss"] = float(consistency_loss.mean().detach().item())

            if self.flow_contrastive_weight > 0:
                flow_contrastive_loss = compute_contrastive_loss(
                    obs_latents.view(batch_size, -1), action_latents_pred.view(batch_size, -1)
                )
                per_sample_loss = per_sample_loss + self.flow_contrastive_weight * flow_contrastive_loss
                metrics["flow_contrastive_loss"] = float(flow_contrastive_loss.detach().item())

            if self.flow_action_recon_weight > 0 and not self.freeze_action_decoder:
                actions_recon = self.action_decoder(action_latents_pred)
                flow_recon_loss = self.recon_loss_fn(actions_recon, gt_actions, reduction="none").mean(dim=(1, 2))
                per_sample_loss = per_sample_loss + self.flow_action_recon_weight * flow_recon_loss
                metrics["flow_action_recon_loss"] = float(flow_recon_loss.mean().detach().item())

        if self.enc_action_recon_weight > 0 and not self.freeze_action_decoder:
            actions_recon = self.action_decoder(action_latents)
            enc_recon_loss = self.recon_loss_fn(actions_recon, gt_actions, reduction="none").mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + self.enc_action_recon_weight * enc_recon_loss
            metrics["enc_action_recon_loss"] = float(enc_recon_loss.mean().detach().item())

        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()
        metrics["loss"] = float(loss.mean().detach().item())
        return loss, metrics

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        obs_features = self.observer(batch)
        obs_latents = self.obs_encoder(obs_features)
        action_latents_pred = self.FM.sample(
            self.flow_net,
            shape=(batch_size, self.latent_dim),
            device=obs_latents.device,
            start=obs_latents,
            num_steps=self.num_sampling_steps,
        )
        with torch.no_grad() if self.freeze_action_decoder else torch.enable_grad():
            actions_pred = self.action_decoder(action_latents_pred)
        return actions_pred


class ResNetObserver(nn.Module):
    def __init__(self, config: VitaConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.robot_state_feature.shape[0]
        self.resize_shape = tuple(config.resize_shape)
        self.crop_shape = tuple(config.crop_shape)
        backbone_weights = resolve_torchvision_weights(config.pretrained_backbone_weights)
        norm_layer = FrozenBatchNorm2d if config.use_frozen_batch_norm else nn.BatchNorm2d
        backbone = getattr(torchvision.models, config.vision_backbone)(weights=backbone_weights, norm_layer=norm_layer)
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = self._infer_output_dim()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        states = batch[OBS_STATE]
        if states.ndim == 2:
            states = states.unsqueeze(1)
        states = states.flatten(start_dim=1)

        images = batch[OBS_IMAGES]
        if images.ndim == 5:
            images = images.unsqueeze(1)
        elif images.ndim != 6:
            raise ValueError(
                f"vita expects `{OBS_IMAGES}` to be 5D or 6D, got shape={tuple(images.shape)}"
            )

        b, s, n = images.shape[:3]
        images = einops.rearrange(images, "b s n c h w -> (b s n) c h w")
        if images.shape[-2:] != self.resize_shape:
            images = F.interpolate(images, size=self.resize_shape, mode="bilinear", align_corners=False)
        ch, cw = self.crop_shape
        _, _, h, w = images.shape
        if ch <= h and cw <= w:
            top = torch.randint(0, h - ch + 1, (1,), device=images.device).item()
            left = torch.randint(0, w - cw + 1, (1,), device=images.device).item()
            images = images[..., top : top + ch, left : left + cw]
        img_features = self.pool(self.backbone(images)["feature_map"])
        img_features = einops.rearrange(img_features, "(b s n) c h w -> b (s n c h w)", b=b, s=s, n=n)
        return torch.cat([states, img_features], dim=1)

    def _infer_output_dim(self) -> int:
        image_shape = next(iter(self.config.image_features.values())).shape
        dummy_batch = {
            OBS_STATE: torch.zeros(1, self.config.obs_horizon, self.state_dim, dtype=torch.float32),
            OBS_IMAGES: torch.zeros(
                1,
                self.config.obs_horizon,
                len(self.config.image_features),
                image_shape[0],
                image_shape[1],
                image_shape[2],
                dtype=torch.float32,
            ),
        }
        with torch.inference_mode():
            output = self.forward(dummy_batch)
        return output.shape[-1]


class DiagonalGaussianDistribution:
    def __init__(self, parameters: Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> Tensor:
        if self.deterministic:
            return self.mean
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self) -> Tensor:
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.mean.device, dtype=self.mean.dtype)
        kl = 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=-1)
        return kl


def build_action_autoencoder(config: VitaConfig, action_dim: int) -> tuple[nn.Module, nn.Module]:
    if config.use_variational:
        encoder: nn.Module
        if config.action_encoder_type == "cnn":
            encoder = CNNVariationalActionEncoder(
                pred_horizon=config.pred_horizon,
                action_dim=action_dim,
                latent_dim=config.latent_dim,
                hidden_dim=config.action_ae_enc_hidden_dim,
                num_layers=config.action_ae_num_layers,
            )
        elif config.action_encoder_type == "transformer":
            encoder = TransformerVariationalActionEncoder(
                enc_hidden_dim=config.action_ae_enc_hidden_dim,
                latent_dim=config.latent_dim,
                num_heads=config.action_ae_num_heads,
                pred_horizon=config.pred_horizon,
                action_dim=action_dim,
                num_layers=config.action_ae_num_layers,
                mlp_ratio=config.action_ae_mlp_ratio,
                dropout=config.action_ae_dropout,
            )
        else:
            raise ValueError(f"Unsupported variational encoder type: {config.action_encoder_type}")
    else:
        if config.action_encoder_type == "cnn":
            encoder = CNNActionEncoder(
                pred_horizon=config.pred_horizon,
                action_dim=action_dim,
                latent_dim=config.latent_dim,
                hidden_dim=config.action_ae_enc_hidden_dim,
                num_layers=config.action_ae_num_layers,
            )
        elif config.action_encoder_type == "transformer":
            encoder = TransformerActionEncoder(
                enc_hidden_dim=config.action_ae_enc_hidden_dim,
                latent_dim=config.latent_dim,
                num_heads=config.action_ae_num_heads,
                pred_horizon=config.pred_horizon,
                action_dim=action_dim,
                num_layers=config.action_ae_num_layers,
                mlp_ratio=config.action_ae_mlp_ratio,
                dropout=config.action_ae_dropout,
            )
        elif config.action_encoder_type == "simple":
            encoder = SimpleActionEncoder(
                latent_dim=config.latent_dim,
                pred_horizon=config.pred_horizon,
                action_dim=action_dim,
                num_layers=config.action_ae_num_layers,
                use_attention=config.action_ae_use_attention,
            )
        else:
            raise ValueError(f"Unsupported action encoder type: {config.action_encoder_type}")

    if config.action_decoder_type == "cnn":
        decoder = CNNActionDecoder(
            pred_horizon=config.pred_horizon,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.action_ae_dec_hidden_dim,
            num_layers=config.action_ae_num_layers,
        )
    else:
        decoder = SimpleActionDecoder(
            dec_hidden_dim=config.action_ae_dec_hidden_dim,
            latent_dim=config.latent_dim,
            pred_horizon=config.pred_horizon,
            action_dim=action_dim,
            num_layers=config.action_ae_num_layers,
            dropout=config.action_ae_dropout,
        )
    return encoder, decoder


def compute_contrastive_loss(image_features: Tensor, action_features: Tensor, temperature: float = 0.07) -> Tensor:
    batch_size = image_features.size(0)
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)
    logits = torch.matmul(image_features, action_features.T) / temperature
    labels = torch.arange(batch_size, device=logits.device)
    loss_i2a = F.cross_entropy(logits, labels)
    loss_a2i = F.cross_entropy(logits.T, labels)
    return (loss_i2a + loss_a2i) / 2


def resolve_torchvision_weights(weights_name: str | None):
    if not weights_name:
        return None
    if "." not in weights_name:
        return getattr(torchvision.models, weights_name)
    enum_name, member_name = weights_name.split(".", 1)
    enum_cls = getattr(torchvision.models, enum_name)
    return getattr(enum_cls, member_name)


def weights_init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


class VitaMlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dimension))
    table = torch.zeros(num_positions, dimension, dtype=torch.float32)
    table[:, 0::2] = torch.sin(positions * div)
    table[:, 1::2] = torch.cos(positions * div)
    return table


class TransformerActionEncoder(nn.Module):
    def __init__(
        self,
        enc_hidden_dim: int,
        latent_dim: int,
        num_heads: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.cls_embed = nn.Embedding(1, enc_hidden_dim)
        self.action_input_proj = nn.Linear(action_dim, enc_hidden_dim)
        ff_dim = int(enc_hidden_dim * mlp_ratio)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=enc_hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(enc_hidden_dim)
        self.latent_output_proj = nn.Linear(enc_hidden_dim, latent_dim)
        self.register_buffer("pos_embed", create_sinusoidal_pos_embedding(1 + pred_horizon, enc_hidden_dim).unsqueeze(0))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        cls_token = einops.repeat(self.cls_embed.weight, "1 d -> b 1 d", b=batch_size)
        action_tokens = self.action_input_proj(actions)
        x = torch.cat([cls_token, action_tokens], dim=1)
        x = x + self.pos_embed[:, : x.shape[1]]
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.latent_output_proj(x[:, 0])


class TransformerVariationalActionEncoder(TransformerActionEncoder):
    def __init__(self, *args, latent_dim: int, **kwargs):
        super().__init__(*args, latent_dim=latent_dim, **kwargs)
        self.latent_output_proj = nn.Linear(self.latent_output_proj.in_features, latent_dim * 2)

    def forward(self, actions: Tensor, deterministic: bool = False) -> tuple[DiagonalGaussianDistribution, Tensor]:
        params = super().forward(actions)
        posterior = DiagonalGaussianDistribution(params, deterministic=deterministic)
        return posterior, posterior.sample()


class SimpleActionEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        use_attention: bool = False,
    ):
        super().__init__()
        self.action_latent_dim = latent_dim // pred_horizon
        self.input_proj = nn.Linear(action_dim, self.action_latent_dim)
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        if use_attention:
            self.layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=self.action_latent_dim,
                        nhead=8,
                        dim_feedforward=4 * self.action_latent_dim,
                        dropout=0.0,
                        activation="gelu",
                        batch_first=True,
                        norm_first=True,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [VitaMlp(self.action_latent_dim, 4 * self.action_latent_dim, self.action_latent_dim) for _ in range(num_layers)]
            )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        x = self.input_proj(actions.view(batch_size, self.pred_horizon, self.action_dim))
        for layer in self.layers:
            x = layer(x)
        return x.reshape(batch_size, -1)


class SimpleActionDecoder(nn.Module):
    def __init__(
        self,
        dec_hidden_dim: int,
        latent_dim: int,
        pred_horizon: int,
        action_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, dec_hidden_dim)
        self.output_proj = nn.Linear(dec_hidden_dim, pred_horizon * action_dim)
        self.layers = nn.ModuleList(
            [VitaMlp(dec_hidden_dim, dec_hidden_dim, dec_hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, z: Tensor) -> Tensor:
        x = self.input_proj(z)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x.view(-1, self.pred_horizon, self.action_dim)


class CNNActionEncoder(nn.Module):
    def __init__(
        self,
        pred_horizon: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = action_dim
        for i in range(num_layers):
            conv_in = current_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(conv_in, hidden_dim, kernel_size=5, stride=2, padding=2))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        with torch.no_grad():
            dummy = torch.zeros(1, action_dim, pred_horizon, dtype=torch.float32)
            encoded = self.encoder(dummy)
            flattened_dim = encoded.reshape(1, -1).shape[1]
        self.latent_proj = nn.Linear(flattened_dim, latent_dim)
        self.apply(weights_init_encoder)

    def forward(self, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        x = actions.transpose(1, 2)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        return self.latent_proj(x)


class CNNVariationalActionEncoder(CNNActionEncoder):
    def __init__(self, *args, latent_dim: int, **kwargs):
        super().__init__(*args, latent_dim=latent_dim, **kwargs)
        in_features = self.latent_proj.in_features
        self.latent_proj = nn.Linear(in_features, latent_dim * 2)
        self.apply(weights_init_encoder)

    def forward(self, actions: Tensor, deterministic: bool = False) -> tuple[DiagonalGaussianDistribution, Tensor]:
        batch_size = actions.shape[0]
        x = actions.transpose(1, 2)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        params = self.latent_proj(x)
        posterior = DiagonalGaussianDistribution(params, deterministic=deterministic)
        return posterior, posterior.sample()


class CNNActionDecoder(nn.Module):
    def __init__(
        self,
        pred_horizon: int,
        action_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        layers: list[nn.Module] = []
        for i in range(num_layers):
            if i == num_layers - 1:
                layers.append(
                    nn.ConvTranspose1d(hidden_dim, action_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
                )
            else:
                layers.append(
                    nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=5, stride=2, padding=2, output_padding=1)
                )
                layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

        conv_output_length = max(1, pred_horizon // (2**num_layers))
        while True:
            with torch.no_grad():
                dummy = torch.zeros(1, hidden_dim, conv_output_length, dtype=torch.float32)
                decoded = self.decoder(dummy)
            if decoded.shape[-1] >= pred_horizon:
                break
            conv_output_length += 1

        self.conv_output_length = conv_output_length
        self.latent_proj = nn.Linear(latent_dim, hidden_dim * conv_output_length)
        self.apply(weights_init_encoder)

    def forward(self, z: Tensor) -> Tensor:
        batch_size = z.shape[0]
        x = self.latent_proj(z)
        x = x.view(batch_size, self.hidden_dim, self.conv_output_length)
        x = self.decoder(x)
        actions = x.transpose(1, 2)
        if actions.shape[1] != self.pred_horizon:
            actions = F.interpolate(x, size=self.pred_horizon, mode="linear", align_corners=False).transpose(1, 2)
        return actions


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=x.device, dtype=torch.float32)
        emb = torch.exp(-torch.log(torch.tensor(10000.0, device=x.device)) * emb / max(half_dim - 1, 1))
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FlowNetLayer(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = VitaMlp(dim, int(dim * mlp_ratio), dim, dropout=dropout)
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        features = self.time_modulator(t).view(x.shape[0], 3, self.dim).unbind(1)
        gamma, scale, shift = features
        x_norm = self.norm(x)
        x_norm = x_norm.mul(scale.add(1)).add_(shift)
        return x + self.mlp(x_norm).mul_(gamma)


class SimpleFlowNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.layers = nn.ModuleList(
            [FlowNetLayer(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)
        for block in self.layers:
            x = block(x, t)
        x = self.norm(x)
        return self.out_proj(x)


class TorchFlowMatcher:
    def __init__(self, fm, num_sampling_steps: int = 6):
        self.fm = fm
        self.num_sampling_steps = num_sampling_steps

    def compute_loss(self, model: nn.Module, target: Tensor, start: Tensor | None = None, **kwargs) -> tuple[Tensor, dict]:
        x0 = torch.randn_like(target) if start is None else start
        timestep, xt, ut = self.fm.sample_location_and_conditional_flow(x0, target)
        vt = model(xt, timestep, **kwargs)
        per_sample_loss = ((vt - ut) ** 2).mean(dim=-1)
        return per_sample_loss, {"loss": float(per_sample_loss.mean().detach().item())}

    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        device: torch.device | str,
        num_steps: int | None = None,
        return_traces: bool = False,
        start: Tensor | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, tuple[list[Tensor], list[Tensor]]]:
        if num_steps is None:
            num_steps = self.num_sampling_steps
        x = torch.randn(shape, device=device) if start is None else start
        dt = 1.0 / num_steps
        if return_traces:
            traj_history = [x.detach().clone().cpu()]
            vel_history = [torch.zeros_like(x).cpu()]
        for step in range(num_steps):
            timestep = torch.full((x.shape[0],), step / num_steps, device=x.device, dtype=x.dtype)
            vt = model(x, timestep, **kwargs)
            x = x + vt * dt
            if return_traces:
                traj_history.append(x.detach().clone().cpu())
                vel_history.append(vt.detach().clone().cpu())
        if return_traces:
            return x, (traj_history, vel_history)
        return x


class TorchFlowMatcherFactory:
    @staticmethod
    def make(name: str, sigma: float, num_sampling_steps: int) -> TorchFlowMatcher:
        if TorchCFMConditionalFlowMatcher is None:
            raise ImportError(
                "VITA requires `torchcfm` to match the original implementation. "
                "Install it before training or inference, e.g. `pip install torchcfm`."
            )

        mapping = {
            "conditional": TorchCFMConditionalFlowMatcher,
            "target": TorchCFMTargetMatcher,
            "schrodinger": TorchCFMSchrodingerMatcher,
            "exact": TorchCFMExactMatcher,
        }
        if name not in mapping:
            raise ValueError(f"Invalid flow matcher name: {name}")
        return TorchFlowMatcher(mapping[name](sigma=sigma), num_sampling_steps=num_sampling_steps)
