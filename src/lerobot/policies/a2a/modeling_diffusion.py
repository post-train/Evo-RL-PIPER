#!/usr/bin/env python

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import pyarrow.dataset as pa_ds
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from lerobot.configs.types import NormalizationMode
from lerobot.policies.a2a.configuration_diffusion import A2AConfig
from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax, _replace_submodules
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher as TorchCFMConditionalFlowMatcher
except ModuleNotFoundError:
    TorchCFMConditionalFlowMatcher = None


def compute_a2a_delta_stats(dataset_meta, n_action_steps: int) -> tuple[Tensor, Tensor]:
    cached = getattr(dataset_meta, "_a2a_delta_stats_cache", None)
    if cached is not None and cached.get("n_action_steps") == n_action_steps:
        return cached["mean"], cached["std"]

    data_root = Path(dataset_meta.root) / "data"
    paths = sorted(data_root.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under dataset data root: {data_root}")

    table = pa_ds.dataset(paths, format="parquet").to_table(columns=["episode_index", OBS_STATE, ACTION])
    episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    states = np.asarray(table[OBS_STATE].to_pylist(), dtype=np.float32)
    actions = np.asarray(table[ACTION].to_pylist(), dtype=np.float32)

    action_dim = actions.shape[1]
    delta_sum = np.zeros(action_dim, dtype=np.float64)
    delta_sq_sum = np.zeros(action_dim, dtype=np.float64)
    delta_count = 0

    episode_ids, episode_starts = np.unique(episode_index, return_index=True)
    episode_starts = list(episode_starts) + [len(episode_index)]

    for i, _ in enumerate(episode_ids):
        start = episode_starts[i]
        end = episode_starts[i + 1]
        states_ep = states[start:end]
        actions_ep = actions[start:end]
        valid = len(states_ep) - n_action_steps + 1
        if valid <= 0:
            continue
        anchor_states = states_ep[:valid]
        for step_ahead in range(n_action_steps):
            deltas = actions_ep[step_ahead : step_ahead + valid] - anchor_states
            delta_sum += deltas.sum(axis=0)
            delta_sq_sum += np.square(deltas).sum(axis=0)
            delta_count += valid

    if delta_count == 0:
        raise ValueError("Failed to compute A2A delta stats: no valid windows found.")

    delta_mean = delta_sum / delta_count
    delta_var = np.maximum(delta_sq_sum / delta_count - np.square(delta_mean), 1e-12)
    delta_std = np.sqrt(delta_var)

    result = {
        "n_action_steps": n_action_steps,
        "mean": torch.as_tensor(delta_mean, dtype=torch.float32),
        "std": torch.as_tensor(delta_std, dtype=torch.float32),
    }
    dataset_meta._a2a_delta_stats_cache = result
    return result["mean"], result["std"]


class A2APolicy(PreTrainedPolicy):
    config_class = A2AConfig
    name = "a2a"

    def __init__(self, config: A2AConfig, dataset_stats: dict | None = None, dataset_meta=None, **kwargs):
        super().__init__(config)
        del kwargs
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = A2AModel(config, dataset_stats=dataset_stats, dataset_meta=dataset_meta)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._steps_since_replan = self.config.action_queue_refresh_steps
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        stacked = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.model.generate_actions(stacked, noise=noise)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        should_replan = len(self._queues[ACTION]) == 0 or self._steps_since_replan >= self.config.action_queue_refresh_steps
        if should_replan:
            action_chunk = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].clear()
            self._queues[ACTION].extend(action_chunk.transpose(0, 1))
            self._steps_since_replan = 0

        action = self._queues[ACTION].popleft()
        self._steps_since_replan += 1
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.model.compute_loss(batch, reduction=reduction)


class A2AModel(nn.Module):
    def __init__(self, config: A2AConfig, dataset_stats: dict | None = None, dataset_meta=None):
        super().__init__()
        self.config = config

        self.obs_encoder = A2AMultiObsEncoder(config)
        self.obs_projector = nn.Linear(self.obs_encoder.output_dim * config.n_obs_steps, config.latent_dim)

        action_dim = config.action_feature.shape[0]
        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=config.n_obs_steps,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.history_hidden_dim,
            num_layers=config.history_num_layers,
        )
        self.action_encoder = CNNActionEncoder(
            pred_horizon=config.n_action_steps,
            action_dim=action_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.action_ae_enc_hidden_dim,
            num_layers=config.history_num_layers,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=config.action_ae_dec_hidden_dim,
            latent_dim=config.latent_dim,
            pred_horizon=config.n_action_steps,
            action_dim=action_dim,
            num_layers=config.action_ae_num_layers,
            dropout=config.action_ae_dropout,
        )
        self.flow_net = SimpleFlowNet(
            input_dim=config.latent_dim,
            hidden_dim=config.flow_hidden_dim,
            output_dim=config.latent_dim,
            num_layers=config.flow_num_layers,
            mlp_ratio=config.flow_mlp_ratio,
            dropout=config.flow_dropout,
            time_embed_dim=config.flow_time_embed_dim,
            condition_dim=config.latent_dim,
        )
        self.flow_matcher = A2ATorchFlowMatcher(config.flow_sigma, config.num_sampling_steps)
        self._init_state_action_norm_buffers(dataset_stats, dataset_meta)

    def _init_state_action_norm_buffers(self, dataset_stats: dict | None, dataset_meta=None) -> None:
        action_dim = self.config.action_feature.shape[0]
        zeros = torch.zeros(action_dim, dtype=torch.float32)
        ones = torch.ones(action_dim, dtype=torch.float32)

        self.register_buffer("state_min", zeros.clone(), persistent=True)
        self.register_buffer("state_max", ones.clone(), persistent=True)
        self.register_buffer("state_mean", zeros.clone(), persistent=True)
        self.register_buffer("state_std", ones.clone(), persistent=True)
        self.register_buffer("action_min", zeros.clone(), persistent=True)
        self.register_buffer("action_max", ones.clone(), persistent=True)
        self.register_buffer("action_mean", zeros.clone(), persistent=True)
        self.register_buffer("action_std", ones.clone(), persistent=True)
        self.register_buffer("delta_mean", zeros.clone(), persistent=True)
        self.register_buffer("delta_std", ones.clone(), persistent=True)

        if dataset_stats is None:
            return

        def _copy_stat(buffer_name: str, key: str, stat_name: str, default: Tensor) -> None:
            stat = dataset_stats.get(key, {}).get(stat_name)
            value = default if stat is None else torch.as_tensor(stat, dtype=torch.float32)
            getattr(self, buffer_name).copy_(value)

        _copy_stat("state_min", OBS_STATE, "min", zeros)
        _copy_stat("state_max", OBS_STATE, "max", ones)
        _copy_stat("state_mean", OBS_STATE, "mean", zeros)
        _copy_stat("state_std", OBS_STATE, "std", ones)
        _copy_stat("action_min", ACTION, "min", zeros)
        _copy_stat("action_max", ACTION, "max", ones)
        _copy_stat("action_mean", ACTION, "mean", zeros)
        _copy_stat("action_std", ACTION, "std", ones)

        if dataset_meta is not None and self.config.action_prediction_mode == "delta":
            delta_mean, delta_std = compute_a2a_delta_stats(dataset_meta, self.config.n_action_steps)
            self.delta_mean.copy_(delta_mean)
            self.delta_std.copy_(delta_std)

    def _normalize_delta_targets(self, deltas: Tensor) -> Tensor:
        if self.config.normalize_delta_targets:
            return (deltas - self.delta_mean.view(1, 1, -1)) / (self.delta_std.view(1, 1, -1) + self.config.delta_stats_eps)
        return deltas * self.config.delta_action_scale

    def _denormalize_delta_predictions(self, deltas: Tensor) -> Tensor:
        if self.config.normalize_delta_targets:
            return deltas * (self.delta_std.view(1, 1, -1) + self.config.delta_stats_eps) + self.delta_mean.view(1, 1, -1)
        return deltas / self.config.delta_action_scale

    def _state_to_raw(self, state: Tensor) -> Tensor:
        state_norm = self.config.normalization_mapping.get(self.config.robot_state_feature.type, NormalizationMode.IDENTITY)
        if state_norm == NormalizationMode.IDENTITY:
            return state
        if state_norm == NormalizationMode.MIN_MAX:
            denom = torch.where(
                (self.state_max - self.state_min) == 0,
                torch.full_like(self.state_max, 1e-8),
                self.state_max - self.state_min,
            )
            return (state + 1) / 2 * denom + self.state_min
        if state_norm == NormalizationMode.MEAN_STD:
            return state * self.state_std + self.state_mean
        raise ValueError(f"Unsupported state normalization for A2A delta target: {state_norm}")

    def _raw_to_action(self, raw: Tensor) -> Tensor:
        action_norm = self.config.normalization_mapping.get(self.config.action_feature.type, NormalizationMode.IDENTITY)
        if action_norm == NormalizationMode.IDENTITY:
            return raw
        if action_norm == NormalizationMode.MIN_MAX:
            denom = torch.where(
                (self.action_max - self.action_min) == 0,
                torch.full_like(self.action_max, 1e-8),
                self.action_max - self.action_min,
            )
            return 2 * (raw - self.action_min) / denom - 1
        if action_norm == NormalizationMode.MEAN_STD:
            return (raw - self.action_mean) / (self.action_std + 1e-8)
        raise ValueError(f"Unsupported action normalization for A2A delta target: {action_norm}")

    def _state_as_action_space(self, state: Tensor) -> Tensor:
        return self._raw_to_action(self._state_to_raw(state))

    def _current_state_action_anchor(self, batch: dict[str, Tensor]) -> Tensor:
        current_state = batch[OBS_STATE][:, self.config.n_obs_steps - 1, :]
        return self._state_as_action_space(current_state)

    def _delta_targets(self, future_actions: Tensor, batch: dict[str, Tensor]) -> Tensor:
        anchor = self._current_state_action_anchor(batch).unsqueeze(1)
        return self._normalize_delta_targets(future_actions - anchor)

    def _action_targets(self, future_actions: Tensor, batch: dict[str, Tensor]) -> Tensor:
        if self.config.action_prediction_mode == "delta":
            return self._delta_targets(future_actions, batch)
        if self.config.action_prediction_mode == "absolute":
            return future_actions
        raise ValueError(f"Unsupported action_prediction_mode: {self.config.action_prediction_mode}")

    def _decode_to_absolute_actions(self, decoded_actions: Tensor, batch: dict[str, Tensor]) -> Tensor:
        if self.config.action_prediction_mode == "delta":
            decoded_actions = self._denormalize_delta_predictions(decoded_actions)
            action_anchor = self._current_state_action_anchor(batch).unsqueeze(1)
            return decoded_actions + action_anchor
        if self.config.action_prediction_mode == "absolute":
            return decoded_actions
        raise ValueError(f"Unsupported action_prediction_mode: {self.config.action_prediction_mode}")

    def _prepare_obs_cond(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        obs_features = self.obs_encoder(batch)
        obs_features = obs_features.reshape(batch_size, -1)
        return self.obs_projector(obs_features)

    def _future_action_slice(self, actions: Tensor) -> Tensor:
        future_start = self.config.n_obs_steps - 1
        future_end = future_start + self.config.n_action_steps
        return actions[:, future_start:future_end, :]

    def generate_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} obs steps, got {n_obs_steps}.")

        obs_latents = self._prepare_obs_cond(batch)
        history_states = batch[OBS_STATE][:, : self.config.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        if noise is not None:
            if noise.ndim == 3:
                noise = self.action_encoder(noise[:, : self.config.n_action_steps, :])
            elif noise.ndim != 2:
                raise ValueError(f"Unsupported noise shape for A2A sampling: {tuple(noise.shape)}.")

        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(batch_size, self.config.latent_dim),
            device=obs_latents.device,
            num_steps=self.config.num_sampling_steps,
            start=history_latents,
            global_cond=obs_latents,
            noise=noise,
        )
        decoded_actions = self.action_decoder(action_latents_pred)
        return self._decode_to_absolute_actions(decoded_actions, batch)

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        required = {OBS_STATE, ACTION}
        if not required.issubset(batch):
            raise ValueError(f"A2A batch is missing required keys: {required - set(batch)}")

        actions = batch[ACTION]
        if actions.shape[1] != self.config.horizon:
            raise ValueError(f"Expected action horizon {self.config.horizon}, got {actions.shape[1]}.")

        batch_size = actions.shape[0]
        obs_latents = self._prepare_obs_cond(batch)

        history_states = batch[OBS_STATE][:, : self.config.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)

        future_actions = self._future_action_slice(actions)
        future_action_targets = self._action_targets(future_actions, batch)
        future_action_latents = self.action_encoder(future_action_targets)

        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=history_latents,
            global_cond=obs_latents,
        )

        per_sample_loss = flow_loss
        metrics["flow_loss"] = float(flow_loss.mean().detach().item())

        if self.config.enc_contrastive_weight > 0:
            contrastive_loss = self._compute_contrastive_loss(
                obs_latents.view(batch_size, -1), future_action_latents.view(batch_size, -1)
            )
            per_sample_loss = per_sample_loss + self.config.enc_contrastive_weight * contrastive_loss
            metrics["enc_contrastive_loss"] = float(contrastive_loss.detach().item())

        if self.config.decode_flow_latents:
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.config.latent_dim),
                device=obs_latents.device,
                start=history_latents,
                num_steps=self.config.num_sampling_steps,
                global_cond=obs_latents,
            )

            if self.config.consistency_weight > 0:
                consistency_loss = F.mse_loss(
                    action_latents_pred, future_action_latents, reduction="none"
                ).mean(dim=-1)
                per_sample_loss = per_sample_loss + self.config.consistency_weight * consistency_loss
                metrics["consistency_loss"] = float(consistency_loss.mean().detach().item())

            if self.config.flow_contrastive_weight > 0:
                flow_contrastive_loss = self._compute_contrastive_loss(
                    obs_latents.view(batch_size, -1), action_latents_pred.view(batch_size, -1)
                )
                per_sample_loss = per_sample_loss + self.config.flow_contrastive_weight * flow_contrastive_loss
                metrics["flow_contrastive_loss"] = float(flow_contrastive_loss.detach().item())

            if self.config.flow_recon_weight > 0:
                action_recon = self.action_decoder(action_latents_pred)
                flow_recon_loss = F.l1_loss(action_recon, future_action_targets, reduction="none").mean(dim=(1, 2))
                per_sample_loss = per_sample_loss + self.config.flow_recon_weight * flow_recon_loss
                metrics["flow_action_recon_loss"] = float(flow_recon_loss.mean().detach().item())
        else:
            action_latents_pred = future_action_latents

        if self.config.enc_recon_weight > 0:
            action_recon = self.action_decoder(future_action_latents)
            enc_recon_loss = F.l1_loss(action_recon, future_action_targets, reduction="none").mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + self.config.enc_recon_weight * enc_recon_loss
            metrics["enc_action_recon_loss"] = float(enc_recon_loss.mean().detach().item())

        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()
        metrics["loss"] = float(loss.mean().detach().item())
        return loss, metrics

    @staticmethod
    def _compute_contrastive_loss(image_features: Tensor, action_features: Tensor, temperature: float = 0.07) -> Tensor:
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)
        logits = torch.matmul(image_features, action_features.T) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)
        return (loss_i2a + loss_a2i) / 2


class A2AMultiObsEncoder(nn.Module):
    def __init__(self, config: A2AConfig):
        super().__init__()
        self.config = config
        self.rgb_keys = list(config.image_features.keys())
        self.low_dim_keys = [OBS_STATE]
        if config.env_state_feature is not None:
            self.low_dim_keys.append(OBS_ENV_STATE)

        if self.rgb_keys:
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList([A2ARgbEncoder(config) for _ in self.rgb_keys])
                self.rgb_feature_dim = self.rgb_encoder[0].feature_dim * len(self.rgb_keys)
            else:
                self.rgb_encoder = A2ARgbEncoder(config)
                self.rgb_feature_dim = self.rgb_encoder.feature_dim * len(self.rgb_keys)
        else:
            self.rgb_encoder = None
            self.rgb_feature_dim = 0

        self.low_dim_total = sum(config.input_features[key].shape[0] for key in self.low_dim_keys if key in config.input_features)
        self.output_dim = self.rgb_feature_dim + self.low_dim_total

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        features: list[Tensor] = []

        if self.rgb_keys:
            images = batch[OBS_IMAGES]
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(images, "b s n c h w -> n (b s) c h w")
                img_features = torch.cat(
                    [encoder(camera_imgs) for encoder, camera_imgs in zip(self.rgb_encoder, images_per_camera, strict=True)],
                    dim=-1,
                )
            else:
                encoded = self.rgb_encoder(einops.rearrange(images, "b s n c h w -> (b s n) c h w"))
                img_features = einops.rearrange(
                    encoded, "(b s n) d -> (b s) (n d)", b=batch_size, s=n_obs_steps, n=len(self.rgb_keys)
                )
            features.append(img_features)

        for key in self.low_dim_keys:
            if key in batch:
                features.append(batch[key].reshape(batch_size * n_obs_steps, -1))

        if not features:
            raise ValueError("A2A observation encoder received no usable input features.")

        return torch.cat(features, dim=-1)


class A2AMlp(nn.Module):
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


class A2ARgbEncoder(nn.Module):
    def __init__(self, config: A2AConfig):
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
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the pretrained weights."
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=module.num_features // 16,
                    num_channels=module.num_features,
                ),
            )

        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = self._get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(self.feature_dim, self.feature_dim)
        self.relu = nn.ReLU()

        if config.imagenet_norm:
            self.register_buffer(
                "imagenet_mean",
                torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "imagenet_std",
                torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
        else:
            self.imagenet_mean = None
            self.imagenet_std = None

    @staticmethod
    def _get_output_shape(module: nn.Module, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        dummy_input = torch.zeros(size=input_shape)
        with torch.inference_mode():
            output = module(dummy_input)
        return tuple(output.shape)

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(x) if self.training else self.center_crop(x)
        if self.imagenet_mean is not None and self.imagenet_std is not None:
            mean = self.imagenet_mean.to(device=x.device, dtype=x.dtype)
            std = self.imagenet_std.to(device=x.device, dtype=x.dtype)
            x = (x - mean) / std
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x


def _weights_init_encoder(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)
    elif isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.orthogonal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.fill_(0.0)


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
        if flattened_dim <= 0:
            raise ValueError(
                f"Invalid flattened encoder dim for pred_horizon={pred_horizon}, num_layers={num_layers}."
            )
        self.latent_proj = nn.Linear(flattened_dim, latent_dim)
        self.apply(_weights_init_encoder)

    def forward(self, actions: Tensor) -> Tensor:
        batch_size = actions.shape[0]
        x = actions.transpose(1, 2)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        return self.latent_proj(x)


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
        self.layers = nn.ModuleList(
            [A2AMlp(dec_hidden_dim, dec_hidden_dim, dec_hidden_dim, dropout=dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Linear(dec_hidden_dim, pred_horizon * action_dim)
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


class SinusoidalPosEmb(nn.Module):
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
        self.mlp = A2AMlp(dim, int(dim * mlp_ratio), dim, dropout=dropout)
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
        condition_dim: int | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )
        self.cond_embed = nn.Linear(condition_dim, hidden_dim) if condition_dim is not None else None
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

    def forward(self, x: Tensor, t: Tensor, global_cond: Tensor | None = None) -> Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)
        if global_cond is not None and self.cond_embed is not None:
            t = t + self.cond_embed(global_cond)
        for block in self.layers:
            x = block(x, t)
        x = self.norm(x)
        return self.out_proj(x)


@dataclass
class _FallbackConditionalFlowMatcher:
    sigma: float = 0.0

    def sample_location_and_conditional_flow(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0) * self.sigma if self.sigma > 0 else 0.0
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        if isinstance(noise, Tensor):
            xt = xt + noise
        ut = x1 - x0
        return t, xt, ut


class A2ATorchFlowMatcher:
    def __init__(self, sigma: float = 0.0, num_sampling_steps: int = 6):
        self.num_sampling_steps = num_sampling_steps
        if TorchCFMConditionalFlowMatcher is not None:
            self.fm = TorchCFMConditionalFlowMatcher(sigma=sigma)
        else:
            self.fm = _FallbackConditionalFlowMatcher(sigma=sigma)

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
        noise: Tensor | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, tuple[list[Tensor], list[Tensor]]]:
        if num_steps is None:
            num_steps = self.num_sampling_steps

        if start is not None and noise is not None:
            x = noise
        elif start is not None:
            x = start
        elif noise is not None:
            x = noise
        else:
            x = torch.randn(shape, device=device)

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
