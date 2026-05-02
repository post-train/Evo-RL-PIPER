#!/usr/bin/env python

from __future__ import annotations

from collections import deque

import einops
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn

from lerobot.policies.diffusion.modeling_diffusion import SpatialSoftmax, _replace_submodules
from lerobot.policies.original_a2a.configuration_diffusion import OriginalA2AConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

try:
    from torchcfm.conditional_flow_matching import ConditionalFlowMatcher as TorchCFMConditionalFlowMatcher
except ModuleNotFoundError:
    TorchCFMConditionalFlowMatcher = None


class OriginalA2APolicy(PreTrainedPolicy):
    config_class = OriginalA2AConfig
    name = "original_a2a"

    def __init__(self, config: OriginalA2AConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        del kwargs
        config.validate_features()
        self.config = config
        self._queues = None
        self.model = OriginalA2AModel(config, dataset_stats=dataset_stats)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.model.parameters()

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_obs_steps),
        }
        self._action_plan = deque([], maxlen=self.config.n_action_steps)
        self._steps_since_replan = self.config.action_queue_refresh_steps
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        stacked = {k: torch.stack(list(queue), dim=1) for k, queue in self._queues.items() if len(queue) > 0}
        return self.model.generate_actions(stacked)

    def _build_history_action_seed(self, batch: dict[str, Tensor]) -> Tensor:
        # Before enough real actions have been executed, fall back to the current state
        # so the model still receives a full-length history tensor.
        current_state = batch[OBS_STATE]
        if current_state.ndim == 3:
            current_state = current_state[:, -1, :]
        return current_state

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del noise
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            seed = self._build_history_action_seed(batch)
            for _ in range(self.config.n_obs_steps):
                self._queues[ACTION].append(seed)

        should_replan = len(self._action_plan) == 0 or self._steps_since_replan >= self.config.action_queue_refresh_steps
        if should_replan:
            action_chunk = self.predict_action_chunk(batch)
            self._action_plan.clear()
            self._action_plan.extend(action_chunk.transpose(0, 1))
            self._steps_since_replan = 0

        action = self._action_plan.popleft()
        self._queues[ACTION].append(action)
        self._steps_since_replan += 1
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        return self.model.compute_loss(batch, reduction=reduction)


class OriginalA2AModel(nn.Module):
    def __init__(self, config: OriginalA2AConfig, dataset_stats: dict | None = None):
        super().__init__()
        self.config = config
        self.obs_encoder = OriginalA2AMultiObsEncoder(config)
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
        self.flow_matcher = OriginalA2ATorchFlowMatcher(config.flow_sigma, config.num_sampling_steps)
        del dataset_stats

    def _prepare_obs_cond(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        obs_features = self.obs_encoder(batch)
        obs_features = obs_features.reshape(batch_size, -1)
        return self.obs_projector(obs_features)

    def _future_action_slice(self, actions: Tensor) -> Tensor:
        future_start = self.config.n_obs_steps - 1
        future_end = future_start + self.config.n_action_steps
        return actions[:, future_start:future_end, :]

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        if n_obs_steps != self.config.n_obs_steps:
            raise ValueError(f"Expected {self.config.n_obs_steps} obs steps, got {n_obs_steps}.")
        if ACTION not in batch:
            raise ValueError("original_a2a requires historical actions at inference time.")
        if batch[ACTION].shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} historical action steps, got {batch[ACTION].shape[1]}."
            )

        obs_latents = self._prepare_obs_cond(batch)
        history_actions = batch[ACTION][:, : self.config.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_actions)
        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(batch_size, self.config.latent_dim),
            device=obs_latents.device,
            num_steps=self.config.num_sampling_steps,
            start=history_latents,
            global_cond=obs_latents,
        )
        return self.action_decoder(action_latents_pred)

    def compute_loss(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        required = {OBS_STATE, ACTION}
        if not required.issubset(batch):
            raise ValueError(f"original_a2a batch is missing required keys: {required - set(batch)}")

        actions = batch[ACTION]
        if actions.shape[1] != self.config.horizon:
            raise ValueError(f"Expected action horizon {self.config.horizon}, got {actions.shape[1]}.")

        batch_size = actions.shape[0]
        obs_latents = self._prepare_obs_cond(batch)
        history_actions = actions[:, : self.config.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_actions)
        future_actions = self._future_action_slice(actions)
        future_action_latents = self.action_encoder(future_actions)

        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=history_latents,
            global_cond=obs_latents,
        )
        per_sample_loss = flow_loss
        metrics["flow_loss"] = float(flow_loss.mean().detach().item())

        if self.config.enc_contrastive_weight > 0:
            contrastive_loss = self._compute_contrastive_loss(obs_latents.view(batch_size, -1), future_action_latents)
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
                flow_contrastive_loss = self._compute_contrastive_loss(obs_latents.view(batch_size, -1), action_latents_pred)
                per_sample_loss = per_sample_loss + self.config.flow_contrastive_weight * flow_contrastive_loss
                metrics["flow_contrastive_loss"] = float(flow_contrastive_loss.detach().item())

            if self.config.flow_recon_weight > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                flow_recon_loss = F.l1_loss(actions_recon, future_actions, reduction="none").mean(dim=(1, 2))
                per_sample_loss = per_sample_loss + self.config.flow_recon_weight * flow_recon_loss
                metrics["flow_action_recon_loss"] = float(flow_recon_loss.mean().detach().item())

        if self.config.enc_recon_weight > 0:
            actions_recon = self.action_decoder(future_action_latents)
            enc_recon_loss = F.l1_loss(actions_recon, future_actions, reduction="none").mean(dim=(1, 2))
            per_sample_loss = per_sample_loss + self.config.enc_recon_weight * enc_recon_loss
            metrics["enc_action_recon_loss"] = float(enc_recon_loss.mean().detach().item())

        loss = per_sample_loss if reduction == "none" else per_sample_loss.mean()
        metrics["loss"] = float(loss.mean().detach().item())
        return loss, metrics

    @staticmethod
    def _compute_contrastive_loss(image_features: Tensor, action_features: Tensor, temperature: float = 0.07) -> Tensor:
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features.view(batch_size, -1), dim=1)
        logits = torch.matmul(image_features, action_features.T) / temperature
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)
        return (loss_i2a + loss_a2i) / 2


class OriginalA2AMultiObsEncoder(nn.Module):
    def __init__(self, config: OriginalA2AConfig):
        super().__init__()
        self.config = config
        self.rgb_keys = list(config.image_features.keys())
        self.low_dim_keys = [OBS_STATE]
        if config.env_state_feature is not None:
            self.low_dim_keys.append(OBS_ENV_STATE)

        if self.rgb_keys:
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList([OriginalA2ARgbEncoder(config) for _ in self.rgb_keys])
                self.rgb_feature_dim = self.rgb_encoder[0].feature_dim * len(self.rgb_keys)
            else:
                self.rgb_encoder = OriginalA2ARgbEncoder(config)
                self.rgb_feature_dim = self.rgb_encoder.feature_dim * len(self.rgb_keys)
        else:
            self.rgb_encoder = None
            self.rgb_feature_dim = 0

        self.low_dim_total = sum(
            config.input_features[key].shape[0] for key in self.low_dim_keys if key in config.input_features
        )
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
            raise ValueError("original_a2a observation encoder received no usable input features.")

        return torch.cat(features, dim=-1)


class OriginalA2ARgbEncoder(nn.Module):
    def __init__(self, config: OriginalA2AConfig):
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
                raise ValueError("You can't replace BatchNorm in a pretrained model without ruining the pretrained weights.")
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


class _FallbackConditionalFlowMatcher:
    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma

    def sample_location_and_conditional_flow(self, x0: Tensor, x1: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        noise = torch.randn_like(x0) * self.sigma if self.sigma > 0 else 0.0
        xt = (1.0 - t.unsqueeze(-1)) * x0 + t.unsqueeze(-1) * x1
        if isinstance(noise, Tensor):
            xt = xt + noise
        ut = x1 - x0
        return t, xt, ut


class OriginalA2ATorchFlowMatcher:
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
