#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("a2a")
@dataclass
class A2AConfig(PreTrainedConfig):
    """Configuration for the Action-to-Action Flow Matching policy."""

    n_obs_steps: int = 16
    horizon: int = 32
    n_action_steps: int = 16
    action_prediction_mode: str = "delta"
    delta_action_scale: float = 1.0
    normalize_delta_targets: bool = True
    delta_stats_eps: float = 1e-6
    action_queue_refresh_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 1

    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = True
    imagenet_norm: bool = True

    latent_dim: int = 512

    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    flow_time_embed_dim: int = 256
    flow_sigma: float = 0.0
    num_sampling_steps: int = 6

    history_hidden_dim: int = 512
    history_num_layers: int = 3

    action_ae_enc_hidden_dim: int = 512
    action_ae_dec_hidden_dim: int = 512
    action_ae_num_layers: int = 4
    action_ae_dropout: float = 0.0

    decode_flow_latents: bool = True
    consistency_weight: float = 1.0
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0
    enc_recon_weight: float = 0.5
    flow_recon_weight: float = 0.5

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self) -> None:
        super().__post_init__()

        self.drop_n_last_frames = max(0, self.horizon - self.n_action_steps - self.n_obs_steps + 1)

        if self.robot_state_feature is not None and self.robot_state_feature.shape is not None:
            if len(self.robot_state_feature.shape) != 1:
                raise ValueError("`observation.state` must be a 1D feature for A2A.")
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")
        if self.horizon < self.n_obs_steps - 1 + self.n_action_steps:
            raise ValueError(
                "`horizon` must satisfy horizon >= n_obs_steps - 1 + n_action_steps for A2A future slicing."
            )
        if self.n_obs_steps < 2**self.history_num_layers:
            raise ValueError(
                "`n_obs_steps` is too small for the history CNN encoder depth. "
                f"Got n_obs_steps={self.n_obs_steps}, history_num_layers={self.history_num_layers}."
            )
        if self.n_action_steps < 2**self.history_num_layers:
            raise ValueError(
                "`n_action_steps` is too small for the action CNN encoder depth. "
                f"Got n_action_steps={self.n_action_steps}, history_num_layers={self.history_num_layers}."
            )
        if self.latent_dim <= 0:
            raise ValueError("`latent_dim` must be positive.")
        if self.num_sampling_steps <= 0:
            raise ValueError("`num_sampling_steps` must be positive.")
        if self.action_prediction_mode not in {"delta", "absolute"}:
            raise ValueError(
                f"`action_prediction_mode` must be one of ['delta', 'absolute'], got {self.action_prediction_mode}."
            )
        if self.delta_action_scale <= 0:
            raise ValueError("`delta_action_scale` must be positive.")
        if self.delta_stats_eps <= 0:
            raise ValueError("`delta_stats_eps` must be positive.")
        if self.action_queue_refresh_steps <= 0:
            raise ValueError("`action_queue_refresh_steps` must be positive.")
        if self.action_queue_refresh_steps > self.n_action_steps:
            raise ValueError(
                "`action_queue_refresh_steps` must be <= `n_action_steps` for online chunk replanning."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError("A2A requires `observation.state` as an input feature.")
        if self.action_feature is None:
            raise ValueError("A2A requires `action` as an output feature.")
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but A2A expects all image shapes to match."
                    )
                if self.crop_shape is not None:
                    if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                        raise ValueError(
                            f"`crop_shape` should fit within image shapes. Got {self.crop_shape} for `{key}`."
                        )

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
