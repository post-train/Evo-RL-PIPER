#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("vita")
@dataclass
class VitaConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    horizon: int = 16
    n_action_steps: int = 8
    action_queue_refresh_steps: int = 8
    drop_n_last_frames: int | None = None

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MIN_MAX,
            "ENV": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    resize_shape: tuple[int, int] = (240, 320)
    crop_shape: tuple[int, int] = (224, 308)
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    use_frozen_batch_norm: bool = True
    latent_dim: int = 512

    flow_matcher_name: str = "exact"
    flow_sigma: float = 0.0
    num_sampling_steps: int = 6
    flow_hidden_dim: int = 512
    flow_num_layers: int = 4
    flow_mlp_ratio: float = 4.0
    flow_dropout: float = 0.0
    flow_time_embed_dim: int = 256

    decode_flow_latents: bool = True
    consistency_weight: float = 1.0
    enc_contrastive_weight: float = 0.0
    flow_contrastive_weight: float = 0.0

    use_variational: bool = False
    action_encoder_type: str = "cnn"
    action_decoder_type: str = "simple"
    freeze_action_encoder: bool = False
    freeze_action_decoder: bool = False
    action_kl_weight: float = 0.0
    action_recon_loss_type: str = "l1"
    flow_action_recon_weight: float = 0.5
    enc_action_recon_weight: float = 0.5

    action_ae_enc_hidden_dim: int = 512
    action_ae_dec_hidden_dim: int = 512
    action_ae_num_layers: int = 4
    action_ae_num_heads: int = 8
    action_ae_mlp_ratio: float = 4.0
    action_ae_dropout: float = 0.0
    action_ae_use_attention: bool = False

    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500
    use_ema: bool = True
    ema_power: float = 0.75
    rtc_config: RTCConfig | None = None
    infer_safe_delta_enabled: bool = True
    infer_safe_delta_max_norm: float = 0.02
    infer_no_rtc_replan_every_step: bool = True
    infer_no_rtc_refresh_steps: int = 2
    infer_no_rtc_blend_steps: int = 2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps <= 0 or self.horizon <= 0:
            raise ValueError("`horizon` and `n_action_steps` must be positive.")
        if self.horizon < self.n_action_steps:
            raise ValueError("`horizon` must be >= `n_action_steps` for vita.")
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = max(self.horizon - self.n_action_steps - self.n_obs_steps + 1, 0)
        if self.action_queue_refresh_steps <= 0:
            raise ValueError("`action_queue_refresh_steps` must be positive.")
        if self.action_queue_refresh_steps > self.n_action_steps:
            raise ValueError("`action_queue_refresh_steps` must be <= `n_action_steps`.")
        if self.action_encoder_type not in {"cnn", "transformer", "simple"}:
            raise ValueError(f"Unsupported action_encoder_type: {self.action_encoder_type}")
        if self.action_decoder_type not in {"simple", "cnn"}:
            raise ValueError(f"Unsupported action_decoder_type: {self.action_decoder_type}")
        if self.action_recon_loss_type not in {"l1", "l2"}:
            raise ValueError(f"Unsupported action_recon_loss_type: {self.action_recon_loss_type}")
        if self.action_encoder_type == "cnn" and self.horizon < 2**self.action_ae_num_layers:
            raise ValueError(
                "`horizon` is too small for cnn action encoder depth. "
                f"Got horizon={self.horizon}, action_ae_num_layers={self.action_ae_num_layers}."
            )
        if self.action_decoder_type == "cnn" and self.horizon < 2**self.action_ae_num_layers:
            raise ValueError(
                "`horizon` is too small for cnn action decoder depth. "
                f"Got horizon={self.horizon}, action_ae_num_layers={self.action_ae_num_layers}."
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
            raise ValueError("vita requires `observation.state` as an input feature.")
        if self.action_feature is None:
            raise ValueError("vita requires `action` as an output feature.")
        if not self.image_features:
            raise ValueError("vita requires at least one image feature.")
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(f"`{key}` does not match `{first_image_key}`, but vita expects all image shapes to match.")

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.obs_horizon))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.pred_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def obs_horizon(self) -> int:
        return self.n_obs_steps

    @property
    def action_horizon(self) -> int:
        return self.n_action_steps

    @property
    def pred_horizon(self) -> int:
        return self.horizon
