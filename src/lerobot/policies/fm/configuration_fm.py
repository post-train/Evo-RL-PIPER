#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig
from lerobot.policies.rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("flow_matching")
@dataclass
class FlowMatchingConfig(PreTrainedConfig):
    n_obs_steps: int = 1
    horizon: int = 64
    n_action_steps: int = 64

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 0

    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = None
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    backbone_lr_scale: float = 1.0

    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8
    time_embed_dim: int = 256
    use_film_scale_modulation: bool = True

    num_inference_steps: int = 14
    solver_type: str = "euler"
    ode_atol: float = 1e-5
    ode_rtol: float = 1e-5
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    compile_velocity_net: bool = True
    compile_mode: str = "reduce-overhead"
    compile_warmup_num_chunks: int = 1

    do_mask_loss_for_padding: bool = False
    rollout_consistency_weight: float = 0.1
    rollout_consistency_num_steps: int = 14

    use_ema: bool = True
    ema_power: float = 0.75

    rtc_config: RTCConfig | None = None

    optimizer_lr: float = 3e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}.")

        supported_solvers = ["euler", "dopri5"]
        if self.solver_type not in supported_solvers:
            raise ValueError(f"`solver_type` must be one of {supported_solvers}. Got {self.solver_type}.")

        supported_compile_modes = ["default", "reduce-overhead", "max-autotune"]
        if self.compile_mode not in supported_compile_modes:
            raise ValueError(
                f"`compile_mode` must be one of {supported_compile_modes}. Got {self.compile_mode}."
            )
        if self.compile_warmup_num_chunks < 0:
            raise ValueError("`compile_warmup_num_chunks` must be >= 0.")
        if self.rollout_consistency_weight < 0:
            raise ValueError("`rollout_consistency_weight` must be >= 0.")
        if self.rollout_consistency_num_steps < 0:
            raise ValueError("`rollout_consistency_num_steps` must be >= 0.")

        if self.solver_type == "dopri5":
            try:
                import torchdiffeq  # noqa: F401
            except ImportError:
                raise ImportError(
                    "torchdiffeq is required for solver_type='dopri5'. Install it with: pip install torchdiffeq"
                ) from None

        downsampling_factor = 2 ** len(self.down_dims)
        if self.horizon % downsampling_factor != 0:
            raise ValueError(
                "The horizon should be an integer multiple of the downsampling factor "
                f"(determined by len(down_dims)). Got {self.horizon=} and {self.down_dims=}"
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
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
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. "
                        f"Got {self.crop_shape} for `{key}` with shape {image_ft.shape}."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
