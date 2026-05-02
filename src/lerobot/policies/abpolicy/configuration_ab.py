import math
import logging
from dataclasses import dataclass, field

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@LRSchedulerConfig.register_subclass("ab_cosine")
@dataclass
class ABCosineSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    num_cycles: float = 0.5

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def lr_lambda(current_step: int) -> float:
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            progress = float(current_step - self.num_warmup_steps) / float(
                max(1, num_training_steps - self.num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * self.num_cycles * 2.0 * progress)))

        return LambdaLR(optimizer, lr_lambda, -1)


@PreTrainedConfig.register_subclass("abpolicy")
@dataclass
class ABPolicyConfig(PreTrainedConfig):
    n_obs_steps: int = 8
    img_obs_steps: int = 1
    action_history_horizon: int = 8
    action_horizon: int = 32
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    drop_n_last_frames: int | None = None

    image_encoder_name: str = "facebook/dinov2-base"
    image_encoder_pooled: bool = False
    image_encoder_freeze: bool = True
    image_backbone_used_layers: int = 8
    obs_dim: int = 512

    image_crop_shape: tuple[int, int] | None = (480, 480)
    image_crop_jitter_max: tuple[int, int] = (0, 0)
    image_brightness_contrast: tuple[float, float] = (0.1, 0.1)
    image_noise_std: float = 2.0
    img_size: int = 224
    img_patch_size: int = 14

    qpos_encoder_dim_hidden: int = 256
    qpos_encoder_norm_type: str = "layer"

    perceiver_in_channels: int = 1024
    perceiver_mid_channels: int = 512

    backbone_num_blocks: int = 6
    backbone_conv_kernel_size: int = 3
    backbone_num_attn_heads: int = 8

    flow_matching_sigma: float = 0.0
    num_inference_steps: int = 10

    use_bspline: bool = True
    bspline_degree: int = 3
    bspline_num_ctrl_points: int = 8
    refit_n_free: int = 4
    refit_last_pt_weight: float = 0.05
    control_freq: int = 30

    qpos_noise_std: float = 0.01

    optimizer_lr: float = 1e-4
    optimizer_perceiver_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_perceiver_betas: tuple[float, float] = (0.9, 0.99)
    optimizer_weight_decay: float = 0.0
    scheduler_warmup_steps: int = 100
    scheduler_num_cycles: float = 0.5

    @property
    def horizon(self) -> int:
        return self.action_history_horizon + self.action_horizon

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.drop_n_last_frames is None:
            self.drop_n_last_frames = max(self.action_horizon - 1, 0)

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> ABCosineSchedulerConfig:
        return ABCosineSchedulerConfig(
            num_warmup_steps=self.scheduler_warmup_steps,
            num_cycles=self.scheduler_num_cycles,
        )

    def validate_features(self) -> None:
        if self.robot_state_feature is None:
            raise ValueError(f"ABPolicy requires `{OBS_STATE}` in `input_features`.")
        if self.action_feature is None:
            raise ValueError(f"ABPolicy requires `{ACTION}` in `output_features`.")
        if len(self.image_features) == 0:
            raise ValueError("ABPolicy requires at least one image feature.")
        if self.image_encoder_pooled:
            raise ValueError("ABPolicy expects patch tokens, set `image_encoder_pooled=False`.")
        if self.img_size % self.img_patch_size != 0:
            raise ValueError("`img_size` must be divisible by `img_patch_size`.")

        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but ABPolicy expects all image shapes to match."
                )

        if self.image_crop_shape is not None:
            crop_h, crop_w = self.image_crop_shape
            img_h, img_w = first_image_ft.shape[1], first_image_ft.shape[2]
            if crop_h > img_h or crop_w > img_w:
                clamped_crop = (min(crop_h, img_h), min(crop_w, img_w))
                logging.warning(
                    "`image_crop_shape` %s exceeds image shape %s. Clamping crop to %s.",
                    self.image_crop_shape,
                    (img_h, img_w),
                    clamped_crop,
                )
                self.image_crop_shape = clamped_crop

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(-self.action_history_horizon, self.action_horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
