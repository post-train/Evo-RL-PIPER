from __future__ import annotations

import math
import os
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.models.attention import Attention
from diffusers.models.embeddings import Timesteps, get_2d_sincos_pos_embed, get_timestep_embedding
from einops import rearrange
from timm.models.vision_transformer import Mlp
from torch import Tensor, nn
from transformers import AutoModel

from lerobot.policies.abpolicy.configuration_ab import ABPolicyConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import get_device_from_parameters, get_dtype_from_parameters, populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


class BSplineProjector(nn.Module):
    def __init__(self, sequence_length: int, degree: int, num_ctrl_points: int):
        super().__init__()
        if num_ctrl_points < degree + 1:
            raise ValueError("num_ctrl_points must be at least degree + 1.")

        x = torch.arange(sequence_length, dtype=torch.float32)
        t_internal = torch.linspace(x[0], x[-1], num_ctrl_points - degree + 1)
        knots = torch.cat([x.new_full((degree,), x[0]), t_internal, x.new_full((degree,), x[-1])])

        basis = self._build_basis(x, knots, degree, num_ctrl_points)
        self.register_buffer("basis", basis)
        self.register_buffer("basis_pinv", torch.linalg.pinv(basis))

    @staticmethod
    def _build_basis(x: Tensor, knots: Tensor, degree: int, num_ctrl_points: int) -> Tensor:
        num_samples = x.numel()
        basis = torch.zeros(num_ctrl_points, num_samples, dtype=x.dtype)
        for i in range(num_ctrl_points):
            left = knots[i]
            right = knots[i + 1]
            mask = (x >= left) & (x < right)
            if i == num_ctrl_points - 1:
                mask = mask | (x == knots[-1])
            basis[i] = mask.to(x.dtype)

        for p in range(1, degree + 1):
            next_basis = torch.zeros_like(basis)
            for i in range(num_ctrl_points):
                left_denom = knots[i + p] - knots[i]
                right_denom = knots[i + p + 1] - knots[i + 1]
                left_term = 0.0
                right_term = 0.0
                if left_denom > 0:
                    left_term = ((x - knots[i]) / left_denom) * basis[i]
                if right_denom > 0 and i + 1 < num_ctrl_points:
                    right_term = ((knots[i + p + 1] - x) / right_denom) * basis[i + 1]
                next_basis[i] = left_term + right_term
            basis = next_basis

        return basis.transpose(0, 1).contiguous()

    def fit_batch(self, values: Tensor) -> Tensor:
        return torch.einsum("nt,btd->bnd", self.basis_pinv.to(values.dtype), values)

    def rebuild_batch(self, ctrl_points: Tensor) -> Tensor:
        return torch.einsum("tn,bnd->btd", self.basis.to(ctrl_points.dtype), ctrl_points)

    def refit_prefix_w(
        self,
        prefix_actions: Tensor,
        ctrl_points: Tensor,
        *,
        n_prefix: int,
        n_free: int,
        last_pt_weight: float,
    ) -> Tensor:
        phi = self.basis[:n_prefix]
        phi_free = phi[:, :n_free]
        phi_fixed = phi[:, n_free:]

        ctrl_fixed = ctrl_points[:, n_free:]
        y_fixed = torch.einsum("tf,bfd->btd", phi_fixed.to(ctrl_points.dtype), ctrl_fixed)
        residual = prefix_actions - y_fixed

        if last_pt_weight > 1e-9:
            sqrt_w = math.sqrt(last_pt_weight)
            penalty_row = torch.zeros(1, n_free, device=ctrl_points.device, dtype=ctrl_points.dtype)
            penalty_row[0, n_free - 1] = sqrt_w
            phi_aug = torch.cat([phi_free.to(ctrl_points.dtype), penalty_row], dim=0)
            penalty_target = sqrt_w * ctrl_points[:, n_free - 1 : n_free]
            target_aug = torch.cat([residual, penalty_target], dim=1)
            phi_pinv = torch.linalg.pinv(phi_aug)
            ctrl_free = torch.einsum("fn,bnd->bfd", phi_pinv, target_aug)
        else:
            phi_pinv = torch.linalg.pinv(phi_free.to(ctrl_points.dtype))
            ctrl_free = torch.einsum("fn,bnd->bfd", phi_pinv, residual)

        new_ctrl = ctrl_points.clone()
        new_ctrl[:, :n_free] = ctrl_free
        return new_ctrl


class JointMLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, norm_type: str):
        super().__init__()
        if norm_type == "batch":
            norm = lambda dim: nn.BatchNorm1d(dim)
        elif norm_type == "group":
            norm = lambda dim: nn.GroupNorm(num_groups=8, num_channels=dim)
        elif norm_type == "layer":
            norm = lambda dim: nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.mlp = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            norm(dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_out),
            norm(dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        bsz, steps, dim = x.shape
        x = self.mlp(x.reshape(bsz * steps, dim))
        return x.reshape(bsz, steps, self.dim_out)


class ImgObsPerceiver(nn.Module):
    def __init__(self, initial_n: int, in_channels: int, out_channels: int, mid_channels: int):
        super().__init__()
        side = int(math.sqrt(initial_n))
        if side * side != initial_n:
            raise ValueError("initial_n must be a perfect square.")
        self.side = side
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b (h w) c -> b c h w", h=self.side, w=self.side)
        x = self.conv_layers(x)
        return rearrange(x, "b c h w -> b (h w) c")


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.norm0 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn0 = Attention(hidden_size, heads=num_heads, dim_head=hidden_size // num_heads, dropout=0.0)
        self.attn1 = Attention(
            hidden_size,
            cross_attention_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=0.0,
        )
        self.attn2 = Attention(
            hidden_size,
            cross_attention_dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            dropout=0.0,
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 4,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0.0,
        )
        self.ada_ln = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, hidden_size * 5, bias=True))

    def forward(self, x: Tensor, emb_t: Tensor, emb_c: Tensor, emb_q: Tensor) -> Tensor:
        shift_msa, scale_msa, shift_mlp, scale_mlp, gate_mlp = self.ada_ln(emb_t).chunk(5, dim=1)
        x = self.attn0(self.norm0(x)) + x
        xq = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.attn1(xq, emb_c) + self.attn2(xq, emb_q)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class CondDiT(nn.Module):
    def __init__(self, config: ABPolicyConfig, input_dim: int, input_len: int):
        super().__init__()
        obs_dim = config.obs_dim
        self.timestep_emb = Timesteps(obs_dim, flip_sin_to_cos=False, downscale_freq_shift=1)
        self.timestep_proj = nn.Sequential(
            nn.Linear(obs_dim, obs_dim * 4),
            nn.SiLU(),
            nn.Linear(obs_dim * 4, obs_dim),
        )
        self.action_time_emb = nn.Parameter(
            get_timestep_embedding(torch.arange(input_len), input_dim).reshape(1, input_len, input_dim)
        )
        self.cond_norm = nn.LayerNorm(obs_dim)
        self.qpos_norm = nn.LayerNorm(obs_dim)
        self.conv_in = nn.Conv1d(
            input_dim,
            obs_dim,
            kernel_size=config.backbone_conv_kernel_size,
            padding=config.backbone_conv_kernel_size // 2,
        )
        self.blocks = nn.ModuleList([DiTBlock(obs_dim, config.backbone_num_attn_heads) for _ in range(config.backbone_num_blocks)])
        self.layer_norm_out = nn.LayerNorm(obs_dim)
        self.activation_out = nn.Mish()
        self.conv_out = nn.Conv1d(
            obs_dim,
            input_dim,
            kernel_size=config.backbone_conv_kernel_size,
            padding=config.backbone_conv_kernel_size // 2,
        )
        self.apply(self._init_weights)

    def forward(self, sample: Tensor, timesteps: Tensor, cond: Tensor, qpos_cond: Tensor) -> Tensor:
        sample = sample + self.action_time_emb.to(sample.dtype)
        sample = sample.permute(0, 2, 1)
        if timesteps.ndim == 0:
            timesteps = timesteps[None]
        timesteps = timesteps.to(sample.device).expand(sample.shape[0])

        t_emb = self.timestep_proj(self.timestep_emb(timesteps).to(sample.dtype))
        cond = self.cond_norm(cond.flatten(1, -2))
        qpos_cond = self.qpos_norm(qpos_cond.flatten(1, -2))

        x = self.conv_in(sample).permute(0, 2, 1)
        for block in self.blocks:
            x = block(x, t_emb, cond, qpos_cond)
        x = self.layer_norm_out(x).permute(0, 2, 1)
        x = self.activation_out(x)
        x = self.conv_out(x)
        return x.permute(0, 2, 1)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], num_kp: int):
        super().__init__()
        channels, height, width = input_shape
        self.nets = nn.Conv2d(channels, num_kp, kernel_size=1)
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, -1))
        self.height = height
        self.width = width
        self.num_kp = num_kp

    def forward(self, feature: Tensor) -> Tensor:
        feature = self.nets(feature).reshape(-1, self.height * self.width)
        attention = F.softmax(feature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        return torch.cat([expected_x, expected_y], dim=1).view(-1, self.num_kp, 2)


def replace_submodules(root_module: nn.Module, predicate, func) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    targets = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parent, child in targets:
        parent_module = root_module if not parent else root_module.get_submodule(".".join(parent))
        src = parent_module[int(child)] if isinstance(parent_module, nn.Sequential) else getattr(parent_module, child)
        dst = func(src)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(child)] = dst
        else:
            setattr(parent_module, child, dst)
    return root_module


def wrap_resnet(resnet: nn.Module, out_dim: int) -> nn.Module:
    resnet = replace_submodules(
        root_module=resnet,
        predicate=lambda module: isinstance(module, nn.BatchNorm2d),
        func=lambda module: nn.GroupNorm(max(1, module.num_features // 16), module.num_features),
    )
    resnet.pooler = nn.Sequential(
        SpatialSoftmax(input_shape=(2048, 7, 7), num_kp=out_dim // 2),
        nn.Flatten(),
    )
    return resnet


class ResNetEncoderWrapper(nn.Module):
    def __init__(self, resnet: nn.Module, pooled: bool, out_dim: int):
        super().__init__()
        self.center_crop = torchvision.transforms.CenterCrop((224, 224))
        self.model = wrap_resnet(resnet, out_dim=out_dim)
        self.pooled = pooled
        hidden_size = resnet.config.hidden_sizes[-1]
        self.linear = nn.Identity() if pooled else nn.Linear(hidden_size, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.center_crop(x)
        output = self.model(x)
        if self.pooled:
            output = output.pooler_output.unsqueeze(1)
        else:
            bsz, channels, height, width = output.last_hidden_state.shape
            pos_emb = get_2d_sincos_pos_embed(embed_dim=channels, grid_size=(height, width))
            pos_emb = torch.tensor(pos_emb, device=x.device, dtype=x.dtype).unsqueeze(0)
            output = rearrange(output.last_hidden_state, "b c h w -> b (h w) c")
            output = output + pos_emb
        return self.linear(output)


class DinoV2EncoderWrapper(nn.Module):
    def __init__(self, dino: nn.Module, pooled: bool, out_dim: int):
        super().__init__()
        self.model = dino
        self.pooled = pooled
        hidden_size = dino.config.hidden_size
        self.linear = nn.Linear(hidden_size * 2 if pooled else hidden_size, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        output = self.model(x)
        if self.pooled:
            cls_token = output.last_hidden_state[:, :1]
            avg_token = output.last_hidden_state[:, 1:].mean(dim=1, keepdim=True)
            output = torch.cat([cls_token, avg_token], dim=-1)
        else:
            output = output.last_hidden_state[:, 1:]
        return self.linear(output)


class ABModel(nn.Module):
    def __init__(self, config: ABPolicyConfig):
        super().__init__()
        self.config = config
        self.num_cameras = len(config.image_features)
        offline_mode = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"

        image_encoder = AutoModel.from_pretrained(
            config.image_encoder_name,
            local_files_only=offline_mode,
        )
        if config.image_encoder_freeze:
            image_encoder.requires_grad_(False)

        encoder_layers = getattr(getattr(image_encoder, "encoder", None), "layer", None)
        if encoder_layers is not None:
            image_encoder.encoder.layer = encoder_layers[: config.image_backbone_used_layers]

        wrapper_cls = DinoV2EncoderWrapper if "dino" in config.image_encoder_name.lower() else ResNetEncoderWrapper
        self.obs_encoder = wrapper_cls(image_encoder, pooled=config.image_encoder_pooled, out_dim=config.obs_dim)
        self.perceiver = ImgObsPerceiver(
            initial_n=(config.img_size // config.img_patch_size) ** 2,
            in_channels=config.obs_dim * self.num_cameras,
            out_channels=config.obs_dim,
            mid_channels=config.perceiver_mid_channels,
        )
        self.qpos_encoder = JointMLP(
            dim_in=config.robot_state_feature.shape[0],
            dim_hidden=config.qpos_encoder_dim_hidden,
            dim_out=config.obs_dim,
            norm_type=config.qpos_encoder_norm_type,
        )
        self.qpos_time_emb = nn.Parameter(
            get_timestep_embedding(torch.arange(config.n_obs_steps), config.obs_dim).reshape(
                1, config.n_obs_steps, config.obs_dim
            )
        )
        action_len = config.bspline_num_ctrl_points if config.use_bspline else config.horizon
        self.backbone = CondDiT(config, input_dim=config.action_feature.shape[0], input_len=action_len)
        self.projector = BSplineProjector(config.horizon, config.bspline_degree, config.bspline_num_ctrl_points)

    def preprocess_images(self, images: Tensor) -> Tensor:
        bsz, cams = images.shape[:2]
        x = images.reshape(bsz * cams, *images.shape[2:]).float()
        if self.training and self.config.image_noise_std > 0:
            x = torch.clamp(x + torch.randn_like(x) * self.config.image_noise_std, 0.0, 255.0)

        if self.config.image_crop_shape is not None:
            crop_h, crop_w = self.config.image_crop_shape
            _, _, img_h, img_w = x.shape
            top = (img_h - crop_h) // 2
            left = (img_w - crop_w) // 2
            if self.training:
                jitter_h, jitter_w = self.config.image_crop_jitter_max
                if jitter_h > 0:
                    top += int(torch.randint(-jitter_h, jitter_h + 1, ()).item())
                if jitter_w > 0:
                    left += int(torch.randint(-jitter_w, jitter_w + 1, ()).item())
                top = max(0, min(top, img_h - crop_h))
                left = max(0, min(left, img_w - crop_w))
            x = x[:, :, top : top + crop_h, left : left + crop_w]

        x = F.interpolate(x, size=(self.config.img_size, self.config.img_size), mode="bilinear", align_corners=False)
        return (x / 255.0).reshape(bsz, cams, *x.shape[1:])

    def encode_observations(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        qpos = batch[OBS_STATE]
        images = self.preprocess_images(batch[OBS_IMAGES][:, -1])
        image_embeds = [self.obs_encoder(images[:, cam_idx]) for cam_idx in range(images.shape[1])]
        obs_emb = self.perceiver(torch.cat(image_embeds, dim=-1))
        qpos_emb = self.qpos_encoder(qpos) + self.qpos_time_emb.to(qpos.dtype)
        return obs_emb, qpos_emb

    def forward_vector_field(self, batch: dict[str, Tensor], noisy_actions: Tensor, timesteps: Tensor) -> Tensor:
        obs_emb, qpos_emb = self.encode_observations(batch)
        return self.backbone(noisy_actions, timesteps, cond=obs_emb, qpos_cond=qpos_emb)

    def generate_ctrl_points(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)
        action_dim = self.config.action_feature.shape[0]
        action_len = self.config.bspline_num_ctrl_points if self.config.use_bspline else self.config.horizon
        ctrl = (
            noise
            if noise is not None
            else torch.randn(batch[OBS_STATE].shape[0], action_len, action_dim, device=device, dtype=dtype)
        )
        obs_emb, qpos_emb = self.encode_observations(batch)
        t_vals = torch.linspace(0.0, 1.0, self.config.num_inference_steps + 1, device=device, dtype=dtype)
        for idx in range(self.config.num_inference_steps):
            t = torch.full((ctrl.size(0),), t_vals[idx], device=device, dtype=dtype)
            r = torch.full((ctrl.size(0),), t_vals[idx + 1], device=device, dtype=dtype)
            velocity = self.backbone(ctrl, t, cond=obs_emb, qpos_cond=qpos_emb)
            ctrl = ctrl - (t - r).view(-1, 1, 1) * velocity
        return ctrl


class ABPolicy(PreTrainedPolicy):
    config_class = ABPolicyConfig
    name = "abpolicy"

    def __init__(self, config: ABPolicyConfig, dataset_stats: dict | None = None, **kwargs):
        super().__init__(config)
        del kwargs
        config.validate_features()
        self.config = config
        self.model = ABModel(config)
        self.cage = self.model

        action_stats = (dataset_stats or {}).get(ACTION, {})
        action_dim = config.action_feature.shape[0]
        action_min = action_stats.get("min", torch.full((action_dim,), -1.0))
        action_max = action_stats.get("max", torch.full((action_dim,), 1.0))
        self.register_buffer("action_min", torch.as_tensor(action_min, dtype=torch.float32))
        self.register_buffer("action_max", torch.as_tensor(action_max, dtype=torch.float32))
        self.register_buffer("action_range", self.action_max - self.action_min)
        self.reset()

    def get_optim_params(self):
        return [
            {
                "params": [p for p in self.model.obs_encoder.parameters() if p.requires_grad],
                "weight_decay": self.config.optimizer_weight_decay,
                "lr": self.config.optimizer_lr,
                "betas": self.config.optimizer_betas,
            },
            {
                "params": list(self.model.perceiver.parameters()) + list(self.model.qpos_encoder.parameters()),
                "weight_decay": self.config.optimizer_weight_decay,
                "lr": self.config.optimizer_perceiver_lr,
                "betas": self.config.optimizer_perceiver_betas,
            },
            {
                "params": self.model.backbone.parameters(),
                "weight_decay": self.config.optimizer_weight_decay,
                "lr": self.config.optimizer_lr,
                "betas": self.config.optimizer_betas,
            },
        ]

    def _normalize_action_like(self, tensor: Tensor) -> Tensor:
        scale = self.action_range.to(tensor.device, tensor.dtype)
        minimum = self.action_min.to(tensor.device, tensor.dtype)
        normalized = (tensor - minimum) / (scale + 1e-8)
        return normalized * 2.0 - 1.0

    def _denormalize_action_like(self, tensor: Tensor) -> Tensor:
        scale = self.action_range.to(tensor.device, tensor.dtype)
        minimum = self.action_min.to(tensor.device, tensor.dtype)
        return ((tensor + 1.0) * 0.5) * scale + minimum

    def _stack_image_inputs(self, batch: dict[str, Tensor]) -> Tensor:
        first_key = next(iter(self.config.image_features))
        first_img = batch[first_key]
        images = [batch[key].unsqueeze(1) if first_img.dim() == 4 else batch[key] for key in self.config.image_features]
        return torch.stack(images, dim=2)

    def _prepare_model_batch(self, batch: dict[str, Tensor], *, for_training: bool) -> dict[str, Tensor]:
        state = batch[OBS_STATE]
        if for_training and self.config.qpos_noise_std > 0:
            state = state + torch.randn_like(state) * self.config.qpos_noise_std
        prepared = {
            OBS_STATE: self._normalize_action_like(state),
            OBS_IMAGES: self._stack_image_inputs(batch),
        }
        if ACTION in batch:
            prepared[ACTION] = batch[ACTION]
        return prepared

    def _compute_flow_target(self, target_ctrl: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        noise = torch.randn_like(target_ctrl)
        timestep = torch.rand(target_ctrl.shape[0], device=target_ctrl.device, dtype=target_ctrl.dtype)
        xt = (1.0 - timestep).view(-1, 1, 1) * noise + timestep.view(-1, 1, 1) * target_ctrl
        ut = target_ctrl - noise
        return timestep, xt, ut

    def reset(self):
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
            OBS_IMAGES: deque(maxlen=self.config.n_obs_steps),
        }
        self._latest_actions_queue = deque([], maxlen=self.config.horizon + self.config.n_action_steps)
        self._pending_actions = deque()
        self._obs_timestamp_queue = deque(maxlen=self.config.n_obs_steps)

    def sync_executed_actions_through(self, timestep: int) -> None:
        while self._pending_actions and self._pending_actions[0][0] <= timestep:
            _, action = self._pending_actions.popleft()
            self._latest_actions_queue.append(action.detach().cpu().numpy())

    def stage_action_chunk(self, start_timestep: int, actions: Tensor) -> None:
        if actions.ndim == 3:
            actions = actions[0]
        self._pending_actions = deque((start_timestep + idx, action.detach().cpu()) for idx, action in enumerate(actions))

    def _maybe_refit_ctrl_points(self, ctrl_points: Tensor) -> Tensor:
        needed = self.config.action_history_horizon
        if len(self._latest_actions_queue) < needed:
            return ctrl_points
        prefix = torch.as_tensor(
            list(self._latest_actions_queue)[-needed:],
            device=ctrl_points.device,
            dtype=ctrl_points.dtype,
        ).unsqueeze(0)
        return self.model.projector.refit_prefix_w(
            prefix,
            ctrl_points,
            n_prefix=needed,
            n_free=self.config.refit_n_free,
            last_pt_weight=self.config.refit_last_pt_weight,
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if not self.config.image_features and hasattr(self.cage, "generate_ctrl_points"):
            if OBS_STATE in batch and len(self._queues.get(OBS_STATE, [])) == 0:
                self._obs_timestamp_queue.append(len(self._obs_timestamp_queue))
                ctrl_points = self.cage.generate_ctrl_points({OBS_STATE: batch[OBS_STATE].unsqueeze(1)})
                return ctrl_points[:, : self.config.n_action_steps]

        queue_batch = {OBS_STATE: torch.stack(list(self._queues[OBS_STATE]), dim=1)}
        if OBS_IMAGES in self._queues and len(self._queues[OBS_IMAGES]) > 0:
            queue_batch[OBS_IMAGES] = torch.stack(list(self._queues[OBS_IMAGES]), dim=1)
        self._obs_timestamp_queue.append(len(self._obs_timestamp_queue))
        ctrl_points = self.model.generate_ctrl_points(queue_batch, noise=noise)
        if self.config.use_bspline:
            ctrl_points = self._denormalize_action_like(ctrl_points)
            ctrl_points = self._maybe_refit_ctrl_points(ctrl_points)
            full_actions = self.model.projector.rebuild_batch(ctrl_points)
        else:
            full_actions = self._denormalize_action_like(ctrl_points)
        start = self.config.action_history_horizon
        end = start + self.config.n_action_steps
        return full_actions[:, start:end]

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        if ACTION in batch:
            batch = dict(batch)
            batch.pop(ACTION)
        batch = dict(batch)
        batch[OBS_IMAGES] = self._stack_image_inputs(batch)
        self._queues = populate_queues(self._queues, batch)
        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch, noise=noise)
            self._queues[ACTION].extend(actions.transpose(0, 1))
        action = self._queues[ACTION].popleft()
        self._latest_actions_queue.append(action.detach().cpu().numpy())
        return action

    def forward(self, batch: dict[str, Tensor], reduction: str = "mean") -> tuple[Tensor, dict]:
        prepared = self._prepare_model_batch(batch, for_training=True)
        action_target = self.model.projector.fit_batch(prepared[ACTION]) if self.config.use_bspline else prepared[ACTION]
        action_target = self._normalize_action_like(action_target)
        timestep, xt, ut = self._compute_flow_target(action_target)
        prediction = self.model.forward_vector_field(prepared, xt, timestep)
        loss_per_sample = F.mse_loss(prediction, ut, reduction="none").mean(dim=(1, 2))
        if reduction == "none":
            return loss_per_sample, {"loss": float(loss_per_sample.mean().item())}
        loss = loss_per_sample.mean()
        return loss, {"loss": float(loss.item())}
