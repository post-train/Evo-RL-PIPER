#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import logging
from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.a2a.configuration_diffusion import A2AConfig
from lerobot.policies.abpolicy.configuration_ab import ABPolicyConfig
from lerobot.policies.cage.configuration_cage import CAGEConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.evo1.configuration_evo1 import Evo1Config
from lerobot.policies.fm.configuration_fm import FlowMatchingConfig
from lerobot.policies.original_a2a.configuration_diffusion import OriginalA2AConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.policies.vita.configuration_diffusion import VitaConfig
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """
    Retrieves a policy class by its registered name.

    This function uses dynamic imports to avoid loading all policy classes into memory
    at once, improving startup time and reducing dependencies.

    Args:
        name: The name of the policy. Supported names are "diffusion", "act",
              "pi0", "pi05", "wall_x", and other registered policies.

    Returns:
        The policy class corresponding to the given name.

    Raises:
        NotImplementedError: If the policy name is not recognized.
    """
    if name == "abpolicy":
        from lerobot.policies.abpolicy.modeling_ab import ABPolicy

        return ABPolicy
    elif name == "a2a":
        from lerobot.policies.a2a.modeling_diffusion import A2APolicy

        return A2APolicy
    elif name == "original_a2a":
        from lerobot.policies.original_a2a.modeling_diffusion import OriginalA2APolicy

        return OriginalA2APolicy
    elif name == "cage":
        from lerobot.policies.cage.modeling_cage import CAGEPolicy

        return CAGEPolicy
    elif name == "flow_matching":
        from lerobot.policies.fm.modeling_fm import FlowMatchingPolicy

        return FlowMatchingPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy
    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "evo1":
        from lerobot.policies.evo1.modeling_evo1 import EVO1Policy

        return EVO1Policy
    elif name == "vita":
        from lerobot.policies.vita.modeling_diffusion import VitaPolicy

        return VitaPolicy
    elif name == "wall_x":
        from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy

        return WallXPolicy
    else:
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """
    Instantiates a policy configuration object based on the policy type.

    This factory function simplifies the creation of policy configuration objects by
    mapping a string identifier to the corresponding config class.

    Args:
        policy_type: The type of the policy. Supported types include "diffusion",
                     "act", "pi0", "pi05", "wall_x", and other registered policies.
        **kwargs: Keyword arguments to be passed to the configuration class constructor.

    Returns:
        An instance of a `PreTrainedConfig` subclass.

    Raises:
        ValueError: If the `policy_type` is not recognized.
    """
    if policy_type == "abpolicy":
        return ABPolicyConfig(**kwargs)
    elif policy_type == "a2a":
        return A2AConfig(**kwargs)
    elif policy_type == "original_a2a":
        return OriginalA2AConfig(**kwargs)
    elif policy_type == "cage":
        return CAGEConfig(**kwargs)
    elif policy_type == "flow_matching":
        return FlowMatchingConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "evo1":
        return Evo1Config(**kwargs)
    elif policy_type == "vita":
        return VitaConfig(**kwargs)
    elif policy_type == "wall_x":
        return WallXConfig(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


class ProcessorConfigKwargs(TypedDict, total=False):
    """
    A TypedDict defining the keyword arguments for processor configuration.

    This provides type hints for the optional arguments passed to `make_pre_post_processors`,
    improving code clarity and enabling static analysis.

    Attributes:
        preprocessor_config_filename: The filename for the preprocessor configuration.
        postprocessor_config_filename: The filename for the postprocessor configuration.
        preprocessor_overrides: A dictionary of overrides for the preprocessor configuration.
        postprocessor_overrides: A dictionary of overrides for the postprocessor configuration.
        dataset_stats: Dataset statistics for normalization.
    """

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """
    Create or load pre- and post-processor pipelines for a given policy.

    This function acts as a factory. It can either load existing processor pipelines
    from a pretrained path or create new ones from scratch based on the policy
    configuration. Each policy type has a dedicated factory function for its
    processors (e.g., `make_tdmpc_pre_post_processors`).

    Args:
        policy_cfg: The configuration of the policy for which to create processors.
        pretrained_path: An optional path to load pretrained processor pipelines from.
            If provided, pipelines are loaded from this path.
        **kwargs: Keyword arguments for processor configuration, as defined in
            `ProcessorConfigKwargs`.

    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.

    Raises:
        NotImplementedError: If a processor factory is not implemented for the given
            policy configuration type.
    """
    if pretrained_path:
        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # Create a new processor based on policy type
    if isinstance(policy_cfg, A2AConfig):
        from lerobot.policies.a2a.processor_diffusion import make_a2a_pre_post_processors

        processors = make_a2a_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    elif isinstance(policy_cfg, OriginalA2AConfig):
        from lerobot.policies.original_a2a.processor_diffusion import make_original_a2a_pre_post_processors

        processors = make_original_a2a_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, (ABPolicyConfig, CAGEConfig)):
        from lerobot.policies.abpolicy.processor_ab import make_ab_pre_post_processors

        processors = make_ab_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, FlowMatchingConfig):
        from lerobot.policies.fm.processor_fm import make_flow_matching_pre_post_processors

        processors = make_flow_matching_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        processors = make_pi0_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI05Config):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

        processors = make_pi05_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, Evo1Config):
        from lerobot.policies.evo1.processor_evo1 import make_evo1_pre_post_processors

        processors = make_evo1_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )
    elif isinstance(policy_cfg, VitaConfig):
        from lerobot.policies.vita.processor_diffusion import make_vita_pre_post_processors

        processors = make_vita_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, WallXConfig):
        from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors

        processors = make_wall_x_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_policy_config(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """
    Instantiate a policy model.

    This factory function handles the logic of creating a policy, which requires
    determining the input and output feature shapes. These shapes can be derived
    either from a `LeRobotDatasetMetadata` object or an `EnvConfig` object. The function
    can either initialize a new policy from scratch or load a pretrained one.

    Args:
        cfg: The configuration for the policy to be created. If `cfg.pretrained_path` is
             set, the policy will be loaded with weights from that path.
        ds_meta: Dataset metadata used to infer feature shapes and types. Also provides
                 statistics for normalization layers.
        env_cfg: Environment configuration used to infer feature shapes and types.
                 One of `ds_meta` or `env_cfg` must be provided.
        rename_map: Optional mapping of dataset or environment feature keys to match
                 expected policy feature names (e.g., `"left"` → `"camera1"`).

    Returns:
        An instantiated and device-placed policy model.

    Raises:
        ValueError: If both or neither of `ds_meta` and `env_cfg` are provided.
        NotImplementedError: If attempting to use an unsupported policy-backend
                             combination (e.g., VQBeT with 'mps').
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        if cfg.type == "abpolicy":
            preferred_image_keys = [
                "observation.images.cam_high_left",
                "observation.images.cam_high_right",
                "observation.images.left_left_arm",
                "observation.images.right_right_arm",
                "observation.images.wide_angle",
            ]
            selected_input_features: dict[str, Any] = {}

            if "observation.state" in features:
                selected_input_features["observation.state"] = features["observation.state"]

            for key in preferred_image_keys:
                if key in features:
                    selected_input_features[key] = features[key]

            if "observation.state" not in selected_input_features:
                raise ValueError("ABPolicy requires `observation.state` in the dataset features.")

            image_keys = [k for k in selected_input_features if k.startswith("observation.images.")]
            if len(image_keys) < 2:
                fallback_image_keys = [
                    key for key, ft in features.items() if ft.type is FeatureType.VISUAL and key not in image_keys
                ]
                for key in fallback_image_keys:
                    selected_input_features[key] = features[key]
                    image_keys.append(key)
                    if len(image_keys) >= 2:
                        break

            if len(image_keys) < 2:
                raise ValueError("ABPolicy requires at least two camera streams.")

            cfg.input_features = selected_input_features
            logging.info("ABPolicy selected input features: %s", list(cfg.input_features))
        else:
            cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    # Pass dataset_stats to the policy if available (needed for some policies like SARM)
    if ds_meta is not None and hasattr(ds_meta, "stats"):
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    if cfg.pretrained_path and not cfg.use_peft:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # Load a pretrained PEFT model on top of the policy. The pretrained path points to the folder/repo
        # of the adapter and the adapter's config contains the path to the base policy. So we need the
        # adapter config first, then load the correct policy and then apply PEFT.
        from peft import PeftConfig, PeftModel

        logging.info("Loading policy's PEFT adapter.")

        peft_pretrained_path = cfg.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            # This means that there's a bug or we trained a policy from scratch using PEFT.
            # It is more likely that this is a bug so we'll raise an error.
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)

    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - add a check_state(cfg, features) and check_action(cfg, features)

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """Get policy class from its registered name using dynamic imports.

    This is used as a helper function to import policies from 3rd party lerobot plugins.

    Args:
        name: The name of the policy.
    Returns:
        The policy class corresponding to the given name.
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # e.g., DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # e.g., DiffusionConfig -> DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # e.g., configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """Create pre- and post-processors from a policy configuration using dynamic imports.

    This is used as a helper function to import processor factories from 3rd party lerobot plugins.

    Args:
        config: The policy configuration object.
        dataset_stats: Dataset statistics for normalization.
    Returns:
        A tuple containing the input (pre-processor) and output (post-processor) pipelines.
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )  # e.g., configuration_diffusion -> processor_diffusion
    logging.debug(
        f"Instantiating pre/post processors using function '{function_name}' from module '{module_path}'"
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
