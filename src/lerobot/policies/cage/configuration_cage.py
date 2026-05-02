from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.abpolicy.configuration_ab import ABCosineSchedulerConfig, ABPolicyConfig


@PreTrainedConfig.register_subclass("cage")
@dataclass
class CAGEConfig(ABPolicyConfig):
    pass


CAGESchedulerConfig = ABCosineSchedulerConfig
