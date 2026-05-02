from lerobot.policies.abpolicy.modeling_ab import ABPolicy
from lerobot.policies.cage.configuration_cage import CAGEConfig


class CAGEPolicy(ABPolicy):
    config_class = CAGEConfig
    name = "cage"
