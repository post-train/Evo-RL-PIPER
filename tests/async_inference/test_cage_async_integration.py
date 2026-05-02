from __future__ import annotations

from collections import deque
from queue import Queue
import threading
from types import SimpleNamespace

import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction
from lerobot.policies.abpolicy.configuration_ab import ABPolicyConfig
from lerobot.policies.factory import get_policy_class, make_policy_config
from lerobot.policies.cage.modeling_cage import CAGEPolicy
from lerobot.utils.constants import ACTION, OBS_STATE
from tests.mocks.mock_robot import MockRobotConfig


class _MockCAGEModel:
    def __init__(self, horizon: int, action_dim: int):
        self.horizon = horizon
        self.action_dim = action_dim
        self.action_fitter = None

    def generate_ctrl_points(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = batch[OBS_STATE].shape[0]
        return torch.zeros(batch_size, self.horizon, self.action_dim)


def _make_policy_stub(n_obs_steps: int = 4, n_action_steps: int = 6, action_dim: int = 3) -> CAGEPolicy:
    policy = CAGEPolicy.__new__(CAGEPolicy)
    horizon = n_obs_steps - 1 + n_action_steps
    policy.config = SimpleNamespace(
        image_features={},
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        horizon=horizon,
        action_history_horizon=4,
        use_bspline=False,
        control_freq=30,
        refit_n_free=2,
        refit_last_pt_weight=0.05,
    )
    policy._queues = {
        OBS_STATE: deque(maxlen=n_obs_steps),
        ACTION: deque(maxlen=n_action_steps),
    }
    policy._latest_actions_queue = deque([torch.zeros(action_dim).numpy() for _ in range(20)], maxlen=24)
    policy._ctrl_y_queue = deque(maxlen=1)
    policy._obs_timestamp_queue = deque(maxlen=n_obs_steps)
    policy._pending_actions = deque()
    policy.cage = _MockCAGEModel(horizon=horizon, action_dim=action_dim)
    return policy


def test_cage_predict_action_chunk_supports_async_path():
    policy = _make_policy_stub()
    batch = {OBS_STATE: torch.ones(1, 3)}
    original_latest_action = policy._latest_actions_queue[-1].copy()
    original_queue_len = len(policy._latest_actions_queue)

    actions = policy.predict_action_chunk(batch)

    assert actions.shape == (1, policy.config.n_action_steps, 3)
    assert len(policy._obs_timestamp_queue) == 1
    assert len(policy._latest_actions_queue) == original_queue_len
    assert (policy._latest_actions_queue[-1] == original_latest_action).all()


def test_cage_pending_actions_sync_matches_executed_history():
    policy = _make_policy_stub()
    actions = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=torch.float32)

    policy.stage_action_chunk(5, actions)
    policy.sync_executed_actions_through(6)

    assert len(policy._pending_actions) == 1
    assert policy._pending_actions[0][0] == 7
    assert policy._latest_actions_queue[-1][0] == 2.0


def test_cage_async_client_defaults_to_latest_only():
    cfg = RobotClientConfig(
        server_address="localhost:8080",
        robot=MockRobotConfig(),
        chunk_size_threshold=0.0,
        policy_type="cage",
        pretrained_name_or_path="test-model",
        actions_per_chunk=16,
    )

    assert cfg.aggregate_fn_name == "latest_only"


def test_cage_async_client_defaults_to_full_refresh_threshold():
    cfg = RobotClientConfig(
        server_address="localhost:8080",
        robot=MockRobotConfig(),
        policy_type="cage",
        pretrained_name_or_path="test-model",
        actions_per_chunk=16,
    )

    assert cfg.chunk_size_threshold == 1.0


def test_cage_async_queue_replaces_pending_chunk():
    from lerobot.async_inference.robot_client import RobotClient

    client = RobotClient.__new__(RobotClient)
    client.config = SimpleNamespace(policy_type="cage")
    client.action_queue_lock = threading.Lock()
    client.latest_action_lock = threading.Lock()
    client.latest_action = 4
    client.action_queue = Queue()
    client.action_queue.put(TimedAction(timestamp=0.0, timestep=5, action=torch.tensor([0.0])))
    client.action_queue.put(TimedAction(timestamp=0.0, timestep=6, action=torch.tensor([0.0])))

    incoming = [
        TimedAction(timestamp=0.1, timestep=5, action=torch.tensor([1.0])),
        TimedAction(timestamp=0.1, timestep=6, action=torch.tensor([2.0])),
        TimedAction(timestamp=0.1, timestep=7, action=torch.tensor([3.0])),
    ]

    RobotClient._aggregate_action_queues(client, incoming)

    assert client.action_queue.qsize() == 3
    got = [client.action_queue.get_nowait().get_timestep() for _ in range(3)]
    assert got == [5, 6, 7]


def test_abpolicy_factory_registration():
    cfg = make_policy_config("abpolicy")
    cls = get_policy_class("abpolicy")

    assert isinstance(cfg, ABPolicyConfig)
    assert cls.name == "abpolicy"
