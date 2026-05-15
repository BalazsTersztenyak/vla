"""Integration tests: full episode with the real OpenVLA model.

Requires: GPU, ~16 GB VRAM, and the 'vla' optional dependencies.
Run with: uv run pytest -m gpu
"""
import numpy as np
import pytest

from vla.runner import EpisodeResult, run_episode

pytestmark = [pytest.mark.integration, pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def openvla_policy():
    from vla.policy.openvla import OpenVLAPolicy
    return OpenVLAPolicy(unnorm_key="bridge_orig", device="cuda:0")


class TestOpenVLAEpisode:
    def test_policy_loads(self, openvla_policy):
        assert openvla_policy is not None

    def test_policy_predict_shape(self, openvla_policy, dummy_image):
        action = openvla_policy.predict(dummy_image, "pick up the cube")
        assert action.shape == (7,)

    def test_policy_predict_dtype(self, openvla_policy, dummy_image):
        action = openvla_policy.predict(dummy_image, "pick up the cube")
        assert action.dtype == np.float32

    def test_full_episode_10_steps(self, pick_cube_env, openvla_policy):
        result = run_episode(
            pick_cube_env,
            openvla_policy,
            instruction="pick up the cube",
            max_steps=10,
            seed=42,
        )
        assert isinstance(result, EpisodeResult)
        assert 1 <= result.steps <= 10
        assert np.isfinite(result.total_reward)
        assert isinstance(result.success, bool)

    def test_avg_step_time_is_reasonable(self, pick_cube_env, openvla_policy):
        result = run_episode(pick_cube_env, openvla_policy, max_steps=5, seed=0)
        # Each step should complete in under 60s even on slow hardware
        assert result.avg_step_time < 60.0
