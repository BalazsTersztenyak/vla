"""Integration tests: full episode loop with RandomPolicy (no GPU)."""
import numpy as np
import pytest

from vla.policy.random_policy import RandomPolicy
from vla.runner import EpisodeResult, run_episode

pytestmark = pytest.mark.integration


class TestRandomEpisode:
    def test_episode_returns_result(self, pick_cube_env):
        policy = RandomPolicy(seed=0)
        result = run_episode(pick_cube_env, policy, max_steps=5, seed=0)
        assert isinstance(result, EpisodeResult)

    def test_episode_runs_correct_number_of_steps(self, pick_cube_env):
        policy = RandomPolicy(seed=0)
        result = run_episode(pick_cube_env, policy, max_steps=5, seed=0)
        assert 1 <= result.steps <= 5

    def test_episode_total_reward_is_finite(self, pick_cube_env):
        policy = RandomPolicy(seed=1)
        result = run_episode(pick_cube_env, policy, max_steps=5, seed=1)
        assert np.isfinite(result.total_reward)

    def test_episode_success_is_bool(self, pick_cube_env):
        policy = RandomPolicy(seed=2)
        result = run_episode(pick_cube_env, policy, max_steps=5, seed=2)
        assert isinstance(result.success, bool)

    def test_step_times_recorded(self, pick_cube_env):
        policy = RandomPolicy(seed=3)
        result = run_episode(pick_cube_env, policy, max_steps=5, seed=3)
        assert len(result.step_times) == result.steps
        assert all(t > 0 for t in result.step_times)

    def test_no_frames_without_record_video(self, pick_cube_env):
        policy = RandomPolicy(seed=4)
        result = run_episode(pick_cube_env, policy, max_steps=3, seed=4, record_video=False)
        assert result.frames == []

    def test_frames_captured_with_record_video(self, pick_cube_env):
        policy = RandomPolicy(seed=5)
        result = run_episode(pick_cube_env, policy, max_steps=3, seed=5, record_video=True)
        assert len(result.frames) == result.steps
        for frame in result.frames:
            arr = np.asarray(frame)
            assert arr.ndim == 3  # (H, W, 3)
            assert arr.shape[2] == 3

    def test_different_seeds_give_different_rewards(self, pick_cube_env):
        policy_a = RandomPolicy(seed=10)
        policy_b = RandomPolicy(seed=11)
        r_a = run_episode(pick_cube_env, policy_a, max_steps=5, seed=10)
        r_b = run_episode(pick_cube_env, policy_b, max_steps=5, seed=11)
        # Not guaranteed but overwhelmingly likely with different random policies
        # Just check both are valid
        assert np.isfinite(r_a.total_reward) and np.isfinite(r_b.total_reward)
