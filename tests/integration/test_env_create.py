"""Integration tests: environment creation and basic API surface."""
import numpy as np
import pytest

pytestmark = pytest.mark.integration


class TestEnvCreation:
    def test_env_makes_without_error(self, pick_cube_env):
        assert pick_cube_env is not None

    def test_action_space_is_6d(self, pick_cube_env):
        assert pick_cube_env.action_space.shape == (6,)

    def test_action_space_bounds(self, pick_cube_env):
        space = pick_cube_env.action_space
        assert (space.low == -1.0).all()
        assert (space.high == 1.0).all()

    def test_reset_returns_obs_and_info(self, pick_cube_env):
        obs, info = pick_cube_env.reset(seed=0)
        assert obs is not None
        assert isinstance(info, dict)

    def test_obs_contains_sensor_data(self, pick_cube_env):
        obs, _ = pick_cube_env.reset(seed=0)
        assert "sensor_data" in obs

    def test_obs_rgb_shape(self, pick_cube_env):
        obs, _ = pick_cube_env.reset(seed=0)
        for cam_data in obs["sensor_data"].values():
            rgb = cam_data.get("rgb")
            if rgb is not None:
                arr = np.asarray(rgb)
                # shape is (batch, H, W, 3) or (H, W, 3)
                assert arr.ndim in (3, 4)
                assert arr.shape[-1] == 3
                return
        pytest.fail("No RGB found in sensor_data")

    def test_render_returns_array(self, pick_cube_env):
        pick_cube_env.reset(seed=0)
        frame = pick_cube_env.render()
        assert frame is not None
        arr = np.asarray(frame)
        assert arr.ndim in (3, 4)


class TestEnvStep:
    def test_zero_action_step(self, pick_cube_env):
        pick_cube_env.reset(seed=0)
        action = np.zeros(6, dtype=np.float32)
        obs, reward, terminated, truncated, info = pick_cube_env.step(action)
        assert obs is not None
        assert "success" in info

    def test_reward_is_scalar_or_batchable(self, pick_cube_env):
        pick_cube_env.reset(seed=0)
        action = pick_cube_env.action_space.sample()
        _, reward, _, _, _ = pick_cube_env.step(action)
        # reward may be a tensor; must be convertible to float
        float(reward.mean() if hasattr(reward, "mean") else reward)

    def test_truncated_after_max_steps(self, pick_cube_env):
        pick_cube_env.reset(seed=0)
        action = np.zeros(6, dtype=np.float32)
        done = False
        for _ in range(100):
            _, _, terminated, truncated, _ = pick_cube_env.step(action)
            if terminated or truncated:
                done = True
                break
        assert done, "Episode should terminate/truncate within max_episode_steps=100"
