"""Unit tests for vla.action — no SAPIEN, no GPU."""
import numpy as np
import pytest
from gymnasium import spaces

from vla.action import adapt_action, clip_deltas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _space(n: int) -> spaces.Box:
    return spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)


# ---------------------------------------------------------------------------
# adapt_action
# ---------------------------------------------------------------------------

class TestAdaptAction:
    def test_truncates_7d_to_6d(self):
        vla = np.ones(7, dtype=np.float32) * 0.5
        out = adapt_action(vla, _space(6))
        assert out.shape == (6,)
        np.testing.assert_array_almost_equal(out, np.ones(6) * 0.5)

    def test_gripper_dim_dropped_for_6dof(self):
        vla = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.99], dtype=np.float32)
        out = adapt_action(vla, _space(6))
        np.testing.assert_array_almost_equal(out, vla[:6])

    def test_clips_to_action_space_bounds(self):
        vla = np.ones(7, dtype=np.float32) * 2.0  # above upper bound of 1.0
        out = adapt_action(vla, _space(6))
        assert (out <= 1.0).all()

    def test_pads_with_zeros_when_env_wider_than_vla(self):
        vla = np.ones(4, dtype=np.float32) * 0.3
        out = adapt_action(vla, _space(7))
        assert out.shape == (7,)
        np.testing.assert_array_almost_equal(out[:4], 0.3)
        np.testing.assert_array_almost_equal(out[4:], 0.0)

    def test_output_dtype_is_float32(self):
        vla = np.ones(7, dtype=np.float64)
        out = adapt_action(vla, _space(6))
        assert out.dtype == np.float32

    def test_exact_match_no_clipping_needed(self):
        vla = np.array([0.0, -0.5, 0.5, -0.1, 0.1, 0.0, 1.0], dtype=np.float32)
        out = adapt_action(vla, _space(6))
        np.testing.assert_array_almost_equal(out, vla[:6])


# ---------------------------------------------------------------------------
# clip_deltas
# ---------------------------------------------------------------------------

class TestClipDeltas:
    def test_pos_dims_clipped(self):
        delta = np.array([1.0, -1.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        out = clip_deltas(delta, pos_limit=0.05)
        assert abs(out[0]) <= 0.05
        assert abs(out[1]) <= 0.05
        assert abs(out[2]) <= 0.05

    def test_rot_dims_clipped(self):
        delta = np.array([0.0, 0.0, 0.0, 5.0, -5.0, 5.0], dtype=np.float32)
        out = clip_deltas(delta, rot_limit=0.1)
        assert abs(out[3]) <= 0.1
        assert abs(out[4]) <= 0.1
        assert abs(out[5]) <= 0.1

    def test_within_limits_unchanged(self):
        delta = np.array([0.01, -0.02, 0.03, 0.05, -0.05, 0.05], dtype=np.float32)
        out = clip_deltas(delta)
        np.testing.assert_array_almost_equal(out, delta)

    def test_does_not_modify_input(self):
        delta = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        original = delta.copy()
        clip_deltas(delta)
        np.testing.assert_array_equal(delta, original)

    def test_custom_limits(self):
        delta = np.ones(6, dtype=np.float32)
        out = clip_deltas(delta, pos_limit=0.2, rot_limit=0.3)
        assert (out[:3] <= 0.2).all()
        assert (out[3:] <= 0.3).all()
