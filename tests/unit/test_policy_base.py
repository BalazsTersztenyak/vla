"""Unit tests for policy implementations — no GPU, no env."""
import numpy as np
import pytest
from PIL import Image

from vla.policy.base import Policy
from vla.policy.random_policy import RandomPolicy


def _dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


class TestRandomPolicy:
    def test_satisfies_policy_protocol(self):
        policy = RandomPolicy()
        assert isinstance(policy, Policy)

    def test_predict_returns_ndarray(self):
        policy = RandomPolicy(seed=0)
        action = policy.predict(_dummy_image(), "pick up the cube")
        assert isinstance(action, np.ndarray)

    def test_predict_shape(self):
        policy = RandomPolicy(seed=0)
        action = policy.predict(_dummy_image(), "pick up the cube")
        assert action.shape == (7,)

    def test_predict_dtype(self):
        policy = RandomPolicy(seed=0)
        action = policy.predict(_dummy_image(), "any instruction")
        assert action.dtype == np.float32

    def test_predict_in_range(self):
        policy = RandomPolicy(seed=42)
        for _ in range(20):
            action = policy.predict(_dummy_image(), "test")
            assert (action >= -1.0).all() and (action <= 1.0).all()

    def test_seeded_reproducibility(self):
        a = RandomPolicy(seed=7).predict(_dummy_image(), "task")
        b = RandomPolicy(seed=7).predict(_dummy_image(), "task")
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        a = RandomPolicy(seed=1).predict(_dummy_image(), "task")
        b = RandomPolicy(seed=2).predict(_dummy_image(), "task")
        assert not np.array_equal(a, b)

    def test_instruction_does_not_affect_output_shape(self):
        policy = RandomPolicy(seed=0)
        for instruction in ["", "pick up the red cube", "move left", "open gripper"]:
            action = policy.predict(_dummy_image(), instruction)
            assert action.shape == (7,)


class TestMockPolicy:
    """Any object with a compatible predict() signature satisfies Policy."""

    def test_lambda_style_mock_satisfies_protocol(self):
        class FixedPolicy:
            def predict(self, image, instruction):
                return np.zeros(7, dtype=np.float32)

        policy = FixedPolicy()
        assert isinstance(policy, Policy)
        action = policy.predict(_dummy_image(), "test")
        assert action.shape == (7,)
