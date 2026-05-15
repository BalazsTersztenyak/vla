"""Shared fixtures for the test suite."""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from PIL import Image

# EGL is Linux-only; on Windows SAPIEN uses Vulkan by default
if sys.platform != "win32":
    os.environ.setdefault("SAPIEN_RENDERER", "egl")


@pytest.fixture(scope="session")
def pick_cube_env():
    """A PickCube-UR10-v1 env shared across the integration test session."""
    import gymnasium as gym

    # Side-effect: registers PickCube-UR10-v1 and the UR10 agent
    import vla.env.pick_cube  # noqa: F401

    env = gym.make(
        "PickCube-UR10-v1",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids="ur10",
        render_backend="cpu",
    )
    yield env
    env.close()


@pytest.fixture()
def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
