"""Random policy — no GPU required, used for integration testing."""
from __future__ import annotations

import numpy as np
from PIL import Image


class RandomPolicy:
    """Always returns a random action in [-1, 1]^7."""

    ACTION_DIM = 7

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        return self._rng.uniform(-1.0, 1.0, size=(self.ACTION_DIM,)).astype(np.float32)
