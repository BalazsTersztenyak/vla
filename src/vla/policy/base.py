"""Policy interface. Any callable that satisfies Policy can be used with run_episode."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class Policy(Protocol):
    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        """Return a 7-D action vector [dx, dy, dz, drx, dry, drz, gripper]."""
        ...
