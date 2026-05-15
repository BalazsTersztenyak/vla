"""Observation extraction helpers — converts ManiSkill obs dicts to PIL images."""
from __future__ import annotations

import numpy as np
from PIL import Image


def _to_numpy(x) -> np.ndarray:
    """Convert tensor or array-like to a CPU numpy array without blocking."""
    if hasattr(x, "cpu"):      # torch.Tensor (any device)
        return x.cpu().numpy()
    return np.asarray(x)


def obs_to_pil(obs: dict | np.ndarray, env) -> Image.Image:
    """Extract a PIL RGB image from a ManiSkill observation.

    Tries sensor_data cameras first, falls back to env.render().
    """
    if isinstance(obs, dict):
        for cam_data in obs.get("sensor_data", {}).values():
            rgb = cam_data.get("rgb") if isinstance(cam_data, dict) else None
            if rgb is not None:
                arr = _to_numpy(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                return Image.fromarray(arr.astype(np.uint8))

    frame = env.render()
    if frame is None:
        raise RuntimeError("Could not extract image: env.render() returned None")
    arr = _to_numpy(frame)
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(arr.astype(np.uint8))
