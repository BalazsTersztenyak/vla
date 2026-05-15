"""Pure action-space utilities — no SAPIEN/ManiSkill imports."""
from __future__ import annotations

import numpy as np


def adapt_action(vla_action: np.ndarray, action_space) -> np.ndarray:
    """Truncate/pad a VLA action vector to match the environment action space.

    VLA outputs shape (7,): [dx, dy, dz, drx, dry, drz, gripper].
    UR10 (no gripper) has a 6-DOF action space, so the gripper dimension is
    silently dropped. If the env ever has more dims than VLA, they stay at 0.
    """
    n = action_space.shape[0]
    adapted = np.zeros(n, dtype=np.float32)
    copy_n = min(n, len(vla_action))
    adapted[:copy_n] = vla_action[:copy_n]
    return np.clip(adapted, action_space.low, action_space.high)


def clip_deltas(
    delta: np.ndarray,
    pos_limit: float = 0.05,
    rot_limit: float = 0.1,
) -> np.ndarray:
    """Clip a 6-D end-effector delta [dx,dy,dz,drx,dry,drz] to safe ranges."""
    out = delta.copy()
    out[:3] = np.clip(out[:3], -pos_limit, pos_limit)
    out[3:6] = np.clip(out[3:6], -rot_limit, rot_limit)
    return out
