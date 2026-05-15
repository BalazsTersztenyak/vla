"""UR10 ManiSkill agent — 6-DOF arm, no gripper."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import sapien

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent

# URDF lives at <repo_root>/assets/ur10/ur10.urdf.
# __file__ is src/vla/robot/ur10_agent.py → go up 4 levels to reach the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
URDF_PATH = _REPO_ROOT / "assets" / "ur10" / "ur10.urdf"

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Home position: arm up, out of collision with table
HOME_QPOS = np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0], dtype=np.float32)


@register_agent()
class UR10(BaseAgent):
    uid = "ur10"
    urdf_path = str(URDF_PATH)
    urdf_config = dict(
        _materials=dict(
            default=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        )
    )

    ee_link_name = "tool0"

    keyframes = dict(
        rest=Keyframe(
            qpos=HOME_QPOS,
            pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        )
    )

    @property
    def tcp(self):
        return self.robot.find_link_by_name(self.ee_link_name)

    @property
    def tcp_pose(self):
        return self.robot.find_link_by_name(self.ee_link_name).pose

    @property
    def _controller_configs(self):
        pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=JOINT_NAMES,
            lower=-0.1,
            upper=0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=100.0,
            normalize_action=True,
            use_delta=True,
        )
        return dict(pd_joint_delta_pos=pd_joint_delta_pos)
