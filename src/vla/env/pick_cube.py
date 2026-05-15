"""PickCube-UR10 environment. Import this module to register the env with gym."""
from __future__ import annotations

import torch

from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

# Side-effect import: registers the UR10 agent
from vla.robot.ur10_agent import UR10  # noqa: F401


@register_env("PickCube-UR10-v1", max_episode_steps=100)
class PickCubeUR10Env(PickCubeEnv):
    SUPPORTED_ROBOTS = ["ur10"]
    agent: UR10

    def evaluate(self):
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        dist = torch.linalg.norm(cube_pos - goal_pos, dim=-1)
        return dict(
            is_grasped=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            cube_at_goal=dist < 0.025,
            success=dist < 0.025,
        )

    def compute_dense_reward(self, obs, action, info):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        reach = 1.0 - torch.tanh(5.0 * torch.linalg.norm(cube_pos - tcp_pos, dim=-1))
        place = 1.0 - torch.tanh(5.0 * torch.linalg.norm(goal_pos - cube_pos, dim=-1))
        reward = reach + place
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0

    def _get_obs_extra(self, info):
        obs = dict(
            goal_pos=self.goal_site.pose.p,
            obj_pose=self.cube.pose.raw_pose,
        )
        try:
            use_state = self.obs_mode_struct.use_state
        except AttributeError:
            use_state = "state" in self.obs_mode
        if use_state:
            tcp_pose = self.agent.tcp.pose.raw_pose
            obs.update(
                tcp_pose=tcp_pose,
                tcp_to_obj_pos=self.cube.pose.p - tcp_pose[..., :3],
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
            )
        return obs
