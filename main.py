import gymnasium
import vla.env.pick_cube
from vla.policy.openvla import OpenVLAPolicy
from vla.runner import run_episode, save_video

env = gymnasium.make(
    "PickCube-UR10-v1",
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
    robot_uids="ur10",
    render_backend="cpu",
)
policy = OpenVLAPolicy(device="cuda:0")

result = run_episode(env, policy, instruction="pick up the cube", max_steps=80, record_video=True)
print(f"Success: {result.success}  Reward: {result.total_reward:.3f}")

if result.frames:
    save_video(result.frames, "openvla_episode.mp4")

env.close()
