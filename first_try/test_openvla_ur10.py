"""
OpenVLA-7B + UR10 in ManiSkill PickCube — Episode Test + Video Recorder
=======================================================================
Requirements:
    pip install mani-skill transformers timm tokenizers pillow imageio[ffmpeg] torch

Usage:
    python test_openvla_ur10.py --skip_vla
    python test_openvla_ur10.py --instruction "pick up the cube" --max_steps 80 --output video.mp4
"""

import argparse
import logging
import os
import time
from pathlib import Path

import gymnasium as gym
import imageio
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

import mani_skill.envs
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env

import sapien

os.environ["SAPIEN_RENDERER"] = "egl"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. UR10 Robot
# ══════════════════════════════════════════════════════════════════════════════

@register_agent()
class UR10(BaseAgent):
    uid = "ur10"

    # ── SET YOUR URDF PATH HERE ───────────────────────────────────────────────
    urdf_path = "mnt/d/dev/VLA/ur10_fixed.urdf"
    # ─────────────────────────────────────────────────────────────────────────

    urdf_config = dict(
        _materials=dict(
            default=dict(static_friction=1.0, dynamic_friction=1.0, restitution=0.0)
        ),
    )

    ARM_JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    ee_link_name = "tool0"

    @property
    def tcp(self):
        return self.robot.find_link_by_name(self.ee_link_name)

    @property
    def tcp_pose(self):
        return self.robot.find_link_by_name(self.ee_link_name).pose

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array([0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]),
            pose=sapien.Pose(p=[0.0, 0.0, 0.0]),
        )
    )

    @property
    def _controller_configs(self):
        pd_joint_delta_pos = PDJointPosControllerConfig(
            joint_names=self.ARM_JOINT_NAMES,
            lower=-0.1,
            upper=0.1,
            stiffness=1e4,
            damping=1e3,
            force_limit=100.0,
            normalize_action=True,
            use_delta=True,
        )
        return dict(pd_joint_delta_pos=pd_joint_delta_pos)


# ══════════════════════════════════════════════════════════════════════════════
# 2. PickCube-UR10 Environment
# ══════════════════════════════════════════════════════════════════════════════

@register_env("PickCube-UR10-v1", max_episode_steps=100)
class PickCubeUR10Env(PickCubeEnv):
    SUPPORTED_ROBOTS = ["ur10"]
    agent: UR10

    def evaluate(self):
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        dist = torch.linalg.norm(cube_pos - goal_pos, dim=-1)
        cube_at_goal = dist < 0.025
        return dict(
            is_grasped=torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            cube_at_goal=cube_at_goal,
            success=cube_at_goal,
        )

    def compute_dense_reward(self, obs, action, info):
        tcp_pos = self.agent.tcp.pose.p
        cube_pos = self.cube.pose.p
        goal_pos = self.goal_site.pose.p
        dist_tcp_to_cube = torch.linalg.norm(cube_pos - tcp_pos, dim=-1)
        dist_cube_to_goal = torch.linalg.norm(goal_pos - cube_pos, dim=-1)
        reward = 1.0 - torch.tanh(5.0 * dist_tcp_to_cube)
        reward += 1.0 - torch.tanh(5.0 * dist_cube_to_goal)
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


# ══════════════════════════════════════════════════════════════════════════════
# 3. OpenVLA Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class OpenVLAPolicy:
    MODEL_ID = "openvla/openvla-7b"

    def __init__(self, unnorm_key="bridge_orig", device="cuda:0"):
        self.unnorm_key = unnorm_key
        self.device = device
        log.info("Loading OpenVLA processor ...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID, trust_remote_code=True)
        log.info("Loading OpenVLA model (bfloat16) ...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.MODEL_ID,
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.model.eval()
        log.info("OpenVLA model ready.")

    def predict(self, image: Image.Image, instruction: str) -> np.ndarray:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs, unnorm_key=self.unnorm_key, do_sample=False
            )
        return np.array(action, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Helpers
# ══════════════════════════════════════════════════════════════════════════════

def obs_to_pil(obs, env):
    if isinstance(obs, dict):
        for cam_data in obs.get("sensor_data", {}).values():
            rgb = cam_data.get("rgb")
            if rgb is not None:
                arr = np.array(rgb)
                if arr.ndim == 4:
                    arr = arr[0]
                return Image.fromarray(arr.astype(np.uint8))
    frame = env.render()
    if frame is not None:
        arr = np.array(frame)
        if arr.ndim == 4:
            arr = arr[0]
        return Image.fromarray(arr.astype(np.uint8))
    raise RuntimeError("Could not extract image from environment.")


def adapt_action(vla_action: np.ndarray, action_space) -> np.ndarray:
    action_dim = action_space.shape[0]
    adapted = np.zeros(action_dim, dtype=np.float32)
    copy_dims = min(action_dim, len(vla_action))
    adapted[:copy_dims] = vla_action[:copy_dims]
    return np.clip(adapted, action_space.low, action_space.high)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main episode loop
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    instruction="pick up the cube",
    max_steps=80,
    output_video="openvla_ur10_episode.mp4",
    seed=42,
    unnorm_key="bridge_orig",
    fps=10,
    device="cuda:0",
    skip_vla=False,
):
    log.info(f"URDF path : {UR10.urdf_path}")
    log.info(f"URDF exists: {os.path.exists(UR10.urdf_path)}")

    import sapien.render
    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")
    sapien.render.set_ray_tracing_samples_per_pixel(1)

    log.info("Calling gym.make ...")
    env = gym.make(
        "PickCube-UR10-v1",
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
        robot_uids="ur10",
    )
    log.info("gym.make done.")

    log.info("Calling env.reset ...")
    obs, info = env.reset(seed=seed)
    log.info("env.reset done.")
    log.info(f"Action space : {env.action_space}")
    log.info(f"Obs shape    : {list(obs.keys()) if isinstance(obs, dict) else obs.shape}")

    if skip_vla:
        log.info("Skipping VLA — using random actions.")
        policy = None
    else:
        policy = OpenVLAPolicy(unnorm_key=unnorm_key, device=device)

    frames = []

    def record_frame():
        if env.render_mode is None:
            return
        log.info("Calling env.render() ...")
        frame = env.render()
        log.info(f"env.render() returned: {type(frame)}, shape: {getattr(frame, 'shape', 'N/A')}")
        if frame is not None:
            arr = np.array(frame)
            if arr.ndim == 4:
                arr = arr[0]
            frames.append(arr.astype(np.uint8))

    log.info(f"Starting episode — instruction: '{instruction}', max_steps: {max_steps}")

    total_reward = 0.0
    success = False
    step_times = []

    for step in range(max_steps):
        t0 = time.time()

        if policy is not None:
            log.info("Getting action from OpenVLA policy ...")
            pil_img = obs_to_pil(obs, env)
            vla_action = policy.predict(pil_img, instruction)
            action = adapt_action(vla_action, env.action_space)
        else:
            log.info("No policy — sampling random action.")
            action = env.action_space.sample()

        log.info(f"Step {step+1}: stepping env ...")
        obs, reward, terminated, truncated, info = env.step(action)
        log.info(f"Step {step+1}: env.step done. Recording frame ...")
        reward_val = float(reward.mean()) if hasattr(reward, 'mean') else float(reward)
        total_reward += reward_val
        log.info(f"Step {step+1}: reward={reward_val:.4f}. Recording frame ...")
        record_frame()
        log.info(f"Step {step+1}: frame recorded.")

        elapsed = time.time() - t0
        step_times.append(elapsed)
        success_raw = info.get("success", False)
        success = bool(success_raw.any()) if hasattr(success_raw, 'any') else bool(success_raw)

        log.info(f"step {step+1:03d} | reward={reward_val:.4f} | success={success} | {elapsed:.2f}s")

        if terminated or truncated:
            log.info(f"Episode ended early at step {step+1}.")
            break

    log.info("=" * 50)
    log.info(f"Total reward : {total_reward:.4f}")
    log.info(f"Success      : {success}")
    log.info(f"Steps taken  : {step+1}")
    if step_times:
        log.info(f"Avg step time: {np.mean(step_times):.2f}s")
    log.info("=" * 50)

    env.close()

    if frames:
        out_path = Path(output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving {len(frames)} frames → {out_path} (fps={fps})")
        imageio.mimwrite(str(out_path), frames, fps=fps, quality=8)
        log.info(f"Video saved: {out_path}")
    else:
        log.warning("No frames captured — video not saved.")

    return {"success": success, "total_reward": total_reward, "steps": step + 1}


# ══════════════════════════════════════════════════════════════════════════════
# 6. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Test OpenVLA-7B with UR10 in PickCube")
    p.add_argument("--instruction", type=str, default="pick up the cube")
    p.add_argument("--max_steps", type=int, default=80)
    p.add_argument("--output", type=str, default="openvla_ur10_episode.mp4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unnorm_key", type=str, default="bridge_orig")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--skip_vla", action="store_true")
    p.add_argument("--log_level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    run_episode(
        instruction=args.instruction,
        max_steps=args.max_steps,
        output_video=args.output,
        seed=args.seed,
        unnorm_key=args.unnorm_key,
        fps=args.fps,
        device=args.device,
        skip_vla=args.skip_vla,
    )