"""
Gym-style SAPIEN 3.x environment — Jaco2 arm + tabletop scene.
Extended for OpenVLA: image observations, language instructions,
EE delta-action control, and a ready-to-use step_with_openvla() helper.
"""

from __future__ import annotations

import os
import pathlib
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np
import sapien
import sapien.physx as physx
from sapien import Pose
from sapien.utils import Viewer
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import qmult, qinverse

# ---------------------------------------------------------------------------
# URDF helpers  (unchanged)
# ---------------------------------------------------------------------------
_CACHE   = pathlib.Path.home() / ".cache" / "jaco2_urdf"
_ZIP_URL = "https://github.com/Kinovarobotics/kinova-ros/archive/refs/heads/master.zip"


def _ensure_jaco_urdf() -> pathlib.Path:
    override = os.environ.get("JACO2_URDF")
    if override:
        p = pathlib.Path(override)
        if not p.exists():
            raise FileNotFoundError(f"JACO2_URDF={override} not found.")
        return p
    hits = list(_CACHE.rglob("j2n6s300_standalone.urdf"))
    if hits:
        return hits[0]
    print("[JacoTableEnv] Downloading Kinova URDF (~200 MB, one-time) …")
    _CACHE.mkdir(parents=True, exist_ok=True)
    zp = _CACHE / "kinova-ros.zip"
    urllib.request.urlretrieve(_ZIP_URL, zp)
    with zipfile.ZipFile(zp) as zf:
        members = [m for m in zf.namelist()
                   if "kinova_description" in m or "j2n6s300" in m]
        zf.extractall(_CACHE, members)
    hits = list(_CACHE.rglob("j2n6s300_standalone.urdf"))
    if not hits:
        raise FileNotFoundError(
            "URDF not found after extraction. "
            "Set JACO2_URDF to your local j2n6s300_standalone.urdf."
        )
    print(f"[JacoTableEnv] URDF cached at {hits[0]}")
    return hits[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quat_from_euler(xyz_rad: np.ndarray) -> np.ndarray:
    w, x, y, z = euler2quat(*xyz_rad, axes="sxyz")
    return np.array([w, x, y, z], dtype=np.float64)


def _mat(rgb: List[float]):
    import sapien.render as sr
    m = sr.RenderMaterial()
    m.set_base_color([*rgb, 1.0])
    m.metallic  = 0.0
    m.roughness = 0.6
    return m


# ---------------------------------------------------------------------------
# OpenVLA action dimensions
# ---------------------------------------------------------------------------
# OpenVLA outputs 7 values: [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper]
# gripper: 1.0 = open, 0.0 = closed  (we threshold at 0.5)
OPENVLA_ACTION_DIM = 7
OPENVLA_IMAGE_SIZE = 224   # model expects 224×224 RGB


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class JacoTableEnv(gym.Env):
    """
    Observation space
    -----------------
    Dict:
      "image"       : uint8  (224, 224, 3)   — wrist + overhead cameras
      "wrist_image" : uint8  (224, 224, 3)   — wrist-mounted camera
      "state"       : float32 (20,)
                        [qpos(6), qvel(6), ee_pos(3), ee_quat_wxyz(4), gripper(1)]
      "instruction" : str    — current task language goal

    Action space  (OpenVLA mode)
    ----------------------------
      float32 (7,) : [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper ∈ {0,1}]
      Δ values are clipped to ±_EE_DELTA_LIMIT before applying.

    Action space  (joint-velocity mode, kept for RL baselines)
    ----------------------------
      float32 (6,) : joint velocity targets clipped to ±π rad/s
    """

    metadata = {"render_modes": ["human", "none"]}

    TABLE_H  = 0.76
    TABLE_W  = 1.20
    TABLE_D  = 0.70
    TABLE_T  = 0.04
    LEG_R    = 0.03
    N_JOINTS = 6

    _EE_DELTA_LIMIT    = 0.05   # max EE translation per step [m]
    _EE_ROT_LIMIT      = 0.2    # max EE rotation per step [rad]
    _IK_ITER           = 100    # damped-LS IK iterations
    _IK_DAMPING        = 0.05

    # Task descriptions shown to OpenVLA
    TASK_INSTRUCTIONS = {
        "reach_red_cube"   : "Pick up the red cube.",
        "reach_blue_sphere": "Pick up the blue sphere.",
        "reach_green_cyl"  : "Pick up the green cylinder.",
    }

    def __init__(
        self,
        render_mode : str  = "none",
        control_freq: int  = 20,
        sim_freq    : int  = 500,
        task        : str  = "reach_red_cube",
        control_mode: str  = "ee_delta",   # "ee_delta" | "joint_vel"
    ) -> None:
        super().__init__()
        assert render_mode  in self.metadata["render_modes"]
        assert task         in self.TASK_INSTRUCTIONS
        assert control_mode in ("ee_delta", "joint_vel")

        self.render_mode  = render_mode
        self.control_freq = control_freq
        self.sim_freq     = sim_freq
        self._ctrl_steps  = sim_freq // control_freq
        self.task         = task
        self.control_mode = control_mode

        # ── Observation space ──────────────────────────────────────────
        img_space = gym.spaces.Box(0, 255,
                                   shape=(OPENVLA_IMAGE_SIZE,
                                          OPENVLA_IMAGE_SIZE, 3),
                                   dtype=np.uint8)
        # 6 qpos + 6 qvel + 3 ee_pos + 4 ee_quat + 1 gripper = 20
        state_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(20,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "image"      : img_space,
            "wrist_image": img_space,
            "state"      : state_space,
            "instruction": gym.spaces.Text(min_length=1, max_length=256),
        })

        # ── Action space ───────────────────────────────────────────────
        if control_mode == "ee_delta":
            self.action_space = gym.spaces.Box(
                low  = np.array([-self._EE_DELTA_LIMIT] * 3
                                + [-self._EE_ROT_LIMIT] * 3
                                + [0.0], dtype=np.float32),
                high = np.array([self._EE_DELTA_LIMIT] * 3
                                + [self._EE_ROT_LIMIT] * 3
                                + [1.0], dtype=np.float32),
            )
        else:
            n   = self.N_JOINTS
            self.action_space = gym.spaces.Box(
                -np.full(n, np.pi, dtype=np.float32),
                 np.full(n, np.pi, dtype=np.float32),
            )

        self._viewer: Optional[Viewer] = None
        self._gripper_open = True
        self._setup()

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        physx.set_scene_config(
            gravity=np.array([0.0, 0.0, -9.81], dtype=np.float32),
            bounce_threshold=2.0,
            enable_ccd=False,
        )
        self._scene = sapien.Scene()
        self._scene.set_timestep(1.0 / self.sim_freq)

        self._scene.set_ambient_light([0.4, 0.4, 0.4])
        self._scene.add_directional_light([-1, -1, -2], [1.0, 1.0, 1.0], shadow=True)
        self._scene.add_point_light([0.5, 0.5, 2.0], [0.8, 0.8, 0.8])

        self._add_ground()
        self._add_table()
        self._objects    = self._add_objects()
        self._robot      = self._add_robot()
        self._arm_joints = self._get_arm_joints()
        self._joint_to_idx = {
            j: i for i, j in enumerate(self._robot.get_active_joints())
        }

        # ── Cameras ───────────────────────────────────────────────────
        # Overhead camera (main OpenVLA input)
        self._cam = self._scene.add_camera(
            name="overhead", width=640, height=480,
            fovy=np.deg2rad(60), near=0.01, far=10.0,
        )
        self._cam.set_entity_pose(Pose(
            p=[0.0, -1.6, self.TABLE_H + 1.0],
            q=_quat_from_euler(np.deg2rad([-55, 0, 0])),
        ))

        # Wrist camera — attached to the last link
        self._wrist_cam = self._scene.add_camera(
            name="wrist", width=640, height=480,
            fovy=np.deg2rad(70), near=0.01, far=2.0,
        )
        # Pose updated each step in _update_wrist_cam()
        self._update_wrist_cam()

    # ── Ground / Table / Objects  (unchanged from original) ───────────
    def _add_ground(self) -> None:
        phys = self._scene.create_physical_material(1.0, 0.8, 0.1)
        b    = self._scene.create_actor_builder()
        b.add_plane_collision(material=phys)
        b.add_plane_visual(material=_mat([0.25, 0.25, 0.25]))
        g = b.build_static(name="ground")
        g.set_pose(Pose(q=_quat_from_euler([0, np.pi / 2, 0])))

    def _add_table(self) -> None:
        w, d, t = self.TABLE_W, self.TABLE_D, self.TABLE_T
        h, lr   = self.TABLE_H, self.LEG_R
        lh      = (h - t) / 2
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[w/2, d/2, t/2])
        b.add_box_visual(half_size=[w/2, d/2, t/2], material=_mat([0.55, 0.35, 0.15]))
        for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            lx = sx * (w/2 - lr - 0.02)
            ly = sy * (d/2 - lr - 0.02)
            lz = -(t/2 + lh)
            p  = Pose(p=[lx, ly, lz])
            b.add_box_collision(half_size=[lr, lr, lh], pose=p)
            b.add_box_visual(half_size=[lr, lr, lh],
                             material=_mat([0.4, 0.28, 0.18]), pose=p)
        tbl = b.build_static(name="table")
        tbl.set_pose(Pose(p=[0.0, 0.0, h - t / 2]))

    def _add_objects(self) -> Dict[str, sapien.Entity]:
        z    = self.TABLE_H
        phys = self._scene.create_physical_material(0.5, 0.4, 0.01)
        out: Dict[str, sapien.Entity] = {}

        s = 0.04
        b = self._scene.create_actor_builder()
        b.add_box_collision(half_size=[s]*3, material=phys)
        b.add_box_visual(half_size=[s]*3, material=_mat([0.85, 0.1, 0.1]))
        obj = b.build(name="red_cube")
        obj.set_pose(Pose(p=[0.25, 0.05, z + s]))
        out["red_cube"] = obj

        r = 0.035
        b = self._scene.create_actor_builder()
        b.add_sphere_collision(radius=r, material=phys)
        b.add_sphere_visual(radius=r, material=_mat([0.1, 0.2, 0.9]))
        obj = b.build(name="blue_sphere")
        obj.set_pose(Pose(p=[0.10, -0.10, z + r]))
        out["blue_sphere"] = obj

        cr, cl = 0.025, 0.05
        up = Pose(q=_quat_from_euler([0, np.pi / 2, 0]))
        b  = self._scene.create_actor_builder()
        b.add_capsule_collision(radius=cr, half_length=cl, material=phys, pose=up)
        b.add_capsule_visual(radius=cr, half_length=cl,
                             material=_mat([0.1, 0.75, 0.2]), pose=up)
        obj = b.build(name="green_cylinder")
        obj.set_pose(Pose(p=[0.30, -0.15, z + cl + cr]))
        out["green_cylinder"] = obj
        return out

    def _add_robot(self):
        urdf   = _ensure_jaco_urdf()
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        loader.scale = 1.0
        robot  = loader.load(str(urdf))
        robot.set_root_pose(Pose(p=[-0.55, 0.0, self.TABLE_H + 0.005]))

        active       = robot.get_active_joints()
        joint_to_idx = {j: i for i, j in enumerate(active)}

        neutral = np.array([0.0, np.pi, np.pi, 0.0, np.pi, 0.0])
        qpos    = np.zeros(robot.dof)
        for joint, angle in zip(self._get_arm_joints(robot), neutral):
            joint.set_drive_property(stiffness=400, damping=40)
            joint.set_drive_target(float(angle))
            qpos[joint_to_idx[joint]] = angle
        robot.set_qpos(qpos)
        return robot

    def _get_arm_joints(self, robot=None) -> List:
        r = robot or self._robot
        return [j for j in r.get_active_joints()
                if "finger" not in j.name.lower()][:self.N_JOINTS]

    def _get_finger_joints(self) -> List:
        return [j for j in self._robot.get_active_joints()
                if "finger" in j.name.lower()]

    # ------------------------------------------------------------------
    # EE helpers
    # ------------------------------------------------------------------
    def _ee_pose(self) -> Pose:
        """Return the pose of the end-effector (last link)."""
        return self._robot.get_links()[-1].get_entity_pose()

    def _update_wrist_cam(self) -> None:
        """Keep wrist camera rigidly attached to the EE link."""
        ee = self._ee_pose()
        # Small offset so the camera looks forward from the wrist
        offset = Pose(p=[0.0, 0.0, 0.12],
                      q=_quat_from_euler([np.pi, 0, np.pi / 2]))
        self._wrist_cam.set_entity_pose(ee * offset)

    def _apply_ee_delta(self, delta: np.ndarray) -> None:
        d_pos = np.clip(delta[:3], -self._EE_DELTA_LIMIT, self._EE_DELTA_LIMIT)
        d_rot = np.clip(delta[3:6], -self._EE_ROT_LIMIT,  self._EE_ROT_LIMIT)
        gripper_open = bool(delta[6] >= 0.5)

        ee  = self._ee_pose()
        t_p = np.array(ee.p) + d_pos
        dq  = _quat_from_euler(d_rot)
        t_q = qmult(dq, np.array(ee.q))
        t_q /= np.linalg.norm(t_q)

        idx  = [self._joint_to_idx[j] for j in self._arm_joints]
        qpos = self._robot.get_qpos().copy()
        lam2 = self._IK_DAMPING ** 2

        for _ in range(self._IK_ITER):
            ee_now  = self._ee_pose()
            pos_err = t_p - np.array(ee_now.p)

            q_now = np.array(ee_now.q)
            q_rel = qmult(t_q, qinverse(q_now))
            if q_rel[0] < 0:
                q_rel = -q_rel
            angle   = 2.0 * np.arccos(np.clip(q_rel[0], -1, 1))
            axis    = q_rel[1:] / (np.linalg.norm(q_rel[1:]) + 1e-9)
            rot_err = axis * angle

            err6 = np.concatenate([pos_err, rot_err])
            if np.linalg.norm(err6) < 1e-4:
                break

            J    = self._numerical_jacobian(qpos, idx)          # no scene.step inside
            JJT  = J @ J.T + lam2 * np.eye(6)
            dq_j = J.T @ np.linalg.solve(JJT, err6)
            qpos[idx] += dq_j * 0.5
            self._robot.set_qpos(qpos)                          # FK updates immediately

        # ONE physics step after IK converges
        for j in self._arm_joints:
            j.set_drive_target(float(qpos[self._joint_to_idx[j]]))
        for _ in range(self._ctrl_steps):
            self._scene.step()

        self._set_gripper(gripper_open)
        self._gripper_open = gripper_open


    def _numerical_jacobian(
        self, qpos: np.ndarray, idx: List[int], eps: float = 1e-4
    ) -> np.ndarray:
        """
        Purely kinematic Jacobian — set_qpos triggers FK immediately in SAPIEN 3.
        No scene.step() calls here, so the simulation clock does not advance.
        """
        n  = len(idx)
        J  = np.zeros((6, n))
        ee0 = self._ee_pose()
        p0  = np.array(ee0.p)
        q0  = np.array(ee0.q)

        for col, i in enumerate(idx):
            qp = qpos.copy()
            qp[i] += eps
            self._robot.set_qpos(qp)          # FK only, no physics tick
            ee1 = self._ee_pose()

            dp    = (np.array(ee1.p) - p0) / eps
            q_rel = qmult(np.array(ee1.q), qinverse(q0))
            if q_rel[0] < 0:
                q_rel = -q_rel
            angle = 2.0 * np.arccos(np.clip(q_rel[0], -1, 1))
            axis  = q_rel[1:] / (np.linalg.norm(q_rel[1:]) + 1e-9)
            dr    = axis * angle / eps
            J[:, col] = np.concatenate([dp, dr])

        self._robot.set_qpos(qpos)            # restore, still no physics tick
        return J

    def _set_gripper(self, open_: bool) -> None:
        target = 0.0 if open_ else 1.0   # Jaco2 fingers: 0=open, ~1=closed
        for j in self._get_finger_joints():
            j.set_drive_target(target)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        rng = self.np_random
        z   = self.TABLE_H

        self._objects["red_cube"].set_pose(Pose(p=[
            0.25 + rng.uniform(-0.05, 0.05),
            0.05 + rng.uniform(-0.05, 0.05),
            z + 0.04,
        ]))
        self._objects["blue_sphere"].set_pose(Pose(p=[
            0.10 + rng.uniform(-0.05, 0.05),
           -0.10 + rng.uniform(-0.05, 0.05),
            z + 0.035,
        ]))
        self._objects["green_cylinder"].set_pose(Pose(p=[
            0.30 + rng.uniform(-0.05, 0.05),
           -0.15 + rng.uniform(-0.05, 0.05),
            z + 0.075,
        ]))

        active       = self._robot.get_active_joints()
        joint_to_idx = {j: i for i, j in enumerate(active)}

        neutral = np.array([0.0, np.pi, np.pi, 0.0, np.pi, 0.0])
        qpos    = np.zeros(self._robot.dof)
        for joint, angle in zip(self._arm_joints, neutral):
            joint.set_drive_target(float(angle))
            qpos[joint_to_idx[joint]] = angle
        self._robot.set_qpos(qpos)
        self._robot.set_qvel(np.zeros(self._robot.dof))
        self._set_gripper(True)
        self._gripper_open = True

        self._scene.step()
        self._update_wrist_cam()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)

        if self.control_mode == "ee_delta":
            self._apply_ee_delta(action)
        else:
            for joint, vel in zip(self._arm_joints, action):
                joint.set_drive_velocity_target(float(vel))
            for _ in range(self._ctrl_steps):
                self._scene.step()

        self._update_wrist_cam()
        obs     = self._obs()
        reward  = self._reward()
        terminated = self._check_success()
        return obs, reward, terminated, False, {"success": terminated}

    # ------------------------------------------------------------------
    # OpenVLA convenience method
    # ------------------------------------------------------------------
    def step_with_openvla(self, model, processor) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Single-step helper: grab obs → call OpenVLA → apply action.

        Usage:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            processor = AutoProcessor.from_pretrained("openvla/openvla-7b", ...)
            model     = AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b", ...)
            model.to("cuda")

            obs, _ = env.reset()
            for _ in range(200):
                obs, r, done, _, info = env.step_with_openvla(model, processor)
                if done:
                    break
        """
        from PIL import Image as PILImage
        import torch

        obs = self._obs()

        # OpenVLA expects a PIL image + text instruction
        pil_img     = PILImage.fromarray(obs["image"])
        instruction = obs["instruction"]

        inputs = processor(
            images=pil_img,
            text=f"In: What action should the robot take to {instruction}\nOut:",
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        # OpenVLA returns raw action tokens; processor decodes to float array
        action_tokens = model.predict_action(**inputs)
        action = processor.decode_actions(action_tokens)   # shape (7,)
        action = np.array(action, dtype=np.float32)

        return self.step(action)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _grab_camera(self, cam) -> np.ndarray:
        """Render camera → uint8 (224,224,3) for OpenVLA."""
        self._scene.update_render()
        cam.take_picture()
        rgba = cam.get_picture("Color")                          # float [0,1]
        rgb  = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        return cv2.resize(rgb, (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
                          interpolation=cv2.INTER_LINEAR)

    def _obs(self) -> Dict:
        idx   = [self._joint_to_idx[j] for j in self._arm_joints]
        qpos  = self._robot.get_qpos()[idx].astype(np.float32)
        qvel  = self._robot.get_qvel()[idx].astype(np.float32)
        ee    = self._ee_pose()
        ee_p  = np.array(ee.p, dtype=np.float32)
        ee_q  = np.array(ee.q, dtype=np.float32)              # [w,x,y,z]
        grip  = np.array([1.0 if self._gripper_open else 0.0],
                          dtype=np.float32)
        state = np.concatenate([qpos, qvel, ee_p, ee_q, grip]) # (20,)

        return {
            "image"      : self._grab_camera(self._cam),
            "wrist_image": self._grab_camera(self._wrist_cam),
            "state"      : state,
            "instruction": self.TASK_INSTRUCTIONS[self.task],
        }

    # ------------------------------------------------------------------
    # Reward / success
    # ------------------------------------------------------------------
    _TARGET_MAP = {
        "reach_red_cube"   : "red_cube",
        "reach_blue_sphere": "blue_sphere",
        "reach_green_cyl"  : "green_cylinder",
    }
    _SUCCESS_DIST = 0.05   # [m] EE must be within this to count as success

    def _reward(self) -> float:
        ee  = np.array(self._ee_pose().p)
        tgt = np.array(self._objects[self._TARGET_MAP[self.task]].get_pose().p)
        dist = float(np.linalg.norm(ee - tgt))
        # Dense: negative distance + bonus for being close
        return -dist + (1.0 if dist < self._SUCCESS_DIST else 0.0)

    def _check_success(self) -> bool:
        ee  = np.array(self._ee_pose().p)
        tgt = np.array(self._objects[self._TARGET_MAP[self.task]].get_pose().p)
        return bool(np.linalg.norm(ee - tgt) < self._SUCCESS_DIST)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = Viewer()
                self._viewer.set_scene(self._scene)
                self._viewer.set_camera_xyz(1.0, -1.2, 1.4)
                self._viewer.set_camera_rpy(0.0, -0.6, 0.8)
            self._viewer.render()
            return None
        return self._grab_camera(self._cam)

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def main() -> None:
    print("Building JacoTableEnv (ee_delta, reach_red_cube) …")
    env = JacoTableEnv(render_mode="none", control_mode="ee_delta",
                       task="reach_red_cube")

    obs, _ = env.reset(seed=0)
    print(f"Image shape      : {obs['image'].shape}")
    print(f"Wrist image shape: {obs['wrist_image'].shape}")
    print(f"State shape      : {obs['state'].shape}")
    print(f"Instruction      : {obs['instruction']}")

    total = 0.0
    for step in range(50):
        action = env.action_space.sample() * 0.3
        obs, r, terminated, _, info = env.step(action)
        total += r
        if (step + 1) % 10 == 0:
            print(f"  step {step+1:3d}  r={r:+.4f}  Σr={total:+.4f}  "
                  f"success={info['success']}")
        if terminated:
            print("  ✓ Task succeeded!")
            break

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()