# pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import sapien
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# Quaternion helpers (SAPIEN uses [w,x,y,z], scipy uses [x,y,z,w])           #
# --------------------------------------------------------------------------- #

def quat_sapien_to_scipy(q):
    """[w,x,y,z] → [x,y,z,w]"""
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)

def quat_scipy_to_sapien(q):
    """[x,y,z,w] → [w,x,y,z]"""
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Environment                                                                  #
# --------------------------------------------------------------------------- #

def build_sapien_env():
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    scene.add_ground(altitude=0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [1, 1, 1])

    cam = scene.add_camera(
        name="robot_cam", width=224, height=224,
        fovy=np.deg2rad(60), near=0.01, far=10,
    )
    tilt = R.from_euler('xyz', [np.deg2rad(-30), 0, 0])
    cam.set_local_pose(sapien.Pose(
        p=np.array([0.5, 0, 0.5], dtype=np.float32),
        q=quat_scipy_to_sapien(tilt.as_quat()),
    ))
    return scene, cam


def get_observation(scene, cam) -> Image.Image:
    scene.update_render()
    cam.take_picture()
    rgba = cam.get_picture("Color")          # (H, W, 4) float32
    rgb  = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(rgb)


# --------------------------------------------------------------------------- #
# Robot                                                                        #
# --------------------------------------------------------------------------- #

EE_LINK_CANDIDATES = [
    "j2s7s300_end_effector",
    "j2s7s300_link_7",
    "j2s7s300_link_finger_tip_1",
]
FINGER_OPEN   = 0.0
FINGER_CLOSED = 1.4   # radians

def load_robot(scene: sapien.Scene):
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True

    robot = loader.load("assets/jaco2/jaco2.urdf")
    robot.set_root_pose(sapien.Pose(
        p=np.zeros(3, dtype=np.float32),
        q=np.array([1, 0, 0, 0], dtype=np.float32),
    ))

    init_qpos = np.deg2rad([180, 180, 180, 228, 0, 270, 180, 0, 0, 0, 0, 0, 0])
    robot.set_qpos(init_qpos)

    for joint in robot.get_active_joints()[:7]:
        joint.set_drive_property(stiffness=800, damping=150)
    for joint in robot.get_active_joints()[7:]:
        joint.set_drive_property(stiffness=200, damping=50)

    return robot


def get_ee_link(robot):
    link_map = {l.get_name(): l for l in robot.get_links()}
    for name in EE_LINK_CANDIDATES:
        if name in link_map:
            print(f"Using EE link: {name}")
            return link_map[name]
    raise ValueError(f"No EE link found. Available: {list(link_map)}")


# --------------------------------------------------------------------------- #
# Action application                                                           #
# --------------------------------------------------------------------------- #

def apply_delta(current_pose: sapien.Pose, ee_delta: np.ndarray) -> sapien.Pose:
    pos_delta = np.clip(ee_delta[:3], -0.05, 0.05)
    rot_delta = np.clip(ee_delta[3:6], -0.1,  0.1)

    r_new = R.from_quat(quat_sapien_to_scipy(current_pose.q)) * R.from_euler('xyz', rot_delta)
    return sapien.Pose(
        p=(current_pose.p + pos_delta).astype(np.float32),
        q=quat_scipy_to_sapien(r_new.as_quat()),
    )


def set_gripper(robot, gripper_action: float):
    val    = FINGER_CLOSED if gripper_action > 0.5 else FINGER_OPEN
    target = robot.get_drive_target().copy()
    target[7:10]  = val           # base finger joints
    target[10:13] = val * 0.7     # tip joints (mimic)
    robot.set_drive_target(target)


def apply_action(robot, ee_link, pinocchio_model, ee_link_idx, action: np.ndarray):
    new_pose = apply_delta(ee_link.get_pose(), action[:6])

    new_qpos = pinocchio_model.compute_inverse_kinematics(
        link_index=ee_link_idx,
        pose=new_pose,
        initial_qpos=robot.get_qpos(),
        active_qmask=np.array([1,1,1,1,1,1,1, 0,0,0, 0,0,0], dtype=bool),
        max_iterations=100,
    )

    target = robot.get_drive_target().copy()
    target[:7] = new_qpos[:7]
    robot.set_drive_target(target)
    set_gripper(robot, action[6])


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

def load_openvla(model_id="openvla/openvla-7b"):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda")
    return model, processor


def get_action(model, processor, image: Image.Image, instruction: str, device="cuda") -> np.ndarray:
    prompt  = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs  = processor(prompt, image).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    return action   # (7,): [dx, dy, dz, drx, dry, drz, gripper]


# --------------------------------------------------------------------------- #
# Main loop                                                                    #
# --------------------------------------------------------------------------- #

def run(instruction="pick up the red block"):
    scene, cam       = build_sapien_env()
    robot            = load_robot(scene)
    model, processor = load_openvla()

    ee_link          = get_ee_link(robot)
    pinocchio_model  = robot.create_pinocchio_model()          # build once
    ee_link_idx      = [l.get_name() for l in robot.get_links()].index(ee_link.get_name())

    for _ in range(200):
        image  = get_observation(scene, cam)
        action = get_action(model, processor, image, instruction)
        apply_action(robot, ee_link, pinocchio_model, ee_link_idx, action)
        for _ in range(10):
            scene.step()


if __name__ == "__main__":
    run()