# VLA — Vision-Language-Action on UR10

Run [OpenVLA-7B](https://huggingface.co/openvla/openvla-7b) (or any custom policy) on a simulated Universal Robots UR10 arm inside [ManiSkill](https://maniskill.readthedocs.io). The task: pick up a cube and move it to a goal site.

---

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| [uv](https://docs.astral.sh/uv/) | Package manager |
| SAPIEN (via ManiSkill) | Physics simulation backend |
| NVIDIA GPU w/ 16 GB+ VRAM | Only needed for OpenVLA; CPU / random policy works without |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/BalazsTersztenyak/vla.git
cd vla

# Install core dependencies (random policy, simulation)
uv sync

# If you want to run OpenVLA (GPU required, downloads ~14 GB model on first run)
uv sync --extra vla
```

---

## Quick Start

### Random policy (no GPU needed)

```python
import gymnasium
import vla.env.pick_cube   # registers the PickCube-UR10-v1 env
from vla.policy.random_policy import RandomPolicy
from vla.runner import run_episode, save_video

env = gymnasium.make(
    "PickCube-UR10-v1",
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
    robot_uids="ur10",
    render_backend="cpu",
)
policy = RandomPolicy(seed=42)

result = run_episode(env, policy, instruction="pick up the cube", max_steps=80, record_video=True)
print(f"Success: {result.success}  Reward: {result.total_reward:.3f}  Steps: {result.steps}")

if result.frames:
    save_video(result.frames, "random_episode.mp4")

env.close()
```

### OpenVLA policy (GPU required)

```python
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
```

---

## Project Layout

```
src/vla/
├── env/pick_cube.py        # PickCube-UR10-v1 Gymnasium environment
├── robot/ur10_agent.py     # UR10 ManiSkill agent (6-DOF, URDF-based)
├── policy/
│   ├── base.py             # Policy Protocol (implement predict() to plug in your own)
│   ├── openvla.py          # OpenVLA-7B wrapper
│   └── random_policy.py    # Random baseline
├── action.py               # adapt_action() + clip_deltas() safety utilities
├── obs.py                  # obs_to_pil() — converts sim observations to PIL images
└── runner.py               # run_episode() + save_video()

assets/ur10/                # UR10 URDF and meshes used by ManiSkill
tests/
├── unit/                   # Fast tests, no GPU or SAPIEN required
└── integration/            # Full environment + episode tests
```

---

## Writing Your Own Policy

Implement the `Policy` protocol from `vla.policy.base`:

```python
from PIL.Image import Image
from numpy import ndarray

class MyPolicy:
    def predict(self, image: Image, instruction: str) -> ndarray:
        # Return a 7-D action: [dx, dy, dz, drx, dry, drz, gripper]
        # Values are clipped to safe ranges automatically by run_episode()
        ...
```

Pass an instance to `run_episode()` and it will work without any changes to the runner.

---

## Environment Details

**ID:** `PickCube-UR10-v1`

| Property | Value |
|---|---|
| Robot | Universal Robots UR10 (6 joints) |
| Action space | 6-D joint delta positions (no gripper) |
| Observation | RGB image from wrist/head camera |
| Success condition | Cube center within 2.5 cm of goal site |
| Reward | Shaped: distance-to-cube + distance-cube-to-goal |
| Home pose | `[0, -π/2, π/2, -π/2, -π/2, 0]` |

The action adapter (`vla.action.adapt_action`) strips the gripper dimension from the 7-D VLA output before sending it to the 6-DOF UR10.

---

## Running Tests

```bash
# Unit tests — fast, no GPU or SAPIEN
uv run pytest tests/unit

# Integration tests — requires ManiSkill / SAPIEN (no GPU needed)
uv run pytest tests/integration -m "not gpu"

# OpenVLA episode test — requires CUDA GPU (~16 GB VRAM) and downloaded model
uv run pytest tests/integration -m gpu

# Everything
uv run pytest
```

---

## Acknowledgements

- [ManiSkill](https://github.com/haosulab/ManiSkill) — simulation framework
- [OpenVLA](https://github.com/openvla/openvla) — vision-language-action model
- [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description) — official UR URDF models (included as a submodule)
