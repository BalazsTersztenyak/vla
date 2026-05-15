"""Episode runner — wires env, policy, and recording together."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from vla.action import adapt_action
from vla.obs import obs_to_pil
from vla.policy.base import Policy

log = logging.getLogger(__name__)


@dataclass
class EpisodeResult:
    success: bool
    total_reward: float
    steps: int
    frames: list[np.ndarray] = field(default_factory=list)
    step_times: list[float] = field(default_factory=list)

    @property
    def avg_step_time(self) -> float:
        return float(np.mean(self.step_times)) if self.step_times else 0.0


def run_episode(
    env,
    policy: Policy,
    instruction: str = "pick up the cube",
    max_steps: int = 80,
    seed: int = 42,
    record_video: bool = False,
    verbose: bool = True,
) -> EpisodeResult:
    """Run one episode and return structured results.

    Args:
        env: A Gymnasium environment (PickCube-UR10-v1 or compatible).
        policy: Any object with a predict(image, instruction) -> ndarray method.
        instruction: Natural-language task description passed to the policy.
        max_steps: Hard step limit.
        seed: Environment reset seed.
        record_video: Whether to capture frames (requires render_mode != None).
        verbose: Print per-step progress to stdout.
    """
    obs, _ = env.reset(seed=seed)
    result = EpisodeResult(success=False, total_reward=0.0, steps=0)

    if verbose:
        print(f"Episode start — instruction: '{instruction}' | max_steps={max_steps}")

    for step in range(max_steps):
        t0 = time.perf_counter()

        if verbose:
            print(f"  step {step + 1:03d}/{max_steps} | obs→PIL …", end="", flush=True)

        pil_img = obs_to_pil(obs, env)

        if verbose:
            print(f"\r  step {step + 1:03d}/{max_steps} | inferring …", end="", flush=True)

        vla_action = policy.predict(pil_img, instruction)
        action = adapt_action(vla_action, env.action_space)

        obs, reward, terminated, truncated, info = env.step(action)

        reward_val = float(reward.mean()) if hasattr(reward, "mean") else float(reward)
        result.total_reward += reward_val
        result.step_times.append(time.perf_counter() - t0)

        success_raw = info.get("success", False)
        result.success = bool(success_raw.any()) if hasattr(success_raw, "any") else bool(success_raw)

        if record_video and env.render_mode is not None:
            frame = env.render()
            if frame is not None:
                arr = np.asarray(frame)
                result.frames.append(arr[0] if arr.ndim == 4 else arr)

        log.debug("step %03d | reward=%.4f | success=%s", step + 1, reward_val, result.success)
        if verbose:
            status = "SUCCESS" if result.success else ("TERMINATED" if terminated else ("TRUNCATED" if truncated else ""))
            suffix = f" [{status}]" if status else ""
            print(f"\r  step {step + 1:03d}/{max_steps} | reward={reward_val:+.4f} | total={result.total_reward:+.4f} | t={result.step_times[-1]:.2f}s{suffix}")

        if terminated or truncated:
            break

    result.steps = step + 1
    if verbose:
        print(f"Episode done — steps={result.steps} | total_reward={result.total_reward:.4f} | success={result.success}")
    return result


def save_video(frames: list[np.ndarray], path: str | Path, fps: int = 10) -> None:
    """Write a list of uint8 RGB frames to an mp4 file."""
    import imageio

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out), [f.astype(np.uint8) for f in frames], fps=fps, quality=8)
    log.info("Saved %d frames → %s", len(frames), out)
