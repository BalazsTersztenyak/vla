"""
visualizer.py — save camera feeds + metrics to disk for JacoTableEnv.

Output (all in ./viz_output/):
  episode_<N>.mp4   — side-by-side overhead | wrist frames with HUD overlay
  episode_<N>.png   — reward / EE-distance / gripper metrics plot
  episode_<N>_raw/  — individual PNG frames (optional, set SAVE_FRAMES=True)

Usage:
    python visualizer.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── Import your environment ────────────────────────────────────────────────
from env2 import JacoTableEnv   # adjust if your file is named differently

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR     = Path("viz_output")
SAVE_FRAMES = False             # also dump individual PNGs
FPS         = 20
VIDEO_W     = 1280              # final video width  (two 640-wide panels)
VIDEO_H     = 540               # final video height
HUD_H       = 60                # pixels reserved for HUD bar below cameras
FONT        = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------------------
# Metrics recorder
# ---------------------------------------------------------------------------
class EpisodeMetrics:
    def __init__(self) -> None:
        self.rewards:    List[float] = []
        self.cumulative: List[float] = []
        self.ee_dists:   List[float] = []
        self.gripper:    List[float] = []   # 1=open, 0=closed
        self._cum = 0.0

    def record(self, reward: float, ee_dist: float, gripper_open: bool) -> None:
        self.rewards.append(reward)
        self._cum += reward
        self.cumulative.append(self._cum)
        self.ee_dists.append(ee_dist)
        self.gripper.append(1.0 if gripper_open else 0.0)

    @property
    def n_steps(self) -> int:
        return len(self.rewards)


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------
def _draw_hud(
    frame: np.ndarray,
    step: int,
    reward: float,
    cum_reward: float,
    ee_dist: float,
    gripper_open: bool,
    success: bool,
    task: str,
) -> np.ndarray:
    """
    Paste a black HUD bar below the camera strip.
    frame: (VIDEO_H - HUD_H, VIDEO_W, 3) camera strip.
    Returns (VIDEO_H, VIDEO_W, 3).
    """
    bar = np.zeros((HUD_H, VIDEO_W, 3), dtype=np.uint8)

    # Background gradient
    for x in range(VIDEO_W):
        t = x / VIDEO_W
        bar[:, x] = (int(20 * (1 - t)), int(20 * (1 - t)), int(35 * t + 20))

    col1 = f"Step: {step:4d}   Task: {task}"
    col2 = f"r={reward:+.3f}   Sigma_r={cum_reward:+.1f}"
    col3 = f"EE dist: {ee_dist:.3f} m   Gripper: {'OPEN' if gripper_open else 'CLOSED'}"
    suc  = "  ** SUCCESS **" if success else ""

    cv2.putText(bar, col1, (12, 22),  FONT, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, col2, (12, 44),  FONT, 0.55, (120, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(bar, col3, (500, 22), FONT, 0.55, (255, 220, 120), 1, cv2.LINE_AA)
    if suc:
        cv2.putText(bar, suc, (500, 44), FONT, 0.65, (0, 255, 100), 2, cv2.LINE_AA)

    return np.vstack([frame, bar])


def _label_panel(img: np.ndarray, label: str) -> np.ndarray:
    """Stamp a label in the top-left corner of a camera panel."""
    out = img.copy()
    cv2.rectangle(out, (0, 0), (len(label) * 11 + 8, 22), (0, 0, 0), -1)
    cv2.putText(out, label, (4, 16), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ---------------------------------------------------------------------------
# Per-step frame builder
# ---------------------------------------------------------------------------
def build_frame(
    overhead: np.ndarray,   # (224,224,3) uint8
    wrist:    np.ndarray,   # (224,224,3) uint8
    cam_w:    int,
    cam_h:    int,
    **hud_kwargs,
) -> np.ndarray:
    """Resize both cameras, place side-by-side, add HUD bar."""
    left  = cv2.resize(overhead, (cam_w, cam_h))
    right = cv2.resize(wrist,   (cam_w, cam_h))

    left  = _label_panel(left,  "Overhead")
    right = _label_panel(right, "Wrist")

    strip = np.hstack([left, right])                # (cam_h, VIDEO_W, 3)
    full  = _draw_hud(strip, **hud_kwargs)          # + HUD bar
    return full                                     # (VIDEO_H, VIDEO_W, 3)


# ---------------------------------------------------------------------------
# Metrics plot
# ---------------------------------------------------------------------------
def save_metrics_plot(metrics: EpisodeMetrics, path: Path, task: str) -> None:
    steps = np.arange(metrics.n_steps)

    fig = plt.figure(figsize=(14, 8), facecolor="#0e1117")
    gs  = gridspec.GridSpec(3, 1, hspace=0.45)

    style = dict(linewidth=1.6)

    # ── Reward ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(steps, metrics.rewards,    color="#4af0a0", label="Step reward",      **style)
    ax1.plot(steps, metrics.cumulative, color="#f0a04a", label="Cumulative reward", **style, linestyle="--")
    ax1.set_ylabel("Reward", color="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax1.set_title(f"Episode metrics — {task}", color="white", fontsize=11)

    # ── EE distance ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(steps, metrics.ee_dists, color="#4ab4f0", **style)
    ax2.axhline(JacoTableEnv._SUCCESS_DIST, color="#ff5555",
                linestyle=":", linewidth=1.2, label="Success threshold")
    ax2.set_ylabel("EE distance (m)", color="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    # ── Gripper state ─────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.fill_between(steps, metrics.gripper, step="post",
                     color="#c084fc", alpha=0.7, label="1=open  0=closed")
    ax3.set_ylim(-0.05, 1.15)
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Closed", "Open"], color="white")
    ax3.set_ylabel("Gripper", color="white")
    ax3.set_xlabel("Step", color="white")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] metrics plot → {path}")


# ---------------------------------------------------------------------------
# Main recording loop
# ---------------------------------------------------------------------------
def record_episode(
    env: JacoTableEnv,
    episode: int = 0,
    n_steps: int = 200,
    seed: int = 0,
    random_action_scale: float = 0.3,
) -> EpisodeMetrics:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base    = OUT_DIR / f"episode_{episode:03d}"
    vid_path = base.with_suffix(".mp4")
    plt_path = base.with_suffix(".png")

    # Camera panel size (two side by side = VIDEO_W)
    cam_w = VIDEO_W // 2
    cam_h = VIDEO_H - HUD_H

    writer = cv2.VideoWriter(
        str(vid_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (VIDEO_W, VIDEO_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {vid_path}. "
                           "Check that opencv-python is installed.")

    if SAVE_FRAMES:
        frames_dir = Path(f"{base}_raw")
        frames_dir.mkdir(exist_ok=True)

    metrics = EpisodeMetrics()
    obs, _  = env.reset(seed=seed)
    success = False

    print(f"[viz] Recording episode {episode} ({n_steps} steps) …")
    t0 = time.time()

    for step in range(n_steps):
        # ── Action ────────────────────────────────────────────────────
        action = env.action_space.sample() * random_action_scale
        obs, reward, terminated, _, info = env.step(action)

        # ── Metrics ───────────────────────────────────────────────────
        state     = obs["state"]
        ee_pos    = state[12:15]                        # indices in (20,) state
        tgt_name  = JacoTableEnv._TARGET_MAP[env.task]
        tgt_pos   = np.array(env._objects[tgt_name].get_pose().p)
        ee_dist   = float(np.linalg.norm(ee_pos - tgt_pos))
        gripper_open = bool(state[19] > 0.5)

        metrics.record(reward, ee_dist, gripper_open)
        success = info.get("success", False)

        # ── Frame ─────────────────────────────────────────────────────
        frame = build_frame(
            overhead     = obs["image"],
            wrist        = obs["wrist_image"],
            cam_w        = cam_w,
            cam_h        = cam_h,
            step         = step,
            reward       = reward,
            cum_reward   = metrics.cumulative[-1],
            ee_dist      = ee_dist,
            gripper_open = gripper_open,
            success      = success,
            task         = env.task,
        )

        # cv2 uses BGR
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if SAVE_FRAMES:
            cv2.imwrite(
                str(frames_dir / f"frame_{step:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        if terminated:
            print(f"  [viz] SUCCESS at step {step}")
            break

    writer.release()
    elapsed = time.time() - t0
    print(f"  [viz] video → {vid_path}  ({step+1} frames in {elapsed:.1f}s)")

    save_metrics_plot(metrics, plt_path, env.task)
    return metrics


# ---------------------------------------------------------------------------
# Entry point — records one episode per task
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    TASKS = ["reach_red_cube", "reach_blue_sphere", "reach_green_cyl"]

    for ep, task in enumerate(TASKS):
        env = JacoTableEnv(
            render_mode  = "none",
            control_mode = "ee_delta",
            task         = task,
        )
        try:
            record_episode(env, episode=ep, n_steps=150, seed=ep)
        finally:
            env.close()

    print("\nAll episodes saved to", OUT_DIR.resolve())