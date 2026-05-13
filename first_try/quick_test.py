import os
os.environ["SAPIEN_RENDERER"] = "egl"

import gymnasium as gym
import mani_skill.envs

env = gym.make("PickCube-v1", render_mode="rgb_array")
obs, _ = env.reset()

frame = env.render()
print(frame.shape)
