import sapien.core as sapien

import gymnasium as gym
from gymnasium.utils import seeding
import cv2


class SapienEnv(gym.Env):
    """Superclass for Sapien environments."""

    def __init__(self, control_freq, timestep):
        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep

        self._scene = sapien.Scene()
        self._scene.set_timestep(timestep)

        self._build_world()
        self.viewer = None
        self.seed()

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pass  # release viewer

    def render(self, mode='human'):
        if mode == 'human':
            if self.viewer is None:
                self._setup_viewer()
            self._scene.update_render()
            if not self.viewer.closed:
                self.viewer.render()
        else:
            raise NotImplementedError('Unsupported render mode {}.'.format(mode))

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_actor(self, name):
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f'Not a unique name for actor: {name}')
        elif len(actor) == 0:
            raise RuntimeError(f'Actor not found: {name}')
        return actor[0]

    def get_articulation(self, name):
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        return self.timestep * self.control_freq
    
class AntEnv(SapienEnv):
    def __init__(self):
        super().__init__(control_freq=5, timestep=0.01)

        self.actuator = self.get_articulation('ant')
        self._scene.step()  # simulate one step for steady state
        self._init_state = self._scene.physx_system.pack()

        dof = self.actuator.dof
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=[5 + dof + 6 + dof], dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=[dof], dtype=np.float32)
        # Following the original implementation, we scale the action (qf)
        self._action_scale_factor = 50.0

    # ---------------------------------------------------------------------------- #
    # Simulation world
    # ---------------------------------------------------------------------------- #
    def _build_world(self):
        physical_material = self._scene.create_physical_material(1.0, 1.0, 0.0)
        self._scene.default_physical_material = physical_material
        render_material = sapien.render.RenderMaterial()
        render_material.set_base_color([0.8, 0.9, 0.8, 1])
        self._scene.add_ground(0.0, render_material=render_material)
        ant = self.create_ant(self._scene)
        ant.set_pose(Pose([0., 0., 0.55]))

    def step(self, action):
        ant = self.actuator

        x_before = ant.pose.p[0]
        ant.set_qf(action * self._action_scale_factor)
        for i in range(self.control_freq):
            self._scene.step()
        x_after = ant.pose.p[0]

        forward_reward = (x_after - x_before) / self.dt
        ctrl_cost = 0.5 * np.square(action).sum()
        survive_reward = 1.0
        # Note that we do not include contact cost as the original version
        reward = forward_reward - ctrl_cost + survive_reward

        state = self.state_vector()
        is_healthy = (np.isfinite(state).all() and 0.2 <= state[2] <= 1.0)
        done = not is_healthy

        obs = self._get_obs()

        return obs, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_survive=survive_reward)

    def reset(self):
        self._scene.physx_system.unpack(self._init_state)
        # add some random noise
        init_qpos = self.actuator.get_qpos()
        init_qvel = self.actuator.get_qvel()
        qpos = init_qpos + self.np_random.uniform(size=self.actuator.dof, low=-0.1, high=0.1)
        qvel = init_qvel + self.np_random.normal(size=self.actuator.dof) * 0.1
        self.actuator.set_qpos(qpos)
        self.actuator.set_qvel(qvel)
        obs = self._get_obs()
        return obs
    
def main():
    print('Testing AntEnv...')
    env = AntEnv()
    env.reset()
    observations = []
    for step in range(1000):
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        observations.append(obs)
        if done:
            print(f'Done at step {step}')
            obs = env.reset()
    env.close()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = observations[0].shape[:2]
    out = cv2.VideoWriter('observations.mp4', fourcc, 30.0, (width, height))
    for obs in observations:
        frame = (obs * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

if __name__ == '__main__':
    main()