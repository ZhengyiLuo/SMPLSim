import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

class GymVectEnv:
    def __init__(
        self,
        env,
        clip_observations: float = float(np.inf),
    ) -> None:
        self._env = env
        self._clip_obs = clip_observations
        self._autoreset_envs = None

        self.single_observation_space = spaces.Box(np.ones(self._env.num_obs) * -np.Inf, np.ones(self._env.num_obs) * np.Inf)
        self.single_action_space = spaces.Box(np.ones(self._env.num_actions) * -1., np.ones(self._env.num_actions) * 1.)

        # Initialise the obs and action space based on the single versions and num of sub-environments
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    def reset(self, seed=None, options=None):
        self._autoreset_envs = None
        self._env.reset(self._autoreset_envs)
        obs = self._get_clipped_obs()
        return obs, {}

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def close(self) -> None:
        pass

    def step(self, actions):
        self._env.step(actions)

        obs = self._get_clipped_obs()
        reward = self._env.rew_buf
        extras = self._env.extras

        # Whether the agent reaches the terminal state (as defined under the MDP of the task) which can be positive or negative. An example is reaching the goal state or moving into the lava from the Sutton and Barton, Gridworld. If true, the user needs to call reset().
        terminated = extras["terminate"]

        # Whether the truncation condition outside the scope of the MDP is satisfied. Typically, this is a timelimit, but could also be used to indicate an agent physically going out of bounds. Can be used to end the episode prematurely before a terminal state is reached. If true, the user needs to call reset()
        truncated = self._env.reset_buf

        # vectorized environment should autoreset
        self._autoreset_envs = truncated.nonzero(as_tuple=False)[:,0]
        self._env.reset(self._autoreset_envs)
        obs_after_reset = self._get_clipped_obs()

        info = extras
        if len(self._autoreset_envs) > 0:
            info["final_observation"] = obs

        return obs_after_reset, reward, terminated, truncated, info

    def _get_clipped_obs(self) -> torch.Tensor:
        return torch.clamp(self._env.obs_buf, -self._clip_obs, self._clip_obs)
