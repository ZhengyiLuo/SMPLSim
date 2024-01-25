from typing import Any
import mujoco
from absl import logging
from ..smplenv import SMPLHumanoid


class SMPLHumanoidReplay(SMPLHumanoid):
    def __init__(
        self,
        motions: Any,
        data_dir: str = "",
        physics_steps_per_control_step: int = 6,  # 30Hz
        sim_timestep: float = 1.0 / 180.0,
    ) -> None:
        max_episode_length = max(map(lambda v: v.shape[0], motions.values()))
        super().__init__(
            motions=motions,
            max_episode_length=max_episode_length,
            pid_controlled=False,
            data_dir=data_dir,
            physics_steps_per_control_step=physics_steps_per_control_step,
            sim_timestep=sim_timestep,
            initial_position="mocap",
        )
        self._num_total_motions = len(self._motion_keys)
        self._current_clip_index = -1

    def reset(self) -> None:
        self.cur_t = 0
        self._current_clip_index = (
            self._current_clip_index + 1
        ) % self._num_total_motions
        clip_id = self._motion_keys[self._current_clip_index]
        self._init_walker_from_mocap(clip_id, random_timestep=False)
        logging.info(
            f"Showing clip {self._current_clip_index+1} of {self._num_total_motions}, clip id: {clip_id}"
        )
        self.episode = {
            "qpos": [],
            "qvel": [],
            "torque": [],
            "action": [],
        }  # debugging TODO:remove
        return self.get_observation()

    def step(self, action):
        self.cur_t += 1
        self.mj_data.qpos[:] = self.expert_qpos[self.cur_t]
        # TODO: get qvel
        # self.mj_data.qvel[:] = self.expert_qvel[0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        done = False
        if self.cur_t + 1 == self.expert_qpos.shape[0]:
            done = True
        obs = self.get_observation()
        reward = self.get_reward(obs, action)
        assert self.cur_t <= self._max_episode_length

        return obs, reward, done, {}
