from typing import Any, Sequence
import numpy as np
from collections import OrderedDict

from smpl_sim.envs.humanoid_task import HumanoidTask
import smpl_sim.utils.np_transform_utils as npt_utils
from smpl_sim.utils.mujoco_utils import add_visual_capsule

def height_reward(tar_height, root_pos):

    pos_err_scale = 4.0
    pos_diff = tar_height - root_pos[:, 2]
    pos_err = pos_diff * pos_diff
    pos_reward = np.exp(-pos_err_scale * pos_err)
    
    reward = pos_reward[0]

    return reward

def compute_height_observations(tar_height, upright_start, humanoid_type):
    obs_dict = OrderedDict()
    obs_dict['tar_height'] = tar_height
    return obs_dict


class HumanoidGetup(HumanoidTask):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._tar_height_min = cfg.env.tar_height_min
        self._tar_height_max = cfg.env.tar_height_max
        self._height_change_steps_min = cfg.env.height_change_steps_min
        self._height_change_steps_max = cfg.env.height_change_steps_max
        
        self._recovery_steps = cfg.env.recovery_steps
        self._height_change_steps = np.zeros((1, 1))
        self._tar_height = np.zeros((1, 1))
        self._recovery_counter = np.zeros((1, 1))

    def create_task_visualization(self):
        if self.viewer is not None: # this implies that headless == False
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.05, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        if self.renderer is not None:
            add_visual_capsule(self.renderer.scene, np.zeros(3), np.array([0.05, 0, 0]), 0.05, np.array([245, 40, 145, 0.8]))
    
    def draw_task(self):
        def draw_obj(scene):
            root_pos = self.get_root_pos()
            root_pos[2] = self._tar_height[0]
            scene.geoms[scene.ngeom - 1].pos = root_pos

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)
    
    def get_task_obs_size(self):
        return 1 # tar_height
    
    def compute_reset(self):
        pass_time = self.cur_t > self.max_episode_length
        terminated, truncated = False, False
        if self._recovery_counter > 0:
            self._recovery_counter -= 1
            reset = False
        else:
            select_floor = self.mj_data.contact.geom1 == self.floor_idx
            all_legal_contacts = np.isin(self.mj_data.contact.geom2[select_floor], self.contact_bodies_ids).all()
            truncated = pass_time 
            terminated = not all_legal_contacts
            
        return terminated, truncated
    
    def reset(self, seed=None, options=None):
        self._recovery_counter[:] = self._recovery_steps
        return super().reset(seed=seed, options=options)
    
    def update_task(self):
        if self.cur_t >= self._height_change_steps:
            self.reset_task()
        return

    def reset_task(self, options = None):
        tar_height = (self._tar_height_max - self._tar_height_min) * np.random.random() + self._tar_height_min
        change_steps = np.random.randint(low=self._height_change_steps_min, high=self._height_change_steps_max)
        
        self._tar_height[:, :] = tar_height
        self._height_change_steps = self.cur_t + change_steps
        return
    
    def compute_task_obs(self):
        root_rot = self.get_qpos()[None, 3:7]
        tar_height = self._tar_height
        
        obs_dict = compute_height_observations(tar_height, self.upright_start, self.humanoid_type)
        # return obs_dict # we need to change obs space to deal with dictionaries
        return np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=self.dtype)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        return

    def compute_reward(self, actions):
        body_pos = self.get_body_xpos()
        root_pos = body_pos[None, 0]

        reward = height_reward(tar_height=self._tar_height, root_pos=root_pos)[0]  # ZL: the [0] is a little ugly
        return reward
