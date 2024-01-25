from typing import Any, Sequence
import numpy as np
from collections import OrderedDict

from smpl_sim.envs.humanoid_task import HumanoidTask
import smpl_sim.utils.np_transform_utils as npt_utils
from smpl_sim.utils.mujoco_utils import add_visual_capsule


def reach_reward(reach_body_pos, tar_pos):
    pos_err_scale = 4.0
    
    pos_diff = tar_pos - reach_body_pos
    pos_err = np.sum(pos_diff * pos_diff, axis=-1)
    pos_reward = np.exp(-pos_err_scale * pos_err)
    
    reward = pos_reward

    return reward

def compute_location_observations(root_pos, root_rot, tar_pos, upright_start, humanoid_type):
    if not upright_start:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    local_tar_pos = tar_pos - root_pos
    
    local_tar_pos = npt_utils.quat_rotate(heading_rot_inv, local_tar_pos)

    return {"tar_pos": local_tar_pos}


class HumanoidReach(HumanoidTask):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._tar_dist_max = cfg.env.tar_dist_max
        self._tar_height_min = cfg.env.tar_height_min
        self._tar_height_max = cfg.env.tar_height_max
        self._tar_change_steps_min = cfg.env.tar_change_steps_min
        self._tar_change_steps_max = cfg.env.tar_change_steps_max

        self._tar_pos = np.zeros((1, 3))
        self._tar_change_steps = np.zeros((1, 1))
        
        self._reach_body_idx = self.body_names_orig.index(cfg.env.reach_body_name)

    def create_task_visualization(self):
        if self.viewer is not None: # this implies that headless == False
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        if self.renderer is not None:
            add_visual_capsule(self.renderer.scene, np.zeros(3), np.array([0.05, 0, 0]), 0.05, np.array([245, 40, 145, 0.8]))
    
    def draw_task(self):
        def draw_obj(scene):
            scene.geoms[scene.ngeom - 1].pos = self._tar_pos

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)
    
    def get_task_obs_size(self):
        return 3 # local_tar_dir + tar_speed
    
    def compute_reset(self):
        terminated, truncated = False, False
        pass_time = self.cur_t > self.max_episode_length
        select_floor = self.mj_data.contact.geom1 == self.floor_idx
        all_legal_contacts = np.isin(self.mj_data.contact.geom2[select_floor], self.contact_bodies_ids).all()
        truncated = pass_time 
        terminated = (not all_legal_contacts)
        return terminated, truncated
    
    def update_task(self):
        if self.cur_t >= self._tar_change_steps:
            self.reset_task()
        return

    def reset_task(self, options = None):
        rand_pos = np.random.random((1, 3))
        rand_pos[..., 0:2] = self._tar_dist_max * (2.0 * rand_pos[..., 0:2] - 1.0)
        rand_pos[..., 2] = (self._tar_height_max - self._tar_height_min) * rand_pos[..., 2] + self._tar_height_min
        
        
        change_steps = np.random.randint(low=self._tar_change_steps_min, high=self._tar_change_steps_max)
        
        self._tar_pos = rand_pos
        self._tar_change_steps = self.cur_t + change_steps
        return
    
    def compute_task_obs(self):
        qpos = self.get_qpos()
        root_pos = qpos[None, 0:3]
        root_rot = qpos[None, 3:7]
        obs_dict = compute_location_observations(root_pos, root_rot, self._tar_pos, self.upright_start, self.humanoid_type)
        # return obs_dict # we need to change obs space to deal with dictionaries
        return np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=self.dtype)


    def compute_reward(self, actions):
        reach_body_pos = self.get_body_xpos_by_id(self._reach_body_idx)[None, ]
        reward = reach_reward(reach_body_pos, self._tar_pos)[0]  # ZL: the [0] is a little ugly
        return reward
