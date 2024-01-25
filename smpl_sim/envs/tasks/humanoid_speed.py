from typing import Any, Sequence
import numpy as np
from collections import OrderedDict

from smpl_sim.envs.humanoid_task import HumanoidTask
import smpl_sim.utils.np_transform_utils as npt_utils
from smpl_sim.utils.mujoco_utils import add_visual_capsule

def forward_reward(tar_speed, root_pos, prev_root_pos, dt):

    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = root_vel[..., 0]
    tangent_speed = root_vel[..., 1]

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = np.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    reward = dir_reward

    return reward

def compute_speed_observations(root_rot, tar_speed, upright_start, humanoid_type):
    if not upright_start:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    tar_dir3d = np.zeros((1, 3))
    tar_dir3d[..., 0] = 1
    heading_rot = npt_utils.calc_heading_quat_inv(root_rot)
    
    local_tar_dir = npt_utils.quat_rotate(heading_rot, tar_dir3d)
    local_tar_dir = local_tar_dir[..., 0:2]
    obs_dict = OrderedDict()
    obs_dict['local_tar_dir'] = local_tar_dir
    obs_dict['tar_speed'] = np.array([[tar_speed]])
    return obs_dict


class HumanoidSpeed(HumanoidTask):
    
    def __init__(self, cfg):
        super().__init__(cfg)
        self._tar_speed_min = cfg.env.tar_speed_min
        self._tar_speed_max = cfg.env.tar_speed_max
        self._speed_change_steps_min = cfg.env.speed_change_steps_min
        self._speed_change_steps_max = cfg.env.speed_change_steps_max
        
        self._prev_root_pos = np.zeros((1, 3))
        self._speed_change_steps = np.zeros((1, 1))
        self._tar_speed = np.zeros((1, 1))

    def create_task_visualization(self):
        if self.viewer is not None: # this implies that headless == False
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.05, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        if self.renderer is not None:
            add_visual_capsule(self.renderer.scene, np.zeros(3), np.array([0.05, 0, 0]), 0.05, np.array([245, 40, 145, 0.8]))
    
    def draw_task(self):
        def draw_obj(scene):
            root_pos = self.get_root_pos()
            root_pos[0] += 0.5 + 0.2 * self._tar_speed
            root_pos[2] = 0
            scene.geoms[scene.ngeom - 1].pos = root_pos

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
        if self.cur_t >= self._speed_change_steps:
            self.reset_task()
        return

    def reset_task(self, options = None):
        tar_speed = (self._tar_speed_max - self._tar_speed_min) * np.random.random() + self._tar_speed_min
        change_steps = np.random.randint(low=self._speed_change_steps_min, high=self._speed_change_steps_max)
        
        self._tar_speed = tar_speed
        self._speed_change_steps = self.cur_t + change_steps
        return
    
    def compute_task_obs(self):
        root_rot = self.get_qpos()[None, 3:7]
        tar_speed = self._tar_speed
        
        obs_dict = compute_speed_observations(root_rot, tar_speed, self.upright_start, self.humanoid_type)
        # return obs_dict # we need to change obs space to deal with dictionaries
        return np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=self.dtype)

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        body_pos = self.get_body_xpos()
        self._prev_root_pos[:] = body_pos[None, 0]
        return

    def compute_reward(self, actions):
        body_pos = self.get_body_xpos()
        root_pos = body_pos[None, 0]

        reward = forward_reward(tar_speed=self._tar_speed, root_pos=root_pos, prev_root_pos=self._prev_root_pos, dt=self.dt)[0]  # ZL: the [0] is a little ugly
        return reward
