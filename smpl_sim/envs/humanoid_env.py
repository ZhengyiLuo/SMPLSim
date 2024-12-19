import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import numpy as np
from collections import OrderedDict
import gymnasium as gym
import mujoco
from scipy.spatial.transform import Rotation as sRot
from enum import Enum
from collections import defaultdict
import torch

from smpl_sim.envs.base_env import BaseEnv
import smpl_sim.envs.controllers as ctrls
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib import smpl_xml_addons as smplxadd
from smpl_sim.smpllib.motion_lib_base import FixHeightMode
import smpl_sim.utils.np_transform_utils as npt_utils
import smpl_sim.utils.mujoco_utils as mj_utils

try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
    
    
_AVAILABLE_CONTROLLERS = ["uhc_pd", "simple_pid", "pd", "torque"]


GAINS = {}
GAINS["stablepd"] = {
    # "L_Hip":        [500, 50, 1, 500, 10, 2],
    # "L_Knee":       [500, 50, 1, 500, 10, 2],
    # "L_Ankle":      [500, 50, 1, 500, 10, 2],
    # "L_Toe":        [200, 20, 1, 500, 1, 1],
    # "R_Hip":        [500, 50, 1, 500, 10, 2],
    # "R_Knee":       [500, 50, 1, 500, 10, 2],
    # "R_Ankle":      [500, 50, 1, 500, 10, 2],
    # "R_Toe":        [200, 20, 1, 500, 1, 1],
    # "Torso":        [1000, 100, 1, 500, 10, 2],
    # "Spine":        [1000, 100, 1, 500, 10, 2],
    # "Chest":        [1000, 100, 1, 500, 10, 2],
    # "Neck":         [100, 10, 1, 250, 50, 4],
    # "Head":         [100, 10, 1, 250, 50, 4],
    # "L_Thorax":     [400, 40, 1, 500, 50, 4],
    # "L_Shoulder":   [400, 40, 1, 500, 50, 4],
    # "L_Elbow":      [300, 30, 1, 150, 10, 2],
    # "L_Wrist":      [100, 10, 1, 150, 1, 1],
    # "L_Hand":       [100, 10, 1, 150, 1, 1],
    # "R_Thorax":     [400, 40, 1, 150, 10, 2],
    # "R_Shoulder":   [400, 40, 1, 250, 10, 2],
    # "R_Elbow":      [300, 30, 1, 150, 10, 2],
    # "R_Wrist":      [100, 10, 1, 150, 1, 1],
    # "R_Hand":       [100, 10, 1, 150, 1, 1],
    
    # # Much bigger
    "L_Hip":      [800, 80, 1, 1000],
    "L_Knee":     [800, 80, 1, 1000],
    "L_Ankle":    [800, 80, 1, 1000],
    "L_Toe":      [500, 50, 1, 500],
    "R_Hip":      [800, 80, 1, 1000],
    "R_Knee":     [800, 80, 1, 1000],
    "R_Ankle":    [800, 80, 1, 1000],
    "R_Toe":      [500, 50, 1, 500],
    "Torso":      [1000, 100, 1, 500],
    "Spine":      [1000, 100, 1, 500],
    "Chest":      [1000, 100, 1, 500],
    "Neck":       [500, 50, 1, 250],
    "Head":       [500, 50, 1, 250],
    "L_Thorax":   [500, 50, 1, 1000],
    "L_Shoulder": [500, 50, 1, 1000],
    "L_Elbow":    [500, 50, 1, 250],
    "L_Wrist":    [300, 30, 1, 250],
    "L_Hand":     [300, 30, 1, 250],
    "R_Thorax":   [500, 50, 1, 1000],
    "R_Shoulder": [500, 50, 1, 1000],
    "R_Elbow":    [500, 50, 1, 250],
    "R_Wrist":    [300, 30, 1, 250],
    "R_Hand":     [300, 30, 1, 250],
    
    # Much Smaller
    # "L_Hip":            [250, 2.5, 1, 500, 10, 2],
    # "L_Knee":           [250, 2.5, 1, 500, 10, 2],
    # "L_Ankle":          [150, 2.5, 1, 500, 10, 2],
    # "L_Toe":            [150, 1, 1, 500, 1, 1],
    # "R_Hip":            [250, 2.5, 1, 500, 10, 2],
    # "R_Knee":           [250, 2.5, 1, 500, 10, 2],
    # "R_Ankle":          [150, 1, 1, 500, 10, 2],
    # "R_Toe":            [150, 1, 1, 500, 1, 1],
    # "Torso":            [500, 5, 1, 500, 10, 2],
    # "Spine":            [500, 5, 1, 500, 10, 2],
    # "Chest":            [500, 5, 1, 500, 10, 2],
    # "Neck":             [150, 1, 1, 250, 50, 4],
    # "Head":             [150, 1, 1, 250, 50, 4],
    # "L_Thorax":         [200, 2, 1, 500, 50, 4],
    # "L_Shoulder":       [200, 2, 1, 500, 50, 4],
    # "L_Elbow":          [150, 1, 1, 150, 10, 2],
    # "L_Wrist":          [100, 1, 1, 150, 1, 1],
    # "L_Hand":           [50, 1, 1, 150, 1, 1],
    # "R_Thorax":         [200, 2, 1, 150, 10, 2],
    # "R_Shoulder":       [200, 2, 1, 250, 10, 2],
    # "R_Elbow":          [150, 1, 1, 150, 10, 2],
    # "R_Wrist":          [100, 1, 1, 150, 1, 1],
    # "R_Hand":           [50, 1, 1, 150, 1, 1],
}

# simple pd:
GAINS["simplepd"] = {
    "L_Hip":            [250, 5, 1, 500, 10, 2],
    "L_Knee":           [250, 5, 1, 500, 10, 2],
    "L_Ankle":          [150, 5, 1, 500, 10, 2],
    "L_Toe":            [150, 3, 1, 500, 1, 1],
    "R_Hip":            [250, 5, 1, 500, 10, 2],
    "R_Knee":           [250, 5, 1, 500, 10, 2],
    "R_Ankle":          [150, 5, 1, 500, 10, 2],
    "R_Toe":            [150, 3, 1, 500, 1, 1],
    "Torso":            [500, 10, 1, 500, 10, 2],
    "Spine":            [500, 10, 1, 500, 10, 2],
    "Chest":            [500, 10, 1, 500, 10, 2],
    "Neck":             [150, 1, 1, 250, 50, 4],
    "Head":             [150, 1, 1, 250, 50, 4],
    "L_Thorax":         [200, 4, 1, 500, 50, 4],
    "L_Shoulder":       [200, 4, 1, 500, 50, 4],
    "L_Elbow":          [150, 3, 1, 150, 10, 2],
    "L_Wrist":          [100, 1, 1, 150, 1, 1],
    "L_Hand":           [50, 1, 1, 150, 1, 1],
    "R_Thorax":         [200, 4, 1, 150, 10, 2],
    "R_Shoulder":       [200, 4, 1, 250, 10, 2],
    "R_Elbow":          [150, 3, 1, 150, 10, 2],
    "R_Wrist":          [100, 1, 1, 150, 1, 1],
    "R_Hand":           [50, 1, 1, 150, 1, 1],
}

class HumanoidEnv(BaseEnv):

    class StateInit(Enum):
        Default = 0
        Fall = 1
        MoCap = 2
        DefaultAndFall = 3
        FallAndMoCap = 4

    def __init__(self, cfg):
        self.cfg = cfg
        super().__init__(cfg=self.cfg)
        self.load_humanoid_configs(cfg)

        self.control_mode = self.cfg.env.control_mode
        self.power_scale = self.cfg.env.power_scale
        assert self.control_mode in _AVAILABLE_CONTROLLERS, f"{self.control_mode} is not a valid controller {_AVAILABLE_CONTROLLERS}"

        self.max_episode_length = self.cfg.env.episode_length
        self._root_height_obs = self.cfg.env.root_height_obs
        self._enable_early_termination = self.cfg.env.enable_early_termination
        self.self_obs_v = self.cfg.env.self_obs_v
        self.dtype = np.float32  # need to come from cfg.
        self.state_init = HumanoidEnv.StateInit[cfg.env.state_init]

        self._create_humanoid_robot(cfg=self.cfg)
        self.create_sim(
            self.default_xml_str
        )  # Create sim first, then intialize the base env.
        self.setup_humanoid_properties()
        self.setup_controller()
        # self.create_viewer() # viewer is created when you call render, no need to create it before
        self.state_record = defaultdict(list)
        self.reward_info = {}
        
        self.observation_space = gym.spaces.Box(
            -np.inf * np.ones(self.get_obs_size()),
            np.inf * np.ones(self.get_obs_size()),
            dtype=self.dtype,
        )
        
        self.action_space = gym.spaces.Box(
            low=-np.ones(self.get_action_size()) if  self.clip_actions else -np.inf * np.ones(self.get_action_size()),
            high=np.ones(self.get_action_size()) if  self.clip_actions else np.inf * np.ones(self.get_action_size()),
            dtype=self.dtype,
        )
        
        

    def load_humanoid_configs(self, cfg):
        self.humanoid_type = cfg.robot.humanoid_type
        self.contact_bodies = self.cfg.env.contact_bodies
        
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            self.load_smpl_configs(cfg)
        else:
            raise NotImplementedError(f"humanoid_type: {self.humanoid_type}")

    def load_smpl_configs(self, cfg):
        self.load_common_humanoid_configs(cfg)
        self.upright_start = cfg.robot.has_upright_start
        self._smpl_data_dir = cfg.robot.get("smpl_data_dir", "data/smpl")

    def load_common_humanoid_configs(self, cfg):
        self._has_shape_obs = cfg.robot.has_shape_obs
        self._has_limb_weight_obs = cfg.robot.has_weight_obs
        self.has_shape_variation = cfg.robot.has_shape_variation

        self._kp_scale = cfg.env.kp_scale
        self._kd_scale = cfg.env.kd_scale
        self.cycle_motion = cfg.env.cycle_motion
        self.power_reward = cfg.env.power_reward

        height_fix_mode = cfg.robot.height_fix_mode

        if height_fix_mode == "full":
            self.height_fix_mode = FixHeightMode.full_fix
        elif height_fix_mode == "ankle":
            self.height_fix_mode = FixHeightMode.ankle_fix
    
    def _create_humanoid_robot(self, cfg):
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            robot_cfg = {
                "mesh": cfg.robot.has_mesh,
                "replace_feet": cfg.robot.replace_feet,
                "rel_joint_lm": cfg.robot.has_jt_limit,
                "remove_toe": cfg.robot.get("remove_toe", False),
                "freeze_hand": cfg.robot.get("freeze_hand", False),
                "real_weight_porpotion_capsules": cfg.robot.real_weight_porpotion_capsules,
                "real_weight_porpotion_boxes": cfg.robot.real_weight_porpotion_boxes,
                "real_weight": cfg.robot.real_weight,
                "master_range": cfg.robot.get("master_range", 30),
                "big_ankle": cfg.robot.big_ankle,
                "box_body": cfg.robot.box_body,
                "masterfoot": cfg.robot.get("masterfoot", False),
                "upright_start": self.upright_start,
                "model": self.humanoid_type,
                "create_vel_sensors": cfg.robot.create_vel_sensors,
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }
            if os.path.exists(self._smpl_data_dir):
                self.robot = SMPL_Robot(
                    robot_cfg,
                    data_dir=self._smpl_data_dir,
                )
                
                self.default_xml_str = self.robot.export_xml_string().decode("utf-8")
            else:
                print("Missing SMPL Files!!!!! Using mean netural body ")
                default_smpl_file = files('smpl_sim').joinpath('data/assets/mjcf/smpl_humanoid.xml')
                with open(default_smpl_file, 'r') as file:
                    self.default_xml_str = file.read()
                self.robot = None
            
            if self.render_mode == "rgb_array":
                # this is temp fix for rendering without visualizer, should we add a camera directly in SMPL_robot
                self.default_xml_str = smplxadd.smpl_add_camera(self.default_xml_str)
        else:
            raise NotImplementedError(f"humanoid_type: {self.humanoid_type}")

    def setup_humanoid_properties(self):
        self.mj_body_names = []
        for i in range(self.mj_model.nbody):  # the first one is always world
            body_name = self.mj_model.body(i).name
            self.mj_body_names.append(body_name)
        if self.robot is not None:
            self.body_names_orig = self.robot.joint_names 
        else:
            self.body_names_orig = self.mj_body_names[1:] # making some assumptions about the xml file here. 
            
        self.num_rigid_bodies = len(self.body_names_orig)
        self.num_vel_limit = self.num_rigid_bodies * 3
        self.dof_names = self.body_names_orig[1:] # first joint is not actuated.
        self.actuator_names = mj_utils.get_actuator_names(self.mj_model)
        self.body_qposaddr = mj_utils.get_body_qposaddr(self.mj_model)
        self.body_qveladdr = mj_utils.get_body_qveladdr(self.mj_model)
        self.robot_body_idxes = [
            self.mj_body_names.index(name) for name in self.body_names_orig
        ]
        self.robot_idx_start = self.robot_body_idxes[0]
        self.robot_idx_end = self.robot_body_idxes[-1] + 1

        self.qpos_lim = np.max(self.mj_model.jnt_qposadr) + self.mj_model.jnt_qposadr[-1] - self.mj_model.jnt_qposadr[-2]
        self.qvel_lim = np.max(self.mj_model.jnt_dofadr) + self.mj_model.jnt_dofadr[-1] - self.mj_model.jnt_dofadr[-2]
        
        geom_type_id = mujoco.mju_str2Type("geom")
        self.contact_bodies_ids = [mujoco.mj_name2id(self.mj_model, geom_type_id, name) for name in self.contact_bodies]
        self.floor_idx = mujoco.mj_name2id(self.mj_model, geom_type_id, "floor")
        
        ################## Humanoid Character Properties ##################
        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            if self.self_obs_v == 1:
                self._num_self_obs = (1 if self._root_height_obs else 0) \
                    + len(self.dof_names) * 3 + len(self.body_names_orig)  * 6 + 3 + 3  + len(self.dof_names) * 3
            elif self.self_obs_v == 2:
                assert(self.cfg.robot.create_vel_sensors)
                self._num_self_obs = (1 if self._root_height_obs else 0) \
                    + len(self.dof_names) * 3 + len(self.body_names_orig)  * (6 + 3 + 3) 
                    
            else:
                raise NotImplementedError(f"self_obs_v: {self.self_obs_v}")
            
            if self.has_shape_variation:
                self._num_self_obs += 10  # self._num_self_obs = np.sum([v.flatten().shape[-1] for k, v in self.compute_proprioception().items()])
            self.dof_size = len(self.dof_names) * 3
        else:
            raise NotImplementedError(f"humanoid_type: {self.humanoid_type}")
        #####################################################################

            
    def setup_controller(self):
        self.build_pd_action_scale()
        if self.control_mode == "uhc_pd":
            self.ctrler = ctrls.StablePDController(self._pd_action_scale, self._pd_action_offset, self.qvel_lim, self.torque_lim, self.jkp / self.cfg.env.pdp_scale, self.jkd / self.cfg.env.pdd_scale)
        if self.control_mode == "pd":
            self.ctrler = ctrls.PIDController(self._pd_action_scale, self._pd_action_offset, self.torque_lim, self.jkp / self.cfg.env.pdp_scale, self.jkd / self.cfg.env.pdd_scale, np.zeros_like(self.jkd))
        elif self.control_mode == "simple_pid":
            self.ctrler = ctrls.SimplePID(self.jkp/10, np.ones_like(self.jkp), self.jkd/10, self.mj_model.opt.timestep*self.control_freq_inv,self.torque_lim, self._pd_action_scale, self._pd_action_offset)
        elif self.control_mode == "torque":
            self.ctrler = ctrls.SimpleTorqueController(self.power_scale*self.torque_lim, self.torque_lim)
        
    def build_pd_action_scale(self):
        lim_high = np.zeros(self.dof_size)
        lim_low = np.zeros(self.dof_size)
        self.jkp = np.zeros(self.dof_size)
        self.jkd = np.zeros(self.dof_size)
        self.torque_lim = np.zeros(self.dof_size)
        for idx, n in enumerate(self.actuator_names):
            joint_config = self.mj_model.joint(n)
                
            low, high = joint_config.range
            curr_low = low
            curr_high = high
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 1.2 * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[idx] = -curr_scale
            lim_high[idx] = curr_scale
            
        if self.control_mode in ["pd", "uhc_pd"]:
            for idx, n in enumerate(self.actuator_names):
                joint = "_".join(n.split("_")[:-1])
                self.jkp[idx] = GAINS["stablepd"][joint][0]
                self.jkd[idx] = GAINS["stablepd"][joint][1]
                self.torque_lim[idx] = GAINS["stablepd"][joint][3]
            # self.jkp = self.jkp * 2
            # self.jkd = self.jkd / 5
        if self.control_mode == "simple_pid":
            self.jki = np.zeros(self.dof_size)
            for idx, n in enumerate(self.actuator_names):
                joint = "_".join(n.split("_")[:-1])
                self.jkp[idx] = GAINS["stablepd"][joint][0]
                self.jkd[idx] = GAINS["stablepd"][joint][1]
                # self.jki[idx] = GAINS["simplepd"][joint][0]
                self.torque_lim[idx] = GAINS["stablepd"][joint][3]
            # self.jkp = self.jkp
            # self.jkd = self.jkd / 20
            # self.jki = self.jki * 1.5
        if self.clip_actions:
            self._pd_action_scale = 0.5 * (lim_high - lim_low)
            self._pd_action_offset = 0.5 * (lim_high + lim_low)
        else:
            self._pd_action_scale = np.ones_like(lim_high)
            self._pd_action_offset = np.zeros_like(lim_high)

    def get_action_size(self):
        return self.dof_size

    def get_obs_size(self):
        return self.get_self_obs_size()

    def get_self_obs_size(self):
        return self._num_self_obs

    def compute_observations(self):
        obs = self.compute_proprioception()
        return obs
    
    def compute_info(self):
        return {}

    def compute_proprioception(self):
        mujoco.mj_kinematics(self.mj_model, self.mj_data)  # update xpos to the latest simulation values
        
        qpos = self.get_qpos()[None,]  # TODO why qpos is not used in the proprioception?
        qvel = self.get_qvel()[None,]
        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]

        if self.self_obs_v == 1:
            obs_dict =  compute_humanoid_self_obs_v1(qpos, qvel, body_pos, body_rot, self.upright_start, self._root_height_obs,  humanoid_type = self.humanoid_type)
        elif self.self_obs_v == 2:
            body_vel = self.get_body_linear_vel()[None,]
            body_ang_vel = self.get_body_angular_vel()[None,]
            obs_dict =  compute_humanoid_self_obs_v2(body_pos, body_rot, body_vel, body_ang_vel, self.upright_start, self._root_height_obs,  humanoid_type = self.humanoid_type)
            
        return np.concatenate([v.ravel() for v in obs_dict.values()], axis=0, dtype=self.dtype)
        # return obs # we need to change the definition of spaces if we want to keep the dictionary
        
    
    def compute_torque(self, ctrl):
        ctrl_joint = ctrl[:self.dof_size]
        # ctrl_joint[:] = 0; # ctrl_joint[self.actuator_names.index("L_Ankle_x")] = 1/2 # Debugging
        torque = self.ctrler.control(ctrl_joint, self.mj_model, self.mj_data)
        
        # np.set_printoptions(precision=4, suppress=1)
        # print(torque, torque.max(), torque.min())
        return torque
    
    def get_body_xpos(self):
        return self.mj_data.xpos.copy()[self.robot_idx_start : self.robot_idx_end]
    
    def get_body_xpos_by_id(self, body_id):
        return self.mj_data.xpos[self.robot_idx_start + body_id]

    def get_body_xquat(self):
        return self.mj_data.xquat.copy()[self.robot_idx_start : self.robot_idx_end]

    def compute_reward(self, actions):
        reward = 0
        return reward

    def compute_reset(self):
        if self.cur_t > self.max_episode_length:
            return False, True
        else:
            return False, False

    def pre_physics_step(self, actions):
        pass

    def physics_step(self, actions):
        # if not self.action_space.contains(actions):
        #     breakpoint()
        # self.render() This is done in the base class
        self.curr_power_usage = []
        for i in range(self.control_freq_inv):
            if not self.paused:
                torque = self.compute_torque(actions)
                # np.set_printoptions(precision=4, suppress=1); print( np.abs(torque).max(), np.abs(torque).min())
                self.mj_data.ctrl[:] = torque
                mujoco.mj_step(self.mj_model, self.mj_data)
                self.curr_power_usage.append(np.abs(torque * self.get_qvel()[6:]))
                
                # self.mj_data.qpos[2] = 1.5
                # mujoco.mj_forward(self.mj_model, self.mj_data)
        return

    def post_physics_step(self, actions):
        if not self.paused:
            self.cur_t += 1
        obs = self.compute_observations()
        reward = self.compute_reward(actions)
        terminated, truncated = self.compute_reset()
        if self.disable_reset:
            terminated, truncated = False, False
        info = {}
        info.update(self.reward_info)
        return obs, reward, terminated, truncated, info
    
    def init_humanoid(self):
        if self.state_init == HumanoidEnv.StateInit.Default:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                self.mj_data.qpos[:] = 0
                self.mj_data.qvel[:] = 0
                self.mj_data.qpos[2] = 0.94
                self.mj_data.qpos[3:7] = np.array([0.5, 0.5, 0.5, 0.5])
        elif self.state_init == HumanoidEnv.StateInit.Fall:
            if self.humanoid_type in ["smpl", "smplh", "smplx"]:
                self.mj_data.qpos[:] = 0
                self.mj_data.qvel[:] = 0
                self.mj_data.qpos[2] = 0.3
                self.mj_data.qpos[3:7] = np.array([1, 0, 0, 0])
                mujoco.mj_forward(self.mj_model, self.mj_data)
                for _ in range(3):
                    # on purpose this is always done in torque space
                    action = (self.np_random.random(self.get_action_size()) - 0.5 ) * 1
                    for _ in range(self.control_freq_inv):
                        torque = self.compute_torque(action)
                        self.mj_data.ctrl[:] = torque
                        mujoco.mj_step(self.mj_model, self.mj_data)                


    def reset_humanoid(self):
        self.init_humanoid()
        
        # Heading in varance check for proprioception
        # for _ in range(10):
        #     self.mj_data.qpos[:] = np.random.random()
        #     self.reset_sim(); self.render()
        #     prop1 = self.compute_proprioception()
        #     self.mj_data.qpos[3:7] = npt_utils.xyzw_to_wxyz((sRot.from_euler('xyz', [0, 0, np.random.random() * np.pi], degrees=False) * sRot.from_quat(np.array([0.5, 0.5, 0.5, 0.5])) ).as_quat()); self.reset_sim(); self.render()
        #     prop2 = self.compute_proprioception()
        #     diff = np.concatenate([v.flatten() for v in prop1.values()]) - np.concatenate([v.flatten() for v in prop2.values()]); np.abs(diff).sum()
        
    def reset_sim(self):
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def reset(self, seed=None, options=None):
        self.reset_humanoid()
        self.reset_sim()
        return super().reset(seed=seed, options=options)

    def get_body_jac_by_id(self, id, jacp=None, jacr=None):
        mujoco.mj_jacBody(self.mj_model, self.mj_data, jacp, jacr, id)
        return jacp, jacr

    def get_body_jacp_by_name(self, name, jacp=None, jacr=None):
        id = self.mj_model.body(name, jacp=jacp, jacr=jacr).id
        return self.get_body_jac_by_id(id)

    def get_body_xvelp(self, name):
        qvel = self.mj_data.qvel.copy()
        jacp_temp = np.zeros((3, self.mj_model.nv))
        jacp = self.get_body_jacp_by_name(name, jacp=jacp_temp).reshape(
            (3, self.mj_model.nv)
        )
        xvelp = np.dot(jacp, qvel)
        return xvelp

    def get_body_xvelr(self, name):
        qvel = self.mj_data.qvel.copy()
        jacr_temp = np.zeros((3, self.mj_model.nv)) # TODO this is not used
        jacr = self.get_body_jacp_by_name(name).reshape((3, self._model.nv))
        xvelr = np.dot(jacr, qvel)
        return xvelr
    
    # Getting the body linear velocity in the world frame using sensors. This is required for self_obs_v2. This function expect sensors to be first linear then angular. 
    def get_body_linear_vel(self): 
        return self.mj_data.sensordata[:self.num_vel_limit].reshape(self.num_rigid_bodies, 3).copy()
    
    # Getting the body angular velocity in the world frame using sensors. This is required for self_obs_v2. This function expect sensors to be first linear then angular. 
    def get_body_angular_vel(self):
        return self.mj_data.sensordata[self.num_vel_limit:].reshape(self.num_rigid_bodies, 3).copy()
    
        
    def get_qpos(self):
        return self.mj_data.qpos.copy()[: self.qpos_lim]

    def get_qvel(self):
        return self.mj_data.qvel.copy()[:self.qvel_lim]
    
    def get_root_pos(self):
        return self.get_body_xpos()[0].copy()
    
    def get_root_state(self):
        return np.concatenate([self.get_qpos()[:7], self.get_qvel()[:6]]).copy()
    
    def record_states(self):
        self.state_record['qpos'].append(self.get_qpos())
        self.state_record['qvel'].append(self.get_qvel())
        
    
    
def compute_humanoid_self_obs_v1(qpos, qvel, body_pos, body_rot, upright_start, root_height_obs, humanoid_type):
    obs = OrderedDict()
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    
    if not upright_start:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    if root_height_obs:
        obs["root_h_obs"] = root_h
    
    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    ###### Velocity ######
    # Here different from PHC, we only have access to the global linear and angular velocity of the root in qvel (computating global ones would be expensive)
    # so we will use those.
    # TODO: what about https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html?highlight=subtree_linvel#mj-subtreevel
    root_velp = qvel[:, None, 0:3]
    root_velr = qvel[:, None, 3:6]
    body_vel = qvel[:, 6:]

    flat_root_vel = root_velp.reshape(
        root_velp.shape[0] * root_velp.shape[1], root_velp.shape[2]
    )
    flat_local_body_vel = npt_utils.quat_rotate(heading_rot_inv, flat_root_vel)
    obs["local_root_vel"] = flat_local_body_vel.reshape(
        body_vel.shape[0], root_velp.shape[1] * root_velp.shape[2]
    )

    flat_root_ang_vel = root_velr.reshape(
        root_velr.shape[0] * root_velr.shape[1], root_velr.shape[2]
    )
    flat_local_root_ang_vel = npt_utils.quat_rotate(
        heading_rot_inv, flat_root_ang_vel
    )
    obs["local_root_ang_vel"] = flat_local_root_ang_vel.reshape(
        root_velr.shape[0], root_velr.shape[1] * root_velr.shape[2]
    )
    
    obs["body_ang_vel"] = body_vel
    return obs


# This function is an excat replica of PHC's. 
def compute_humanoid_self_obs_v2(body_pos, body_rot, body_vel, body_ang_vel, upright_start, root_height_obs, humanoid_type):
    obs = OrderedDict()
    
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]
    
    if not upright_start:
        root_rot = npt_utils.remove_base_rot(root_rot, humanoid_type)

    heading_rot_inv = npt_utils.calc_heading_quat_inv(root_rot)
    root_h = root_pos[:, 2:3]

    if root_height_obs:
        obs["root_h_obs"] = root_h
    
    heading_rot_inv_expand = heading_rot_inv[..., None, :]
    heading_rot_inv_expand = heading_rot_inv_expand.repeat(body_pos.shape[1], axis=1)
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1],heading_rot_inv_expand.shape[2],)

    root_pos_expand = root_pos[..., None, :]
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = npt_utils.quat_rotate(
        flat_heading_rot_inv, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    obs["local_body_pos"] = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )  # This is global rotation of the body
    flat_local_body_rot = npt_utils.quat_mul(flat_heading_rot_inv, flat_body_rot)
    flat_local_body_rot_obs = npt_utils.quat_to_tan_norm(flat_local_body_rot)
    obs["local_body_rot_obs"] = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    ###### Velocity ######
    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_vel)
    obs["local_body_vel"]  = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = npt_utils.quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    obs["local_body_ang_vel"] = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    
    return obs