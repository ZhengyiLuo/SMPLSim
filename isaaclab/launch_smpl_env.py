# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/02_scene/create_scene.py --num_envs 32

"""

"""Launch Isaac Sim Simulator first."""


import argparse
from omni.isaac.lab.app import AppLauncher
from smpl_sim.utils.rotation_conversions import xyzw_to_wxyz, wxyz_to_xyzw
import sys

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import DirectRLEnvCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext, PhysxCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from omni.isaac.lab.devices.keyboard.se2_keyboard import Se2Keyboard
import omni.isaac.lab.utils.math as lab_math_utils
import carb
import imageio
from carb.input import KeyboardEventType



##
# Pre-defined configs
##
# from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from isaaclab.smpl_config import SMPL_CFG, SMPLX_CFG

from collections.abc import Sequence
from phc.utils.flags import flags
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp
from phc.env.tasks.humanoid_funcs import compute_humanoid_observations_smpl_max, compute_imitation_observations_v6
from rl_games.algos_torch import torch_ext
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from rl_games.algos_torch.players import rescale_actions
import torch
import joblib
from easydict import EasyDict
import numpy as np
import copy
from scipy.spatial.transform import Rotation as sRot
import time



flags.test=False

@configclass
class SMPLSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())


    
    
@configclass 
class SMPLEnvCfg(DirectRLEnvCfg):
    num_actions = 69
    num_observations = 1
    num_states = 1
    
    decimation = 2
    
    sim = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=1 / 60,
        # decimation will be set in the task config
        # up axis will always be Z in isaac sim
        # use_gpu_pipeline is deduced from the device
        gravity=(0.0, 0.0, -9.81),
        physx = PhysxCfg(
            # num_threads is no longer needed
            solver_type=1,
            # use_gpu is deduced from the device
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            # moved to actor config
            # moved to actor config
            bounce_threshold_velocity=0.2,
            # moved to actor config
            # default_buffer_size_multiplier is no longer needed
            gpu_max_rigid_contact_count=2**23,
            # num_subscenes is no longer needed
            # contact_collection is no longer needed
        )
    )
    

    viewer = ViewerCfg(eye=(20.0, 20.0, 20.0))
    
    # smpl_robot: ArticulationCfg = SMPL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    smpl_robot: ArticulationCfg = SMPLX_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(-4.0, 0.0, 2.0), rot=(0.9961947, 0, 0.0871557, 0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.0, focus_distance=50.0, horizontal_aperture=5, clipping_range=(0.1, 20.0)
        ),
        width=512,
        height=512,
    )
    
    # scene
    scene: InteractiveSceneCfg = SMPLSceneCfg(num_envs=args_cli.num_envs, env_spacing=20.0, replicate_physics=True)
    

class SMPLEnv(DirectRLEnv):
    cfg: SMPLEnvCfg
    
    def __init__(self, cfg: SMPLEnvCfg, render_mode: str | None = None, **kwargs):
        self.cfg = cfg
        self.humanoid_type = "smplx"   
        super().__init__(cfg, render_mode, **kwargs)
        
        SMPL_NAMES = SMPLH_MUJOCO_NAMES if self.humanoid_type == "smplx" else SMPL_MUJOCO_NAMES
        sim_joint_names = self._robot.data.joint_names
        sim_body_names = self._robot.data.body_names
        gym_joint_names = [f"{j}_{axis}" for j in SMPL_NAMES[1:] for axis in ["x", "y", "z"]]
        
        self.sim_to_gym_body = [sim_body_names.index(n) for n in SMPL_NAMES]
        self.sim_to_gym_dof = [sim_joint_names.index(n) for n in gym_joint_names]
        
        self.gym_to_sim_dof = [gym_joint_names.index(n) for n in sim_joint_names]
        self.gym_to_sim_body = [SMPL_NAMES.index(n) for n in sim_body_names]
        
        keyboard_interface = Se2Keyboard()
        keyboard_interface.add_callback("R", self.reset)
        
    
    def close(self):
        super().close()
        
    def _configure_gym_env_spaces(self):
        self.num_actions = self.cfg.num_actions
        self.num_observations = self.cfg.num_observations
        self.num_states = self.cfg.num_states
        
        
    def _setup_scene(self):
        self._robot = robot = Articulation(self.cfg.smpl_robot)
        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[])

        # add articultion and sensors to scene
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["tiled_camera"] = self._tiled_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        
        # motion_file = "data/amass/pkls/singles/hard_z_fut_1_1_2_upright_slim.pkl"
        # motion_file = "data/amass/pkls/singles/0-CMU_88_88_05_poses_upright_slim.pkl"
        motion_file = "data/amass_x/mouse.pkl"
        self._load_motion(motion_file)
        
        
    def _load_motion(self, motion_file):
        
        time_steps = 1
        self._has_upright_start = False
        motion_lib_cfg = EasyDict({
                        "has_upright_start": self._has_upright_start,
                        "motion_file": motion_file,
                        "fix_height": FixHeightMode.full_fix,
                        "min_length": -1,
                        "max_length": 3000,
                        "im_eval": flags.im_eval,
                        "multi_thread": False ,
                        "smpl_type": self.humanoid_type,
                        "randomrize_heading": True,
                        "device": self.device,
                    })
        robot_cfg = {
                "mesh": False,
                "rel_joint_lm": False,
                "upright_start": self._has_upright_start,
                "remove_toe": False,
                "real_weight_porpotion_capsules": True,
                "real_weight_porpotion_boxes": True,
                "model": self.humanoid_type,
                "big_ankle": True, 
                "box_body": True, 
                "body_params": {},
                "joint_params": {},
                "geom_params": {},
                "actuator_params": {},
            }
        smpl_robot = SMPL_Robot(
            robot_cfg,
            data_dir="data/smpl",
        )
            
        # gender_beta = np.zeros((17))
        gender_beta = np.zeros((16))
        smpl_robot.load_from_skeleton(betas=torch.from_numpy(gender_beta[None, 1:]), gender=gender_beta[0:1], objs_info=None)
        test_good = f"/tmp/smpl/test_good.xml"
        smpl_robot.write_xml(test_good)
        sk_tree = SkeletonTree.from_mjcf(test_good)
        num_motions = self.num_envs
        skeleton_trees = [sk_tree] * num_motions
        start_idx = 0
        
        motion_lib = MotionLibSMPL(motion_lib_cfg)
        motion_lib.load_motions(skeleton_trees=skeleton_trees, 
                                gender_betas=[torch.from_numpy(gender_beta)] * num_motions,
                                limb_weights=[np.zeros(10)] * num_motions,
                                random_sample=False,
                                start_idx = start_idx, 
                                max_len=-1)
        self._motion_lib = motion_lib
        self._motion_id, self._motion_time = torch.arange(self.num_envs).to(self.device).long(), torch.zeros(self.num_envs).to(self.device).float()
        
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions
        
        # robot_root = self._robot.data.root_pos_w
        # eyes = robot_root + torch.tensor([0, -4, 0]).to(robot_root)
        # targets = robot_root
        # self._tiled_camera.set_world_poses_from_view(eyes, targets)
        
        # head_pos = self._robot.data.body_pos_w[:, self._robot.data.body_names.index("Head")]
        # head_rot = self._robot.data.body_quat_w[:, self._robot.data.body_names.index("Head")]
        # eye_offset = torch.tensor([[0.0, 0.075, 0.1]] * self.num_envs, device=head_rot.device)
        
        # eye_pos = head_pos + lab_math_utils.quat_rotate(head_rot, eye_offset)
        # self._tiled_camera.set_world_poses(eye_pos, head_rot)
        pass
    
    def _post_physics_step(self) -> None:
        
        
        pass
        
    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.actions, joint_ids=None)
        
    def _get_observations(self) -> dict:
        
        self._motion_time = (self.episode_length_buf + 1) * self.step_dt
        
        motion_res = self._motion_lib.get_motion_state(self._motion_id, self._motion_time)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                        motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        body_pos = self._robot.data.body_pos_w[:, self.sim_to_gym_body ]
        body_rot = wxyz_to_xyzw(self._robot.data.body_quat_w[:, self.sim_to_gym_body])
        body_vel = self._robot.data.body_lin_vel_w[:, self.sim_to_gym_body]
        body_ang_vel = self._robot.data.body_ang_vel_w[:, self.sim_to_gym_body]
        
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]
        body_pos_subset = body_pos
        body_rot_subset = body_rot
        body_vel_subset = body_vel
        body_ang_vel_subset = body_ang_vel
        ref_rb_pos_subset = ref_rb_pos
        ref_rb_rot_subset = ref_rb_rot
        ref_body_vel_subset = ref_body_vel
        ref_body_ang_vel_subset = ref_body_ang_vel
        ref_joint_pos = ref_dof_pos[:, self.gym_to_sim_dof]
        ref_joint_vel = ref_dof_vel[:, self.gym_to_sim_dof]
        
        # Data replay
        
        # ref_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
        # robot.write_root_state_to_sim(ref_root_state, env_ids)
        # robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, None, env_ids)
        
        self_obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, ref_smpl_params, ref_limb_weights, True, True, self._has_upright_start, False, False)
        task_obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 1, self._has_upright_start)

        return {
            "self_obs": self_obs,
            "task_obs": task_obs,
        }
    
    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs)
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(self.num_envs), torch.zeros(self.num_envs)
    
    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._motion_time[env_ids] = 0
        
        motion_res = self._motion_lib.get_motion_state(self._motion_id, self._motion_time)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                        motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
        init_joint_pos = ref_dof_pos[:, self.gym_to_sim_dof]
        init_joint_vel = ref_dof_vel[:, self.gym_to_sim_dof]
        
        # ref_joint_pos = torch.from_numpy(sRot.from_rotvec(ref_joint_pos.reshape(-1, 3)).as_euler("xyz")).to(ref_joint_pos).reshape(scene.num_envs, -1)
        
        init_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
        self._robot.write_root_state_to_sim(init_root_state, env_ids)
        self._robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, None, env_ids)
    
    
def main():
    """Main function."""
    env_cfg = SMPLEnvCfg()
    env = SMPLEnv(env_cfg)
    
    
    device = env.device
    if env.humanoid_type == "smplx":
        policy_path = "output/HumanoidIm/phc_x_pnn/Humanoid.pth"
    else:
        policy_path = "output/HumanoidIm/phc_3/Humanoid.pth"
    
    policy_path = "output/HumanoidIm/phc_x_pnn/Humanoid.pth"
    check_points = [torch_ext.load_checkpoint(policy_path)]
    pnn = load_pnn(check_points[0], num_prim = 3, has_lateral = False, activation = "silu", device = device)
    running_mean, running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
    if env.humanoid_type == "smplx":
        action_offset = joblib.load("data/action_offset_smplx.pkl")
    else:
        action_offset = joblib.load("data/action_offset_smpl.pkl")
        
    pd_action_offset = action_offset[0]
    pd_action_scale = action_offset[1]
    
    writer = imageio.get_writer('tiled_camera_output.mp4', fps=30)
    time = 0 
    obs_dict, extras = env.reset()
    while True:
        
        #### Test Rendering #####
        # camera_data = env.scene['tiled_camera'].data.output['rgb']
        # camera_data = (camera_data).byte().cpu().numpy()
        # batch_size = camera_data.shape[0]
        # grid_size = int(np.ceil(np.sqrt(batch_size)))
        # frame_height, frame_width = camera_data.shape[1], camera_data.shape[2]
        # collage = np.zeros((grid_size * frame_height, grid_size * frame_width, 3), dtype=np.uint8)
        # for i in range(batch_size):
        #     row = i // grid_size
        #     col = i % grid_size
        #     collage[row * frame_height:(row + 1) * frame_height, col * frame_width:(col + 1) * frame_width, :] = camera_data[i]
        # frame = collage
        
        # writer.append_data(frame)
        
        # time += 1
        # if time > 200:
        #     import ipdb; ipdb.set_trace()
        #     writer.close()
        #     break
        #### Test Rendering #####
        
        self_obs, task_obs = obs_dict["self_obs"], obs_dict["task_obs"]
        full_obs = torch.cat([self_obs, task_obs], dim = -1)
        full_obs = ((full_obs - running_mean.float()) / torch.sqrt(running_var.float() + 1e-05))
        full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)
        
        with torch.no_grad():
            actions, _ = pnn(full_obs, idx=0)
            actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
            actions = actions * pd_action_scale + pd_action_offset
            actions = actions[:, env.gym_to_sim_dof]
        
        obs_dict, _, _, _, _ = env.step(actions)
    


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()