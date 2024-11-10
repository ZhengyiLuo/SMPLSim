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
from phc.utils.rotation_conversions import xyzw_to_wxyz, wxyz_to_xyzw
import sys

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
# args_cli = parser.parse_args()
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext, PhysxCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.devices.keyboard.se2_keyboard import Se2Keyboard
import carb
##
# Pre-defined configs
##
# from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from isaaclab.smpl_config import SMPL_CFG
from isaaclab.atlas_21jts_config import Atlas_21jts_CFG


from phc.utils.flags import flags
from phc.utils.motion_lib_real import MotionLibReal
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from phc.learning.network_loader import load_z_encoder, load_z_decoder, load_pnn, load_mcp_mlp
from phc.env.tasks.humanoid_funcs import compute_humanoid_observations_smpl_max, compute_imitation_observations_v6
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.players import rescale_actions
import torch
import joblib
from easydict import EasyDict
import numpy as np
import open3d as o3d
import copy
from scipy.spatial.transform import Rotation as sRot
import time
from omegaconf import DictConfig, OmegaConf
from multiprocessing import Pool
import hydra
from phc.utils.torch_humanoid_batch import Humanoid_Batch

@configclass
class SMPLSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = Atlas_21jts_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(cfg, sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    def test_keyboard():
        import ipdb; ipdb.set_trace()
        print('....')
    
    keyboard = Se2Keyboard()
    keyboard.add_callback(carb.input.KeyboardInput.W, test_keyboard)
    
    def reset_robot(robot):
        robot.write_root_state_to_sim(init_root_state, env_ids)
        robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, None, env_ids)
    
    humanoid_fk = Humanoid_Batch(cfg.robot)
    
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    motion_file = "data/e_atlas_nohand/v1/amass_phc.pkl"
    dt = 1/30
    time_steps = 1
    has_upright_start = True
    motion_lib_cfg = EasyDict({
                        "has_upright_start": has_upright_start,
                        "motion_file": motion_file,
                        "fix_height": FixHeightMode.full_fix,
                        "min_length": -1,
                        "max_length": 3000,
                        "im_eval": flags.im_eval,
                        "multi_thread": False ,
                        "randomrize_heading": True,
                        "device": device,
                        "robot": cfg.robot,
                    })
    
        
    gender_beta = np.zeros((17))
    sk_tree = SkeletonTree.from_mjcf(cfg.robot.asset.assetFileName)
    num_motions = 10
    skeleton_trees = [sk_tree] * num_motions
    start_idx = 0
    motion_lib = MotionLibReal(motion_lib_cfg)
    motion_lib.load_motions(skeleton_trees=skeleton_trees, 
                            gender_betas=[torch.from_numpy(gender_beta)] * num_motions,
                            limb_weights=[np.zeros(10)] * num_motions,
                            random_sample=False,
                            start_idx = start_idx, 
                            max_len=-1)
    motion_id, time_steps = torch.zeros(1).to(device).long(), torch.zeros(1).to(device).float()
    motion_id[:] = 2
    motion_len = motion_lib.get_motion_length(motion_id).item()
    motion_time = time_steps % motion_len

    policy_path = "output/HumanoidIm/eatlas_phc_1002/Humanoid.pth"
    check_points = [torch_ext.load_checkpoint(policy_path)]
    mlp = load_mcp_mlp(check_points[0],  activation = "silu", device = device)
    running_mean, running_var = check_points[-1]['running_mean_std']['running_mean'], check_points[-1]['running_mean_std']['running_var']
    action_offset = joblib.load("atlas_pd_action_offset_scale.pkl")
    pd_action_offset = action_offset[0]
    pd_action_scale = action_offset[1]
    #######
    
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    
    ###########################################################
    sim_joint_names = robot.data.joint_names
    sim_body_names = robot.data.body_names
    body_names_gym = cfg.robot.body_names
    dof_names_gym = ['larm_shy', 'larm_shx', 'larm_shz', 'larm_el', 'lleg_hpy',
       'lleg_hpx', 'lleg_hpz', 'lleg_kn', 'lleg_aky', 'lleg_akx',
       'rarm_shy', 'rarm_shx', 'rarm_shz', 'rarm_el', 'rleg_hpy',
       'rleg_hpx', 'rleg_hpz', 'rleg_kn', 'rleg_aky', 'rleg_akx']
    
    sim_to_gym_body = [sim_body_names.index(n) for n in body_names_gym]
    sim_to_gym_dof = [sim_joint_names.index(n) for n in dof_names_gym]
    
    gym_to_sim_body = [body_names_gym.index(n) for n in sim_body_names]
    gym_to_sim_dof = [dof_names_gym.index(n) for n in sim_joint_names]
    
    motion_res = motion_lib.get_motion_state(motion_id, motion_time)
    ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                    motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                    motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]
    init_joint_pos = ref_dof_pos[:, gym_to_sim_dof]
    init_joint_vel = ref_dof_vel[:, gym_to_sim_dof]
    
    # ref_joint_pos = torch.from_numpy(sRot.from_rotvec(ref_joint_pos.reshape(-1, 3)).as_euler("xyz")).to(ref_joint_pos).reshape(scene.num_envs, -1)
    env_ids = torch.zeros(scene.num_envs).to(device).long()
    
    init_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
    robot.write_root_state_to_sim(init_root_state, env_ids)
    robot.write_joint_state_to_sim(init_joint_pos, init_joint_vel, None, env_ids)
    time_step = 3
    
    # Simulation loop
    while simulation_app.is_running():
        start_time = time.time()
        # Apply random action
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)
        
        ############################## PHC ########################################
        
        motion_time = time_steps % motion_len
        time_internals = torch.arange(time_step).to(device).repeat(scene.num_envs).view(-1, time_step) * 1/30
        motion_time_steps = motion_time + time_internals 
        env_ids_steps = motion_id.repeat_interleave(time_step)
        # motion_time_steps = motion_time
        # env_ids_steps = motion_id
            
        motion_res = motion_lib.get_motion_state(env_ids_steps, motion_time_steps)
        ref_root_pos, ref_root_rot, ref_dof_pos, ref_root_vel, ref_root_ang_vel, ref_dof_vel, ref_smpl_params, ref_limb_weights, ref_pose_aa, ref_rb_pos, ref_rb_rot, ref_body_vel, ref_body_ang_vel = \
                        motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
                        motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]

        body_pos = robot.data.body_pos_w[:, sim_to_gym_body ]
        body_rot = wxyz_to_xyzw(robot.data.body_quat_w[:, sim_to_gym_body])
        body_vel = robot.data.body_lin_vel_w[:, sim_to_gym_body]
        body_ang_vel = robot.data.body_ang_vel_w[:, sim_to_gym_body]
        
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
        
        # Data replay
        # ref_joint_pos = ref_dof_pos[..., gym_to_sim_dof]
        # ref_joint_vel = ref_dof_vel[..., gym_to_sim_dof]
        # ref_root_state = torch.cat([ref_root_pos, xyzw_to_wxyz(ref_root_rot), ref_root_vel, ref_root_ang_vel], dim = -1)
        # robot.write_root_state_to_sim(ref_root_state, env_ids)
        # robot.write_joint_state_to_sim(ref_joint_pos, ref_joint_vel, None, env_ids)
        
        self_obs = compute_humanoid_observations_smpl_max(body_pos, body_rot, body_vel, body_ang_vel, ref_smpl_params, ref_limb_weights, True, False, has_upright_start, False, False)
        task_obs = compute_imitation_observations_v6(root_pos, root_rot, body_pos_subset, body_rot_subset, body_vel_subset, body_ang_vel_subset, ref_rb_pos_subset, ref_rb_rot_subset, ref_body_vel_subset, ref_body_ang_vel_subset, 3, has_upright_start)

        full_obs = torch.cat([self_obs, task_obs], dim = -1)
        full_obs = ((full_obs - running_mean.float()) / torch.sqrt(running_var.float() + 1e-05))
        full_obs = torch.clamp(full_obs, min=-5.0, max=5.0)

        with torch.no_grad():
            actions = mlp(full_obs)
            actions = rescale_actions(-1, 1, torch.clamp(actions, -1, 1))
            actions = actions * pd_action_scale + pd_action_offset
            actions = actions[:, gym_to_sim_dof]
            actions[:] = 0
        
        robot.set_joint_position_target(actions, None, env_ids)
        
        ############################## PHC ########################################
        
        if time_steps > motion_len:
            print("resetting")
            reset_robot(robot)
            time_steps[:] = 0
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # sim.render()
        # Increment counter
        time_steps += sim_dt
        # Update buffers
        scene.update(sim_dt)
        
        # Measure wall time and ensure 30 fps
        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / 30.0) - elapsed_time)
        # time.sleep(sleep_time)

@hydra.main(version_base=None, config_path="../phc/data/cfg", config_name="config")
def main(cfg : DictConfig) -> None:
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(
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
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_cfg = SMPLSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(cfg, sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()