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
from phc.utils.rotation_conversions import xyzw_to_wxyz

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_FINISHED = 3

##
# Pre-defined configs
##
# from omni.isaac.lab_assets import CARTPOLE_CFG  # isort:skip
from isaaclab.smpl_config import SMPL_CFG, SMPLX_CFG
from isaaclab.atlas_21jts_config import Atlas_21jts_CFG


from phc.utils.flags import flags
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from phc.utils.motion_lib_base import FixHeightMode
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPLX_BONE_ORDER_NAMES, SMPLH_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES, SMPLH_MUJOCO_NAMES
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
    # robot: ArticulationCfg = Atlas_21jts_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot: ArticulationCfg = SMPLX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    robot = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    env_ids = torch.zeros(scene.num_envs).to(sim.device).long()
    
    current_dof = 0
    anim_state = ANIM_SEEK_LOWER
    anim_timer = 0.0
    anim_duration = 4.0  # Duration to animate each joint
    incremental = 0.1

    
    # Simulation loop
    
    sim_dof_names = robot.data.joint_names
    sim_body_names = robot.data.body_names
    ref_root_state = torch.zeros((scene.num_envs, 13)).to(sim.device)
    joint_pos = torch.zeros((scene.num_envs, len(sim_dof_names))).to(sim.device)
    joint_vel = torch.zeros((scene.num_envs, len(sim_dof_names))).to(sim.device)
    ref_root_state[:, 2] = 0.9
    ref_root_state[:, 3] = 1.3
    
    while simulation_app.is_running():
        # Apply random action
        # -- generate random joint efforts
        # efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        # -- apply action to the robot
        # robot.set_joint_effort_target(efforts)
        
        ############################## PHC ########################################
        current_dof_val = joint_pos[:, current_dof].clone()  # Preserve current_dof position
        joint_pos.zero_()  # Efficiently set all positions to zero
        joint_pos[:, current_dof] = current_dof_val  # Restore current_dof position
        if anim_state == ANIM_SEEK_LOWER:
            joint_pos[:, current_dof] -= incremental  # Adjust the increment as needed
            if joint_pos[:, current_dof] <= -np.pi:
                joint_pos[:, current_dof] = -np.pi
                anim_state = ANIM_SEEK_UPPER
        elif anim_state == ANIM_SEEK_UPPER:
            joint_pos[:, current_dof] += incremental
            if joint_pos[:, current_dof] >= np.pi:
                joint_pos[:, current_dof] = np.pi
                anim_state = ANIM_FINISHED
        elif anim_state == ANIM_FINISHED:
            current_dof = (current_dof + 1) % len(sim_dof_names)
            anim_state = ANIM_SEEK_LOWER
        print("Animating DOF %d ('%s')" % (current_dof, sim_dof_names[current_dof]), anim_timer)
        # Update the timer
        anim_timer += sim_dt
        if anim_timer >= anim_duration:
            anim_timer = 0.0
            current_dof = (current_dof + 1) % len(sim_dof_names)
            anim_state = ANIM_SEEK_LOWER
            
        robot.write_root_state_to_sim(ref_root_state, env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        
        ############################## PHC ########################################
        
        # -- write data to sim
        scene.write_data_to_sim()
        # Perform step
        # sim.step()
        sim.render()
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
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
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()