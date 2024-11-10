# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a simple Cartpole robot."""


import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
 

Atlas_21jts_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"data/assets/usd/Atlas/atlas_21jts.usda",
        
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
      "legs": ImplicitActuatorCfg(
            joint_names_expr=["lleg_hpy", "rleg_hpy", "lleg_hpx", "rleg_hpx", "lleg_hpz", "rleg_hpz", "lleg_kn", "rleg_kn", "lleg_aky", "rleg_aky", "lleg_akx", "rleg_akx"],
            stiffness={
                "lleg_hpy": 500,
                "rleg_hpy": 500,
                "lleg_hpx": 500,
                "rleg_hpx": 500,
                "lleg_hpz": 500,
                "rleg_hpz": 500,
                "lleg_kn": 500,
                "rleg_kn": 500,
                "lleg_aky": 500,
                "rleg_aky": 500,
                "lleg_akx": 500,
                "rleg_akx": 500,
            },
            damping={
                "lleg_hpy": 50,
                "rleg_hpy": 50,
                "lleg_hpx": 50,
                "rleg_hpx": 50,
                "lleg_hpz": 50,
                "rleg_hpz": 50,
                "lleg_kn": 50,
                "rleg_kn": 50,
                "lleg_aky": 50,
                "rleg_aky": 50,
                "lleg_akx": 50,
                "rleg_akx": 50,
            },
        ),
        
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["larm_shy", "rarm_shy", "larm_shx", "rarm_shx", "larm_shz", "rarm_shz", "larm_el", "rarm_el"],
            stiffness={
                "larm_shy": 500,
                "rarm_shy": 500,
                "larm_shx": 500,
                "rarm_shx": 500,
                "larm_shz": 500,
                "rarm_shz": 500,
                "larm_el": 500,
                "rarm_el": 500,
            },
            damping={
                "larm_shy": 50,
                "rarm_shy": 50,
                "larm_shx": 50,
                "rarm_shx": 50,
                "larm_shz": 50,
                "rarm_shz": 50,
                "larm_el": 50,
                "rarm_el": 50,
            },
        )
    },
)