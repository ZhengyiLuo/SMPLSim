import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
import time
# os.environ["MUJOCO_GL"] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import joblib
import numpy as np
import imageio

### Mujoco-related imports
import mujoco
print(f"mujoco.__version__: {mujoco.__version__}")
### SMPL
from smpl_sim.envs.humanoid_env import HumanoidEnv
import yaml
try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
import hydra
from scipy.spatial.transform import Rotation as sRot
from omegaconf import DictConfig, OmegaConf
from smpl_sim.smpllib.motion_lib_smpl import MotionLibSMPL
import torch
from easydict import EasyDict
from smpl_sim.smpllib.motion_lib_base import FixHeightMode

@hydra.main(version_base=None, config_path=str(files('smpl_sim').joinpath('data/cfg')), config_name="config")
def main(cfg : DictConfig) -> None:
    # motions = joblib.load("sample_data/amass_isaac_standing_upright_slim.pkl")
    device = torch.device("cpu")
    motion_file = "data/amass/singles/0-KIT_3_walking_slow08_poses.pkl"
    # motion_file = "data/amass/singles/0-SSM_synced_20160930_50032_punch_kick_sync_poses.pkl"
    motion_lib_cfg = EasyDict({
        "motion_file": motion_file,
        "device": device,
        "fix_height": FixHeightMode.full_fix,
        "min_length": -1,
        "max_length": -1,
        "im_eval": False,
        "multi_thread": False,
        "smpl_type": "smpl",
        "randomrize_heading": True,
    })
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    shape_params = np.zeros(17)
    motion_lib.load_motions(m_cfg = motion_lib_cfg, shape_params = [shape_params] )
    
    env = HumanoidEnv(cfg)
    env.reset()
    cur_t, T = 0, 0
    while True:
        motion_ids = np.array([0])
        motion_times = np.array([(cur_t % motion_lib.get_motion_num_steps()[0]) * 1/30])
        motion_return = motion_lib.get_motion_state_intervaled(motion_ids, motion_times)
        
        env.mj_data.qpos[:] = motion_return.qpos.flatten()
        mujoco.mj_forward(env.mj_model, env.mj_data)
        cur_t += 1
        env.render()
        
        # action = motion_return.dof_pos.flatten() / np.pi
        # cur_t += 1
        # env.step(action=action)
        # env.render()
        

if __name__ == "__main__":
    main()

 

