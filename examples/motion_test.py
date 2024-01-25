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

@hydra.main(version_base=None, config_path=str(files('smpl_sim').joinpath('data/cfg')), config_name="config")
def main(cfg : DictConfig) -> None:
    # motions = joblib.load("sample_data/amass_isaac_standing_upright_slim.pkl")
    
    motions = joblib.load("/hdd/zen/data/ActBound/AMASS/amass_copycat_take6_test.pkl")
    data_key = "0-Transitions_mocap_mazen_c3d_JOOF_walkbackwards_poses"
    pose_aa = motions[data_key]['pose_aa'][:, :66]
    root_pos = motions[data_key]['trans']
    pose_aa = np.concatenate([pose_aa, np.zeros((pose_aa.shape[0], 6))], axis=1).reshape(-1, 24, 3)
    pose_aa = pose_aa[:, [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 22, 14, 17, 19, 21, 23]]
    body_pose_aa = pose_aa[:, 1:]
    root_pose_aa = pose_aa[:, 0]
    root_rot = sRot.from_rotvec(root_pose_aa.reshape(-1, 3)).as_quat()[:, [3, 0, 1, 2]]
    body_pos = sRot.from_rotvec(body_pose_aa.reshape(-1, 3)).as_euler("XYZ").reshape(-1, 69)
    
    print(f"number of motions: {len(motions)}")
    env = HumanoidEnv(cfg)
    env.reset()
    cur_t, T = 0, 0
    while True:
        # action = np.random.random(env.mj_data.ctrl.shape[0]) 
        
        # action /= np.abs(action.max())
        # print(action)
        env.mj_data.qpos[:3] = root_pos[cur_t % body_pos.shape[0]] 
        env.mj_data.qpos[3:7] = root_rot[cur_t % body_pos.shape[0]] 
        env.mj_data.qpos[7:] = body_pos[cur_t % body_pos.shape[0]]
        mujoco.mj_forward(env.mj_model, env.mj_data)
        cur_t += 1
        env.render()
        
        # action = body_pos[cur_t % body_pos.shape[0]] / np.pi
        # cur_t += 1
        # env.step(action=action)
        # env.render()
        

if __name__ == "__main__":
    main()

 

