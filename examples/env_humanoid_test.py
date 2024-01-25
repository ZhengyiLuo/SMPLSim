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
# from smpl_sim.envs.humanoid_env import HumanoidEnv
import yaml
try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
import hydra
from omegaconf import DictConfig, OmegaConf
import mediapy as media
from smpl_sim.envs.tasks import *

@hydra.main(version_base=None, config_path=str(files('smpl_sim').joinpath('data/cfg')), config_name="config")
def main(cfg : DictConfig) -> None:
    # env = HumanoidEnv(cfg)
    cfg.env.camera = "back"
    env = eval(cfg.env.task)(cfg)
    print("environment initialized")
    env.reset()
    cur_t, T = 0, 0
    max_T = np.inf
    if cfg.env.render_mode == "rgb_array":
        max_T = 1
    frames = []
    while True:
        action = np.zeros(env.mj_data.ctrl.shape[0]) 
        action[:] = 0
        cur_t += 1
        if cur_t % 90 == 0:
            T += 1
            
        env.step(action=action)
        img = env.render()
        frames.append(img)

        if T == max_T:
            env.close()
            media.write_video("video.mp4", frames, fps=10)
            break
        

if __name__ == "__main__":
    main()

 

