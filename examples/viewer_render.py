import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())
# os.environ["MUJOCO_GL"] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import joblib
import numpy as np
import imageio

### Mujoco-related imports
import mujoco
print(f"mujoco.__version__: {mujoco.__version__}")
### SMPL
import smpl_sim.envs as mjhum

motions = joblib.load("sample_data/amass_isaac_standing_upright_slim.pkl")
print(f"number of motions: {len(motions)}")

data_dir = "data/smpl"
env = mjhum.SMPLHumanoidMove(
    motions=motions,
    pid_controlled=False,
    data_dir=data_dir,
    max_episode_length=500,
    move_speed=0,
    initial_position="random"
)
env.reset()
 

while True:
    action = np.zeros(env.mj_data.ctrl.shape[0])
    # action[50] = 1
    env.step(action=action)
    env.viewer_render()
