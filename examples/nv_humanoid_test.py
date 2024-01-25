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
import isaacgym

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
from smpl_sim.envs.nv.humanoid import Humanoid
from smpl_sim.envs.nv.gymwrapper import GymVectEnv
import time
from smpl_sim.envs.nv.utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
import torch

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False, 
    'debug': False,
    "real_traj": False,
    "im_eval": False,
    })


def process_config(args):

    cfg_env_name = args.cfg_env.split("/")[-1].split(".")[0]
    args.logdir = args.network_path
    cfg, cfg_train, logdir = load_cfg(args)
    flags.debug, flags.follow, flags.fixed, flags.divide_group, flags.no_collision_check, flags.fixed_path, flags.real_path, flags.small_terrain, flags.show_traj, flags.server_mode, flags.slow, flags.real_traj, flags.im_eval, flags.no_virtual_display, flags.render_o3d = \
        args.debug, args.follow, False, False, False, False, False, args.small_terrain, True, args.server_mode, False, False, args.im_eval, args.no_virtual_display, args.render_o3d

    flags.add_proj = args.add_proj
    flags.has_eval = args.has_eval
    flags.trigger_input = False
    flags.demo = args.demo

    assert not args.server_mode
    if args.server_mode:
        flags.follow = args.follow = True
        flags.fixed = args.fixed = True
        flags.no_collision_check = True
        flags.show_traj = True
        cfg['env']['episodeLength'] = 99999999999999
    
    if args.test and not flags.small_terrain:
        cfg['env']['episodeLength'] = 99999999999999

    if args.real_traj:
        cfg['env']['episodeLength'] = 99999999999999
        flags.real_traj = True


    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.test and not flags.small_terrain:
        cfg['env']['episodeLength'] = 99999999999999

    if args.real_traj:
        cfg['env']['episodeLength'] = 99999999999999
        flags.real_traj = True


    cfg_train['params']['seed'] = set_seed(cfg_train['params'].get("seed", -1), cfg_train['params'].get("torch_deterministic", False))

    if args.horovod:
        cfg_train['params']['config']['multi_gpu'] = args.horovod

    if args.horizon_length != -1:
        cfg_train['params']['config']['horizon_length'] = args.horizon_length

    if args.minibatch_size != -1:
        cfg_train['params']['config']['minibatch_size'] = args.minibatch_size

    if args.motion_file:
        cfg['env']['motion_file'] = args.motion_file
    flags.test = args.test

    # Create default directories for weights and statistics
    cfg_train['params']['config']['network_path'] = args.network_path
    args.log_path = osp.join(args.log_path, cfg['name'], cfg_env_name)
    cfg_train['params']['config']['log_path'] = args.log_path
    cfg_train['params']['config']['train_dir'] = args.log_path

    cfg["flags"] = flags.__dict__
    cfg["data"] = {}
    cfg["data"]["tmp_out_dir"] = osp.abspath("tmp") # please provide an absolute path

    # relative path data: is relative to the root of the nvig_humanoid module
    # you can also specify an absolute path
    cfg["data"]["amass_isaac_gender_betas"] = "sample_data/amass_isaac_gender_betas.pkl"
    cfg["data"]["amass_isaac_gender_betas_unique"] = "sample_data/amass_isaac_gender_betas_unique.pkl"
    cfg["data"]["smpl_template_file"] = "smpl_sim/data/assets/mjcf/humanoid_template_local.xml"
    cfg["data"]["smpl_folder"] = "data/smpl"
    return args, cfg, cfg_train

if __name__ == "__main__":
    droot = ""
    args = get_args()
    args.task = "Humanoid"
    args.cfg_env = osp.join(droot,"smpl_sim/envs/nv/data/cfg/phc_kp_mcp_iccv.yaml")
    args.cfg_train = osp.join(droot,"smpl_sim/envs/nv/data/cfg/train/rlg/im_mcp.yaml")
    # args.motion_file = osp.join(droot,"sample_data/amass_isaac_standing_upright_slim.pkl")
    # args.motion_file = osp.join(droot,"sample_data/amass_isaac_standing_upright_slim.pkl")
    args.num_envs = 10
    args.headless = False

    # get complete configurations
    args, cfg, cfg_train = process_config(args)

    multi_gpu = cfg_train['params']['config'].get('multi_gpu', False)
    assert not multi_gpu
    
    # get simulator params
    sim_params = parse_sim_params(args, cfg, None, cfg["flags"])

    # create tasks
    device_id = args.device_id
    rl_device = args.rl_device
    cfg["seed"] = cfg_train.get("seed", -1)
    env = eval(args.task)(cfg=cfg, sim_params=sim_params, physics_engine=args.physics_engine, device_type=args.device, device_id=device_id, headless=args.headless)
    genv = GymVectEnv(env)

    render_mode = "rgb_array"
    # render_mode = "human"

    obs, info = genv.reset()
    img = genv.render(render_mode)
    media.write_image("image0.png", img)

    cur_t, T = 0, 0
    max_T = np.inf
    if render_mode == "rgb_array":
        max_T = 1
    frames = []
    # env.robot.write_xml("smpl_humanoid.xml")
    while True:
        # action = np.random.random(env.mj_data.ctrl.shape[0]) 
        
        # action /= np.abs(action.max())
        # print(action)
        
        action = torch.tensor(genv.action_space.sample()).to(genv._env.device)
        # action[:] = 1
        cur_t += 1
        if cur_t % 90 == 0:
            T += 1
            
        env.step(action)
        img = genv.render(render_mode)
        frames.append(img)

        if T == max_T:
            genv.close()
            media.write_video("video.mp4", frames, fps=10)
            break
        