import time
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MUJOCO_GL"] = 'egl'
# os.environ['PYOPENGL_PLATFORM'] = 'egl'

import numpy as np

### Mujoco-related imports
import mujoco
### SMPL
# from smpl_sim.envs.humanoid_env import HumanoidEnv
from omegaconf import OmegaConf
import gymnasium as gym
from tqdm import tqdm

def conv_time(seconds, new_format):
    assert new_format in ["s", "ms", "us"]
    if new_format == "s":
        return seconds
    elif new_format == "ms":
        return seconds * 1e3
    elif new_format == "us":
        return seconds * 1e6

YAML_CONFIG_STR = """
env:
    task: HumanoidEnv
    note: this is the default config file for humanoid_env
    episode_length: 300
    sim_timestep_inv: 450
    control_frequency_inv: 15
    control_mode: "uhc_pd"
    power_scale: 1.0
    root_height_obs: true
    enable_early_termination: True
    self_obs_v: 1
    kp_scale: 1.0
    kd_scale: 1.0
    cycle_motion: False
    power_reward: True
    clip_actions: True
    contact_bodies: ["R_Ankle", "L_Ankle", "R_Toe", "L_Toe"]
    render_mode: "rgb_array"
    camera: "side"
    state_init: Default
robot:
    note: "Humanoid robot with SMPL body model."
    humanoid_type: smpl
    has_upright_start: False
    has_shape_obs: False
    has_weight_obs: False
    has_shape_variation: False
    has_mesh: False
    replace_feet: True
    has_jt_limit: False
    height_fix_mode: full
    big_ankle: True
    remove_toe: False
    real_weight_porpotion_capsules: True
    real_weight_porpotion_boxes: True
    real_weight: True
    box_body: True
    smpl_data_dir: data/smpl
headless: False
"""

def make_env_mj(cfg_str, num_envs):
    cfg = OmegaConf.create(cfg_str)

    def thunk():
        from smpl_sim.envs.tasks import HumanoidEnv
        env = eval(cfg.env.task)(cfg)
        return env
    
    
    if num_envs > 1:
        env = gym.vector.AsyncVectorEnv(
                [lambda : thunk() for i in range(num_envs)],
                context='spawn'
            )
        # action = np.random.rand(np.prod(env.action_space.shape)).reshape(env.action_space.shape)
    else:
        env = thunk()
    return env

def measure_function(env, func_name, reps=1000, **kwargs):
    times = []
    _f = getattr(env, func_name)
    for i in tqdm(range(reps)):
        beg = time.perf_counter()
        output = _f(**kwargs)
        times.append(time.perf_counter() - beg)
    return np.array(times)
         

def evaluate_env(env, time_format: str = "ms", reps=100):
    num_envs = env.num_envs if hasattr(env, "num_envs") else 1
    
    action = env.action_space.sample()
    metrics = {}

    times = measure_function(env, "reset", reps=reps, seed=54)
    mean, ci = np.mean(times), np.std(times)/np.sqrt(len(times))
    # print(f"Avg reset time: {conv_time(mean, time_format):.3f}{time_format}")
    metrics["reset/avg_time"] =  mean
    
    if num_envs > 1:
        times = measure_function(env, "step", reps=reps, actions=action)
    else:
        times = measure_function(env, "step", reps=reps, action=action)
    mean, ci = np.mean(times), np.std(times)/np.sqrt(len(times))
    # print(f"Avg step time: {conv_time(mean, time_format):.3f}{time_format}")
    metrics["step/avg_time"] =  mean
    metrics["step/sps"] = num_envs * reps / np.sum(times)
    return metrics
    

if __name__ == "__main__":
    time_format = "ms"
    num_envs = 64
    print(f"mujoco.__version__   : {mujoco.__version__}")
    print(f"gymnasium.__version__: {gym.__version__}")

    env = make_env_mj(YAML_CONFIG_STR, 1)
    # print(f"Num envs = 1")
    # m_single = evaluate_env(env, time_format=time_format)
    # env.close()

    env = make_env_mj(YAML_CONFIG_STR, num_envs)
    print(f"Num envs = {num_envs}")
    m_vect = evaluate_env(env, time_format=time_format)
    print(m_vect)
    env.close()

    # for k in m_single.keys():
    #     print(f"{k} -> {m_vect[k]/m_single[k]:.2f}x slowdown")
    #     print(f"{k} -> {m_single[k]/m_vect[k]*num_envs:.2f}x faster")

    
    
