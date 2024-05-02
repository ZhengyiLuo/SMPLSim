import math
import time
import os
import torch

os.environ["OMP_NUM_THREADS"] = "1"
import joblib
import pickle
from collections import defaultdict
import glob
import os
import sys
import os.path as osp
from tqdm import tqdm
import wandb
import numpy as np
import multiprocessing

from smpl_sim.agents.agent_ppo import AgentPPO
from smpl_sim.learning.memory import Memory
from smpl_sim.learning.policy_gaussian import PolicyGaussian
from smpl_sim.learning.critic import Value
from smpl_sim.learning.policy_mcp import PolicyMCP
from smpl_sim.learning.mlp import MLP
from smpl_sim.learning.logger_txt import create_logger
from smpl_sim.utils.flags import flags
from smpl_sim.learning.learning_utils import to_test, to_device, to_cpu, get_optimizer
from smpl_sim.envs.tasks import *


class AgentHumanoid(AgentPPO):

    def __init__(self, cfg, dtype, device, training=True, checkpoint_epoch=0):
        self.cfg = cfg
        self.cc_cfg = cfg
        self.device = device
        self.dtype = dtype
        self.training = training
        self.max_freq = 50

        self.setup_vars()
        self.setup_data_loader()
        self.setup_env()
        self.setup_policy()
        self.setup_value()
        self.setup_optimizer()
        self.setup_logging()
        self.seed(cfg.seed)
        self.print_config()
        self.load_checkpoint(checkpoint_epoch)
            

        super().__init__(
            env=self.env,
            dtype=dtype,
            device=device,
            mean_action=cfg.test,
            headless=not cfg.headless,
            num_threads=cfg.num_threads,
            policy_net=self.policy_net,
            value_net=self.value_net,
            optimizer_policy=self.optimizer_policy,
            optimizer_value=self.optimizer_value,
            opt_num_epochs=cfg.learning.opt_num_epochs,
            gamma=cfg.learning.gamma,
            tau=cfg.learning.tau,
            clip_epsilon=cfg.learning.clip_epsilon,
            policy_grad_clip=[(self.policy_net, cfg.learning.policy_grad_clip)],
            use_mini_batch=False,
            mini_batch_size=0,
            clip_obs=cfg.learning.clip_obs,
            clip_obs_range = cfg.learning.clip_obs_range,
            clip_actions = cfg.env.clip_actions,
        )

    def setup_vars(self):
        self.epoch = 0
        

    def print_config(self):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        self.logger_txt.info("==========================Agent Humanoid===========================")
        self.logger_txt.info(f"self_obs_v: {self.cfg.env.self_obs_v}")
        self.logger_txt.info(f"humanoid_type: {self.cfg.robot.humanoid_type}")
        self.logger_txt.info(f"State_dim: {self.state_dim}")
        self.logger_txt.info("============================================================")

    def setup_data_loader(self):
        pass

    def setup_env(self):
        self.env = eval(self.cfg.env.task)(self.cfg)

    def setup_policy(self):
        actuators = self.env.actuator_names
        self.state_dim = state_dim = self.env.observation_space.shape[0]
        self.action_dim = action_dim = self.env.action_space.shape[0]
        """define actor and critic"""
        if self.cfg.learning.actor_type == "gauss":
            self.policy_net = PolicyGaussian(self.cfg, action_dim=action_dim, state_dim=state_dim)
        to_device(self.device, self.policy_net)

    def setup_value(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        self.value_net = Value(MLP(state_dim, self.cfg.learning.mlp.units, self.cfg.learning.mlp.activation))
        to_device(self.device, self.value_net)

    def setup_optimizer(self):
        self.optimizer_policy = get_optimizer(self.policy_net, self.cfg.learning.policy_lr, self.cfg.learning.policy_weightdecay, self.cfg.learning.policy_optimizer)
        self.optimizer_value =  get_optimizer(self.value_net,  self.cfg.learning.value_lr, self.cfg.learning.value_weightdecay, self.cfg.learning.value_optimizer,)
            
    def get_nn_weights(self):
        state = {}
        state['policy'] = self.policy_net.state_dict()
        state['value'] = self.value_net.state_dict()
        return state

    def set_nn_weights(self, weights):
        self.policy_net.load_state_dict(weights['policy'])
        self.value_net.load_state_dict(weights['value'])
            
    def get_full_state_weights(self):
        state = self.get_nn_weights()
        state['epoch'] = self.epoch
        state['optimizer_policy'] = self.optimizer_policy.state_dict()
        state['optimizer_value'] = self.optimizer_value.state_dict()
        state['frame'] = self.num_steps
        return state
    
    def set_full_state_weights(self, state):
        self.set_nn_weights(state)
        self.epoch = state['epoch']
        self.optimizer_value.load_state_dict(state['optimizer_value'])
        self.optimizer_policy.load_state_dict(state['optimizer_policy'])
        self.num_steps = state.get('frame', 0)
        print(f"==============================Loading checkpoint model: Epoch {self.epoch}==============================")


    def save_checkpoint(self):
        print(f"==============================Saving checkpoint model: Epoch {self.epoch}==============================")
        torch.save(self.get_full_state_weights(), f"{self.cfg.output_dir}/Humanoid_{self.epoch:08d}.pth")
        

    def save_curr(self):
        print(f"==============================Saving current model: Epoch {self.epoch}==============================")
        torch.save(self.get_full_state_weights(), f"{self.cfg.output_dir}/Humanoid.pth")

    def load_checkpoint(self, epoch):
        if epoch == -1:
            state = torch.load(f"{self.cfg.output_dir}/Humanoid.pth")
            self.set_full_state_weights(state)
        elif epoch > 0:
            state = torch.load(f"{self.cfg.output_dir}/Humanoid_{epoch:08d}.pth")
            self.set_full_state_weights(state)
        else:
            pass
            
        
        to_device(self.device, self.policy_net, self.value_net)

    def setup_logging(self):
        self.logger_txt = create_logger(os.path.join(self.cfg.output_dir, "log.txt"))

    def pre_epoch(self):
        return
    
    def after_epoch(self):
        return 

    def log_train(self, info):
        """logging"""
        loggers = info["loggers"]
        self.num_steps += loggers.num_steps
        reward_str = " ".join([f"{v:.3f}" for k, v in loggers.info_dict.items()])
        
        log_str = f"Ep: {self.epoch} \t {self.cfg.exp_name} T_s {info['T_sample']:.2f}  T_u { info['T_update']:.2f} \t eps_R_avg {loggers.avg_episode_reward:.4f} R_avg {loggers.avg_reward:.4f} R_range ({loggers.min_reward:.4f}, {loggers.max_reward:.4f}) [{reward_str}] \t num_s { self.num_steps} eps_len {loggers.avg_episode_len:.2f}"

        self.logger_txt.info(log_str)
        

        if not self.cfg.no_log:
            log_data = {
                    "avg_episode_reward": loggers.avg_episode_reward,
                    "eps_len": loggers.avg_episode_len,
                    "avg_rwd": loggers.avg_reward,
                    "reward_raw": loggers.info_dict,
                }
            
            if "log_eval" in info:
                log_data.update(info["log_eval"])
            
            wandb.log(data=log_data, step=self.epoch)


    def optimize_policy(self,save_model=True):
        starting_epoch = self.epoch
        for _ in range(starting_epoch, self.cfg.learning.max_epoch):
            info = {}
            t0 = time.time()
            self.pre_epoch()
            batch, loggers = self.sample(self.cfg.learning.min_batch_size)

            """update networks"""
            t1 = time.time()
            self.update_params(batch)
            
            self.epoch += 1
            
            if save_model and (self.epoch) % self.cfg.learning.save_frequency == 0:
                self.save_checkpoint()
                log_eval = self.eval_policy()
                info["log_eval"] = log_eval
            elif save_model and (self.epoch) % self.cfg.learning.save_curr_frequency == 0:
                self.save_curr()
            t2 = time.time()
            info.update({
                "loggers": loggers,
                "T_sample": t1 - t0,
                "T_update": t2 - t1,
                "T_total": t2 - t0,
            })
                
            self.log_train(info)
            self.after_epoch()

    def eval_policy(self, epoch=0, dump=False):
        res_dicts = {}
        return res_dicts
    
    
    def run_policy(self, epoch=0, dump=False):
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                while True:
                    obs_dict, info = self.env.reset()
                    state = self.preprocess_obs(obs_dict)
                    for t in range(10000):
                        actions = self.policy_net.select_action(torch.from_numpy(state).to(self.dtype), True)[0].numpy()
                        
                        next_obs, reward, terminated, truncated, info = self.env.step(self.preprocess_actions(actions))
                        next_state = self.preprocess_obs(next_obs)
                        done = terminated or truncated

                        if done:
                            print(f"Episode finished after {t + 1} timesteps")
                            break 
                        state = next_state
        res_dicts = {}
        return res_dicts

    def seed(self, seed):
        cfg, device, dtype = self.cfg, self.device, self.dtype
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.env.seed(seed)

