import math
import time
import os
import torch
import numpy as np

import multiprocessing

from smpl_sim.learning.memory import Memory
from smpl_sim.learning.trajbatch import TrajBatch
from smpl_sim.learning.logger_rl import LoggerRL
from smpl_sim.learning.learning_utils import to_test, to_cpu, rescale_actions
import random
os.environ["OMP_NUM_THREADS"] = "1"


class Agent:

    def __init__(self, env, policy_net, value_net, dtype, device, gamma,  np_dtype = np.float32,
                 mean_action=False, headless=False, num_threads=1, clip_obs = False, clip_actions = False, clip_obs_range = [-5, 5], env_info = {} ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.dtype = dtype
        self.device = device
        self.np_dtype = np.float32
        self.gamma = gamma
        self.mean_action = mean_action
        self.headless = headless
        self.num_threads = num_threads
        self.noise_rate = 1.0
        self.num_steps = 0
        self.traj_cls = TrajBatch
        self.logger_rl_cls = LoggerRL
        self.sample_modules = [policy_net]
        self.update_modules = [policy_net, value_net]
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions
        self.obs_low = clip_obs_range[0]
        self.obs_high = clip_obs_range[1]
        self._setup_action_space()
        
        
    def _setup_action_space(self):
        action_space = self.env.action_space
        
        self.actions_num = action_space.shape[0]

        # todo introduce device instead of cuda()
        self.actions_low = action_space.low.copy()
        self.actions_high = action_space.high.copy()
        return

    def seed_worker(self, pid):
        if pid > 0:
            seed = random.randint(0, 5000) * pid
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            if hasattr(self.env, 'np_random'):
                self.env.np_random.random(np.random.randint(5000 )* pid)

    def sample_worker(self, pid, queue, min_batch_size):
        self.seed_worker(pid)

        memory = Memory()
        logger = self.logger_rl_cls()
        self.pre_sample()

        while logger.num_steps < min_batch_size:
            
            obs_dict, info = self.env.reset()
            state = self.preprocess_obs(obs_dict) # let's assume that the environment always return a np.ndarray (see https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FlattenObservation)
            logger.start_episode(self.env)

            for t in range(10000):
                mean_action = self.mean_action or self.env.np_random.binomial(1, 1 - self.noise_rate)
                actions = self.policy_net.select_action(torch.from_numpy(state).to(self.dtype), mean_action)[0].numpy()
                next_obs, reward, terminated, truncated, info = self.env.step(self.preprocess_actions(actions)) # action processing should not affect the recorded action
                done = terminated or truncated
                next_state = self.preprocess_obs(next_obs)
                logger.step(self.env, reward, info)

                mask = 0 if done else 1
                exp = 1 - mean_action
                self.push_memory(memory, state.squeeze(), actions, mask, next_state.squeeze(), reward, exp)

                if pid == 0 and not self.headless:
                    self.env.render()
                if done:
                    break
                state = next_state

            
            logger.end_episode(self.env)
        logger.end_sampling()
        
        if queue is not None:
            queue.put([pid, memory, logger])
        else:
            return memory, logger

    def pre_episode(self):
        return

    def push_memory(self, memory, state, action, mask, next_state, reward, exp):
        memory.push(state, action, mask, next_state, reward, exp)

    def pre_sample(self):
        # Function to be called before sampling, unique to each thread.
        return

    def sample(self, min_batch_size):
        t_start = time.time()
        to_test(*self.sample_modules) # Sending test modeuls to cpu!!!
        with to_cpu(*self.sample_modules):
            with torch.no_grad():
                thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
                queue = multiprocessing.Queue()
                memories = [None] * self.num_threads
                loggers = [None] * self.num_threads
                for i in range(self.num_threads-1):
                    worker_args = (i+1, queue, thread_batch_size)
                    worker = multiprocessing.Process(target=self.sample_worker, args=worker_args)
                    worker.start()
                memories[0], loggers[0] = self.sample_worker(0, None, thread_batch_size)

                for i in range(self.num_threads - 1):
                    pid, worker_memory, worker_logger = queue.get()
                    memories[pid] = worker_memory
                    loggers[pid] = worker_logger
                traj_batch = self.traj_cls(memories)
                
                logger = self.logger_rl_cls.merge(loggers)
        
        logger.sample_time = time.time() - t_start
        return traj_batch, logger

    def preprocess_obs(self, obs):
        if self.clip_obs:
            return np.clip(obs.reshape(1, -1), self.obs_low, self.obs_high) 
        else:
            return obs.reshape(-1, 1)

    def preprocess_actions(self, actions):
        actions = int(actions) if self.policy_net.type == 'discrete' else actions.astype(self.np_dtype)
        if self.clip_actions:
            actions = rescale_actions(
                self.actions_low,
                self.actions_high,
                np.clip(actions, self.actions_low, self.actions_high),
            )
        return actions
        
        
    def trans_value(self, states):
        """transform states before going into value net"""
        return states

    def set_noise_rate(self, noise_rate):
        self.noise_rate = noise_rate