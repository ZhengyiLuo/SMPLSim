############################################################################################################
### Functions are from https://github.com/Khrylx/PyTorch-RL
############################################################################################################
import math
from collections import defaultdict
import numpy as np

class LoggerRL:

    def __init__(self):
        self.num_steps = 0
        self.num_episodes = 0
        self.avg_episode_len = 0
        self.total_reward = 0
        self.min_episode_reward = math.inf
        self.max_episode_reward = -math.inf
        self.min_reward = math.inf
        self.max_reward = -math.inf
        self.episode_reward = 0
        self.avg_episode_reward = 0
        self.sample_time = 0
        self.info_dict = defaultdict(list)
        

    def start_episode(self, env):
        self.episode_reward = 0

    def step(self, env, reward, info):
        self.episode_reward += reward
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)
        self.num_steps += 1
        {self.info_dict[k].append(v) for k, v in info.items()}

    def end_episode(self, env):
        self.num_episodes += 1
        self.total_reward += self.episode_reward
        self.min_episode_reward = min(self.min_episode_reward, self.episode_reward)
        self.max_episode_reward = max(self.max_episode_reward, self.episode_reward)

    def end_sampling(self):
        self.avg_episode_len = self.num_steps / self.num_episodes
        self.avg_episode_reward = self.total_reward / self.num_episodes

    @classmethod
    def merge(cls, logger_list):
        logger = cls()
        logger.total_reward = sum([x.total_reward for x in logger_list])
        logger.num_episodes = sum([x.num_episodes for x in logger_list])
        logger.num_steps = sum([x.num_steps for x in logger_list])
        logger.avg_episode_len = logger.num_steps / logger.num_episodes
        logger.avg_episode_reward = logger.total_reward / logger.num_episodes
        logger.max_episode_reward = max([x.max_episode_reward for x in logger_list])
        logger.min_episode_reward = max([x.min_episode_reward for x in logger_list])
        logger.avg_reward = logger.total_reward / logger.num_steps
        logger.max_reward = max([x.max_reward for x in logger_list])
        logger.min_reward = min([x.min_reward for x in logger_list])
        logger.info_dict = {k: np.mean(np.concatenate([np.array(x.info_dict[k]) for x in logger_list])) for k in logger_list[0].info_dict.keys()}
        
        return logger