import math
import numpy as np
import torch

from smpl_sim.learning.learning_utils import to_train, to_test
from smpl_sim.agents.agent_pg import AgentPG


class AgentPPO(AgentPG):

    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, use_mini_batch=False,
                 policy_grad_clip=None, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.use_mini_batch = use_mini_batch
        self.policy_grad_clip = policy_grad_clip

    def update_policy(self, states, actions, returns, advantages, exps):
        """update policy"""
        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(states, actions)

        for _ in range(self.opt_num_epochs):
            if self.use_mini_batch:
                perm = np.arange(states.shape[0])
                np.random.shuffle(perm)
                perm = torch.LongTensor(perm).to(self.device)

                states, actions, returns, advantages, fixed_log_probs, exps = \
                    states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

                optim_iter_num = int(math.floor(states.shape[0] / self.mini_batch_size))
                for i in range(optim_iter_num):
                    ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))
                    states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                        states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                    ind = exps_b.nonzero(as_tuple=False).squeeze(1)
                    self.update_value(states_b, returns_b)
                    surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                    self.optimizer_policy.zero_grad()
                    surr_loss.backward()
                    self.clip_policy_grad()
                    self.optimizer_policy.step()
            else:
                
                ind = exps.nonzero(as_tuple=False).squeeze(1)
                self.update_value(states, returns)
                surr_loss = self.ppo_loss(states, actions, advantages, fixed_log_probs, ind)
                
                self.optimizer_policy.zero_grad()
                
                surr_loss.backward()
                
                ####################################
                # nan_detected = False
                # for name, param in self.policy_net.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any() or torch.isnan(param).any():
                #             print(f"NaN detected in gradients of parameter: {name}")
                #             nan_detected = True

                # if nan_detected:
                #     import ipdb; ipdb.set_trace()
                #     print('....')
                ####################################
                
                
                self.clip_policy_grad()
                
                self.optimizer_policy.step()
                
                # nan_detected = False
                # for name, param in self.policy_net.named_parameters():
                #     if param.grad is not None:
                #         if torch.isnan(param.grad).any() or torch.isnan(param).any():
                #             print(f"NaN detected in gradients of parameter: {name}")
                #             nan_detected = True

                # if nan_detected:
                #     import ipdb; ipdb.set_trace()
                #     print('....')

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for net, max_norm in self.policy_grad_clip:
                total_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
                
                # if torch.isnan(total_norm).any():
                #     import ipdb; ipdb.set_trace()
                #     print('....')
                
                

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        
        # if surr_loss.isinf():
        #     import ipdb; ipdb.set_trace()
        #     print('....')
        
        return surr_loss