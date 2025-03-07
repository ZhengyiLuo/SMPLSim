import torch

from smpl_sim.learning.learning_utils import to_train, to_test
from smpl_sim.learning.learning_utils import estimate_advantages
from smpl_sim.agents.agent import Agent

import time

class AgentPG(Agent):

    def __init__(self, tau=0.95, optimizer_policy=None, optimizer_value=None,
                 opt_num_epochs=1, value_opt_niter=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.optimizer_policy = optimizer_policy
        self.optimizer_value = optimizer_value
        self.opt_num_epochs = opt_num_epochs
        self.value_opt_niter = value_opt_niter

    def update_value(self, critic_states, returns):
        """update critic"""
        for _ in range(self.value_opt_niter):
            values_pred = self.value_net(self.trans_value(critic_states))
            value_loss = (values_pred - returns).pow(2).mean()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def update_policy(self, states, critic_states, actions, returns, advantages, exps):
        """update policy"""
        # use a2c by default
        ind = exps.nonzero().squeeze(1)
        for _ in range(self.opt_num_epochs):
            self.update_value(critic_states, returns)
            log_probs = self.policy_net.get_log_prob(states[ind], actions[ind])
            policy_loss = -(log_probs * advantages[ind]).mean()
            self.optimizer_policy.zero_grad()
            policy_loss.backward()
            self.optimizer_policy.step()

    def update_params(self, batch):
        t0 = time.time()
        to_train(*self.update_modules)
        states = torch.from_numpy(batch.states).to(self.dtype).to(self.device)
        critic_states = torch.from_numpy(batch.critic_states).to(self.dtype).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.dtype).to(self.device)
        rewards = torch.from_numpy(batch.rewards).to(self.dtype).to(self.device)
        not_done = torch.from_numpy(batch.not_done).to(self.dtype).to(self.device)
        not_dead = torch.from_numpy(batch.not_dead).to(self.dtype).to(self.device)
        exps = torch.from_numpy(batch.exps).to(self.dtype).to(self.device)
        with to_test(*self.update_modules):
            with torch.no_grad():
                values = self.value_net(self.trans_value(critic_states))

        """get advantage estimation from the trajectories"""
        advantages, returns = estimate_advantages(rewards, not_done, not_dead, values, self.gamma, self.tau)
        
        self.update_policy(states, critic_states,  actions, returns, advantages, exps)

        return time.time() - t0