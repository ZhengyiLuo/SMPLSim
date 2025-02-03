############################################################################################################
### Functions are from https://github.com/Khrylx/PyTorch-RL
############################################################################################################
import numpy as np
import torch.nn as nn
from smpl_sim.learning.distributions import DiagGaussian
from smpl_sim.learning.policy import Policy
from smpl_sim.learning.mlp import MLP
from smpl_sim.learning.running_norm import RunningNorm
import torch



class PolicyGaussian(Policy):
    def __init__(self, cfg, action_dim, state_dim, net_out_dim=None):
        super().__init__()
        self.type = "gaussian"
        self.norm = RunningNorm(state_dim)
        
        policy_hsize = cfg.learning.mlp.units
        policy_htype = cfg.learning.mlp.activation
        fix_std = cfg.learning.fix_std
        log_std = cfg.learning.log_std
        self.net = net = MLP(state_dim, policy_hsize, policy_htype)
        
        if net_out_dim is None:
            net_out_dim = net.out_dim
        self.action_mean = nn.Linear(net_out_dim, action_dim)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(
            torch.ones(1, action_dim) * log_std, requires_grad=not fix_std
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.net(x)
        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def get_fim(self, x):
        dist = self.forward(x)
        cov_inv = self.action_log_std.exp().pow(-2).squeeze(0).repeat(x.size(0))
        param_count = 0
        std_index = 0
        id = 0
        for name, param in self.named_parameters():
            if name == "action_log_std":
                std_id = id
                std_index = param_count
            param_count += param.view(-1).shape[0]
            id += 1
        return cov_inv.detach(), dist.loc, {"std_id": std_id, "std_index": std_index}
