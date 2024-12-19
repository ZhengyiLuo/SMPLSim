import torch.nn as nn
from smpl_sim.learning.distributions import DiagGaussian
from smpl_sim.learning.policy import Policy
from smpl_sim.learning.policy_gaussian import PolicyGaussian
from smpl_sim.learning.mlp import MLP
import torch

class PolicyMCP(Policy):
    def __init__(self, cfg, action_dim, state_dim, net_out_dim=None):
        super().__init__()
        self.type = 'gaussian'
        policy_hsize = cfg.policy_hsize
        policy_htype = cfg.policy_htype
        fix_std = cfg.fix_std
        log_std = cfg.log_std

        self.nets = nn.ModuleList()
        for i in range(cfg.num_primitive):
            action_mean = nn.Linear(policy_hsize[-1], action_dim)
            action_mean.weight.data.mul_(0.1)
            action_mean.bias.data.mul_(0.0)
            net = nn.Sequential(*[MLP(state_dim, policy_hsize, policy_htype), action_mean])  
            self.nets.append(net)
        
        self.composer = nn.Sequential(*[MLP(state_dim, cfg.get("composer_dim", [300, 200]) + [cfg.num_primitive], policy_htype), nn.Softmax(dim=1)])
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * log_std, requires_grad=not fix_std)

    def forward(self, x):
        x_all = torch.stack([net(x) for net in self.nets], dim=1)
        weight = self.composer(x)
        # self.live_plotter(weight.numpy(), "Weights")
        action_mean = torch.sum(weight[:, :, None] * x_all, dim=1)

        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return DiagGaussian(action_mean, action_std)

    def live_plotter(self, w, identifier='', pause_time=0.00000001):
        import matplotlib.pyplot as plt
        if not hasattr(self, 'lines'):
            size = 100
            self.x_vec = np.linspace(0, 1, size + 1)[0:-1]
            self.y_vecs = [np.array([0] * len(self.x_vec)) for i in range(7)]
            self.lines = [[] for i in range(7)]
            # this is the call to matplotlib that allows dynamic plotting
            plt.ion()

            fig = plt.figure(figsize=(8, 3))
            ax = fig.add_subplot(111)
            # create a variable for the line so we can later update it

            for i in range(7):
                l, = ax.plot(self.x_vec, self.y_vecs[i], '-o', alpha=0.8)
                self.lines[i] = l

            # update plot label/title
            plt.ylabel('Weight')
            plt.title('{}'.format(identifier))
            plt.ylim((0, 1))
            plt.show()

        for i in range(7):
            # after the figure, axis, and line are created, we only need to update the y-data
            self.y_vecs[i][-1] = w[0][i]
            self.lines[i].set_ydata(self.y_vecs[i])
            # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
            self.y_vecs[i] = np.append(self.y_vecs[i][1:], 0.0)
        plt.pause(pause_time)

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
        return cov_inv.detach(), dist.loc, {'std_id': std_id, 'std_index': std_index}


