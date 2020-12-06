import os

import numpy as np
import torch
from torch import nn

from py_cache_emu import torch_utils as  ptu


class TemperatureModel(nn.Module):
    def __init__(self, log_tau, min_entropy, tau_lr=3e-4, log_tau_clip=(-4, 1), **kwargs):
        super(TemperatureModel, self).__init__()
        
        self.min_entropy = min_entropy
        self.log_tau = nn.Parameter(ptu.tensor(float(log_tau)), requires_grad=True)
        self.optim = torch.optim.Adam([self.log_tau], lr=tau_lr)
        self.log_tau_clip = log_tau_clip
    
    def forward(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def __mul__(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def value(self):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip))
    
    def backward(self, probs):
        tau = torch.exp(self.log_tau.clamp(*self.log_tau_clip))
        # 为了避免log(probs)出现nan, probs需要加上eps
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        tau_loss = tau * (entropy.detach() - self.min_entropy)
        
        self.optim.zero_grad()
        tau_loss.backward()
        self.optim.step()
        
        return {"tau"     : tau.cpu().item(),
                "entropy" : entropy.cpu().item(),
                "tau_loss": tau_loss.cpu().item()}
    
    def save_weights(self, path, prefix="", suffix=""):
        ptu.save_model(self, os.path.join(path, prefix + "tau" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        res = ptu.load_model(self, os.path.join(path, prefix + "tau" + suffix + ".pt"))
        return res


class KDWeights(nn.Module):
    def __init__(self, num_agents, kdw_lr=3e-4, kwd_log_tau=-1, min_entropy_ratio=0.98, **kwargs):
        super(KDWeights, self).__init__()
        self.num_agents = num_agents
        self.lr = kdw_lr
        
        self.tau = TemperatureModel(log_tau=kwd_log_tau,
                                    min_entropy=min_entropy_ratio * np.log(num_agents),
                                    **kwargs)
        self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
        self.optim = torch.optim.Adam([self.ws], lr=self.lr)
    
    def forward(self, losses: list, k=0):
        k = self.num_agents if k <= 0 else k
        k = min(k, self.num_agents)
        
        probs = (self.ws / self.tau.value().detach()).softmax(dim=0)
        
        indices = np.random.choice(np.arange(self.num_agents), k,
                                   replace=False,
                                   p=ptu.get_numpy(probs))
        
        loss = ptu.float_tensor(0, requires_grad=True)
        loss_ws = ptu.float_tensor(0, requires_grad=True)
        
        for j in range(k):
            idx = indices[j]
            loss = loss + losses[idx]
            loss_ws = loss_ws - (-losses[idx].detach().abs()).exp() * probs[idx]
        
        # 为了避免log(probs)出现nan, probs需要加上eps
        loss_ws = loss_ws / loss.detach()
        self.tau.backward(probs)
        
        return loss / k + loss_ws
    
    def zero_grad(self):
        self.optim.zero_grad()
    
    def step(self):
        self.optim.step()
    
    def get_dict(self):
        softmax_ws = ptu.get_numpy((self.ws / self.tau.value()).softmax(dim=0))
        return {"KDW{}".format(i): w for i, w in enumerate(softmax_ws)}
    
    def save_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix
        ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix
        res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
        return res
