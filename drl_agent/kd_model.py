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
        entropy = -(probs * torch.log(probs)).sum(dim=-1).mean()
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
        
        self.kd_mode = int(kwargs.get("kd_mode", 0))
        self.ws = ptu.ones(num_agents, dtype=torch.float)
        if self.kd_mode == 2:  # adaptive
            self.ws = nn.Parameter(self.ws + 1e-3 * ptu.randn(num_agents, dtype=torch.float), requires_grad=True)
        else:  # fixed
            self.ws = nn.Parameter(self.ws, requires_grad=True)
        
        self.optim = torch.optim.Adam([self.ws], lr=self.lr)
    
    def forward(self, losses: list, k=0):
        k = self.num_agents if k <= 0 else k
        k = min(k, self.num_agents)
        
        indices = np.random.permutation(np.arange(self.num_agents))[:k]
        loss = ptu.float_tensor(0, requires_grad=True)
        
        ws_sm = (self.ws / self.tau.value().detach()).softmax(dim=0)
        
        if self.kd_mode == 2:
            for idx in indices:
                loss = loss + losses[idx] * ws_sm[idx]
            self.tau.backward(ws_sm)
        else:
            for idx in indices:
                loss = loss + losses[idx]
        
        return loss / k
    
    def zero_grad(self):
        if self.kd_mode == 2:
            self.optim.zero_grad()
    
    def step(self):
        if self.kd_mode == 2:
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
