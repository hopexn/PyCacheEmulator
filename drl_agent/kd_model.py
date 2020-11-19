import os

import numpy as np
import torch
from torch import nn

from py_cache_emu import torch_utils as  ptu


class KDWeights(nn.Module):
    def __init__(self, num_agents, lr, log_tau=1.0, **kwargs):
        super(KDWeights, self).__init__()
        self.num_agents = num_agents
        self.lr = lr
        
        self.tau = np.exp(log_tau)
        self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
        self.optim = torch.optim.Adam([self.ws], lr=lr)
    
    def forward(self, losses: list):
        softmax_ws = (self.ws / self.tau).softmax(dim=0)
        
        loss = 0
        for i in range(self.num_agents):
            loss = loss + softmax_ws[i] * losses[i]
        loss = loss + self.tau * (torch.dot(softmax_ws, softmax_ws.log()))
        
        return loss
    
    def forward_topk(self, losses: list, k):
        softmax_ws = (self.ws / self.tau).softmax(dim=0)
        
        loss = 0
        indices = torch.argsort(softmax_ws, descending=True).cpu().numpy()
        
        for j in range(min(k, self.num_agents)):
            i = indices[j]
            loss = loss + softmax_ws[i] * losses[i]
        
        return loss
    
    def zero_grad(self):
        self.optim.zero_grad()
    
    def step(self):
        self.optim.step()
    
    def get_dict(self):
        softmax_ws = ptu.get_numpy((self.ws / self.tau).softmax(dim=0))
        return {"KDW{}".format(i): w for i, w in enumerate(softmax_ws)}
    
    def save_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix
        ptu.save_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        suffix = "-" + str(self.num_agents) + suffix
        res = ptu.load_model(self, os.path.join(path, prefix + "kd_weights" + suffix + ".pt"))
        return res
