import os

import torch
from torch import nn

from py_cache_emu import torch_utils as ptu


class KDWeightsIF(nn.Module):
    def __init__(self, num_agents, n_neighbors=0, kdw_lr=3e-4, **kwargs):
        super(KDWeightsIF, self).__init__()
        self.num_agents = num_agents
        self.lr = kdw_lr
        
        self.n_neighbors = self.num_agents if n_neighbors <= 0 else n_neighbors
        self.n_neighbors = min(self.n_neighbors, self.num_agents)
        
        self.ws = nn.Parameter(ptu.ones(num_agents, dtype=torch.float), requires_grad=True)
        self.optim = torch.optim.Adam([self.ws], lr=self.lr)
        
        self.ref_ws = None
        self.indices = None
        self.random_index = kwargs.get("random_index", True)
    
    def forward(self, losses: torch.Tensor, **kwargs):
        raise NotImplementedError()
    
    def zero_grad(self):
        self.optim.zero_grad()
    
    def step(self):
        self.optim.step()
    
    def get_dict(self):
        if self.ref_ws is not None:
            return {"KDW{}".format(i): w for i, w in enumerate(ptu.get_numpy(self.ref_ws))}
        return {}
    
    def update_prefix_suffix(self, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        suffix = suffix + "-{}".format(self.num_agents)
        suffix = suffix + "-{}".format(ptu.get_num_trainable_params([self]))
        return prefix, suffix
    
    def save_weights(self, path, prefix="", suffix=""):
        prefix, suffix = self.update_prefix_suffix(prefix, suffix)
        path = os.path.join(path, prefix + "kd_weights" + suffix + ".pt")
        ptu.save_model(self, path)
        print("Weights saved: {}.".format(path))
    
    def load_weights(self, path, prefix="", suffix=""):
        prefix, suffix = self.update_prefix_suffix(prefix, suffix)
        path = os.path.join(path, prefix + "kd_weights" + suffix + ".pt")
        res = ptu.load_model(self, path)
        print("Weights loaded: {}.".format(path))
        return res
