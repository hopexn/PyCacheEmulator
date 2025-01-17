import os

import torch
import torch.nn as nn

from py_cache_emu import torch_utils as ptu


class RLModel(torch.nn.Module):
    def __init__(self, content_dim: int, feature_dim: int, hidden_layer_units: list, lr: float, **kwargs):
        super(RLModel, self).__init__()
        self.content_dim = content_dim
        self.feature_dim = feature_dim
        self.hidden_layer_units = hidden_layer_units
        self.lr = lr
        
        self.net, self.optim, self.loss_fn = self.build_model(**kwargs)
    
    def build_model(self, **kwargs):
        raise NotImplementedError()
    
    def forward(self, x):
        return self.net(x)
    
    # 用于计算target值，默认与forward相同
    def forward_target(self, x):
        return self.forward(x)
    
    # 用于做知识蒸馏，默认与forward相同
    def forward_distilling(self, x):
        return self.net.forward_distilling(x)
    
    # 用于policy_gradient
    def backward(self, **kwargs):
        pass
    
    # 端到端的学习
    def fit(self, xs, ys):
        ys_pred = self.forward(xs)
        
        loss = self.loss_fn(ys_pred, ys.detach())
        
        self.zero_grad()
        loss.backward()
        self.step()
        
        return loss.cpu().item()
    
    def zero_grad(self):
        self.optim.zero_grad()
    
    def step(self):
        self.optim.step()
    
    # 保存参数
    def save_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        num_trainable_params = ptu.get_num_trainable_params([self.net])
        suffix = "-" + str(num_trainable_params) + suffix
        ptu.save_model(self.net, os.path.join(path, prefix + "net" + suffix + ".pt"))
        ptu.save_model(self.optim, os.path.join(path, prefix + "optim" + suffix + ".pt"))
    
    # 加载参数
    def load_weights(self, path, prefix="", suffix=""):
        prefix += self.__class__.__name__ + "_"
        num_trainable_params = ptu.get_num_trainable_params([self.net])
        suffix = "-" + str(num_trainable_params) + suffix
        res1 = ptu.load_model(self.net, os.path.join(path, prefix + "net" + suffix + ".pt"))
        res2 = ptu.load_model(self.optim, os.path.join(path, prefix + "optim" + suffix + ".pt"))
        return res1 and res2


class Temperature(nn.Module):
    def __init__(self, log_tau, min_entropy, lr=3e-4, log_tau_clip=(-4, 1), **kwargs):
        super(Temperature, self).__init__()
        
        self.min_entropy = min_entropy
        self.log_tau = nn.Parameter(ptu.tensor(float(log_tau)), requires_grad=True)
        self.optim = torch.optim.Adam([self.log_tau], lr=lr)
        self.log_tau_clip = log_tau_clip
    
    def forward(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def __mul__(self, other):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip)) * other
    
    def value(self):
        return torch.exp(self.log_tau.clamp(*self.log_tau_clip))
    
    def backward(self, probs):
        tau = self.value()
        # 为了避免log(probs)出现nan, probs需要加上eps
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        tau_loss = tau * (entropy.detach() - self.min_entropy)
        
        self.optim.zero_grad()
        tau_loss.backward()
        self.optim.step()
        
        return {"tau_value": tau.cpu().item(),
                "entropy"  : entropy.cpu().item(),
                "tau_loss" : tau_loss.cpu().item()}
    
    def save_weights(self, path, prefix="", suffix=""):
        ptu.save_model(self, os.path.join(path, prefix + "temperature" + suffix + ".pt"))
    
    def load_weights(self, path, prefix="", suffix=""):
        res = ptu.load_model(self, os.path.join(path, prefix + "temperature" + suffix + ".pt"))
        return res
