import torch
from torch import nn

from cache_emu import torch_utils as ptu


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_layer_units: list, output_units: int, activation=None):
        super(MLP, self).__init__()
        
        self.in_features = in_features
        self.hidden_layer_units = hidden_layer_units
        self.output_units = output_units
        self.activation = activation
        
        self.net = ptu.build_mlp(in_features, hidden_layer_units, output_units)
    
    def forward(self, x):
        y = self.net(x)
        if self.activation is not None:
            return self.activation(y)
        else:
            return y


class EWMLP(nn.Module):
    def __init__(self, in_features: int, hidden_layer_units: list):
        super(EWMLP, self).__init__()
        
        self.in_features = in_features
        self.hidden_layer_units = hidden_layer_units
        
        self.net = ptu.build_mlp(in_features, hidden_layer_units)
        self.output_layer = ptu.build_linear(hidden_layer_units[-1], 1)
    
    def forward(self, x):
        return self.output_layer(self.net(x)).squeeze(-1)
    
    def forward_distilling(self, x):
        return self.net(x).softmax(dim=-1)


class GaussianEWMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_layer_units: list, log_std_min=-20, log_std_max=2):
        super(GaussianEWMLP, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_layer_units = hidden_layer_units
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = ptu.build_mlp(feature_dim, hidden_layer_units)
        self.l_mu = nn.Linear(hidden_layer_units[-1], 1).to(ptu.get_device())
        self.l_log_std = nn.Linear(hidden_layer_units[-1], 1).to(ptu.get_device())
    
    def forward(self, x):
        y = self.net(x)
        mu = self.l_mu(y).squeeze(-1)
        log_std = torch.tanh(self.l_log_std(y).squeeze(-1))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        return mu, log_std
