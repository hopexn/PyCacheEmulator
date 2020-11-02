import os

import numpy as np
import torch

_use_gpu = False
_gpu_id = 0

device = None

# 常量
PI = np.pi
LOG_2PI = np.log(2 * np.pi)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def get_numpy(torch_tensor):
    return torch_tensor.cpu().detach().numpy()


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


# 正态分布
def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


# 均匀分布
def rand(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.rand(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def float_tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device, dtype=torch.float32)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def softmax(data, dim=0):
    max_value = data.max(dim=dim, keepdim=True).values.detach()
    return torch.softmax(data - max_value, dim=dim, dtype=torch.float64).float()


def log_softmax(data, dim):
    max_value = data.max(dim=dim, keepdim=True).values
    return torch.log_softmax(data - max_value, dim=dim)


def to_catagorical(data, num_classes):
    return torch.eye(num_classes, dtype=torch.bool)[data].to(device)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return True
    return False


def join_model_parameters(*models):
    res = []
    for model in models:
        res.extend(model.parameters())
    return res


def gaussian_likelihood(z, mu, log_std, std):
    log_probs = -0.5 * (((z - mu) / (std + 1e-6)) ** 2 + 2 * log_std + LOG_2PI)
    return log_probs


def build_linear(input_units, output_units):
    return torch.nn.Linear(input_units, output_units).to(device)


def build_mlp(input_units, hidden_layer_units, output_units=None):
    layer_units = [input_units]
    layer_units.extend(hidden_layer_units)
    
    layers = []
    for in_features, out_features in zip(layer_units[:-1], layer_units[1:]):
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.ReLU())
    
    if output_units is not None:
        layers.append(torch.nn.Linear(layer_units[-1], output_units))
    
    net = torch.nn.Sequential(*layers).to(device)
    
    return net


def get_num_trainable_params(models: list):
    num_trainable_params = 0
    for model in models:
        num_trainable_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_trainable_params


def cross_entropy_loss(probs_pred, probs):
    return - (probs * torch.log_softmax(probs_pred, dim=-1)).mean()
