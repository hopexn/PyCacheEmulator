import pickle
import random

import numpy as np
import torch
import yaml


# 设置随机种子
def manual_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# 加载yaml配置文件
def load_yaml(path):
    with open(path, 'r', encoding="utf-8") as f:
        config_data = yaml.full_load(f)
    return config_data


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
