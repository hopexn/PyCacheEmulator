import os
import pickle
import random

import numpy as np
import torch
import yaml

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# 设置随机种子
def manual_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def manual_project_root(project_root):
    global _project_root
    _project_root = os.path.expanduser(project_root)


def abs_path(*path):
    global _project_root
    return os.path.join(_project_root, *path)


# 加载yaml配置文件
def load_yaml(path):
    with open(abs_path(path), 'r', encoding="utf-8") as f:
        config_data = yaml.full_load(f)
    return config_data


def load_config(filename):
    return load_yaml(abs_path("asserts/configs/", filename))


def load_data_config(filename):
    return load_yaml(abs_path("asserts/configs/_data/", filename))


def load_feature_config(filename):
    return load_yaml(abs_path("asserts/configs/_feature/", filename))


def load_runner_config(filename):
    return load_yaml(abs_path("asserts/configs/_runner/", filename))


def pickle_dump(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj
