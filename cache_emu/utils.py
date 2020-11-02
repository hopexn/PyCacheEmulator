import os

import yaml

NoneContentType = -1

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def abs_path(*path):
    path = os.path.join(*path)
    if not os.path.isabs(path):
        path = os.path.join(_project_root, "..", path)
    return path


# 加载yaml配置文件
def load_yaml(path):
    with open(abs_path(path), 'r', encoding="utf-8") as f:
        config_data = yaml.full_load(f)
    return config_data
