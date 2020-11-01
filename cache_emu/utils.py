import yaml

NoneContentType = -1


# 加载yaml配置文件
def load_yaml(path):
    with open(path, 'r', encoding="utf-8") as f:
        config_data = yaml.full_load(f)
    return config_data
