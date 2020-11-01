import argparse
import os

import pandas as pd

from cache_baselines import *
from utils import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config = load_yaml(os.path.join(project_root, args.config_path))

runners = []
runners.append(RandomCacheRunner(**config))
runners.append(LruCacheRunner(**config))
runners.append(LfuCacheRunner(**config))
runners.append(OgdOptCacheRunner(**config))
runners.append(OgdLruCacheRunner(**config))
runners.append(OgdLfuCacheRunner(**config))

print(config['feature_config'])

result = {}
for runner in runners:
    res = runner.run()
    result[runner.__class__.__name__] = res

print(pd.DataFrame(result).T)
