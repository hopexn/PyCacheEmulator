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

runners = [RandomCacheRunner(**config), LruCacheRunner(**config), LfuCacheRunner(**config),
           OgdOptCacheRunner(**config), OgdLruCacheRunner(**config), OgdLfuCacheRunner(**config)]

result = {}
for runner in runners:
    res = runner.run()
    runner_name = runner.__class__.__name__
    result[runner_name] = res

print(pd.DataFrame(result).T)
