import argparse
import os

import pandas as pd

from cache_emu.utils import load_yaml
from drl_agent import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config = load_yaml(os.path.join(project_root, args.config_path))

runner_funcs = [
    # OgdOptCacheRunner,  # baseline
    RlCacheRunner
]

runners = {}
for r_func in runner_funcs:
    runner_name = r_func.__name__
    runners[runner_name] = r_func(**config)

[runner.start() for runner in runners.values()]
[runner.join() for runner in runners.values()]
[runner.close() for runner in runners.values()]

results = {runner_name: runner.get_result() for runner_name, runner in runners.items()}
print(pd.DataFrame(results).T)
