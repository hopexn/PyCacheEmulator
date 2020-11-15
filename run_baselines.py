import argparse
import os

import pandas as pd

from cache_emu import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config = proj_utils.load_yaml(os.path.join(project_root, args.config_path))

# 在这里选择运行的baseline
comm_size = config.get("com_size", 7)
runner_ranks = range(comm_size)
runner_funcs = [
    RandomCacheRunner,
    LruCacheRunner,
    LfuCacheRunner,
    OgdOptCacheRunner,
    # OgdLruCacheRunner,
    # OgdLfuCacheRunner,
    # RlCacheRunner # 强化学习方法，也和run_rl_cache.py运行
]

runners = {}
for r_func in runner_funcs:
    for rank in runner_ranks:
        runner = r_func(**config, rank=rank)
        runner_name = "{}/{}".format(runner.main_tag, runner.sub_tag)
        runners[runner_name] = runner

[runner.start() for runner in runners.values()]
[runner.join() for runner in runners.values()]
[runner.close() for runner in runners.values()]

results = {runner_name: runner.get_result() for runner_name, runner in runners.items()}
print(pd.DataFrame(results).T)
