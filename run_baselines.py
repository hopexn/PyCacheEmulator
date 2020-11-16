import argparse
import multiprocessing as mp
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
    [RandomCacheRunner, {}],
    [LruCacheRunner, {}],
    [LfuCacheRunner, {}],
    # [OgdOptCacheRunner, {}],
    # [OgdLruCacheRunner, {}],
    [OgdLfuCacheRunner, {}],
    # [RlCacheRunner, {}] # 强化学习方法，也和run_rl_cache.py运行
]
msg_queue = mp.Queue()
runners = []
for r_func, r_func_kwargs in runner_funcs:
    for rank in runner_ranks:
        runner = r_func(rank=rank, msg_queue=msg_queue, **config, **r_func_kwargs)
        runners.append(runner)

[runner.start() for runner in runners]
[runner.join() for runner in runners]

results = {}
while not msg_queue.empty():
    results.update(msg_queue.get())

print(pd.DataFrame(results).T)
