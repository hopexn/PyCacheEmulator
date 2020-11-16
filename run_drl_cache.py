import argparse
import multiprocessing as mp
import os

import pandas as pd

from cache_emu import *
from cache_emu.utils import mp_utils as mpu
from drl_agent import *

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config = proj_utils.load_yaml(os.path.join(project_root, args.config_path))

comm_size = config.get("com_size", 7)
mpu.init(comm_size)
runner_funcs = [
    [RlCacheRunner, {"enable_distilling": False}],
    [RlCacheRunner, {"enable_distilling": True}]
]

msg_queue = mp.Queue()
runners = []
for r_func, r_func_kwargs in runner_funcs:
    for rank in range(comm_size):
        runner = r_func(rank=rank, msg_queue=msg_queue, **config, **r_func_kwargs)
        runners.append(runner)

[runner.start() for runner in runners]
[runner.join() for runner in runners]

results = {}
while not msg_queue.empty():
    results.update(msg_queue.get())

print(pd.DataFrame(results).T)
