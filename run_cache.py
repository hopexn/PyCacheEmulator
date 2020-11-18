import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from cache_emu import *
from drl_agent import RlCacheRunner

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
config = proj_utils.load_config(args.config_path)

# 用于从子进程传递返回信息
msg_queue = mp.Queue()

# 数据配置
data_config = proj_utils.load_data_config(config.pop("data_config", "iqiyi_pois.yaml"))
# 特征配置
feature_config = proj_utils.load_feature_config(config.pop("feature_config", "drl.yaml"))
# 配置运行的算法
runner_config = proj_utils.load_runner_config(config.get("runner_config", "baselines.yaml"))

comm_size = config.get("comm_size", 1)
mp_utils.init(comm_size)
permute_data = config.get("permute_data", False)

if permute_data:
    n_data_paths = len(data_config.get('data_path'))
    runner_ranks = np.random.permutation(np.arange(n_data_paths))[:min(n_data_paths, comm_size)]
else:
    runner_ranks = np.arange(comm_size)

runners = []
for runner_name, runner_kwargs in runner_config:
    for rank in runner_ranks:
        if runner_name == "RlCacheRunner":
            runner_class = RlCacheRunner
        else:
            runner_class = eval_runner_class(runner_name)
        
        runner = runner_class(
            rank=rank, msg_queue=msg_queue,
            data_config=data_config, feature_config=feature_config,
            **{**config, **runner_kwargs})
        runners.append(runner)

# 启动进程
[runner.start() for runner in runners]
# 等待进程结束
[runner.join() for runner in runners]

# 获取结果
results = {}
while not msg_queue.empty():
    results.update(msg_queue.get())

# 打印结果
print(pd.DataFrame(results).T)