import argparse
import multiprocessing as mp
import os

import numpy as np
import pandas as pd

from drl_agent import RlCacheRunner
from py_cache_emu import *

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="the path of experiment config file.")

parser.add_argument('-s', '--seed', type=int, default=0)
parser.add_argument('-v', '--verbose', action='store_true', default=False)
parser.add_argument('-e', '--eager_mode', action='store_true', default=False)

parser.add_argument('-l', '--log_id', type=str, default=None)
parser.add_argument('--capacity', type=int, default=None)
parser.add_argument('--sparsity', type=float, default=None)
parser.add_argument('--kd_tau', type=float, default=None)
parser.add_argument('--n_neighbors', type=int, default=None)
parser.add_argument('--comm_size', type=int, default=None)
parser.add_argument('--rank', type=int, default=0)

parser.add_argument('--kdw_lr', type=float, default=None)
parser.add_argument('--kwd_log_tau', type=float, default=None)
parser.add_argument('--min_entropy_ratio', type=float, default=None)

args, unkown_args = parser.parse_known_args()

# 手动设置种子
proj_utils.manual_seed(args.seed)

# 根目录路径
config = proj_utils.load_config(args.config_path)
# 数据配置
data_config = proj_utils.load_data_config(config.pop("data_config", "iqiyi_pois.yaml"))
# 特征配置
feature_config = proj_utils.load_feature_config(config.pop("feature_config", "drl.yaml"))
# 配置运行的算法
runner_config = proj_utils.load_runner_config(config.get("runner_config", "baselines.yaml"))

permute_data = config.pop("data_permute", False)
config['verbose'] = args.verbose

if args.log_id is not None:
    config['log_id'] = args.log_id

if args.capacity is not None:
    config['capacity'] = args.capacity

if args.sparsity is not None:
    config['sparsity'] = args.sparsity

if args.kd_tau is not None:
    config['kd_tau'] = args.kd_tau

if args.n_neighbors is not None:
    config['n_neighbors'] = args.n_neighbors

if args.comm_size is not None:
    config['comm_size'] = args.comm_size

if args.kdw_lr is not None:
    config['kdw_lr'] = args.kdw_lr

if args.min_entropy_ratio is not None:
    config['min_entropy_ratio'] = args.min_entropy_ratio

if args.kwd_log_tau is not None:
    config['kwd_log_tau'] = args.kwd_log_tau

print("log_id: {}".format(config['log_id']))

comm_size = config.pop("comm_size", 1)
mp_utils.init(comm_size)
n_data_paths = len(data_config.get('data_path'))
runner_ranks = np.arange(min(n_data_paths, comm_size))
if permute_data:
    runner_ranks = np.random.permutation(runner_ranks)

# 用于从子进程传递返回信息
msg_queue = mp.Queue()

# 获取结果
results = {}
runners = []


def start_runners(runners):
    results = {}
    
    # 启动进程
    [runner.start() for runner in runners]
    # 等待进程结束
    [runner.join() for runner in runners]
    # 保存结果
    while not msg_queue.empty():
        results.update(msg_queue.get())
    # 将进程从队列中清楚
    runners.clear()
    
    return results


for runner_name, runner_config in runner_config:
    runner_kwargs = {**config, **runner_config}
    max_ranks = min(n_data_paths, comm_size)
    for i in range(max_ranks):
        rank = runner_ranks[i] if max_ranks > 1 else (args.rank % n_data_paths)
        
        if runner_name == "RlCacheRunner":
            runner_class = RlCacheRunner
        else:
            runner_class = eval_runner_class(runner_name)
        
        runner = runner_class(
            rank=rank, msg_queue=msg_queue,
            data_config=data_config, feature_config=feature_config,
            **runner_kwargs
        )
        runners.append(runner)
    
    if args.eager_mode or config.get('eager_mode', False):
        # eager execution模式开始运行
        results.update(start_runners(runners))

# 非eager execution模式开始运行
results.update(start_runners(runners))

# 覆盖原有结果
res = pd.DataFrame(results).T
log_path = os.path.expanduser("~/default_log/{}.csv".format(config.get("log_id", "0000")))
if os.path.exists(log_path):
    res_ori = pd.read_csv(log_path, index_col=0)
    res = pd.concat([res_ori, res])
res.to_csv(log_path)
print(res)
