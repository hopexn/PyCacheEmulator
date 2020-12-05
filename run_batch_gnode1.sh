#!/bin/bash

project_root=~/PyCacheEmulator

# 参数选项
#parser.add_argument("config_path", type=str, help="the path of experiment config file.")
#parser.add_argument('-s', '--seed', type=int, default=0)
#parser.add_argument('-v', '--verbose', action='store_true', default=False)
#parser.add_argument('-e', '--eager_mode', action='store_true', default=False)
#parser.add_argument('-l', '--log_id', type=str, default=None)
#parser.add_argument('--capacity', type=int, default=None)
#parser.add_argument('--sparsity', type=float, default=None)

#python $project_root/run_cache.py capacity_zipf_static.yaml -l 400
#python $project_root/run_cache.py capacity_zipf_dynamic10.yaml -l 401
#python $project_root/run_cache.py capacity_zipf_dynamic2.yaml -l 402
python $project_root/run_cache.py capacity_iqiyi_pois.yaml -l 403 --rank=0
python $project_root/run_cache.py capacity_iqiyi_pois.yaml -l 404 --rank=1
python $project_root/run_cache.py capacity_iqiyi12.yaml -l 405 --rank=0
python $project_root/run_cache.py capacity_iqiyi12.yaml -l 406 --rank=1

python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 1.0 -l 500 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.7 -l 501 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.5 -l 502 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.3 -l 503 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.1 -l 504 --seed 0

python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 1.0 -l 505 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 -l 506 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.5 -l 507 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 -l 508 --seed 0
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.1 -l 509 --seed 0

python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 1.0 -l 510 --seed 0 -e
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.7 -l 511 --seed 0 -e
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.5 -l 512 --seed 0 -e
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.3 -l 513 --seed 0 -e
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.1 -l 514 --seed 0 -e
