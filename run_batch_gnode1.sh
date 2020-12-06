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
#python $project_root/run_cache.py capacity_iqiyi_pois.yaml -l 403 --rank=0
#python $project_root/run_cache.py capacity_iqiyi_pois.yaml -l 404 --rank=1
#python $project_root/run_cache.py capacity_iqiyi12.yaml -l 405 --rank=0
#python $project_root/run_cache.py capacity_iqiyi12.yaml -l 406 --rank=1

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 1.0 -l 1500 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.9 -l 1501 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.5 -l 1502 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.3 -l 1503 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.1 -l 1504 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.05 -l 1505 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --sparsity 0.01 -l 1506 --seed 0


#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 0 -l 541 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=2 -l 542 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=3 -l 543 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=4 -l 544 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=6 -l 545 --seed 1

python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 50 --n_neighbors=0 -l 2600
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 100 --n_neighbors=0 -l 2601
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 200 --n_neighbors=0 -l 2602 -e
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 1000 --n_neighbors=0 -l 2603 -e
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 2000 --n_neighbors=0 -l 2604 -e