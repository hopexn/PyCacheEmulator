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

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.9 --kd_mode 0 -l 520 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.9 --kd_mode 0 -l 530 --seed 0

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 0 -l 541 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-2 -l 546 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-4 -l 547 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-6 -l 548 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-8 -l 549 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 2 -l 542 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 4 -l 543 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 6 -l 544 --seed 1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.3 --n_neighbors 8 -l 545 --seed 1

python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors 0 -l 551 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors 2 -l 552 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors 4 -l 553 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors 6 -l 554 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-2 -l 556 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-4 -l 557 --seed 1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --sparsity 0.3 --n_neighbors=-6 -l 558 --seed 1

python $project_root/run_cache.py ewdrl_iqiyi_pois_kd_different_capacity.yaml -l 700 --rank 0 --comm_size=7
python $project_root/run_cache.py ewdrl_iqiyi_pois_kd_different_capacity.yaml -l 701 --rank 1 --comm_size=7
python $project_root/run_cache.py ewdrl_iqiyi12_kd_different_capacity.yaml -l 702 --rank 0
python $project_root/run_cache.py ewdrl_iqiyi12_kd_different_capacity.yaml -l 703 --rank 1
python $project_root/run_cache.py ewdrl_iqiyi12_kd_different_capacity.yaml -l 704 --rank 2
