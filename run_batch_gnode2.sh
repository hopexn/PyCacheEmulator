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

#python $project_root/run_cache.py ewdrl_iqiyi_pois_kd_different_capacity.yaml -l 700 --rank 0 --comm_size=7 -e
#python $project_root/run_cache.py ewdrl_iqiyi_pois_kd_different_capacity.yaml -l 701 --rank 1 --comm_size=7 -e
#python $project_root/run_cache.py ewdrl_iqiyi12_kd_different_capacity.yaml -l 702 --rank 0 --comm_size=7 -e
#python $project_root/run_cache.py ewdrl_iqiyi12_kd_different_capacity.yaml -l 703 --rank 1 --comm_size=7 -e
#
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 --n_neighbors 0 -l 551 --seed 1 --min_entropy_ratio=0.9
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 --n_neighbors 0 -l 552 --seed 1 --min_entropy_ratio=0.8
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 --n_neighbors 0 -l 553 --seed 1 --min_entropy_ratio=0.7
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 --n_neighbors 0 -l 554 --seed 1 --min_entropy_ratio=0.6
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --sparsity 0.7 --n_neighbors 0 -l 555 --seed 1 --min_entropy_ratio=0.5
#
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 1.0 -l 510 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.5 -l 511 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --sparsity 0.1 -l 512 --seed 0

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 1.0 -l 1510 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 1.3 -l 1511 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 1.5 -l 1513 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 1.8 -l 1514 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 2.0 -l 1515 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 2.5 -l 1516 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 3.0 -l 1517 --seed 0
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --kd_tau 5.0 -l 1518 --seed 0

python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 8010 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=1 -l 8011 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=2 -l 8012 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=3 -l 8013 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=4 -l 8014 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=5 -l 8015 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=6 -l 8016 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=7 -l 8017 --seed=1


