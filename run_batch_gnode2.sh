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

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 13620
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=1 -l 13621
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --n_neighbors=2 -l 13622
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 100 --n_neighbors=3 -l 13623
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 200 --n_neighbors=4 -l 13624 -e
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 1000 --n_neighbors=5 -l 13625 -e
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 2000 --n_neighbors=6 -l 13626 -e

#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=0 -l 19020 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=1 -l 19021 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=2 -l 19022 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=3 -l 19023 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=4 -l 19024 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=5 -l 19025 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=6 -l 19026 --seed=1
#
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=0 -l 19030 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=1 -l 19031 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=2 -l 19032 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=3 -l 19033 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=4 -l 19034 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=5 -l 19035 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 20 --n_neighbors=6 -l 19036 --seed=1

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 29020 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=1 -l 29021 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=2 -l 29022 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=3 -l 29023 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=4 -l 29024 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=5 -l 29025 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=6 -l 29026 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=7 -l 29027 --seed=1

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=0 -l 29030 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=1 -l 29031 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=2 -l 29032 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=3 -l 29033 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=4 -l 29034 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=5 -l 29035 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=6 -l 29036 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=7 -l 29037 --seed=1

base_log_id=29040
seed=2
capacity=10

python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=1 -l $((base_log_id)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=2 -l $((base_log_id + 1)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=3 -l $((base_log_id + 2)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=4 -l $((base_log_id + 3)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=5 -l $((base_log_id + 4)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=6 -l $((base_log_id + 5)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=7 -l $((base_log_id + 6)) --seed=$seed
python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity $capacity --n_neighbors=8 -l $((base_log_id + 7)) --seed=$seed


