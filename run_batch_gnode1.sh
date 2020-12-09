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

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 3610
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=0 -l 3611
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --n_neighbors=0 -l 3612
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 100 --n_neighbors=0 -l 3613
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 200 --n_neighbors=0 -l 3614 -e
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 1000 --n_neighbors=0 -l 3615 -e
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 2000 --n_neighbors=0 -l 3616 -e
#
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 3620 --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 20 --n_neighbors=0 -l 3621 --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 50 --n_neighbors=0 -l 3622 --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 100 --n_neighbors=0 -l 3623 --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 200 --n_neighbors=0 -l 3624 -e --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 1000 --n_neighbors=0 -l 3625 -e --seed=2
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 2000 --n_neighbors=0 -l 3626 -e --seed=2

#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=0 -l 9010 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=1 -l 9011 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=2 -l 9012 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=3 -l 9013 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=4 -l 9014 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=5 -l 9015 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=6 -l 9016 --seed=1
#python $project_root/run_cache.py ewdrl_iqiyi12.yaml --capacity 10 --n_neighbors=7 -l 9017 --seed=1

python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=0 -l 19010 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=1 -l 19011 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=2 -l 19012 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=3 -l 19013 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=4 -l 19014 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=5 -l 19015 --seed=1
python $project_root/run_cache.py ewdrl_iqiyi_pois.yaml --capacity 10 --n_neighbors=6 -l 19016 --seed=1
