import argparse
import os

import pandas as pd

from cache_baselines.runner import *
from utils import load_yaml

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config_path", type=str, help="the path of experiment config file.", required=True)
args = parser.parse_args()

# 根目录路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
config = load_yaml(os.path.join(project_root, args.config_path))

runners = [
    # OgdOptCacheRunner(**config),  # baseline
    EwdqnCacheRunner(**config)
]

for runner in runners:
    runner.start()

for runner in runners:
    runner.join()

results = {}

for runner in runners:
    results[runner.__class__.__name__] = runner.get_result()

for runner in runners:
    runner.close()

print(pd.DataFrame(results).T)
