import argparse
import multiprocessing as mp
import os

import numpy as np
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

runner = RlCacheRunner(**config)
res = runner.run()

print(res)
