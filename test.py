import os

import numpy as np

from cache_emu import CacheEnv, utils

# 根目录路径
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))

config = utils.load_yaml(os.path.join(_project_root, "asserts/env_config_tpl.yaml"))

env = CacheEnv(**config)

observation = env.reset()

info = {}
terminal = False

cnt = 0
MOD = int(1e4)

while not terminal:
    action = np.argmin(observation.flatten())
    observation, reward, terminal, info = env.step(action)
    cnt += 1
    if cnt % MOD == 0:
        print(cnt / MOD, info)

print("result:", info)
