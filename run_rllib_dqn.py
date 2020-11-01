import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env
from tqdm import tqdm

import cache_emu

ray.init(num_cpus=1, num_gpus=0, local_mode=True)

config = dqn.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
config["num_workers"] = 1

env_conf = cache_emu.load_yaml("asserts/env_config_tpl.yaml")


def env_creator(env_config):
    return cache_emu.CacheEnv(**env_conf)


register_env("my_env", env_creator)
trainer = dqn.DQNTrainer(env="my_env", config=config)

for i in tqdm(range(100)):
    trainer.train()
