import os

import ray
import ray.rllib.agents.dqn as dqn
from ray.tune.registry import register_env

import py_cache_emu as cache_emu
from py_cache_emu import proj_utils

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

ray.init(num_cpus=2, num_gpus=1)

config = dqn.DEFAULT_CONFIG.copy()
config["framework"] = "torch"
config["hiddens"] = [256, 128]
config["timesteps_per_iteration"] = 10
config["train_batch_size"] = 128
config["dueling"] = False
config["prioritized_replay"] = False
config["gamma"] = 0.1

env_conf = proj_utils.load_yaml("asserts/configs/default.yaml")
env_conf['data_config'] = proj_utils.load_data_config(env_conf.pop("data_config", "iqiyi_pois.yaml"))
env_conf['feature_config'] = proj_utils.load_feature_config(env_conf.pop("feature_config", "drl.yaml"))

register_env("cache_env", env_creator=lambda x: cache_emu.CacheEnv(**env_conf))

trainer = dqn.DQNTrainer(env="cache_env", config=config)
weights_dir = proj_utils.abs_path("assert/weights/")

# if os.path.exists(weights_dir + "latest_ckpt"):
#     with open(weights_dir + "latest_ckpt") as f:
#         ckpt = f.read()
#         trainer.load_checkpoint(ckpt)
#         print("load weight at {}".format(ckpt))

i = 0
try:
    while True:
        i += 1
        result = trainer.train()
        
        if i % 10000 == 0:
            ckpt = trainer.save(checkpoint_dir=proj_utils.abs_path("assert/weights"))
            with open(proj_utils.abs_path("assert/weights/latest_ckpt"), "w") as f:
                print(ckpt, file=f, end="")
finally:
    ckpt = trainer.save(checkpoint_dir=proj_utils.abs_path("assert/weights"))
    with open(proj_utils.abs_path("assert/weights/latest_ckpt"), "w") as f:
        print(ckpt, file=f, end="")
    print("Save weights successfully~")
