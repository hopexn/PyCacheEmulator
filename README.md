# PyCacheEmulator

PycCacheEmulator是一个用于模拟PassiveCache的环境，其接口与OpenAI Gym一致，可以用于测试强化学习算法。

此外，本仓库还集成了几个基本的baseline，它们分别是:
- LRU
- LFU
- Random
- OgdOptimal
- OgdLru
- OgdLfu
- List Wise DQN

## 使用方法

1. 环境安装

```sh
cd /path/to/PyCacheEmulator
# 安装依赖
pip install -r requirements.txt
# 安装cache_emu包
pip install -e .
```

2. 运行baseline

```sh
python run_baselines.py -c asserts/env_config_tpl.yaml
```

3. 以OpenAI Gym形式调用

```python
import cache_emu

extra_params = {
    "main_tag": "main_tag_name",
    "sub_tag" "sub_tag_name"
}

env = CacheEnv(**config, **extra_params) # config内容参考asserts/env_config_tpl.yaml，可使用yaml读取
observation = env.reset()
while not terminal:
   action = agent.forward(observation)
   next_observation, reward, terminal, info = env.step(action)
   agent.backward(reward, terminal, next_observation)
   observation = next_observation
```
