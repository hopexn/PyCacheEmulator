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
pip install -e .
```

2. 运行
```sh
python test.py -c asserts/env_config_tpl.yaml
```