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

4. 项目结构
```
.
├── README.md               //说明文件
├── asserts                 //资源目录
│   ├── configs             //存放配置信息
│   │   ├── _data               //数据配置
│   │   ├── _feature            //特征配置
│   │   ├── _runner             //缓存替换算法配置，例如使用哪些算法进行测试
│   │   ├── capacity_iqiyi12.yaml         //爱奇艺POI数据集（将北京分为12x12块）上不同容量的多节点训练的配置文件
│   │   ├── capacity_zipf_dynamic10.yaml  //在流行度变化的Zipf数据集(流行度变化快)上不同容量的多节点训练的配置文件
│   │   ├── capacity_zipf_dynamic2.yaml   //在流行度变化的Zipf数据集(流行度变化慢)训练的配置文件
│   │   ├── capacity_zipf_static.yaml     //在流行度不变的Zipf数据集训练的配置文件
│   │   ├── default.yaml                  //默认配置，用于参考
│   │   ├── ewdrl_dynamic10.yaml          //
│   │   ├── ewdrl_dynamic2.yaml
│   │   ├── ewdrl_iqiyi12.yaml
│   │   └── ewdrl_iqiyi12_kd_different_capacity.yaml
│   └── data
│       ├── iqiyi_pois
│       └── zipf
├── drl_agent               //Listwise DRL算法包
│   ├── __init__.py 
│   ├── config.py           //DRL智能体默认参数，用处不大，现在参数主要使用yaml从文件中读入
│   ├── ewdrl           
│   │   ├── __init__.py
│   │   ├── agent          //分为ewdqn和eqdnn，两者区别是一个是输出大小为2（两个动作,存与不存）的DNN，另一个是单一输出节点的DNN
│   │   ├── core.py        //DRL抽象接口
│   │   ├── memory.py      //Replay Buffer，用于存储决策路径/经验
│   │   ├── model.py       //常用深度强化学习算法的模型结构,如DQN/DDPG/SAC/SQL等
│   │   ├── nn.py          //最基础的DNN模型
│   │   └── policy.py      //策略函数，根据深度模型的输出得到动作
│   ├── kd                 //知识蒸馏模块，不同的蒸馏方法
│   │   ├── __init__.py     
│   │   ├── common.py      //蒸馏方法的通用接口
│   │   ├── kd_model1.py    //蒸馏方法1
│   │   ├── kd_model2.py    //蒸馏方法2
│   │   ├── kd_model3.py    //蒸馏方法3
│   │   ├── kd_model4.py    //蒸馏方法4
│   │   └── kd_model5.py    //蒸馏方法5
│   ├── kd_callback.py      //知识蒸馏回调类，通过重写可以修改蒸馏方法
│   └── rl_runner.py        //将DRL Agent封装成一个进程接口，方便同时启动多个进程
├── py_cache_emu            //缓存环境
│   ├── __init__.py
│   ├── cache.py            //缓存类
│   ├── callback.py         //缓存回调函数接口，如如何记录日志、如何保存参数等，知识蒸馏回调类也继承该类
│   ├── envs.py             //OpenAI Gym接口，通过将环境封装成这类接口，可以方便使用现有的RL库进行测试
│   ├── feature             //特征包，主要决定RL Agent在决策过程中需要使用哪些类型的特征
│   │   ├── common.py   
│   │   ├── extractors.py
│   │   └── manager.py
│   ├── misc                //别人实现的一些缓存替换算法
│   │   └── arc.py          //Adaptive Replacement Cache
│   ├── request.py          //请求序列加载/生成类
│   ├── runners.py          //传统缓存替换算法进程类，和RL Runner类似
│   └── utils               //工具包
│       ├── __init__.py
│       ├── log_utils.py    //日志工具包
│       ├── mp_utils.py     //多进程通信工具包
│       ├── numpy_utils.py  //numpy工具包
│       ├── proj_utils.py   //项目工具包，如路劲操作、文件读写等
│       └── torch_utils.py  //pytorch工具包，将一些常用的操作进行封装
├── requirements.txt        //项目包依赖，可以使用pip install -r .安装本项目需要的安装包
├── run_batch_gnode1.sh     //批处理脚本，主要用于测试时同时跑不同参数的实验
├── run_batch_gnode2.sh     //同上
├── run_cache.py            //实验入口文件
├── run_rllib.py            //使用ray rllib中的RL算法做缓存决策
├── setup.py                //安装缓存包py_cache_emu到Python环境中
└── td3                     //TD3强化学习算法，暂时用不上
    ├── __init__.py
    ├── env_wrapper.py
    ├── memory.py
    └── td3.py
```